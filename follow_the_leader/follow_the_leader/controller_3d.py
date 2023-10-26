#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import (
    TwistStamped,
    Vector3,
    Vector3Stamped,
    Transform,
    TransformStamped,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
)
import numpy as np
from follow_the_leader.curve_fitting import BezierBasedDetection, Bezier
from follow_the_leader_msgs.msg import StateTransition
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

from std_msgs.msg import Empty, ColorRGBA
from follow_the_leader.utils.ros_utils import TFNode, process_list_as_dict
from tf2_geometry_msgs import do_transform_vector3, do_transform_point
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from follow_the_leader_msgs.msg import TreeModel, States, ControllerParams
from threading import Lock
from scipy.spatial.transform import Rotation


class FollowTheLeaderController_3D_ROS(TFNode):
    """
    This node handles taking in the 3D curve models (simply a list of 3D points)
    and outputting a velocity for the end-effector that moves up the branch while centering the camera.
    """
    def __init__(self):
        super().__init__("ftl_controller_3d", cam_info_topic="/camera/color/camera_info")
        # Config
        self.base_frame = self.declare_parameter("base_frame", "base_link")
        self.tool_frame = self.declare_parameter("tool_frame", "tool0")
        self.min_height = self.declare_parameter("min_height", 0.325)
        self.max_height = self.declare_parameter("max_height", 0.75)
        self.ee_speed = self.declare_parameter("ee_speed", 0.60)
        self.k_centering = self.declare_parameter("k_centering", 1.0)
        self.k_z = self.declare_parameter("k_z", 1.0)
        self.z_desired = self.declare_parameter("z_desired", 0.20)
        self.pan_magnitude_deg = self.declare_parameter("pan_magnitude_deg", 15.0)
        self.pan_frequency = self.declare_parameter("pan_frequency", 1.5)
        self.rotation_speed = self.declare_parameter("rotation_speed", 0.25)

        # State variables
        self.active = False
        self.up = False
        self.init_tf = None
        self.default_action = None
        self.branch_idxs = []
        self.last_curve_pts = None
        self.paused = False
        self.arm_is_rotating = False
        self.rotation_stage = 0
        self.pan_reference = None
        self.to_publish = None
        self.params = {
            "pan_frequency": self.pan_frequency.value,
            "pan_magnitude_deg": self.pan_magnitude_deg.value,
            "z_desired": self.z_desired.value,
            "ee_speed": self.ee_speed.value,
        }

        # ROS2 setup
        self.service_handler_group = ReentrantCallbackGroup()
        self.curve_subscriber_group = ReentrantCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

        self.curve_sub = self.create_subscription(
            TreeModel, "/tree_model", self.process_curve, 1, callback_group=self.curve_subscriber_group
        )
        self.pose_pub = self.create_publisher(PoseStamped, "/camera_pose", 1)
        self.pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.state_announce_pub = self.create_publisher(States, "state_announcement", 1)
        self.params_sub = self.create_subscription(
            ControllerParams,
            "/controller_params",
            self.handle_params_update,
            1,
            callback_group=self.service_handler_group,
        )
        self.transition_sub = self.create_subscription(
            StateTransition,
            "state_transition",
            self.handle_state_transition,
            1,
            callback_group=self.service_handler_group,
        )
        self.diagnostic_pub = self.create_publisher(MarkerArray, "controller_diagnostic", 1)
        self.lock = Lock()
        self.timer = self.create_timer(0.01, self.compute_new_twist)
        self.pub_timer = self.create_timer(
            1 / 200, self.publish_twist_callback, callback_group=self.service_handler_group
        )
        self.reset()

        print("Done loading")
        return

    def handle_params_update(self, msg: ControllerParams):
        for key in self.params:
            self.params[key] = getattr(msg, key)
        return

    def handle_state_transition(self, msg: StateTransition):
        action = process_list_as_dict(msg.actions, "node", "action").get(self.get_name())
        if not action:
            return
    
        if action == "activate":
            if not self.active:
                self.start()
        elif action == "reset":
            if self.active:
                self.stop()
        elif action == "pause":
            if self.active:
                self.pause()
        elif action == "resume":
            if self.active:
                self.resume()

        else:
            raise ValueError("Unknown action {} for node {}".format(action, self.get_name()))
        return

    def reset(self):
        with self.lock:
            self.active = False
            self.default_action = None
            self.up = False
            self.init_tf = None
            self.branch_idxs = []
            self.last_curve_pts = None
            self.paused = False
            self.rotation_stage = 0
            self.arm_is_rotating = False
            self.pan_reference = None
            self.to_publish = None
        return

    def start(self):
        # Initialize movement based on the current location of the arm
        pos = self.lookup_transform(self.base_frame.value, self.tool_frame.value, sync=False, as_matrix=True)[:3, 3]
        z = pos[2]
        lower_dist = z - self.min_height.value
        upper_dist = self.max_height.value - z

        tf = self.lookup_transform(self.base_frame.value, self.camera.tf_frame, sync=False, as_matrix=True)
        self.up = upper_dist > lower_dist
        self.init_tf = tf
        self.pan_reference = None
        self.default_action = np.array([0, -1, 0]) if self.up else np.array([0, 1, 0])

        self.active = True
        self.paused = False
        print("Servoing started!")
        return

    def stop(self):
        if self.active:
            self.reset()
            self.state_announce_pub.publish(States(state=States.IDLE))
            msg = "Servoing stopped!"
            print(msg)
        return

    def pause(self):
        self.paused = True
        return

    def resume(self):
        self.paused = False
        return

    def process_curve(self, msg: TreeModel):
        if not self.active:
            return

        if msg.header.frame_id != self.camera.tf_frame:
            print("Warning! The frame IDs of the 3D curve and camera did not match")
            return

        stamp = msg.header.stamp
        tf = self.lookup_transform(self.base_frame.value, msg.header.frame_id, time=stamp, as_matrix=True)
        curve_pts = np.array([[p.x, p.y, p.z] for p in msg.points])
        ids = msg.ids

        assert len(curve_pts) == len(ids)

        self.branch_idxs = []
        current_id = -1
        for i, id in enumerate(ids):
            if id != current_id:
                current_id = id
                self.branch_idxs.append([])
            self.branch_idxs[-1].append(i)

        if not curve_pts.size:
            return

        curve_pts_base = self.mul_homog(tf, curve_pts)
        self.last_curve_pts = curve_pts_base

        if self.pan_reference is None:
            self.pan_reference = tf[:3, 3]
        return

    def publish_twist_callback(self):
        """Publish a TwistStamped msg"""
        if self.to_publish is None:
            return

        twist_tool = self.to_publish

        cmd = TwistStamped()
        cmd.header.frame_id = self.tool_frame.value
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear = Vector3(x=twist_tool[3], y=twist_tool[4], z=twist_tool[5])
        cmd.twist.angular = Vector3(x=twist_tool[0], y=twist_tool[1], z=twist_tool[2])

        self.pub.publish(cmd)
        return

    def compute_new_twist(self):
        """
        This is the function that gets repeatedly called to output velocity commands.
        The idea is to check if we should stop scanning, and if not, determine the appropriate
        velocity to output.

        There are two "modes" the robot can be in: Either it is attempting to move up the branch,
        or it is attempting to pivot around the current branch target.
        """

        if self.paused or not self.active:
            return

        if not self.camera.tf_frame:
            print("No camera frame has been received!")
            return

        # Check for termination

        pos = self.lookup_transform(self.base_frame.value, self.tool_frame.value, sync=False, as_matrix=True)[:3, 3]
        if (self.up and pos[2] >= self.max_height.value) or (not self.up and pos[2] <= self.min_height.value):
            self.stop()
            return

        current_stamp = self.get_clock().now().to_msg()
        current_tf = self.lookup_transform(
            self.base_frame.value, self.camera.tf_frame, time=current_stamp, as_matrix=True
        )

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame.value
        pose.header.stamp = current_stamp
        tl = current_tf[:3, 3]
        quat = Rotation.from_matrix(current_tf[:3, :3]).as_quat()
        pose.pose.position = Point(x=tl[0], y=tl[1], z=tl[2])
        pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        self.pose_pub.publish(pose)

        with self.lock:
            if self.paused or not self.active:
                return

            self.update_pan_target(current_tf)

            if not self.arm_is_rotating:
                # Scanning up the branch
                # Grab the latest curve model and use it to output a control velocity for the robot
                vel, angular_vel = self.get_vel_from_curve(current_tf)
            else:
                # Rotating the camera around the lookat target
                # Move the camera towards the desired target
                vel, angular_vel = self.get_panning_vel(current_tf)

        if vel is None:
            self.stop()
            return

        tool_tf = self.lookup_transform(self.tool_frame.value, self.camera.tf_frame, time=current_stamp, as_matrix=True)
        twist = np.concatenate([angular_vel, vel])
        twist_tool = adjunct(tool_tf) @ twist

        self.to_publish = twist_tool
        self.publish_markers(current_stamp)
        return

    def update_pan_target(self, tf):
        if self.pan_reference is None:  # Model has not yet been initialized, keep moving up
            return

        # Scanning upwards - Check if we've moved far enough to start panning
        if not self.arm_is_rotating:
            vertical_move = abs(self.pan_reference[2] - tf[2, 3])
            if vertical_move > self.mode_switch_dist:
                theta = self.get_rotation_target(tf)
                print("Rotation target: {:.1f} degrees".format(np.degrees(theta)))
                z = self.params["z_desired"]
                init_frame_offset_vector = np.array([z * np.sin(theta), 0, -z * np.cos(theta)])
                base_offset_vector = self.init_tf[:3, :3] @ init_frame_offset_vector
                base_target = self.mul_homog(tf, [0, 0, z])

                self.arm_is_rotating = True
                self.pan_reference = (base_target + base_offset_vector, base_target)

        else:
            # Arm is currently rotating - Check to make sure if it has moved beyond the desired target position

            cam_target = self.pan_reference[0]
            cam_frame_ref = self.mul_homog(np.linalg.inv(tf), cam_target)
            move_dir = 1 if self.rotation_stage in {0, 1} else -1
            if np.sign(cam_frame_ref[0]) != move_dir:
                self.arm_is_rotating = False
                self.pan_reference = tf[:3, 3]
        return

    def get_rotation_target(self, tf_base_cam):
        theta_mag = np.radians(self.params["pan_magnitude_deg"])

        # Rotation movement starts at center, goes to right, goes to center, goes to left
        self.rotation_stage = (self.rotation_stage + 1) % 4
        target_angle = 0.0
        target_interval = [-theta_mag / 2, theta_mag / 2]  # The range of angles the robot arm can assume

        if self.rotation_stage == 1:
            target_angle = theta_mag
            target_interval = [0, theta_mag]
        elif self.rotation_stage == 3:
            target_angle = -theta_mag
            target_interval = [-theta_mag, 0]

        # Check to see if there are any side branches that are sticking out towards the camera
        # Angles are defined in the -z/x plane
        if not self.branch_idxs[1:]:
            return target_angle

        tf_cam_base = np.linalg.inv(tf_base_cam)
        curve_pts_optical = self.mul_homog(tf_cam_base, self.last_curve_pts)

        branch_angles = []

        print("Num side branches: {}".format(len(self.branch_idxs) - 1))
        for branch_idx in self.branch_idxs[1:]:
            branch_pts_optical = curve_pts_optical[branch_idx]
            px_start = self.camera.project3dToPixel(branch_pts_optical[0])
            px_end = self.camera.project3dToPixel(branch_pts_optical[-1])

            for px in [px_start, px_end]:
                if 0 <= px[0] <= self.camera.width and 0 <= px[1] <= self.camera.height:
                    break
            else:
                # Both branch points are out of frame
                continue

            branch_pts_base = self.last_curve_pts[branch_idx]
            branch_vec_base = branch_pts_base[0] - branch_pts_base[-1]
            branch_vec_base = branch_vec_base / np.linalg.norm(branch_vec_base)

            branch_vec = self.init_tf[:3, :3].T @ branch_vec_base  # In the frame of the initial transform
            theta = np.arctan2(branch_vec[0], -branch_vec[2])

            if target_angle - theta_mag / 2 <= theta <= target_angle + theta_mag:
                branch_angles.append(theta)

        if not branch_angles:
            return target_angle

        # If there are branches that may stick out towards the camera,
        # Find the angle for the camera that maximizes the viewing angle to any branch

        # Candidates for target angles - the interval endpoints or the midpoints in between branches
        branch_angles = np.array(sorted(branch_angles))
        candidate_angles = [target_interval[0], target_interval[1]]
        for angle_start, angle_end in zip(branch_angles[:-1], branch_angles[1:]):
            candidate_angles.append((angle_start + angle_end) / 2)

        best_angle_dist = None
        best_angle = None
        for angle in candidate_angles:
            angle_dist = np.min(np.abs(branch_angles - angle))
            if best_angle_dist is None or angle_dist > best_angle_dist:
                best_angle_dist = angle_dist
                best_angle = angle

        return best_angle

    """
    CURVE ANALYZING METHODS
    """

    def get_vel_from_curve(self, tf):
        """
        Uses the current model of the leader plus the current transform to obtain a velocity vector for moving.
        """

        if self.last_curve_pts is None or len(self.last_curve_pts) < 2:
            return self.default_action * self.params["ee_speed"] / np.linalg.norm(self.default_action), np.zeros(3)

        target_pt, target_px, target_t, curve = self.get_targets_from_curve(np.linalg.inv(tf))
        grad = curve.tangent(target_t)

        # Computes velocity vector based on 3D curve gradient, pixel difference, and desired distance difference
        cx = self.camera.width / 2
        x_diff = np.array([(target_px[0] - cx) / (self.camera.width / 2), 0, 0])
        grad = grad / np.linalg.norm(grad) * self.params["ee_speed"]
        z_diff = np.array([0, 0, target_pt[2] - self.params["z_desired"]])
        linear_vel = grad + x_diff * self.k_centering.value + z_diff * self.k_z.value
        linear_vel = linear_vel / np.linalg.norm(linear_vel) * self.params["ee_speed"]

        # No rotation during scanning
        angular_vel = np.zeros(3)
        return linear_vel, angular_vel

    def get_curve_3d(self, pts_3d):
        """
        Assumes all the points have already been transformed into the camera frame.
        """
        assert len(pts_3d) > 1

        curve_start_idx = None
        curve_end_idx = None

        for idx, pt in enumerate(pts_3d):
            px = self.camera.project3dToPixel(pt)
            if 0 <= px[0] < self.camera.width and 0 <= px[1] < self.camera.height:
                if curve_start_idx is None:
                    curve_start_idx = max(idx - 1, 0)
                curve_end_idx = min(idx + 1, len(pts_3d) - 1)
            else:
                if curve_end_idx:
                    break
        if curve_start_idx is None:
            # No pixels were in the image - Just take the first and last pixels
            curve_start_idx = 0
            curve_end_idx = len(pts_3d) - 1

        pts_to_fit = pts_3d[curve_start_idx : curve_end_idx + 1]
        return Bezier.fit(pts_to_fit, degree=min(3, len(pts_to_fit) - 1))

    def get_targets_from_curve(self, cam_base_tf_mat, samples=100):
        """
        Determines which 3D point in the current model is closest to the vertical center of the camera frame.
        Returns the point, pixel, curve t-value, and curve associated with the vertical center.
        """

        if self.last_curve_pts is None or len(self.last_curve_pts) < 2:
            return None

        curve_pts_optical = self.mul_homog(cam_base_tf_mat, self.last_curve_pts[self.branch_idxs[0]])
        curve_3d = self.get_curve_3d(curve_pts_optical)

        ts = np.linspace(0, 1, num=samples + 1)
        pts = curve_3d(ts)
        pxs = self.camera.project3dToPixel(pts)
        cy = self.camera.height / 2
        idx_c = np.argmin(np.abs(pxs[:, 1] - cy))
        return pts[idx_c], pxs[idx_c], ts[idx_c], curve_3d

    def get_panning_vel(self, tf):
        inv_tf = np.linalg.inv(tf)  # base, current cam
        cam_target, lookat_target = self.pan_reference
        cam_target_cam, lookat_target_cam = self.mul_homog(inv_tf, [cam_target, lookat_target])
        linear_vel = cam_target_cam / np.linalg.norm(cam_target_cam) * self.rotation_speed.value
        angular_vel = self.compute_lookat_rotation(lookat_target_cam, linear_vel, k_adjust=0.5)
        return linear_vel, angular_vel

    @staticmethod
    def compute_lookat_rotation(target, vel, k_adjust=0.0):
        """
        Given a target in the frame of the camera and the desired velocity of the same camera frame,
        computes a rotational speed around the y-axis that will keep the camera looking at the desired target.
        """

        dx, _, dz = target
        vx, _, vz = vel

        # Correction term if the camera is not looking right at the target already
        err = np.pi / 2 - np.arctan2(dz, dx)
        correction = k_adjust * err

        # Derivative of theta with respect to the linear velocity commanded
        d_theta = -(dz * vx - dx * vz) / (dx**2 + dz**2)

        return np.array([0, d_theta + correction, 0])

    """
    UTILITIES
    """

    @property
    def mode_switch_dist(self):
        return self.camera.getDeltaY(self.camera.height, self.params["z_desired"]) / (self.params["pan_frequency"] * 2)

    def publish_markers(self, stamp=None):
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        markers = MarkerArray()
        lookat_marker = Marker()
        lookat_marker.ns = self.get_name()
        lookat_marker.id = 1
        lookat_marker.type = Marker.LINE_LIST
        lookat_marker.header.frame_id = self.camera.tf_frame
        lookat_marker.header.stamp = stamp
        lookat_marker.points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=0.0, y=0.0, z=self.params["z_desired"]),
        ]
        lookat_marker.scale.x = 0.02
        lookat_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        markers.markers.append(lookat_marker)
        self.diagnostic_pub.publish(markers)
        return


def convert_tf_to_pose(tf: TransformStamped):
    pose = PoseStamped()
    pose.header = tf.header
    tl = tf.transform.translation
    pose.pose.position = Point(x=tl.x, y=tl.y, z=tl.z)
    pose.pose.orientation = tf.transform.rotation

    return pose


def adjunct(T):
    R = T[:3, :3]
    p = T[:3, 3]
    final = np.zeros((6, 6))
    final[:3, :3] = R
    final[3:6, 3:6] = R
    final[3:6, :3] = skew_sym(p) @ R

    return final


def skew_sym(x):
    if len(x) != 3:
        raise ValueError("Skew symmetric representation is only valid on a vector of length 3")
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    ctrl = FollowTheLeaderController_3D_ROS()
    rclpy.spin(ctrl, executor)
    return


if __name__ == "__main__":
    main()
