#!/usr/bin/env python

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import TwistStamped, Vector3, Vector3Stamped, Transform, TransformStamped, Point, Pose, PoseStamped
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
from follow_the_leader_msgs.msg import PointList, States
from threading import Lock


class FollowTheLeaderController_3D_ROS(TFNode):

    def __init__(self):
        super().__init__('ftl_controller_3d', cam_info_topic='/camera/color/camera_info')
        # Config

        self.base_frame = self.declare_parameter('base_frame', 'base_link')
        self.tool_frame = self.declare_parameter('tool_frame', 'tool0')
        self.min_height = self.declare_parameter('min_height', 0.325)
        self.max_height = self.declare_parameter('max_height', 0.55)
        self.ee_speed = self.declare_parameter('ee_speed', 0.15)
        self.k_centering = self.declare_parameter('k_centering', 1.0)
        self.k_z = self.declare_parameter('k_z', 1.0)
        self.z_desired = self.declare_parameter('z_desired', 0.20)
        self.lookat = self.declare_parameter('lookat', True)
        self.pan_amplitude = self.declare_parameter('pan_amplitude', 0.075)
        self.pan_period = self.declare_parameter('pan_period', 0.20)

        self.debug_mode = self.declare_parameter('debug_mode', True)

        # State variables
        self.active = False
        self.up = False
        self.init_tf = None
        self.default_action = None
        self.last_curve_pts = None
        self.paused = False

        # ROS2 setup
        self.service_handler_group = ReentrantCallbackGroup()
        self.curve_subscriber_group = ReentrantCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

        self.curve_sub = self.create_subscription(PointList, '/curve_3d', self.process_curve, 1, callback_group=self.curve_subscriber_group)
        self.pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.state_announce_pub = self.create_publisher(States, 'state_announcement', 1)
        self.reset_model_pub = self.create_publisher(Empty, '/reset_model', 1)
        self.transition_sub = self.create_subscription(StateTransition, 'state_transition',
                                                       self.handle_state_transition, 1, callback_group=self.service_handler_group)
        self.diagnostic_pub = self.create_publisher(MarkerArray, 'controller_diagnostic', 1)
        self.lock = Lock()
        self.timer = self.create_timer(0.01, self.twist_callback)
        self.reset()

        if self.debug_mode.value:
            self.load_dummy_camera()

        print('Done loading')

    def handle_state_transition(self, msg: StateTransition):
        action = process_list_as_dict(msg.actions, 'node', 'action').get(self.get_name())
        if not action:
            return

        if action == 'activate':
            if not self.active:
                self.start()
        elif action == 'reset':
            if self.active:
                self.stop()
        elif action == 'pause':
            if self.active:
                self.pause()
        elif action == 'resume':
            if self.active:
                self.resume()

        else:
            raise ValueError('Unknown action {} for node {}'.format(action, self.get_name()))

    def reset(self):
        with self.lock:
            self.active = False
            self.default_action = None
            self.up = False
            self.init_tf = None
            self.last_curve_pts = None
            self.paused = False
            self.reset_model_pub.publish(Empty())

    def start(self):

        # Initialize movement based on the current location of the arm
        pos = self.lookup_transform(self.base_frame.value, self.tool_frame.value, sync=False, as_matrix=True)[:3,3]
        z = pos[2]
        lower_dist = z - self.min_height.value
        upper_dist = self.max_height.value - z

        tf = self.lookup_transform(self.base_frame.value, self.camera.tf_frame, sync=False, as_matrix=True)
        self.up = upper_dist > lower_dist
        self.init_tf = np.linalg.inv(tf)
        self.default_action = np.array([0, -1, 0]) if self.up else np.array([0, 1, 0])

        if self.debug_mode.value:
            self.load_dummy_curve(tf)
            print('Loaded dummy curve!')

        self.active = True
        self.paused = False
        print('Servoing started!')

    def stop(self):
        if self.active:
            self.reset()
            self.state_announce_pub.publish(States(state=States.IDLE))
            msg = 'Servoing stopped!'
            print(msg)
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False

    def process_curve(self, msg: PointList):
        if not self.active:
            return

        if msg.header.frame_id != self.camera.tf_frame:
            print('Warning! The frame IDs of the 3D curve and camera did not match')
            return

        stamp = msg.header.stamp
        tf = self.lookup_transform(self.base_frame.value, msg.header.frame_id, time=stamp, as_matrix=True)
        curve_pts = np.array([[p.x, p.y, p.z] for p in msg.points])
        if not curve_pts.size:
            return

        curve_pts_base = self.mul_homog(tf, curve_pts)
        self.last_curve_pts = curve_pts_base
        print('Curve points processed!')

    def twist_callback(self):
        if self.paused or not self.active:
            return

        if not self.camera.tf_frame:
            print('No camera frame has been received!')
            return

        # Check for termination

        pos = self.lookup_transform(self.base_frame.value, self.tool_frame.value, sync=False, as_matrix=True)[:3,3]
        if self.up:
            print('[DEBUG] Moving up, cur Z = {:.2f}, max Z = {:.2f}'.format(pos[2], self.max_height.value))
        else:
            print('[DEBUG] Moving down, cur Z = {:.2f}, max Z = {:.2f}'.format(pos[2], self.min_height.value))

        if (self.up and pos[2] >= self.max_height.value) or (not self.up and pos[2] <= self.min_height.value):
            self.stop()
            return

        # Grab the latest curve model and use it to output a control velocity for the robot
        current_stamp = self.get_clock().now().to_msg()
        with self.lock:
            vel, angular_vel = self.get_vel_from_curve(current_stamp)

        if vel is None:
            self.stop()
            return

        tool_tf = self.lookup_transform(self.tool_frame.value, self.camera.tf_frame, time=current_stamp,
                                        as_matrix=True)
        twist = np.concatenate([angular_vel, vel])
        twist_tool = adjunct(tool_tf) @ twist

        cmd = TwistStamped()
        cmd.header.frame_id = self.tool_frame.value
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear = Vector3(x=twist_tool[3], y=twist_tool[4], z=twist_tool[5])
        cmd.twist.angular = Vector3(x=twist_tool[0], y=twist_tool[1], z=twist_tool[2])

        self.pub.publish(cmd)
        self.publish_markers(current_stamp)

        print('[DEBUG] Sent vel command: {:.3f}, {:.3f}, {:.3f}'.format(*vel))

    """
    CURVE ANALYZING METHODS
    """

    def get_vel_from_curve(self, stamp=None):
        """
        Uses the current model of the leader plus the current transform to obtain a velocity vector for moving.
        """

        if self.last_curve_pts is None:
            return self.default_action * self.ee_speed.value / np.linalg.norm(self.default_action), np.zeros(3)

        base_cam_tf = self.lookup_transform(self.base_frame.value, self.camera.tf_frame, time=stamp, as_matrix=True)
        target_pt, target_px, target_t, curve = self.get_targets_from_curve(np.linalg.inv(base_cam_tf))
        grad = curve.tangent(target_t)

        # Computes velocity vector based on 3D curve gradient, pixel difference, and desired distance difference
        # TODO: If panning but no lookat, the x_diff will always be large!
        cx = self.camera.width / 2
        grad = grad / np.linalg.norm(grad) * self.ee_speed.value
        x_diff = np.array([(target_px[0] - cx) / cx, 0, 0])
        z_diff = np.array([0, 0, target_pt[2] - self.z_desired.value])
        linear_vel = grad + x_diff * self.k_centering.value + z_diff * self.k_z.value

        # Calculate the velocity component associated with the panning motion
        pan_grad = self.get_panning_gradient(base_cam_tf=base_cam_tf)
        linear_vel += pan_grad * abs(linear_vel[1])
        linear_vel = linear_vel / np.linalg.norm(linear_vel) * self.ee_speed.value

        # Based on the desired velocity, compute the angular velocity needed to continue looking at the model
        angular_vel = np.zeros(3)
        if self.lookat.value:
            angular_vel = self.compute_lookat_rotation(target_pt, linear_vel, k_adjust=0.5)

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

        pts_to_fit = pts_3d[curve_start_idx:curve_end_idx+1]
        return Bezier.fit(pts_to_fit, degree=min(3, len(pts_to_fit) - 1))

    def get_targets_from_curve(self, cam_base_tf_mat, samples=100):
        if self.last_curve_pts is None:
            return None

        curve_pts_optical = self.mul_homog(cam_base_tf_mat, self.last_curve_pts)
        curve_3d = self.get_curve_3d(curve_pts_optical)

        ts = np.linspace(0, 1, num=samples + 1)
        pts = curve_3d(ts)
        pxs = self.camera.project3dToPixel(pts)
        cy = self.camera.height / 2
        idx_c = np.argmin(np.abs(pxs[:, 1] - cy))
        return pts[idx_c], pxs[idx_c], ts[idx_c], curve_3d

    def get_panning_gradient(self, base_cam_tf):

        cur_movement = (self.init_tf @ base_cam_tf)[:3,3]
        t = abs(cur_movement[1] / self.pan_period.value)

        if self.debug_mode.value:
            print('Current T: {:.2f}'.format(t))

        coeff = 2 * np.pi / self.pan_period.value
        return np.array([coeff * self.pan_amplitude.value * np.cos(coeff * t), 0, 0])

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
        print('ERROR: {:.1f} degrees'.format(np.degrees(err)))

        # Derivative of theta with respect to the linear velocity commanded
        d_theta = -(dz * vx - dx * vz) / (dx ** 2 + dz ** 2)

        return np.array([0, d_theta + correction, 0])

    """
    UTILITIES
    """

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
            Point(x=0.0,y=0.0,z=0.0),
            Point(x=0.0,y=0.0,z=self.z_desired.value),
        ]
        lookat_marker.scale.x = 0.02
        lookat_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        markers.markers.append(lookat_marker)

        if self.debug_mode.value:
            model_marker = Marker()
            model_marker.ns = self.get_name()
            model_marker.id = 2
            model_marker.type = Marker.LINE_STRIP
            model_marker.header.frame_id = self.base_frame.value
            model_marker.header.stamp = stamp
            model_marker.points = [
                Point(x=p[0], y=p[1], z=p[2]) for p in self.last_curve_pts
            ]
            model_marker.scale.x = 0.02
            model_marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
            markers.markers.append(model_marker)

        self.diagnostic_pub.publish(markers)

    def load_dummy_curve(self, start_tf_mat):
        if self.last_curve_pts is not None:
            return
        base_world_pt = self.mul_homog(start_tf_mat, np.array([0, 0, self.z_desired.value]))
        last_world_pt = base_world_pt.copy()
        if self.up:
            last_world_pt[2] = self.max_height.value + 0.20
        else:
            last_world_pt[2] = self.min_height.value - 0.20
        last_world_pt += np.random.uniform(-1, 1, 3) * np.array([0.05, 0.05, 0])

        # Intermediate points
        pt_1 = base_world_pt + 0.3 * (last_world_pt - base_world_pt) + np.random.uniform(-1, 1, 3) * np.array([0.05, 0.05, 0.0])
        pt_2 = base_world_pt + 0.7 * (last_world_pt - base_world_pt) + np.random.uniform(-1, 1, 3) * np.array([0.05, 0.05, 0])

        curve = Bezier.fit(np.array([base_world_pt, pt_1, pt_2, last_world_pt]))
        self.last_curve_pts = curve(np.linspace(0, 1, 21))


def convert_tf_to_pose(tf: TransformStamped):
    pose = PoseStamped()
    pose.header = tf.header
    tl = tf.transform.translation
    pose.pose.position = Point(x=tl.x, y=tl.y, z=tl.z)
    pose.pose.orientation = tf.transform.rotation

    return pose


def adjunct(T):
    R = T[:3,:3]
    p = T[:3,3]
    final = np.zeros((6,6))
    final[:3,:3] = R
    final[3:6,3:6] = R
    final[3:6,:3] = skew_sym(p) @ R

    return final


def skew_sym(x):
    if len(x) != 3:
        raise ValueError('Skew symmetric representation is only valid on a vector of length 3')
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    ctrl = FollowTheLeaderController_3D_ROS()
    rclpy.spin(ctrl, executor)


if __name__ == '__main__':
    main()
