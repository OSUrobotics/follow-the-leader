#!/usr/bin/env python3
import time
from copy import deepcopy

import numpy as np
import rclpy
import rclpy.action
import rclpy.parameter
import rclpy.type_support
from cv_bridge import CvBridge
from follow_the_leader.curve_fitting import Bezier, BezierBasedDetection
from follow_the_leader_msgs.msg import StateTransition
from geometry_msgs.msg import (Point, Pose, PoseStamped, Quaternion, Transform,
                               TransformStamped, TwistStamped, Vector3,
                               Vector3Stamped)
from rclpy.action import ActionServer
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Header

bridge = CvBridge()

from threading import Lock

import tf2_geometry_msgs
import transforms3d as t3d
from controller_manager_msgs.srv import SwitchController
from follow_the_leader.utils.fov_distance import constrained_dist
from follow_the_leader.utils.ros_utils import TFNode, process_list_as_dict
from follow_the_leader.utils.speed_overlap import max_speed
from follow_the_leader_msgs.action import RotateAroundPoint
from follow_the_leader_msgs.msg import ControllerParams, States, TreeModel
from follow_the_leader_msgs.srv import Move2Pose
from geometry_msgs.msg import PointStamped
from rclpy.task import Future
from scipy.spatial.transform import Rotation
from std_msgs.msg import ColorRGBA, Empty
from std_srvs.srv import SetBool, Trigger
from tf2_geometry_msgs import do_transform_point, do_transform_vector3
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from transforms3d import _gohlketransforms as gt
from visualization_msgs.msg import Marker, MarkerArray


class FollowTheLeaderController_3D_ROS(TFNode):
    """
    This node handles taking in the 3D curve models (simply a list of 3D points)
    and outputting a velocity for the end-effector that moves up the branch while centering the camera.
    """

    def __init__(self):
        super().__init__(
            "ftl_controller_3d", cam_info_topic="/camera/color/camera_info"
        )
        # Config
        params = {
            "log_path": "/tmp",
            "base_frame": "base_link",
            "tool_frame": "tool0",
            "default_vector": "up",
            "min_height": 0.325,
            "max_height": 0.75,
            "k_centering": 1.0,
            "k_z": 1.0,
            "smallest_feature_size": 5e-3,  # smallest feature size in meters should be discernible on image
            "feature_px": 5,  # how many pixels should smallest feature occupy
            "desired_fov": 0.5,  # desired field of view in meters
            # the following parameters are used to calculate the end effector speed see utils/speed_overlap.py
            "overlap_ratio_mask": 0.8,  # overlap between consecutive mask images
            "overlap_ratio_pixel_tracking": 0.95,  # overlap between consecutive pixel tracking images
            "frame_processing_delay": 0.2,
            "pan_magnitude_deg": 15.0,
            "pan_frequency": 0.0,
            "rotation_speed": 0.0,
            "lookat": False,
            "use_mock_hardware": True,
        }
        self.declare_parameter_dict(**params)

        # calculate params z_desired
        new_z = 0.4
        try:
            new_z = constrained_dist(
                self.camera,
                self.get_param_val("desired_fov"),
                self.get_param_val("feature_px"),
                self.get_param_val("smallest_feature_size"),
            )
        except Exception as e:
            self.get_logger().error(f"Error calculating constrained distance: {e}")
            new_z = 0.1
        self.declare_parameter("z_desired", new_z)
        self.get_logger().info(f"Calculated z_desired: {new_z}")

        # calculate params ee_speed
        min_frame_overlap = min(
            self.get_param_val("overlap_ratio_mask"),
            self.get_param_val("overlap_ratio_pixel_tracking"),
        )

        calculated_speed = max_speed(
            self.camera,
            self.get_param_val("z_desired"),
            min_frame_overlap,
            processing_time=self.get_param_val("frame_processing_delay"),
        )
        self.declare_parameter("ee_speed", calculated_speed)
        self.get_logger().info(f"Calculated ee_speed {self.get_param_val('ee_speed')}")

        if self.get_param_val("use_mock_hardware"):
            self._move_group_controller = "joint_trajectory_controller"
        else:
            self._move_group_controller = "scaled_joint_trajectory_controller"
        self._servo_controller = "forward_velocity_controller"

        self.base_frame = self.get_param_val("base_frame")
        self.tool_frame = self.get_param_val("tool_frame")

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
        self.pan_reference = None  # TODO delete
        self.twist_cmd_to_publish = None

        # ROS2 setup
        self.service_handler_group = ReentrantCallbackGroup()
        self.curve_subscriber_group = ReentrantCallbackGroup()
        self.motion_group = MutuallyExclusiveCallbackGroup()

        self.curve_sub = self.create_subscription(
            TreeModel,
            "/tree_model",
            self.process_curve,
            1,
            callback_group=self.curve_subscriber_group,
        )
        self.pose_pub = self.create_publisher(PoseStamped, "/camera_pose", 1)
        self.lookat_pub = self.create_publisher(PoseStamped, "/lookat_pose", 1)
        self.servo_command_publisher = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10
        )
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
        self.diagnostic_pub = self.create_publisher(
            MarkerArray, "controller_diagnostic", 1
        )
        self.lock = Lock()
        self.timer = self.create_timer(0.01, self.compute_new_twist)
        self.pub_timer = self.create_timer(
            1 / 200,
            self.publish_twist_callback,
            callback_group=self.service_handler_group,
        )

        self._srv_client_start_servo = self.create_client(
            srv_type=Trigger,
            srv_name="/servo_node/start_servo",
            callback_group=self.service_handler_group,
        )

        self._srv_client_stop_servo = self.create_client(
            srv_type=Trigger,
            srv_name="/servo_node/stop_servo",
            callback_group=self.service_handler_group,
        )

        self._srv_switch_ctrls = self.create_client(
            srv_type=SwitchController,
            srv_name="/controller_manager/switch_controller",
            callback_group=self.service_handler_group,
        )
        self.move2pose_cli = self.create_client(
            Move2Pose, "/move2pose", callback_group=self.service_handler_group
        )
        # Action server for rotation
        self._action_server = ActionServer(
            self, RotateAroundPoint, "rotate_around_point", self.execute_rotation, callback_group=self.motion_group
        )

        self.reset()

        print("Done loading")
        return

    def cam_to_pose(self, pose):
        cam_pose = self.convert_tf_to_pose(self.lookup_transform(self.base_frame, self.camera.tf_frame, time=None))
        return self.euclidean_dist_between_poses(cam_pose, pose)

    def get_empty_twist(self, frame_id="tool0"):
        twist = TwistStamped()
        twist.header.frame_id = frame_id
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
        return twist

    async def async_wait_for(self, rel_time: float):
        fut = Future()
        timer = None

        def done_waiting():
            fut.set_result(None)
            timer.cancel()

        timer = self.create_timer(rel_time, done_waiting)
        await fut

    def handle_params_update(self, msg: ControllerParams):
        for key in self.params:
            self.params[key] = getattr(msg, key)
        return

    def handle_state_transition(self, msg: StateTransition):
        action = process_list_as_dict(msg.actions, "node", "action").get(
            self.get_name()
        )
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
            raise ValueError(
                "Unknown action {} for node {}".format(action, self.get_name())
            )
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
            self.twist_cmd_to_publish = None
        return

    def start(self):
        # Initialize movement based on the current location of the arm
        pos = self.lookup_transform(
            self.base_frame, self.tool_frame, sync=False, as_matrix=True
        )[:3, 3]
        x, y, z = pos[0], pos[1], pos[2]
        lower_dist = z - self.get_param_val("min_height")
        upper_dist = self.get_param_val("max_height") - z

        tf = self.lookup_transform(
            self.base_frame, self.camera.tf_frame, sync=False, as_matrix=True
        )
        self.up = upper_dist > lower_dist
        self.init_tf = tf
        self.pan_reference = None
        if self.get_param_val("default_vector") == "up":
            self.default_action = (
                np.array([0, -1, 0]) if self.up else np.array([0, 1, 0])
            )
        elif self.get_param_val("default_vector") == "right":
            self.default_action = (
                np.array([0, 1, 0]) if self.up else np.array([0, -1, 0])
            )                                  

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
        # if not self.active:
        #     return

        if msg.header.frame_id != self.camera.tf_frame:
            print("Warning! The frame IDs of the 3D curve and camera did not match")
            return

        stamp = msg.header.stamp
        tf = self.lookup_transform(
            self.base_frame, msg.header.frame_id, time=stamp, as_matrix=True
        )
        curve_pts = np.array([[p.x, p.y, p.z] for p in msg.points])
        ids = msg.ids

        try:
            assert len(curve_pts) == len(ids)
        except AssertionError:
            self.get_logger().error("Curve points and IDs do not match!")

        self.branch_idxs = []
        current_id = -1
        # create list of lists for branch and points
        for i, id in enumerate(ids):
            if id != current_id:
                current_id = id
                self.branch_idxs.append([])
            self.branch_idxs[-1].append(i)
        self.get_logger().warn(f"Branch indices: {self.branch_idxs} {ids}")

        if not curve_pts.size:
            return

        if tf is None:
            return

        curve_pts_base = self.mul_homog(tf, curve_pts)
        self.last_curve_pts = curve_pts_base

        if self.pan_reference is None:
            self.pan_reference = tf[:3, 3]
        return

    def publish_twist_callback(self):
        if self.twist_cmd_to_publish is None:
            return

        if self.twist_cmd_to_publish.header.frame_id != self.tool_frame:
            self.twist_cmd_to_publish = self.transform_twist(
                self.twist_cmd_to_publish, self.tool_frame
            )

        self.twist_cmd_to_publish.header.stamp = self.get_clock().now().to_msg()
        self.servo_command_publisher.publish(self.twist_cmd_to_publish)
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

        pos = self.lookup_transform(
            self.base_frame, self.tool_frame, sync=False, as_matrix=True
        )[:3, 3]
        if (self.up and pos[2] >= self.get_param_val("max_height")) or (
            not self.up and pos[2] <= self.get_param_val("min_height")
        ):
            self.stop()
            return

        current_stamp = self.get_clock().now().to_msg()
        current_tf = self.lookup_transform(
            self.base_frame, self.camera.tf_frame, time=current_stamp, as_matrix=True
        )

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

        twist_tool = self.transform_twist_for_tool(current_stamp, vel, angular_vel)

        self.twist_cmd_to_publish = twist_tool
        self.publish_markers(current_stamp)
        return

    def update_pan_target(self, tf):
        if (
            self.pan_reference is None
        ):  # Model has not yet been initialized, keep doing default movements
            return

        # Scanning upwards - Check if we've moved far enough to start panning
        if not self.arm_is_rotating:
            vertical_move = abs(self.pan_reference[2] - tf[2, 3])
            if vertical_move > self.mode_switch_dist:
                theta = self.get_rotation_target(tf)
                print("Rotation target: {:.1f} degrees".format(np.degrees(theta)))
                z = self.get_param_val["z_desired"]
                init_frame_offset_vector = np.array(
                    [z * np.sin(theta), 0, -z * np.cos(theta)]
                )
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
        theta_mag = np.radians(self.get_param_val("pan_magnitude_deg"))

        # Rotation movement starts at center, goes to right, goes to center, goes to left
        self.rotation_stage = (self.rotation_stage + 1) % 4
        target_angle = 0.0
        target_interval = [
            -theta_mag / 2,
            theta_mag / 2,
        ]  # The range of angles the robot arm can assume

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

            branch_vec = (
                self.init_tf[:3, :3].T @ branch_vec_base
            )  # In the frame of the initial transform
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
    ACTION SERVER METHODS
    """
    async def execute_rotation(self, goal_handle: rclpy.action.server.ServerGoalHandle):
        self.get_logger().info("Executing rotation action")
        # if not self.arm_is_rotating:
        #     self.get_logger().error("Arm is not rotating")
        #     goal_handle.abort()
        #     result_msg = RotateAroundPoint.Result()
        #     result_msg.error_message = "Arm is not rotating"
        #     return result_msg

        result_msg = RotateAroundPoint.Result()
        if goal_handle.is_cancel_requested:
            self.get_logger().info("Rotation action cancelled")
            goal_handle.canceled()
            result_msg.error_message = "cancelled"
            return result_msg

        if not goal_handle.is_active:
            error_msg = "Goal is not active"
            self.get_logger().error(error_msg)
            result_msg.error_message = error_msg
            return result_msg

        feedback_msg = RotateAroundPoint.Feedback()
        feedback_msg.stage = "Action started"
        goal_handle.publish_feedback(feedback_msg)

        lookat_pose_stamped = PoseStamped()
        time_at = goal_handle.request.header.stamp.sec
        lookat_pose_stamped.header = goal_handle.request.header
        lookat_pose_stamped.pose.position = goal_handle.request.target_point
        lookat_pose_stamped.pose.orientation = goal_handle.request.branch_axis
        lookat_radius = goal_handle.request.radius
        lookat_angle = goal_handle.request.angle
        if lookat_pose_stamped.header.frame_id != self.base_frame:
            lookat_pose_stamped = self.tf_buffer.transform(
                lookat_pose_stamped, self.base_frame
            )

        try:
            # stop servoing and switch controllers
            self.twist_cmd_to_publish = None
            self._srv_client_stop_servo.call_async(request=Trigger.Request())
            switch_ctrlr_req = SwitchController.Request(
                activate_controllers=[self._move_group_controller],
                deactivate_controllers=[self._servo_controller],
                strictness=SwitchController.Request.STRICT,
            )
            ctrl_req = self._srv_switch_ctrls.call_async(request=switch_ctrlr_req)

            # creates a new lookat_frame where the lookat rotation is intended around the z axis
            lookat_tf = self.convert_pose_to_tf(lookat_pose_stamped)
            lookat_tf.header.frame_id = self.base_frame
            lookat_tf.child_frame_id = f"lookat"
            self.tf_buffer.set_transform_static(lookat_tf, "lookat_controller")
            self.static_tf_broadcaster.sendTransform(lookat_tf)
            base_to_lookat = self.lookup_transform(
                self.base_frame, lookat_tf.child_frame_id, as_matrix=True
            )

            # corrects lookat frame so x axis points in new camera z direction
            # new camera z direction points to projection in xy of lookat of the vector from point in base
            projection_in_z = gt.projection_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
            new_x = (
                projection_in_z
                @ base_to_lookat
                @ np.array(
                    [
                        lookat_pose_stamped.pose.position.x,
                        lookat_pose_stamped.pose.position.y,
                        lookat_pose_stamped.pose.position.z,
                        1.0,
                    ]
                )
            ).T

            new_x = (new_x / new_x[-1])[:3]
            new_x = new_x / np.linalg.norm(new_x)
            quat = self.align_vector_with_quaternion(np.array([1.0, 0.0, 0.0]), new_x, normed=True)

            lookat_tf = TransformStamped()
            lookat_tf.header.frame_id = f"lookat"
            lookat_tf.child_frame_id = f"corrected_lookat"
            lookat_tf.transform.translation.x = 0.
            lookat_tf.transform.translation.y = 0.
            lookat_tf.transform.translation.z = 0.
            lookat_tf.transform.rotation = quat
            self.tf_buffer.set_transform_static(lookat_tf, "lookat_controller")
            self.static_tf_broadcaster.sendTransform(lookat_tf)
            feedback_msg.stage = "Lookat frame created"
            goal_handle.publish_feedback(feedback_msg)

            # go to a pose from where anticlockwise rotation starts
            start_pose = PoseStamped()
            start_pose.header.frame_id = lookat_tf.child_frame_id
            sweep_center = np.array([-lookat_radius, 0.0, 0.0])
            rotation_start = t3d.axangles.axangle2mat(
                np.array([0.0, 0.0, 1.0]), -lookat_angle / 2
            )  # half clockwise rotation around z axis
            sweep_start = rotation_start @ sweep_center
            t3d_quat = t3d.quaternions.mat2quat(rotation_start)
            start_pose.pose.position = Point(
                x=sweep_start[0], y=sweep_start[1], z=sweep_start[2]
            )
            start_pose.pose.orientation = Quaternion(x=t3d_quat[1], y=t3d_quat[2], z=t3d_quat[3], w=t3d_quat[0]) # t3d convention
            start_pose = self.tf_buffer.transform(start_pose, self.base_frame)
            self.lookat_pub.publish(start_pose)

            stop_pose = deepcopy(start_pose)
            rotation_stop = t3d.axangles.axangle2mat(
                np.array([0.0, 0.0, 1.0]), lookat_angle / 2
            )
            sweep_stop = rotation_stop @ sweep_center
            t3d_quat = t3d.quaternions.mat2quat(rotation_start)
            stop_pose.pose.position = Point(
                x=sweep_stop[0], y=sweep_stop[1], z=sweep_stop[2]
            )
            stop_pose.pose.orientation = Quaternion(
                x=t3d_quat[1], y=t3d_quat[2], z=t3d_quat[3], w=t3d_quat[0]
            )  # t3d convention

            # TODO if not at lookat pose, go to lookat pose
            self.get_logger().info("Moving to start pose")
            feedback_msg.stage = "Starting move to pose client"
            goal_handle.publish_feedback(feedback_msg)
            req = Move2Pose.Request()
            req.goal_state.pose = start_pose.pose
            req.planner_id = "RRTConnectkConfigDefault"
            req.goal_state.header = start_pose.header

            await ctrl_req
            if ctrl_req.result() is None or ctrl_req.result() is False:
                raise RuntimeError("Switch controller failed")

            move2pose_future = self.move2pose_cli.call_async(req)
            await move2pose_future
            self.get_logger().info("Move to pose returned")
            if (
                move2pose_future.result() is None
                or move2pose_future.result().state is False
            ):
                self.get_logger().info(
                    "Service call failed %r" % (move2pose_future.exception())
                )
                raise RuntimeError(
                    "Move to pose failed: %r" % (move2pose_future.exception())
                )

            feedback_msg.stage = "Finished moving to start pose"
            goal_handle.publish_feedback(feedback_msg)

            switch_ctrlr_req = SwitchController.Request(
                activate_controllers=[self._servo_controller],
                deactivate_controllers=[self._move_group_controller],
                strictness=SwitchController.Request.STRICT,
            )
            self._srv_switch_ctrls.call_async(request=switch_ctrlr_req)
            servo_future = self._srv_client_start_servo.call_async(
                request=Trigger.Request()
            )
            await servo_future
            if servo_future.result() is None or servo_future.result() is False:
                raise RuntimeError("Start servo failed")
            
            time.sleep(1.0)
            # Publish twists to rotate around lookat point
            feedback_msg.stage = "Starting rotation"
            goal_handle.publish_feedback(feedback_msg)
            twist_msg = TwistStamped()
            r_speed = self.get_param_val("ee_speed")
            self.get_logger().info(f"{r_speed}")
            twist_msg.twist.angular = Vector3(x=0.0, y=0.0, z=r_speed)
            twist_msg.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
            twist_msg.header.frame_id = lookat_tf.child_frame_id
            twist_msg.header.stamp = self.get_clock().now().to_msg()

            time_needed = lookat_angle / r_speed  # TODO track angle
            self.get_logger().info(f"Time needed: {time_needed}")
            self.twist_cmd_to_publish = self.transform_twist(twist_msg, self.tool_frame)
            while self.cam_to_pose(stop_pose) > 0.01:
                time.sleep(0.001)
            self.twist_cmd_to_publish = self.get_empty_twist(self.tool_frame)
            feedback_msg.stage = "Finished rotation"
            goal_handle.publish_feedback(feedback_msg)

            switch_ctrlr_req = SwitchController.Request(
                activate_controllers=[self._move_group_controller],
                deactivate_controllers=[self._servo_controller],
                strictness=SwitchController.Request.STRICT,
            )
            ctrl_req = self._srv_switch_ctrls.call_async(request=switch_ctrlr_req)
            await ctrl_req

        except Exception as e:
            error_msg = f"{e}"
            self.get_logger().error(error_msg)
            goal_handle.abort()
            result_msg.error_message = error_msg
            return result_msg

        self.get_logger().info("Rotation action completed")
        self.to_publish = None
        goal_handle.succeed()
        result_msg.success = True
        return result_msg

    """
    CURVE ANALYZING METHODS
    """

    def get_vel_from_curve(self, tf):
        """
        Uses the current model of the leader plus the current transform to obtain a velocity vector for moving.
        """

        if self.last_curve_pts is None or len(self.last_curve_pts) < 2:
            return self.default_action * self.get_param_val(
                "ee_speed"
            ) / np.linalg.norm(self.default_action), np.zeros(3)

        target_pt, target_px, target_t, curve = self.get_targets_from_curve(
            np.linalg.inv(tf)
        )
        grad = curve.tangent(target_t)

        # Computes velocity vector based on 3D curve gradient, pixel difference, and desired distance difference
        cx = self.camera.width / 2
        x_diff = np.array([(target_px[0] - cx) / (self.camera.width / 2), 0, 0])
        grad = grad / np.linalg.norm(grad) * self.get_param_val("ee_speed")
        z_diff = np.array([0, 0, target_pt[2] - self.get_param_val("z_desired")])
        linear_vel = (
            grad
            + x_diff * self.get_param_val("k_centering")
            + z_diff * self.get_param_val("k_z")
        )
        linear_vel = (
            linear_vel / np.linalg.norm(linear_vel) * self.get_param_val("ee_speed")
        )

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

        curve_pts_optical = self.mul_homog(
            cam_base_tf_mat, self.last_curve_pts[self.branch_idxs[0]]
        )
        curve_3d = self.get_curve_3d(curve_pts_optical)

        ts = np.linspace(0, 1, num=samples + 1)
        pts = curve_3d(ts)
        pxs = self.camera.project3dToPixel(pts)
        cy = self.camera.height / 2
        idx_c = np.argmin(np.abs(pxs[:, 1] - cy))
        return pts[idx_c], pxs[idx_c], ts[idx_c], curve_3d

    def get_panning_vel(self, tf):
        inv_tf = np.linalg.inv(tf)  # base in current cam
        cam_target, lookat_target = self.pan_reference
        cam_target_cam, lookat_target_cam = self.mul_homog(
            inv_tf, [cam_target, lookat_target]
        )
        linear_vel = (
            cam_target_cam
            / np.linalg.norm(cam_target_cam)
            * self.get_param_val("rotation_speed")
        )
        angular_vel = self.compute_lookat_rotation(
            lookat_target_cam, linear_vel, k_adjust=0.5
        )

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
        if self.get_param_val("pan_frequency") == 0:
            return np.inf
        return self.camera.getDeltaY(
            self.camera.height,
            self.get_param_val("z_desired") / (self.get_param_val("pan_frequency") * 2),
        )

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
            Point(x=0.0, y=0.0, z=self.get_param_val("z_desired")),
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
    """TODO: Define this method"""
    R = T[:3, :3]
    p = T[:3, 3]
    final = np.zeros((6, 6))
    final[:3, :3] = R
    final[3:6, 3:6] = R
    final[3:6, :3] = skew_sym(p) @ R

    return final


def skew_sym(x):
    if len(x) != 3:
        raise ValueError(
            "Skew symmetric representation is only valid on a vector of length 3"
        )
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = FollowTheLeaderController_3D_ROS()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.dump_params(node.get_param_val("log_path"))
        # do custom cleanup
        node.destroy_node()
        rclpy.shutdown()
    return


if __name__ == "__main__":
    main()
