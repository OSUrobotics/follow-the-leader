#!/usr/bin/env python3
import os.path

import rclpy
from tf2_geometry_msgs import do_transform_vector3
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, TwistStamped, Vector3, Vector3Stamped
from cv_bridge import CvBridge
from follow_the_leader_msgs.msg import (
    Point2D,
    TrackedPointGroup,
    TrackedPointRequest,
    Tracked3DPointGroup,
    Tracked3DPointResponse,
    VisualServoingRequest,
    States,
    StateTransition,
)
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from follow_the_leader.utils.ros_utils import TFNode, process_list_as_dict, call_service_synced
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration

bridge = CvBridge()


class VisualServoingNode(TFNode):
    def __init__(self):
        super().__init__("visual_servoing_node", cam_info_topic="/camera/color/camera_info")

        # Point tracking params
        self.fwd_speed = self.declare_parameter("forward_speed", 0.15)
        self.max_speed = self.declare_parameter("max_speed", 0.20)
        self.k_img = self.declare_parameter("k_img", 1.0)
        self.max_reproj_ignore_threshold = self.declare_parameter("reprojection_error_ignore", 4.0)
        self.stop_dist = self.declare_parameter("stop_dist", 0.15)
        self.base_frame = self.declare_parameter("base_frame", "base_link")
        self.tool_frame = self.declare_parameter("tool_frame", "tool0")
        self.target_frame = self.declare_parameter("target_frame", "")
        self.no_3d_est_scale = self.declare_parameter("no_3d_est_scale", 0.2)
        self.servo_joint_state_dist = self.declare_parameter("servo_joint_state_dist", 0.005)

        # Point tracking state variables
        self.active = False
        self.image_target = None
        self.current_px_estimate = None
        self.current_3d_estimate = None
        self.servo_joint_states = []
        self.last_joint_msg = None
        self.last_tool_pos = None
        self.return_state = States.IDLE

        # ROS2 utils

        self.cb_group = ReentrantCallbackGroup()
        self.point_tracking_name = "vs"
        self.servoing_sub = self.create_subscription(
            VisualServoingRequest,
            "/visual_servoing_request",
            self.handle_servoing_request,
            1,
            callback_group=self.cb_group,
        )
        self.point_response_sub = self.create_subscription(
            Tracked3DPointResponse, "point_tracking_response", self.handle_3d_point_tracking_response, 1
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "joint_states", self.handle_joint_state, 1, callback_group=self.cb_group
        )
        self.transition_sub = self.create_subscription(
            StateTransition, "state_transition", self.handle_state_transition, 1, callback_group=self.cb_group
        )
        self.point_tracking_pub = self.create_publisher(TrackedPointRequest, "point_tracking_request", 1)
        self.servo_pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.state_announce_pub = self.create_publisher(States, "state_announcement", 1)
        self.resource_sync_client = self.create_client(Trigger, "await_resource_ready", callback_group=self.cb_group)
        self.moveit_client = ActionClient(self, ExecuteTrajectory, "execute_trajectory")

        self.timer = self.create_timer(0.01, self.send_servo_command)
        return

    def handle_state_transition(self, msg: StateTransition):
        if msg.state_end == States.VISUAL_SERVOING:
            self.return_state = msg.state_start
        return

    def handle_servoing_request(self, msg: VisualServoingRequest):
        print("Received servoing request!")

        self.image_target = np.array([msg.image_target.x, msg.image_target.y])
        self.state_announce_pub.publish(States(state=States.VISUAL_SERVOING))
        call_service_synced(self.resource_sync_client, Trigger.Request())

        req = TrackedPointRequest()
        req.action = TrackedPointRequest.ACTION_REPLACE
        req.groups.append(TrackedPointGroup(name=self.point_tracking_name, points=msg.points))
        self.point_tracking_pub.publish(req)

        self.active = True
        self.current_px_estimate = np.array([[msg.points[0].x, msg.points[0].y]])
        return

    def handle_rewind(self):
        if not self.servo_joint_states:
            print("No joint states recorded! Rewind is complete")
            self.state_announce_pub.publish(States(state=self.return_state))
            self.reset()
            return

        print("Now rewinding...")
        self.active = False
        self.servo_joint_states.append(self.last_joint_msg)

        self.state_announce_pub.publish(States(state=States.VISUAL_SERVOING_REWIND))
        call_service_synced(self.resource_sync_client, Trigger.Request())

        # Construct the JointTrajectory and send it to the trajectory handler to execute
        ts = self.servo_joint_state_dist.value / self.max_speed.value
        traj = JointTrajectory()
        traj.joint_names = self.servo_joint_states[-1].name
        for i, joints in enumerate(self.servo_joint_states[::-1]):
            start_sec, start_rem = divmod(i * ts, 1)
            duration = Duration(seconds=int(start_sec), nanoseconds=int(start_rem * 1e9))
            traj.points.append(JointTrajectoryPoint(positions=joints.position, time_from_start=duration.to_msg()))
        traj.header.stamp = self.get_clock().now().to_msg()

        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = RobotTrajectory(joint_trajectory=traj)
        self.moveit_client.wait_for_server()
        future = self.moveit_client.send_goal_async(goal_msg)
        future.add_done_callback(self.rewind_done_callback)
        return

    def rewind_done_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            raise Exception("Goal was not accepted!")

        print("Rewind complete! Returning to state {}".format(self.return_state))
        self.state_announce_pub.publish(States(state=self.return_state))
        self.reset()
        return

    def reset(self):
        self.active = False
        self.image_target = None
        self.current_px_estimate = None
        self.current_3d_estimate = None
        self.servo_joint_states = []
        self.last_joint_msg = None
        self.last_tool_pos = None
        self.return_state = States.IDLE
        return

    def handle_3d_point_tracking_response(self, msg: Tracked3DPointResponse):
        for group in msg.groups_2d:
            if group.name == self.point_tracking_name:
                self.current_px_estimate = np.array([[p.x, p.y] for p in group.points])

        for group in msg.groups:
            if group.name == self.point_tracking_name:
                self.current_3d_estimate = np.array([[p.x, p.y, p.z] for p in group.points])
                print("Updated 3D est! Now {:.3f}, {:.3f}, {:.3f}".format(*self.current_3d_estimate[0]))
        return

    def handle_joint_state(self, msg: JointState):
        if not self.active or not msg.position:
            return

        self.last_joint_msg = msg
        cur_pos = self.lookup_transform(
            self.base_frame.value, self.tool_frame.value, rclpy.time.Time(), sync=False, as_matrix=True
        )[:3, 3]
        if (
            self.last_tool_pos is None
            or np.linalg.norm(cur_pos - self.last_tool_pos) > self.servo_joint_state_dist.value
        ):
            self.last_tool_pos = cur_pos
            self.servo_joint_states.append(msg)
        return

    def send_servo_command(self):
        if not self.active:
            return

        if self.current_px_estimate is None:
            print("Warning! Visual servoing is active but there is no 2D estimate? This shouldn't be the case")
            return

        est_2d = self.current_px_estimate[0]
        offscreen = est_2d[0] < 0 or est_2d[1] < 0 or est_2d[0] > self.camera.width or est_2d[1] > self.camera.height

        if self.current_3d_estimate is None:
            print("Warning! No 3D estimate, and no safeguards to stop excessive servoing")
        else:
            est_3d = self.current_3d_estimate[0]
            if est_3d[2] <= self.stop_dist.value:
                print("Done")
                self.handle_rewind()
                return

        if not self.target_frame.value:
            # Pure image-space visual servoing
            if offscreen:
                print("Target is offscreen, ending visual servoing!")
                self.handle_rewind()
                return

            base_vec = np.array(self.camera.projectPixelTo3dRay(est_2d))
            base_vec *= self.fwd_speed.value / base_vec[2]

            if self.current_3d_estimate is None:
                base_vec *= self.no_3d_est_scale.value

        else:
            # Base vector is derived from the 3D estimate and the target frame
            raise NotImplementedError()

        tracked_px = self.current_px_estimate[0]
        diff = (tracked_px - self.image_target) / np.array(
            [self.camera.width, self.camera.height]
        )  # Error in each dim is in [-1, 1]

        img_diff_vec = np.array([diff[0], diff[1], 0]) * self.k_img.value

        final_vec = base_vec + img_diff_vec
        norm = np.linalg.norm(final_vec)
        if norm > self.max_speed.value:
            final_vec *= self.max_speed.value / norm

        self.send_tool_frame_command(final_vec)
        return

    def send_tool_frame_command(self, vec_array, frame=None):
        if frame is None:
            frame = self.camera.tf_frame

        vec = Vector3Stamped()
        vec.header.frame_id = frame
        vec.vector = Vector3(x=vec_array[0], y=vec_array[1], z=vec_array[2])
        tool_tf = self.lookup_transform(self.tool_frame.value, frame, time=rclpy.time.Time(), sync=False)
        tool_frame_vec = do_transform_vector3(vec, tool_tf)

        twist = TwistStamped()
        twist.header.frame_id = self.tool_frame.value
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.twist.linear = tool_frame_vec.vector
        self.servo_pub.publish(twist)
        return


def main(args=None):
    try:
        rclpy.init(args=args)
        executor = MultiThreadedExecutor()
        node = VisualServoingNode()
        rclpy.spin(node, executor=executor)
    finally:
        node.destroy_node()
    return


if __name__ == "__main__":
    main()
