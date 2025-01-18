#!/usr/bin/env python3
import numpy as np
import rclpy
from follow_the_leader.utils.ros_utils import TFNode
from rclpy.action import ActionClient
from follow_the_leader_msgs.action import RotateAroundPoint
from geometry_msgs.msg import PoseStamped


class RotateAroundPointClient(TFNode):

    def __init__(self):
        super().__init__("rotate_around_point_client")
        self._action_client = ActionClient(
            self, RotateAroundPoint, "rotate_around_point"
        )
        self._sub_goal_pose = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.send_goal_from_rviz,
            10
        )

    def send_goal_from_rviz(self, msg):
        self.get_logger().info("Received goal pose, waiting for action server...")
        tf = self.lookup_transform("ur5e__base_link", "mock_pruner__camera0")
        no_rot = tf = self.lookup_transform("ur5e__base_link", "mock_pruner__camera0")
        self.get_logger().info("Transform: %s" % tf)

        goal_msg = RotateAroundPoint.Goal()
        goal_msg.header = msg.header
        goal_msg.target_point = msg.pose.position
        goal_msg.target_point.z = tf.transform.translation.z
        goal_msg.branch_axis.w = 1.0
        goal_msg.angle = np.pi / 6
        goal_msg.radius = 0.1

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("Result: %s" % result)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info("Received feedback: %s" % feedback)

if __name__ == "__main__":
    rclpy.init()
    node = RotateAroundPointClient()
    rclpy.spin(node)
    rclpy.shutdown()
