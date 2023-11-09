#!/usr/bin/env python3
"""
Add safety plane collision objects to the planning scene monitor
"""
import os.path
import sys
import rclpy

from geometry_msgs.msg import Pose
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.srv import ApplyPlanningScene

from rclpy.logging import get_logger
from rclpy.executors import MultiThreadedExecutor

from follow_the_leader.utils.ros_utils import TFNode


class SafetyPlaneNode(TFNode):
    def __init__(self):
        super().__init__("safety_plane_node")
        self.psm_diff_pub = self.create_publisher(PlanningScene, "planning_scene", 1)
        self.psm_cli = self.create_client(ApplyPlanningScene, "apply_planning_scene")
        while not self.psm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again')
        self.get_logger().info('found service')

    def create_plane(self):
        plane_positions = [
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.7),
            (0.0, 0.0, 0.9),
            (0.0, 0.0, 1.1),
        ]
        plane_dimensions = [
            (1.0, 1.0, 0.01),
            (2.0, 2.0, 0.01),
            (3.0, 3.0, 0.01),
            (4.0, 4.0, 0.01),
        ]

        collision_object = CollisionObject()
        collision_object.header.frame_id = "tool0"
        collision_object.id = "safety_plane"
        for position, dimensions in zip(plane_positions, plane_dimensions):
            box_pose = Pose()
            box_pose.position.x = position[0]
            box_pose.position.y = position[1]
            box_pose.position.z = position[2]

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = dimensions
            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD
        
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)

        psm_request = ApplyPlanningScene.Request()
        psm_request.scene = planning_scene
        self.future = self.psm_cli.call_async(psm_request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    try:
        rclpy.init(args=args)
        executor = MultiThreadedExecutor()
        node = SafetyPlaneNode()
        if node.create_plane() == False:
            node.get_logger().info('failed')
        else:
            node.get_logger().info('safety planes created')
    finally:
        node.destroy_node()
    return

if __name__ == "__main__":
    main()
