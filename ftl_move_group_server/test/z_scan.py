import asyncio
import threading
import time
from copy import deepcopy

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from std_msgs.msg import Header, String
from follow_the_leader_msgs.srv import Move2State, Move2Pose, FollowPath
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from moveit_msgs.srv import GetRobotStateFromWarehouse, GetCartesianPath, GetPositionFK


class ZscanService(Node):
    def __init__(self):
        super().__init__("Zscan_service")

        # Setup orchestrator service
        self.srv = self.create_service(Trigger, "/zscan", self.Zscan_callback)

        self.robot_state = RobotState()

        self.create_subscription(JointState, "/joint_states", self.curr_joint_state, 1)

        # Setup clients
        self.getstate_cli = self.create_client(
            GetRobotStateFromWarehouse, "/get_robot_state"
        )
        self.move2state_cli = self.create_client(Move2State, "/move2state")
        self.move2pose_cli = self.create_client(Move2Pose, "/move2pose")
        self.getdrawtraj_cli = self.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )
        self.getpose_cli = self.create_client(GetPositionFK, "/compute_fk")
        self.movecartesian_cli = self.create_client(FollowPath, "/followpath")

        while not self.getstate_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("getstate_cli service not available, waiting...")

        while not self.move2state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("move2state_cli service not available, waiting...")

        while not self.move2pose_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("move2pose_cli service not available, waiting...")

        while not self.getdrawtraj_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "compute_cartesian_path service not available, waiting..."
            )

        while not self.movecartesian_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "movecartesian_cli service not available, waiting..."
            )

    def Zscan_callback(self, request, response):
        threading.Thread(
            target=self.threaded_Zscan_request, args=(request, response)
        ).start()
        return response

    def threaded_Zscan_request(self, request, response):
        # Create a new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Now you can run the async method in this loop
        loop.run_until_complete(self.handle_Zscan_request(request, response))
        loop.close()

    def compute_pose_array(self, start_posestamped: PoseStamped) -> list:
        # red - x green - y blue -z
        path_pose_array = []
        for i in range(7):
            start_posestamped.pose.position.x += 0.01
            pose = deepcopy(start_posestamped.pose)
            path_pose_array.append(pose)
        for i in range(7):
            start_posestamped.pose.position.z += 0.01
            pose = deepcopy(start_posestamped.pose)
            path_pose_array.append(pose)
        for i in range(7):
            start_posestamped.pose.position.x -= 0.01
            pose = deepcopy(start_posestamped.pose)
            path_pose_array.append(pose)
        for i in range(7):
            start_posestamped.pose.position.z += 0.01
            pose = deepcopy(start_posestamped.pose)
            path_pose_array.append(pose)
        return path_pose_array

    async def handle_Zscan_request(self, request, response):

        req = GetPositionFK.Request()
        req.header = Header(frame_id="world")
        req.fk_link_names = ["mock_pruner__camera0"]
        req.robot_state = self.robot_state
        getpose_future = self.getpose_cli.call_async(req)
        await getpose_future
        if (
            getpose_future.result() is None
            or getpose_future.result().error_code.val != MoveItErrorCodes.SUCCESS
        ):
            self.get_logger().info(
                "Service call failed %r" % (getpose_future.exception(),)
            )
            return

        path_pose_array = self.compute_pose_array(
            getpose_future.result().pose_stamped[0]
        )
        # # go to pose 1
        # self.get_logger().info("Move to pose 1")
        # req = Move2Pose.Request()
        # req.goal_state.pose = path_pose_array[1]
        # req.goal_state.header = getpose_future.result().pose_stamped[0].header
        # move2pose_future = self.move2pose_cli.call_async(req)
        # await move2pose_future
        # self.get_logger().info("Move to pose returned")
        # if (
        #     move2pose_future.result() is None
        #     or move2pose_future.result().state is False
        # ):
        #     self.get_logger().info(
        #         "Service call failed %r" % (move2pose_future.exception(),)
        #     )
        #     return

        self.get_logger().info("Compute cartesian path")
        req = GetCartesianPath.Request()
        req.header = getpose_future.result().pose_stamped[0].header
        req.start_state = self.robot_state
        req.group_name = "ur5e__pruning_robot_manipulator"
        req.link_name = "mock_pruner__camera0"
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.waypoints = path_pose_array
        getdrawtraj_future = self.getdrawtraj_cli.call_async(req)
        await getdrawtraj_future
        if getdrawtraj_future.result() is None:
            self.get_logger().info(
                "Service call failed %r" % (getdrawtraj_future.exception())
            )
        elif getdrawtraj_future.result().error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().info(
                f"Error Code {getdrawtraj_future.result().error_code}"
            )
            return
        self.get_logger().info(
            f"Fraction of cartesian path successfully planned {getdrawtraj_future.result().fraction}"
        )

        time.sleep(10)
        self.get_logger().info("Execute cartesian path")
        req = FollowPath.Request()
        req.robot_trajectory = getdrawtraj_future.result().solution
        movecartesian_future = self.movecartesian_cli.call_async(req)
        await movecartesian_future
        if (
            movecartesian_future.result() is None
            or movecartesian_future.result().state is False
        ):
            self.get_logger().info(
                "Service call failed %r" % (movecartesian_future.exception(),)
            )
            return

        self.get_logger().info("COMPLETE")

    def curr_joint_state(self, msg):
        self.robot_state.joint_state = msg


def main(args=None):
    # Initialize rclpy
    rclpy.init(args=args)

    # Create service
    Zscan_service = ZscanService()

    # Spin the service
    rclpy.spin(Zscan_service)

    # Shutdown rclpy
    rclpy.shutdown()

if __name__ == "__main__":
    main()
