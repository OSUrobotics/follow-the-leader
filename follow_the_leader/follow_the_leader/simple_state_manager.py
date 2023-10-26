#!/usr/bin/env python3
import rclpy
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from enum import Enum
from follow_the_leader_msgs.msg import StateTransition, NodeAction, States
from controller_manager_msgs.srv import SwitchController, ListControllers
from follow_the_leader.utils.ros_utils import call_service_synced
import re


class ResourceMode(Enum):
    DEFAULT = 0
    SERVO = 1


class SimpleStateManager(Node):
    def __init__(self):
        super().__init__(node_name="simple_state_manager")

        # ROS2 params
        self.base_ctrl = self.declare_parameter(name="base_controller", value=".*joint_trajectory_controller")
        self.servo_ctrl = self.declare_parameter(name="servo_controller", value="forward_position_controller")

        self.base_ctrl_string = None
        self.servo_ctrl_string = None

        # State variables
        self.current_state = States.IDLE
        self.resource_ready = True

        # ROS2 utils
        self.cb = ReentrantCallbackGroup()
        self.pub = self.create_publisher(StateTransition, "state_transition", 1)
        self.sub = self.create_subscription(
            States, "state_announcement", self.handle_state_announcement, 1, callback_group=self.cb
        )
        self.scan_start_srv = self.create_service(srv_type=Trigger, srv_name="scan_start", callback=self.handle_start, callback_group=self.cb)
        self.scan_stop_srv = self.create_service(Trigger, "scan_stop", self.handle_stop, callback_group=self.cb)
        self.reset_srv = self.create_service(Trigger, "reset_state_machine", self.handle_reset, callback_group=self.cb)
        self.enable_servo = self.create_client(Trigger, "/servo_node/start_servo", callback_group=self.cb)
        self.disable_servo = self.create_client(Trigger, "/servo_node/stop_servo", callback_group=self.cb)
        self.switch_ctrl = self.create_client(
            SwitchController, "/controller_manager/switch_controller", callback_group=self.cb
        )
        self.list_ctrl = self.create_client(
            ListControllers, "/controller_manager/list_controllers", callback_group=self.cb
        )
        self.resource_ready = self.create_service(
            Trigger, "await_resource_ready", self.await_resource_ready, callback_group=self.cb
        )
        self.get_ctrl_string_timer = self.create_timer(0.1, self.get_controller_names, callback_group=self.cb)

        # State definitions
        self.nodes = [
            "image_processor_node",
            "curve_3d_model_node",
            "ftl_controller_3d",
            "point_tracker_node",
            "visual_servoing_node",
        ]

        self.transition_table = {
            (States.IDLE, States.LEADER_SCAN): self.activate_all,
            (States.IDLE, States.VISUAL_SERVOING): {},  # The VS response node handles the state publishing
            (States.LEADER_SCAN, States.IDLE): self.reset_all,
            (States.VISUAL_SERVOING, States.IDLE): self.reset_all,
            (States.LEADER_SCAN, States.VISUAL_SERVOING): {
                "ftl_controller_3d": "pause",
                "curve_3d_model_node": "pause",
            },
            (States.VISUAL_SERVOING, States.LEADER_SCAN): {
                "ftl_controller_3d": "resume",
                "curve_3d_model_node": "resume",
            },
            (States.VISUAL_SERVOING, States.VISUAL_SERVOING_REWIND): {},
            (States.VISUAL_SERVOING_REWIND, States.LEADER_SCAN): {
                "ftl_controller_3d": "resume",
                "curve_3d_model_node": "resume",
            },
            (States.VISUAL_SERVOING_REWIND, States.IDLE): {},
        }

        self.resource_modes = {
            States.IDLE: ResourceMode.DEFAULT,
            States.LEADER_SCAN: ResourceMode.SERVO,
            States.VISUAL_SERVOING: ResourceMode.SERVO,
            States.VISUAL_SERVOING_REWIND: ResourceMode.DEFAULT,
        }
        return

    def get_controller_names(self):
        if self.base_ctrl_string is not None:
            self.get_ctrl_string_timer.destroy()

        if not self.list_ctrl.service_is_ready():
            return

        rez = call_service_synced(self.list_ctrl, ListControllers.Request())

        for ctrl in rez.controller:
            if self.base_ctrl_string is None and re.match(self.base_ctrl.value, ctrl.name):
                self.base_ctrl_string = ctrl.name

            if self.servo_ctrl_string is None and re.match(self.servo_ctrl.value, ctrl.name):
                self.servo_ctrl_string = ctrl.name

        if bool(self.base_ctrl_string) ^ bool(self.servo_ctrl_string):
            print("Only was able to match one of the controllers! Not activating")
            self.base_ctrl_string = None
            self.servo_ctrl_string = None

        elif self.base_ctrl_string is not None:
            print("Located controllers! Base: {}, Servo: {}".format(self.base_ctrl_string, self.servo_ctrl_string))
        return

    def handle_state_announcement(self, msg: States):
        new_state = msg.state
        self.handle_state_transition(self.current_state, new_state)
        return

    def handle_start(self, _, resp):
        if self.current_state != States.IDLE:
            msg = "The system is already running! Not doing anything"
            print(msg)
            resp.message = msg
            resp.success = False

        self.handle_state_transition(self.current_state, States.LEADER_SCAN)
        resp.success = True
        return resp

    def handle_stop(self, _, resp):
        if self.current_state == States.IDLE:
            msg = "The system is already idle! Not doing anything"
            print(msg)
            resp.message = msg
            resp.success = False

        self.handle_state_transition(self.current_state, States.IDLE)
        resp.success = True
        return resp

    def handle_state_transition(self, start_state, end_state):
        """Handle the transitions as defined in the transition_table"""
        if start_state == end_state:
            return

        msg = StateTransition()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state_start = start_state
        msg.state_end = end_state
        actions = self.transition_table.get((start_state, end_state), {})
        for node, action in actions.items():
            msg.actions.append(NodeAction(node=node, action=action))
        self.pub.publish(msg)
        self.get_logger().info(f"[DEBUG] STATE TRANSITION FROM {start_state} to {end_state}. Time: {self.get_clock().now()}")

        # Handle resource management - If you depend on a specific resource, call await_resource_ready
        cur_resource_mode = self.resource_modes.get(start_state, None)
        next_resource_mode = self.resource_modes.get(end_state, None)
        if cur_resource_mode is not None and next_resource_mode is not None:
            self.handle_resource_switch(next_resource_mode)
        self.current_state = end_state
        return

    def handle_resource_switch(self, resource_mode):
        if self.base_ctrl_string is None or self.servo_ctrl_string is None:
            raise Exception("Controllers have not been identified yet!")

        self.resource_ready = False
        if resource_mode == ResourceMode.DEFAULT:
            switch_ctrl_req = SwitchController.Request(
                activate_controllers=[self.base_ctrl_string], deactivate_controllers=[self.servo_ctrl_string]
            )

            call_service_synced(self.disable_servo, Trigger.Request())
            call_service_synced(self.switch_ctrl, switch_ctrl_req)

        elif resource_mode == ResourceMode.SERVO:
            switch_ctrl_req = SwitchController.Request(
                activate_controllers=[self.servo_ctrl_string], deactivate_controllers=[self.base_ctrl_string]
            )
            call_service_synced(self.enable_servo, Trigger.Request())
            call_service_synced(self.switch_ctrl, switch_ctrl_req)

        else:
            raise ValueError("Unknown resource mode {} specified!".format(resource_mode))
        self.resource_ready = True
        return

    def await_resource_ready(self, _, resp):
        rate = self.create_rate(100)
        if not self.resource_ready:
            rate.sleep()
        resp.success = True
        return resp

    @property
    def activate_all(self):
        return {n: "activate" for n in self.nodes}

    @property
    def reset_all(self):
        return {n: "reset" for n in self.nodes}

    def handle_reset(self, *args):
        self.current_state = States.IDLE
        self.handle_resource_switch(ResourceMode.DEFAULT)

        if len(args) == 2:
            resp = args[1]
            resp.success = True
            return resp
        return


def main(args=None):
    rclpy.init(args=args)
    node = SimpleStateManager()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    return


if __name__ == "__main__":
    main()
