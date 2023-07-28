
import rclpy
from std_msgs.msg import Header, Empty, Int16
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
from ur_msgs.msg import IOStates
from sensor_msgs.msg import Joy, JointState
from functools import partial
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from follow_the_leader_msgs.msg import States

class Button:
    def __init__(self, off_state=False, on_callback=None, off_callback=None, switch_on_callback=None, switch_off_callback=None):
        self.off_state = off_state
        self.current_state = off_state

        # Callbacks
        self.on_callback = on_callback
        self.off_callback = off_callback
        self.switch_on_callback = switch_on_callback
        self.switch_off_callback = switch_off_callback

    def process(self, state):
        is_on = state != self.off_state
        if self.current_state != state:
            if is_on:
                self.run_callback(self.switch_on_callback)
            else:
                self.run_callback(self.switch_off_callback)
            self.current_state = state
        else:
            if is_on:
                self.run_callback(self.on_callback)
            else:
                self.run_callback(self.off_callback)

    def run_callback(self, cb):
        if cb is not None:
            cb()


class Axis:

    def __init__(self, low_deadzone, high_deadzone, low_callback=None, high_callback=None):
        self.low_dz = low_deadzone
        self.high_dz = high_deadzone
        self.low_callback = low_callback
        self.high_callback = high_callback

        self.current_state = 0

    def process(self, state):
        mode = 0
        if state <= self.low_dz:
            mode = -1
        if state >= self.high_dz:
            mode = 1

        if mode != self.current_state:
            self.current_state = mode
            if mode == 1 and self.high_callback is not None:
                self.high_callback()
            elif mode == -1 and self.low_callback is not None:
                self.low_callback()


class IOManager(Node):
    def __init__(self):
        super().__init__('io_manager')

        self.service_cb = ReentrantCallbackGroup()
        self.state_publisher = self.create_publisher(States, 'state_announcement', 1)
        self.joint_pub = self.create_publisher(JointState, '/move_joints', 1)
        self.action_pub = self.create_publisher(Int16, '/joy_action', 1)
        self.reset_tree_srv = self.create_client(Trigger, '/initialize_tree_spindle', callback_group=self.service_cb)

        self.buttons = {
            0: Button(off_state=False, switch_on_callback=self.send_stop),
            1: Button(off_state=False, switch_on_callback=self.send_start),
            10: Button(off_state=False, switch_on_callback=self.reset_simulated_tree),
            11: Button(off_state=False, switch_on_callback=self.send_joints_home),
            13: Button(off_state=False, switch_on_callback=partial(self.send_joy_action, 0)),
        }

        self.axes = {
            4: Axis(-0.99, 0.99, low_callback=partial(self.send_joy_action, 2), high_callback=partial(self.send_joy_action, -2)),
            5: Axis(-0.99, 0.99, low_callback=partial(self.send_joy_action, -1), high_callback=partial(self.send_joy_action, 1))
        }

        self.button_sub = self.create_subscription(Joy, '/joy', self.handle_joy, 1, callback_group=self.service_cb)


    def handle_io(self, msg: IOStates):
        for pin_msg in msg.digital_in_states:
            pin = pin_msg.pin
            if pin in self.buttons:
                self.buttons[pin].process(pin_msg.state)

    def handle_joy(self, msg: Joy):
        for i, state in enumerate(msg.buttons):
            if i in self.buttons:
                self.buttons[i].process(bool(state))

        for i, state in enumerate(msg.axes):
            if i in self.axes:
                self.axes[i].process(state)

    def send_joy_action(self, val):
        self.action_pub.publish(Int16(data=val))

    def send_start(self):
        self.state_publisher.publish(States(state=States.LEADER_SCAN))
        print('Sent start request!')


    def send_stop(self):
        self.state_publisher.publish(States(state=States.IDLE))
        print('Sent stop request!')

    def reset_simulated_tree(self):
        if self.reset_tree_srv.wait_for_service(timeout_sec=0.5):
            self.reset_tree_srv.call_async(Trigger.Request())
            print('Reset tree!')
        else:
            print('Reset tree service is not available')

    def send_joints_home(self):
        self.joint_pub.publish(JointState())

def main(args=None):
    rclpy.init(args=args)
    node = IOManager()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)


if __name__ == '__main__':
    main()