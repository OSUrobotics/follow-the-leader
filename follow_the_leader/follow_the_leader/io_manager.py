
import rclpy
from std_msgs.msg import Header, Empty
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
from ur_msgs.msg import IOStates
from functools import partial
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

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


class IOManager(Node):
    def __init__(self):
        super().__init__('io_manager')

        self.service_cb = ReentrantCallbackGroup()
        self.abort_pub = self.create_publisher(Empty, '/abort', 1)
        self.start_servo = self.create_client(Trigger, '/servo_3d_start', callback_group=self.service_cb)

        self.buttons = {
            0: Button(off_state=True, switch_on_callback=self.send_abort),
            1: Button(off_state=True, switch_on_callback=self.send_servo_request),
        }
        self.io_sub = self.create_subscription(IOStates, '/io_and_status_controller/io_states', self.handle_io, 1)

    def handle_io(self, msg: IOStates):
        for pin_msg in msg.digital_in_states:
            pin = pin_msg.pin
            if pin in self.buttons:
                self.buttons[pin].process(pin_msg.state)

    def send_abort(self):
        self.abort_pub.publish(Empty())
        print('Sent abort msg')

    def send_servo_request(self):
        self.start_servo.call(Trigger.Request())
        print('Sent servo start command!')

    def dummy(self, msg):
        print(msg)


def main(args=None):
    rclpy.init(args=args)
    node = IOManager()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)


if __name__ == '__main__':
    main()