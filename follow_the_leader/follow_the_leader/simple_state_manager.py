import rclpy
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from enum import Enum
from follow_the_leader_msgs.msg import StateTransition, NodeAction, States


class SimpleStateManager(Node):
    def __init__(self):
        super().__init__('simple_state_manager')

        # State variables
        self.current_state = States.IDLE

        # ROS2 utils
        self.cb = ReentrantCallbackGroup()
        self.pub = self.create_publisher(StateTransition, 'state_transition', 1)
        self.sub = self.create_subscription(States, 'state_announcement', self.handle_state_announcement, 1)
        self.scan_start_srv = self.create_service(Trigger, 'scan_start', self.handle_start)
        self.scan_stop_srv = self.create_service(Trigger, 'scan_stop', self.handle_stop)

        # State definitions
        self.nodes = [
            'image_processor_node',
            'curve_3d_model_node',
            'ftl_controller_3d',
            'point_tracker_node',
        ]

        self.transition_table = {
            (States.IDLE, States.LEADER_SCAN): self.activate_all,
            (States.IDLE, States.VISUAL_SERVOING): self.activate_all,
            (States.LEADER_SCAN, States.IDLE): self.reset_all,
            (States.VISUAL_SERVOING, States.IDLE): self.reset_all,

            # TODO: Add in proper visual servoing handling
            (States.LEADER_SCAN, States.VISUAL_SERVOING): {
                'ftl_controller_3d': 'pause',
            },
            (States.VISUAL_SERVOING, States.LEADER_SCAN): {
                'ftl_controller_3d': 'resume',
            }
        }

    def handle_state_announcement(self, msg: States):
        new_state = msg.state
        self.handle_state_transition(self.current_state, new_state)

    def handle_start(self, _, resp):
        if self.current_state != States.IDLE:
            msg = 'The system is already running! Not doing anything'
            print(msg)
            resp.message = msg
            resp.success = False

        self.handle_state_transition(self.current_state, States.LEADER_SCAN)
        resp.success = True
        return resp


    def handle_stop(self, _, resp):
        if self.current_state == States.IDLE:
            msg = 'The system is already idle! Not doing anything'
            print(msg)
            resp.message = msg
            resp.success = False

        self.handle_state_transition(self.current_state, States.IDLE)
        resp.success = True
        return resp


    def handle_state_transition(self, start_state, end_state):

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
        print('[DEBUG] STATE TRANSITION FROM {} to {}'.format(start_state, end_state))

        self.current_state = end_state


    @property
    def activate_all(self):
        return {n: 'activate' for n in self.nodes}

    @property
    def reset_all(self):
        return {n: 'reset' for n in self.nodes}




def main(args=None):
    rclpy.init(args=args)
    node = SimpleStateManager()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)

if __name__ == '__main__':
    main()
