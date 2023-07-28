import os.path

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, PlanningOptions, Constraints, JointConstraint, PositionConstraint, OrientationConstraint
import numpy as np
from std_msgs.msg import Int16
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3, Quaternion
from follow_the_leader_msgs.msg import BlenderParams, ControllerParams, States, StateTransition
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup
from follow_the_leader.utils.ros_utils import TFNode
from rclpy.action import ActionClient
from scipy.spatial.transform import Rotation
from functools import partial


class ExperimentManagementNode(TFNode):

    def __init__(self, output_folder, home_joints, sim=True):
        super().__init__('experiment_manager_node')

        self.sim = sim
        self.home_joints = [float(x) for x in home_joints]
        self.folder = output_folder

        self.current_experiment = -1

        self.param_sets = {
            'pan_frequency': [0.0, 1.5, 1.5, 2.5, 2.5],
            'pan_magnitude_deg': [0.0, 22.5, 45.0, 22.5, 45.0],
            'z_desired': [0.20] * 5,
            'ee_speed': [0.05] * 5,
        }

        # ROS utilities
        self.cb = ReentrantCallbackGroup()
        self.moveit_planning_client = ActionClient(self, MoveGroup, 'move_action')
        self.reset_model_srv = self.create_client(Trigger, '/initialize_tree_spindle')
        self.state_announce_pub = self.create_publisher(States, 'state_announcement', 1)
        self.blender_pub = self.create_publisher(BlenderParams, '/blender_params', 1)
        self.controller_pub = self.create_publisher(ControllerParams, '/controller_params', 1)
        self.transition_sub = self.create_subscription(StateTransition, 'state_transition',
                                                       self.handle_state_transition, 1,
                                                       callback_group=self.cb)
        self.joint_state_sub = self.create_subscription(JointState, '/move_joints', partial(self.move_to, None), 1)
        self.joy_action_sub = self.create_subscription(Int16, '/joy_action', self.handle_joy_action, 1, callback_group=self.cb)

    @property
    def n(self):
        return len(self.param_sets['pan_frequency'])

    def handle_joy_action(self, msg):
        action = msg.data

        if action == 0:
            self.execute_experiment()

        if abs(action) == 2:
            # RL - Increase the experiment ID
            self.current_experiment += (1 if action > 0 else -1)
            self.current_experiment = max(0, self.current_experiment)
            self.prepare_experiment()

        else:
            print('Got unknown action value {}'.format(action))
            return

    def send_params_update(self, seed=None, num_branches=None, param_set=None):

        len_params = len(self.param_sets['pan_frequency'])

        if seed is None:
            seed = self.current_experiment // len_params

        if num_branches is None:
            num_branches = self.current_experiment // (2 * len_params)

        print('Experiment {}: Tree ID {} with {} branches'.format(self.current_experiment, seed, num_branches))

        self.blender_pub.publish(BlenderParams(
            seed=seed,
            num_branches=num_branches,
            save_path=self.folder,
            identifier=str(self.current_experiment),
        ))

        if param_set is None:
            idx = self.current_experiment % len(self.param_sets['pan_frequency'])
            param_set = {
                'pan_frequency': self.param_sets['pan_frequency'][idx],
                'pan_magnitude_deg': self.param_sets['pan_magnitude_deg'][idx],
                'z_desired': self.param_sets['z_desired'][idx],
                'ee_speed': self.param_sets['ee_speed'][idx],
                'save_folder': self.folder,
                'identifier': str(self.current_experiment),
            }

        params_message = ControllerParams(**param_set)
        self.controller_pub.publish(params_message)

    def prepare_experiment(self):

        save_path = os.path.join(self.folder, f'{self.current_experiment}_results.pickle')
        if os.path.exists(save_path):
            print('This experiment already looks done!')
            return

        self.send_params_update()

    def execute_experiment(self):

        self.send_params_update()
        print('Running experiment {}'.format(self.current_experiment))
        self.state_announce_pub.publish(States(state=States.LEADER_SCAN))

        # TODO: START BAG RECORDING

    def end_experiment(self):
        pass

        # TODO: END BAG RECORDING

    def level_pose(self):
        tf = self.lookup_transform('base_link', 'tool0', sync=False, as_matrix=True)
        x = tf[:3,0]
        z = tf[:3,2].copy()
        z[2] = 0
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)

        new_tf = tf.copy()
        new_tf[:3,:3] = np.array([x,y,z]).T
        #
        # import pdb
        # pdb.set_trace()

        self.move_to(pose=np.linalg.inv(tf) @ new_tf)

    def move_home(self):
        self.move_to(joints=self.home_joints)

    def move_to(self, pose=None, joints=None):

        if not (pose is None) ^ (joints is None):
            if pose is not None:
                raise ValueError("Please fill in only a pose or a joints value, not both")
            else:
                raise ValueError("Please specify a pose or joints value to move to")

        if joints is not None:
            joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

            if isinstance(joints, JointState):
                joints = joints.position

            if not len(joints):
                joints = self.home_joints

            joint_constraints = [
                JointConstraint(joint_name=n, position=p)
                for n, p in zip(joint_names, joints)
            ]
            kwargs = {
                'joint_constraints': joint_constraints
            }

        else:

            pos = pose[:3,3]
            quat = Rotation.from_matrix(pose[:3,:3]).as_quat()

            # TODO: This doesn't work - why not?
            kwargs = {
                'position_constraints': [
                    PositionConstraint(
                        link_name='tool0',
                        target_point_offset=Vector3(x=pos[0], y=pos[1], z=pos[2])
                    )
                ],
                'orientation_constraints': [
                    OrientationConstraint(
                        link_name='tool0',
                        orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])),
                ]
            }

        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest(
            group_name='ur_manipulator',
            goal_constraints=[Constraints(**kwargs)],
            allowed_planning_time=5.0,
        )
        goal_msg.planning_options = PlanningOptions(
            plan_only=False
        )

        self.moveit_planning_client.wait_for_server()
        future = self.moveit_planning_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_complete)

    def handle_state_transition(self, msg: StateTransition):
        if msg.state_end == States.IDLE:
            self.end_experiment()

    def goal_complete(self, future):
        rez = future.result()
        if not rez.accepted:
            print('Planning failed!')
            return
        else:
            print('Plan succeeded!')



if __name__ == '__main__':

    output_dir = os.path.join(os.path.expanduser('~'), 'data', 'model_the_leader')

    rclpy.init()
    home_joints = [0.0, -1.9936, -2.4379, 1.2682, 1.56252, 0.0]
    node = ExperimentManagementNode(output_dir, home_joints)

    rclpy.spin(node)


