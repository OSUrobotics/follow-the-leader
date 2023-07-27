import os.path

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, PlanningOptions, Constraints, JointConstraint, PositionConstraint, OrientationConstraint
import numpy as np
from std_msgs.msg import Int16
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3, Quaternion
from follow_the_leader_msgs.msg import BlenderParams, States, StateTransition
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

        self.current_tree_id = 0
        self.current_branches = 1

        self.current_param_set = -1
        self.param_sets = {
            'pan_frequency': [0.0, 1.5, 1.5, 2.5, 2.5],
            'pan_magnitude_deg': [0.0, 22.5, 22.5, 45.0, 45.0],
        }

        # ROS utilities
        self.cb = ReentrantCallbackGroup()
        self.moveit_planning_client = ActionClient(self, MoveGroup, 'move_action')
        self.reset_model_srv = self.create_client(Trigger, '/initialize_tree_spindle')
        self.state_announce_pub = self.create_publisher(States, 'state_announcement', 1)
        self.blender_pub = self.create_publisher(BlenderParams, '/blender_params', 1)
        self.transition_sub = self.create_subscription(StateTransition, 'state_transition',
                                                       self.handle_state_transition, 1,
                                                       callback_group=self.cb)
        self.joint_state_sub = self.create_subscription(JointState, '/move_joints', partial(self.move_to, None), 1)
        self.joy_action_sub = self.create_subscription(Int16, '/joy_action', self.handle_joy_action, 1, callback_group=self.cb)

        self.next_experiment_srv = self.create_service(Trigger,  '/next_experiment', self.do_next_experiment)


    @property
    def n(self):
        return len(self.param_sets['pan_frequency'])

    def handle_joy_action(self, msg):
        action = msg.data
        if abs(action) == 2:
            # RL - Increase the tree ID
            self.current_tree_id += (1 if action > 0 else -1)
            self.send_params_update()
        elif abs(action) == 1:
            self.current_branches += (1 if action > 0 else -1)
            self.current_branches = max(self.current_branches, 0)
            self.send_params_update()
        else:
            print('Got unknown action value {}'.format(action))
            return

    def send_params_update(self):

        print('Tree ID {} with {} branches'.format(self.current_tree_id, self.current_branches))

        self.blender_pub.publish(BlenderParams(
            seed=self.current_tree_id,
            num_branches=self.current_branches,
            save_path=self.folder,
        ))

    def do_next_experiment(self, *args):
        if self.current_experiment == -1:
            self.current_experiment = 0
            # Check the output folder to see if there are any experiments that are already done
            existing_files = [int(x.replace('.json', '')) for x in os.listdir(output_dir) if x.endswith('.json')]
            if existing_files:
                self.current_experiment = max(existing_files)

        self.current_param_set = (self.current_param_set + 1) % self.n
        if self.current_param_set == 0:
            self.reset_model_srv.call(Trigger.Request())

        # Update param set
        for param, param_vals in self.param_sets.items():
            # set_param(param, param_vals[self.current_param_set])
            pass

        print('Starting new thing!')
        # TODO: START BAG RECORDING
        self.state_announce_pub.publish(States(state=States.LEADER_SCAN))

        if len(args) == 2:
            # Service call
            resp = args[1]
            resp.success = True
            return resp


    def end_experiment(self):
        pass
        # TODO: END BAG RECORDING
        # TODO: WRITE OUT MODEL

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

            kwargs = {
                'position_constraints': [
                    PositionConstraint(
                        link_name='asdfasdfasdfsda',
                        target_point_offset=Vector3(x=pos[0], y=pos[1], z=pos[2])
                    )
                ],
                'orientation_constraints': [
                    OrientationConstraint(
                        link_name='tooasdfsadfsadfsadfl0',
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


