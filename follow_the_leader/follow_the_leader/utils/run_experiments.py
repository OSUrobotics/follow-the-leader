import os.path
import sys

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, PlanningOptions, Constraints, JointConstraint, PositionConstraint, OrientationConstraint
import numpy as np
from std_msgs.msg import Int16
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3, Quaternion, PoseStamped
from follow_the_leader_msgs.msg import BlenderParams, ControllerParams, States, StateTransition
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup
from follow_the_leader.utils.ros_utils import TFNode
from rclpy.action import ActionClient
from scipy.spatial.transform import Rotation
from functools import partial
import subprocess as sp
import shlex
import shutil
from threading import Lock
import yaml


class ExperimentManagementNode(TFNode):

    def __init__(self, output_folder, home_joints, sim=True):
        super().__init__('experiment_manager_node')

        self.sim = sim
        self.home_joints = [float(x) for x in home_joints]
        self.folder = output_folder

        self.current_experiment = 0
        self.num_branches = 0
        self.custom_seed = None
        self.bag_recording_proc = None
        self.save_counter = 0

        desired_speed = 0.025 if self.sim else 0.15
        self.override_speed = 0     # Increments of 0.05

        self.param_sets = {
            'pan_frequency': [0.0, 1.0, 1.0, 2.0, 2.0],
            'pan_magnitude_deg': [0.0, 22.5, 45.0, 22.5, 45.0],
            'z_desired': [0.20] * 5,
            'ee_speed': [desired_speed] * 5,
        }

        self.camera_poses = []
        self.camera_ts = []
        self.lock = Lock()

        # For real experiments only
        self.branch_id = None if self.sim else 0
        self.probe_mode = False
        self.probes = []

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
        self.camera_pose_sub = self.create_subscription(PoseStamped, '/camera_pose', self.handle_camera_pose, 1)
        self.velocity_reporter_timer = self.create_timer(1.0, self.report_velocity)

    @property
    def n(self):
        return len(self.param_sets['pan_frequency'])

    def handle_joy_action(self, msg):
        action = msg.data

        if action == 0:
            self.execute_experiment()

        elif abs(action) == 1:

            if not self.sim:
                self.override_speed = max(0, self.override_speed + (1 if action > 0 else -1))
                if self.override_speed:
                    print('Set override speed to {:.2f}'.format(self.override_speed * 0.05))
                else:
                    print('Restored default speed')

                self.prepare_experiment()
                return


            self.num_branches += (1 if action > 0 else -1)
            self.num_branches = max(0, self.num_branches)
            self.prepare_experiment()

        elif abs(action) == 2:

            if not self.sim:
                self.current_experiment += (1 if action > 0 else -1)
                self.prepare_experiment()
                return

            self.num_branches = 0
            # RL - Increase the experiment ID
            self.current_experiment += (1 if action > 0 else -1)
            self.current_experiment = max(0, self.current_experiment)
            self.prepare_experiment()

        elif action == 3:

            if not self.sim:
                if not self.probe_mode:
                    return
                # Act as the probing trigger
                pose = self.lookup_transform('base_link', 'tool0', sync=False, as_matrix=True)
                pos = pose[:3,3]
                quat = Rotation.from_matrix(pose[:3,:3]).as_quat()

                if self.probes and np.linalg.norm(pos - self.probes[-1][:3]) < 5e-3:
                    print('Recording null probe!')
                    self.probes.append([0] * len(self.probes[-1]))
                else:
                    while True:
                        try:
                            diameter = float(input('Please enter radius: '))
                            break
                        except ValueError:
                            print('Not a valid input. Try again')

                    combined = np.concatenate([pos, quat, [diameter]])
                    self.probes.append(combined)

                save_folder = os.path.join(self.folder, 'real_data', str(self.branch_id))
                os.makedirs(save_folder, exist_ok=True)
                probe_file = os.path.join(save_folder, 'probes.csv')

                print('Added probe (now at {} probes)'.format(len(self.probes)))
                np.savetxt(probe_file, np.array(self.probes), delimiter=",")

                return

            if self.custom_seed is not None:
                self.custom_seed = None
                self.current_experiment = 0
                print('Disabled custom seed!')
            else:
                # self.custom_seed = 10659
                self.custom_seed = np.random.randint(0, 32767)
                print('Enabled custom seed {}!'.format(self.custom_seed))

            self.prepare_experiment()

        elif abs(action) == 4:
            if self.sim:
                return

            self.branch_id = max(0, self.branch_id + (1 if action > 0 else -1))

            print('Set branch to branch ID {}'.format(self.branch_id))

        elif abs(action) == 5:
            if self.sim:
                return

            self.probes = []
            if not self.probe_mode:
                self.probe_mode = True
                print('Enabled probe mode! (Writing for branch {})'.format(self.branch_id))

            else:
                self.probe_mode = False
                print('Disabled probe mode')



        else:
            print('Got unknown action value {}'.format(action))
            return

    def send_params_update(self, folder=''):

        len_params = len(self.param_sets['pan_frequency'])
        num_branches = self.num_branches

        if not self.sim:
            param_idx = self.current_experiment % len_params

            ee_speed = self.override_speed * 0.05 if self.override_speed else self.param_sets['ee_speed'][param_idx]
            param_set = {
                'pan_frequency': self.param_sets['pan_frequency'][param_idx],
                'pan_magnitude_deg': self.param_sets['pan_magnitude_deg'][param_idx],
                'z_desired': self.param_sets['z_desired'][param_idx],
                'ee_speed': ee_speed,
                'save_folder': folder,
                'identifier': str(self.save_counter),
            }
            print('Prepared for real experiment')

        elif self.custom_seed is not None:

            seed = self.custom_seed
            param_idx = 0
            param_set = {
                'pan_frequency': self.param_sets['pan_frequency'][param_idx],
                'pan_magnitude_deg': self.param_sets['pan_magnitude_deg'][param_idx],
                'z_desired': self.param_sets['z_desired'][param_idx],
                'ee_speed': self.param_sets['ee_speed'][param_idx],
                'save_folder': '',
                'identifier': '',
            }
            print('Running custom experiment with seed {} and {} branches'.format(self.custom_seed, self.num_branches))

            self.blender_pub.publish(BlenderParams(
                seed=seed,
                num_branches=num_branches,
                save_path='',
                identifier='',
            ))

        else:
            seed = self.current_experiment // len_params
            idx = self.current_experiment % len(self.param_sets['pan_frequency'])
            param_set = {
                'pan_frequency': self.param_sets['pan_frequency'][idx],
                'pan_magnitude_deg': self.param_sets['pan_magnitude_deg'][idx],
                'z_desired': self.param_sets['z_desired'][idx],
                'ee_speed': self.param_sets['ee_speed'][idx],
                'save_folder': self.folder,
                'identifier': f'{self.current_experiment}_{num_branches}',
            }

            print('Experiment {}: Setting up tree ID {} with {} branches on param set {}'.format(self.current_experiment, seed, num_branches, idx))

            self.blender_pub.publish(BlenderParams(
                seed=seed,
                num_branches=num_branches,
                save_path=self.folder,
                identifier=str(self.current_experiment),
            ))

        print('Controller parameters:')
        for key in sorted(self.param_sets):
            print(f'\t{key}: {param_set[key]}')

        params_message = ControllerParams(**param_set)
        self.controller_pub.publish(params_message)
        return param_set

    def prepare_experiment(self):

        if not self.sim:
            self.send_params_update()
            return

        if self.custom_seed is None:
            save_path = os.path.join(self.folder, f'{self.current_experiment}_{self.num_branches}_results.pickle')
            if os.path.exists(save_path):
                print('[!] Experiment {} with {} branches is already done! Not loading data'.format(self.current_experiment, self.num_branches))
                return

        self.send_params_update()

    def execute_experiment(self):

        save_params_to = None
        if self.sim:
            if self.custom_seed is not None:
                print('Will not run data collection on custom tree!')
                return

            bag_path = os.path.join(self.folder, f'{self.current_experiment}_{self.num_branches}_data')
            if os.path.exists(bag_path):
                shutil.rmtree(bag_path)

        else:
            branch_data_path = os.path.join(self.folder, 'real_data', str(self.branch_id))
            os.makedirs(branch_data_path, exist_ok=True)
            exp_id = len([x for x in os.listdir(branch_data_path) if os.path.isdir(os.path.join(branch_data_path, x))])
            final_results_dir = os.path.join(branch_data_path, '{:03d}'.format(exp_id))
            save_params_to = final_results_dir
            os.makedirs(final_results_dir, exist_ok=True)

            bag_path = os.path.join(final_results_dir, 'bag_data')
            print('Recording experiment results to:\n{}'.format(final_results_dir))

        topics_to_record = [
            '/camera/color/camera_info',
            '/camera/color/image_rect_raw',
            '/tree_model',
            '/curve_3d_rviz_array',
            '/controller_diagnostic',
            '/image_mask',
            '/joint_states',
            '/model_diagnostic',
            '/point_tracking_response_pc',
            '/tf',
            '/tf_static',
            '/camera_pose',
        ]

        cmd = 'ros2 bag record -o {} {}'.format(
            bag_path, ' '.join(topics_to_record)
        )
        self.bag_recording_proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, shell=False)

        param_set = self.send_params_update(save_params_to or '')
        if save_params_to is not None:
            with open(os.path.join(save_params_to, 'config.yaml'), 'w') as fh:
                yaml.safe_dump(param_set, fh)

        if self.sim:
            print('Running experiment {}'.format(self.current_experiment))
        else:
            print('Running experiment on real arm...')

        self.state_announce_pub.publish(States(state=States.LEADER_SCAN))

    def end_experiment(self):
        if self.bag_recording_proc is not None:
            self.bag_recording_proc.terminate()
            self.bag_recording_proc = None

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

    def handle_camera_pose(self, pose: PoseStamped):
        tl = pose.pose.position
        xyz = np.array([tl.x, tl.y, tl.z])
        stamp = pose.header.stamp
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        with self.lock:
            self.camera_poses.append(xyz)
            self.camera_ts.append(stamp_sec)

    def report_velocity(self):
        with self.lock:
            if self.camera_poses:
                pos = np.array(self.camera_poses)
                dist = np.linalg.norm(pos[1:] - pos[:-1], axis=1).sum()
                elapsed = self.camera_ts[-1] - self.camera_ts[0]
                vel = dist / elapsed
                print('Nominal velocity: {:.3f} m/s'.format(vel))

            self.camera_poses = []
            self.camera_ts = []



if __name__ == '__main__':

    mode = sys.argv[1]
    sim = False
    if mode == 'sim':
        home_joints = [0.0, -1.9936, -2.4379, 1.2682, 1.56252, 0.0]
        sim = True
    elif mode == 'ur5e':
        home_joints = [3.8675 - np.pi, -2.0459, -2.04105, 0.9304, 1.64812, 0.0]
    else:
        raise ValueError('Unsupported value {}'.format(mode))

    output_dir = os.path.join(os.path.expanduser('~'), 'data', 'model_the_leader')

    rclpy.init()

    node = ExperimentManagementNode(output_dir, home_joints, sim=sim)
    rclpy.get_default_context().on_shutdown(node.end_experiment)
    rclpy.spin(node)


