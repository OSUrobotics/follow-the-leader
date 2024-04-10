import os
import sys

from typing import Optional, Tuple, Any, List

from numpy import ndarray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import pybullet

from nptyping import NDArray, Shape, Float
from collections import namedtuple
import random

# ENV is a collection of objects like tree supports and ur5 robot. They interact with each other
# through the env. UR5 class only needs access to pybullet.
class UR5:
    def __init__(self, con, robot_urdf_path: str, pos = [0,0,0]) -> None:
        assert isinstance(robot_urdf_path, str)

        self.con = con
        self.pos = pos
        self.tool0_link_index = None
        self.end_effector_index = None
        self.success_link_index = None
        self.tool_link_index = None
        self.base_index = None
        self.ur5_robot = None
        self.num_joints = None
        self.control_joints = None
        self.joint_type_list = None
        self.joint_info = None
        self.joints = None
        self.init_joint_angles = None
        self.action = None
        self.joint_angles = None
        self.achieved_pos = None
        self.previous_pose = None
        self.init_pos = None
        self.robot_urdf_path = robot_urdf_path
        self.camera_base_offset = np.array([-0.063179, 0.077119, 0.0420027])#np.array([-0.063179, 0.077119, 0.0420027])#
        self.setup_ur5_arm()

    def setup_ur5_arm(self) -> None:
        assert self.ur5_robot is None
        self.tool0_link_index = 8
        self.end_effector_index = 12
        self.success_link_index = 14
        self.base_index = 3
        flags = self.con.URDF_USE_SELF_COLLISION
        #randomize pos TODO
        randomize = False
        if randomize:
            pos = self.pos + np.random.rand(3) * 0.05
            orientation = pybullet.getQuaternionFromEuler(np.random.rand(3) * np.pi/180*10)
        else:
            pos = self.pos
            orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
        self.ur5_robot = self.con.loadURDF(self.robot_urdf_path, pos, orientation, flags=flags)

        self.num_joints = self.con.getNumJoints(self.ur5_robot)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                               "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo",  # type: ignore
                                     ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                      "controllable"])

        self.joints = dict()
        for i in range(self.num_joints):
            info = self.con.getJointInfo(self.ur5_robot, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            print("Joint Name: ", jointName, "Joint ID: ", jointID)
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity, controllable)  # type: ignore
            # print(jointID, jointName)
            if info.type == "REVOLUTE":
                self.con.setJointMotorControl2(self.ur5_robot, info.id, self.con.VELOCITY_CONTROL, targetVelocity=0,
                                               force=0)
            self.joints[info.name] = info

        # self.set_collision_filter()
        self.init_joint_angles = (-np.pi / 2, -2., 2.16, -3.14, -1.57, np.pi)
        self.set_joint_angles(self.init_joint_angles)
        for _ in range(100):
            self.con.stepSimulation()

        self.init_pos = self.con.getLinkState(self.ur5_robot, self.end_effector_index)
        self.action = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)
        self.joint_angles = np.array(self.init_joint_angles).astype(np.float32)
        self.achieved_pos = np.array(self.get_current_pose(self.end_effector_index)[0])
        self.previous_pose = np.array([0, 0, 0, 0, 0, 0, 0]).astype(np.float32)

    def set_collision_filter(self):
        # TO SET CUTTER DISABLE COLLISIONS WITH SELF
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 9, 11, 0)
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 8, 11, 0)
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 10, 11, 0)
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 7, 11, 0)
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 6, 11, 0)

    def unset_collision_filter(self):
        # TO SET CUTTER DISABLE COLLISIONS WITH SELF
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 9, 11, 1)
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 8, 11, 1)
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 10, 11, 1)
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 7, 11, 1)
        self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, 6, 11, 1)

    def disable_self_collision(self):
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, i, j, 0)

    def enable_self_collision(self):
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.con.setCollisionFilterPair(self.ur5_robot, self.ur5_robot, i, j, 1)

    def set_joint_angles(self, joint_angles: Tuple[float, float, float, float, float, float]) -> None:
        """Set joint angles using pybullet motor control"""
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        self.con.setJointMotorControlArray(
            self.ur5_robot, indexes,
            self.con.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.05] * len(poses),
            forces=forces
        )

    # TODO: Decide typing or nptyping or what
    def set_joint_velocities(self, joint_velocities: NDArray[Shape['6, 1'], Float]) -> bool:
        """Set joint velocities using pybullet motor control"""
        velocities = []
        indexes = []
        forces = []
        singularity = False

        # TODO: Find a better way to handle singularity than not moving
        if (abs(joint_velocities) > 0.25).any():
            singularity = True
            joint_velocities = np.zeros(6)
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            velocities.append(joint_velocities[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)
        # print(joint_velocities)
        # if (joint_velocities > 1).any():
        #     input()
        maxForce = 500
        self.con.setJointMotorControlArray(self.ur5_robot,
                                           indexes,
                                           controlMode=self.con.VELOCITY_CONTROL,
                                           targetVelocities=joint_velocities,
                                           )

        return singularity

    def get_joint_velocities(self):
        j = self.con.getJointStates(self.ur5_robot, [3, 4, 5, 6, 7, 8])
        joints = tuple((i[1] for i in j))
        return joints  # type: ignore

    def calculate_jacobian(self):
        jacobian = self.con.calculateJacobian(self.ur5_robot, self.tool0_link_index, [0, 0, 0],
                                              self.get_joint_angles(),
                                              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        jacobian = np.vstack(jacobian)
        return jacobian
    def calculate_joint_velocities_from_ee_velocity(self,
                                                    end_effector_velocity: NDArray[Shape['6, 1'], Float]) -> \
            Tuple[ndarray, ndarray]:
        """Calculate joint velocities from end effector velocity using jacobian"""
        jacobian = self.calculate_jacobian()
        inv_jacobian = np.linalg.pinv(jacobian)
        joint_velocities = np.matmul(inv_jacobian, end_effector_velocity).astype(np.float32)
        return joint_velocities, jacobian

    def get_joint_angles(self) -> Tuple[float, float, float, float, float, float]:
        """Return joint angles"""
        j = self.con.getJointStates(self.ur5_robot, [3, 4, 5, 6, 7, 8])
        joints = tuple((i[0] for i in j))
        return joints  # type: ignore

    def check_collisions(self, tree, tree_support) -> Tuple[bool, dict]:
        """Check if there are any collisions between the robot and the environment
        Returns: Dictionary with information about collisions (Acceptable and Unacceptable)
        """
        collisions_acceptable = self.con.getContactPoints(bodyA=self.ur5_robot, bodyB=tree)
        collisions_unacceptable = self.con.getContactPoints(bodyA=self.ur5_robot, bodyB=tree_support)
        collision_info = {"collisions_acceptable": False, "collisions_unacceptable": False}
        for i in range(len(collisions_unacceptable)):
            # print("collision")
            if collisions_unacceptable[i][-6] < 0:
                collision_info["collisions_unacceptable"] = True
                # print("[Collision detected!] {}, {}".format(collisions[i][-6], collisions[i][3], collisions[i][4]))
                return True, collision_info

        for i in range(len(collisions_acceptable)):
            # print("collision")
            if collisions_acceptable[i][-6] < 0:
                collision_info["collisions_acceptable"] = True
                # print("[Collision detected!] {}, {}".format(collisions[i][-6], collisions[i][3], collisions[i][4]))
                return True, collision_info
        return False, collision_info

    def check_success_collision(self, body_b) -> bool:
        """Check if there are any collisions between the robot and the environment
        Returns: Boolw
        """
        collisions_success = self.con.getContactPoints(bodyA=self.ur5_robot, bodyB=body_b,
                                                       linkIndexA=self.success_link_index)
        for i in range(len(collisions_success)):
            # print("collision")
            if collisions_success[i][-6] < 0:
                # collision_info["collisions_unacceptable"] = True
                # print("[Collision detected!] {}, {}".format(collisions[i][-6], collisions[i][3], collisions[i][4]))
                return True
        return False

    def calculate_ik(self, position: Tuple[float, float, float],
                     orientation: Optional[Tuple[float, float, float, float]]) -> \
            Tuple[float, float, float, float, float, float]:
        """Calculates joint angles from end effector position and orientation using inverse kinematics"""
        lower_limits = [-np.pi] * 6
        upper_limits = [np.pi] * 6
        joint_ranges = [2 * np.pi] * 6

        joint_angles = self.con.calculateInverseKinematics(
            self.ur5_robot, self.end_effector_index, position, orientation,
            jointDamping=[0.01] * 6, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges  # , restPoses=self.init_joint_angles
        )
        return joint_angles

    def get_current_pose(self, index: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Returns current pose of the end effector. Pos wrt end effector, orientation wrt world"""
        link_state: Tuple = self.con.getLinkState(self.ur5_robot, index, computeForwardKinematics=True)
        position, orientation = link_state[4], link_state[5]  # Position wrt end effector, orientation wrt COM
        return position, orientation

    def get_current_vel(self, index: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Returns current pose of the end effector. Pos wrt end effector, orientation wrt world"""
        link_state: Tuple = self.con.getLinkState(self.ur5_robot, index, computeLinkVelocity = True, computeForwardKinematics=True)
        trans, ang = link_state[6], link_state[7]
        return trans, ang

    def create_camera_transform(self, pos, orientation, pan, tilt, xyz_offset) -> np.ndarray:
        """Create rotation matrix for camera"""
        base_offset_tf = np.identity(4)
        base_offset_tf[:3, 3] = self.camera_base_offset + xyz_offset

        ee_transform = np.identity(4)
        ee_rot_mat = np.array(self.con.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        ee_transform[:3, :3] = ee_rot_mat
        ee_transform[:3, 3] = pos

        tilt_tf = np.identity(4)
        tilt_rot = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        tilt_tf[:3, :3] = tilt_rot

        pan_tf = np.identity(4)
        pan_rot = np.array([[np.cos(pan), 0, np.sin(pan)], [0, 1, 0], [-np.sin(pan), 0, np.cos(pan)]])
        pan_tf[:3, :3] = pan_rot


        tf = ee_transform @ pan_tf @ tilt_tf @ base_offset_tf
        return tf

    # TODO: Better types for getCameraImage
    def get_view_mat_at_curr_pose(self, pan, tilt, xyz_offset) -> np.ndarray:
        """Get view matrix at current pose"""
        pose, orientation = self.get_current_pose(self.tool0_link_index)

        camera_tf = self.create_camera_transform(pose, orientation, pan, tilt, xyz_offset)

        # Initial vectors
        camera_vector = np.array([0, 0, 1]) @ camera_tf[:3, :3].T  #
        up_vector = np.array([0, 1, 0]) @ camera_tf[:3, :3].T  #
        # Rotated vectors
        # print(camera_vector, up_vector)
        view_matrix = self.con.computeViewMatrix(camera_tf[:3, 3], camera_tf[:3, 3] + 0.1 * camera_vector, up_vector)
        return view_matrix

    def get_camera_location(self):
        pose, orientation = self.get_current_pose(self.tool0_link_index)
        tilt = np.pi / 180 * 8

        camera_tf = self.create_camera_transform(pose, orientation, tilt)
        return camera_tf

    def get_condition_number(self) -> float:
        # get jacobian
        jacobian = self.con.calculateJacobian(self.ur5_robot, self.tool0_link_index, [0, 0, 0],
                                              self.get_joint_angles(),
                                              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        jacobian = np.vstack(jacobian)
        condition_number = np.linalg.cond(jacobian)
        return condition_number

    def remove_ur5_robot(self):
        self.con.removeBody(self.ur5_robot)
        self.ur5_robot = None
