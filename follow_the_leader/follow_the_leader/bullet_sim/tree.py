import glob
import os
import pickle
from typing import Optional, Tuple, List
import pybullet
import numpy as np
import pywavefront
from nptyping import NDArray, Shape, Float

# # from pruning_sb3.pruning_gym.pruning_env import SUPPORT_AND_POST_PATH
# from pruning_sb3.pruning_gym.helpers import compute_perpendicular_projection_vector

from . import MESHES_AND_URDF_PATH, ROBOT_URDF_PATH, SUPPORT_AND_POST_PATH

class Tree:
    """ Class representing a tree mesh. It is used to sample points on the surface of the tree."""

    def __init__(self, env, pyb, urdf_path: str, obj_path: str,
                 pos: NDArray[Shape['3,1'], Float] = np.array([0, 0, 0]),
                 orientation: NDArray[Shape['4,1'], Float] = np.array([0, 0, 0, 1]),
                 num_points: Optional[int] = None, scale: int = 1, reset_count = 0) -> None:

        self.urdf_path = urdf_path
        self.env = env
        self.pyb = pyb
        self.scale = scale
        self.pos = pos
        self.orientation = orientation
        self.obj_path = obj_path
        self.tree_obj = pywavefront.Wavefront(obj_path, create_materials=True, collect_faces=True)
        self.vertex_and_projection = []
        self.projection_mean = np.array(0.)
        self.projection_std = np.array(0.)
        self.projection_sum_x = np.array(0.)
        self.projection_sum_x2 = np.array(0.)
        self.base_xyz = self.env.ur5.get_current_pose(self.env.ur5.base_index)[0]
        self.num_points = num_points
        self.reachable_points = []
        self.reset_count = reset_count
        self.supports = None
        self.tree_id = None

    def active(self):
        print("activating tree")
        assert self.tree_id is None
        assert self.supports is None
        print('Loading tree from ', self.urdf_path)
        self.supports = self.pyb.con.loadURDF(SUPPORT_AND_POST_PATH, [0, -0.6, 0],
                                              list(self.pyb.con.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2])),
                                              globalScaling=1)
        self.tree_id = self.pyb.con.loadURDF(self.urdf_path, self.pos, self.orientation, globalScaling=self.scale)

    def inactive(self):
        assert self.tree_id is not None
        assert self.supports is not None
        self.pyb.con.removeBody(self.tree_id)
        self.pyb.con.removeBody(self.supports)
        self.tree_id = None
        self.supports = None

    def is_reachable(self, vertice: Tuple[NDArray[Shape['3, 1'], Float], NDArray[Shape['3, 1'], Float]]) -> bool:
        #TODO: Fix this
        ur5_base_pos = np.array(self.base_xyz)


        #Meta condition
        dist = np.linalg.norm(ur5_base_pos - vertice[0], axis=-1)
        projection_length = np.linalg.norm(vertice[1])
        if dist >= 0.8 or (projection_length < self.projection_mean + 0.5 * self.projection_std):
            # print("proj length")
            return False

        j_angles = self.env.ur5.calculate_ik(vertice[0], None)
        self.env.ur5.set_joint_angles(j_angles)
        for i in range(100):
            self.pyb.con.stepSimulation()
        ee_pos, _ = self.env.ur5.get_current_pose(self.env.ur5.end_effector_index)
        dist = np.linalg.norm(np.array(ee_pos) - vertice[0], axis=-1)
        if dist <= 0.05:
            return True

        return False

    @staticmethod
    def make_trees_from_folder(env, pyb, trees_urdf_path: str, trees_obj_path: str, pos: NDArray,
                               orientation: NDArray, scale: int, num_points: int, num_trees: int):
        trees: List[Tree] = []
        for urdf, obj in zip(sorted(glob.glob(trees_urdf_path + '/*.urdf')),
                             sorted(glob.glob(trees_obj_path + '/*.obj'))):
            if len(trees) >= num_trees:
                break
            #randomize position TOOO:
            randomize = True
            if randomize:
                pos = pos + np.random.uniform(low = -1, high=1, size = (3,)) * np.array([0.25, 0.025, 0.25])
                # pos[2] = pos[2] - 0.3
                orientation = np.array([0,0,0,1])#pybullet.getQuaternionFromEuler(np.random.uniform(low = -1, high=1, size = (3,)) * np.pi / 180 * 10)
            trees.append(Tree(env, pyb, urdf_path=urdf, obj_path=obj, pos=pos, orientation=orientation, scale=scale,
                              num_points=num_points))
        return trees

    def reset_tree(self):
        self.reset_count += 1
        if self.reset_count > 2:
            print("Reset count exceeded for tree ", self.urdf_path)
            return
        self.pos = np.array([0,0,0.6]) + np.random.uniform(low = -1, high=1, size = (3,)) * np.array([0.2, 0.1, .2])
        self.pos[2] = self.pos[2] - 2
        orientation = pybullet.getQuaternionFromEuler(np.random.uniform(low = -1, high=1, size = (3,)) * np.pi / 180 * 15)
        self.__init__(self.env, self.pyb, urdf_path=self.urdf_path, obj_path=self.obj_path, pos=self.pos, orientation=orientation, scale=self.scale,
                              num_points=self.num_points, reset_count=self.reset_count)
