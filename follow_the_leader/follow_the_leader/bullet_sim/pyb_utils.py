#!/usr/bin/env python3
import os
import sys
from pybullet_utils import bullet_client as bc
import pybullet as pybullet
from typing import List, Tuple
from nptyping import NDArray

from bullet_sim import MESHES_AND_URDF_PATH
import numpy as np
"""
https://github.com/abhinavj98/pruning_sb3
"""

class pyb_utils:
    def __init__(self, env, renders: bool = False, cam_height: int = 224, cam_width: int = 224) -> None:
        print("initing pyb")
        self.viz_view_matrix = None
        self.viz_proj_matrix = None
        self.renders = renders
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.near_val = 0.01
        self.far_val = 100
        self.env = env

        self.con = None
        # Debug parameters
        self.debug_items_step = []
        self.debug_items_reset = []

        self.setup_pybullet()

    def setup_pybullet(self) -> None:
        # New class for pybullet
        if self.renders:
            self.con = bc.BulletClient(connection_mode=pybullet.GUI)
            print("gui mode")
        else:
            self.con = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.con.setTimeStep(120. / 240.)
        # self.enable_gravity()
        self.con.setRealTimeSimulation(False)
        self.proj_mat = self.con.computeProjectionMatrixFOV(
            fov=60, aspect=self.cam_width / self.cam_height, nearVal=self.near_val,
            farVal=self.far_val)

        self.con.resetDebugVisualizerCamera(cameraDistance=1.06, cameraYaw=-120.3, cameraPitch=-12.48,
                                            cameraTargetPosition=[-0.3, -0.06, 0.4])
        self.create_background()
        self.setup_bird_view_visualizer()

    def disable_gravity(self):
        self.con.setGravity(0, 0, 0)

    def enable_gravity(self):
        self.con.setGravity(0, 0, -10)

    def create_wall_with_texture(self, wall_dim: List, wall_pos: List, euler_rotation: List, wall_texture: int):
        wall_viz = self.con.createVisualShape(shapeType=self.con.GEOM_BOX, halfExtents=wall_dim,
                                              )
        wall_col = self.con.createCollisionShape(shapeType=self.con.GEOM_BOX, halfExtents=wall_dim,
                                                 )
        wall_id = self.con.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_viz,
                                           baseCollisionShapeIndex=wall_col, basePosition=wall_pos,
                                           baseOrientation=list(
                                               self.con.getQuaternionFromEuler(euler_rotation)))
        self.con.changeVisualShape(objectUniqueId=wall_id, linkIndex=-1,
                                   textureUniqueId=wall_texture)
        return wall_id

    def create_background(self) -> None:
        # wall_folder = os.path.join('models', 'wall_textures')
        # TODO: Replace all static paths with os.path.join
        wall_texture_path = os.path.join(MESHES_AND_URDF_PATH, 'textures', 'leaves-dead.png')
        self.wall_texture = self.con.loadTexture(wall_texture_path)

        self.floor_id = self.create_wall_with_texture([0.01, 5, 5], [0, 0, 0], [0, np.pi / 2, 0], self.wall_texture)
        self.wall_id = self.create_wall_with_texture([0.01, 5, 5], [0, -2, 5], [np.pi / 2, 0, np.pi / 2],
                                                     self.wall_texture)
        self.side_wall_1_id = self.create_wall_with_texture([0.01, 5, 5], [-5, 0, 5], [0, 0, 0], self.wall_texture)
        self.side_wall_2_id = self.create_wall_with_texture([0.01, 5, 5], [5, 0, 5], [0, 0, 0], self.wall_texture)
        self.ceil_id = self.create_wall_with_texture([0.01, 5, 5], [0, 0, 10], [0, np.pi / 2, 0], self.wall_texture)

    def remove_debug_items(self, where) -> None:
        if where == 'step':
            for item in self.debug_items_step:
                self.con.removeUserDebugItem(item)
            self.debug_items_step = []
        elif where == 'reset':
            for item in self.debug_items_reset:
                self.con.removeUserDebugItem(item)
            self.debug_items_reset = []

    def add_debug_item(self, type, where, **kwargs):
        debug_item = self.con.addUserDebugLine(**kwargs)
        if where == 'step':
            self.debug_items_step.append(debug_item)
        elif where == 'reset':
            self.debug_items_reset.append(debug_item)
        return debug_item

    def remove_body(self, body_id: int) -> None:
        self.con.removeBody(body_id)

    def add_sphere(self, radius: float, pos: List, rgba: List) -> int:
        colSphereId = -1
        visualShapeId = self.con.createVisualShape(self.con.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        sphereUid = self.con.createMultiBody(0.0, colSphereId, visualShapeId,
                                             pos, [0, 0, 0, 1])
        return sphereUid

    def get_image_at_curr_pose(self, type, view_matrix=None) -> List:
        """Take the current pose of the end effector and set the camera to that pose"""
        if type == 'robot':
            if view_matrix is None:
                raise ValueError('view_matrix cannot be None for robot view')
            return self.con.getCameraImage(width=self.cam_width, height=self.cam_height, viewMatrix=view_matrix,
                                           projectionMatrix=self.proj_mat,
                                           renderer=self.con.ER_BULLET_HARDWARE_OPENGL,
                                           flags=self.con.ER_NO_SEGMENTATION_MASK, lightDirection=[1, 1, 1])
        elif type == 'viz':
            return self.con.getCameraImage(width = self.cam_width, height = self.cam_height, viewMatrix=self.viz_view_matrix,
                                           projectionMatrix=self.viz_proj_matrix,
                                           renderer=self.con.ER_BULLET_HARDWARE_OPENGL,
                                           flags=self.con.ER_NO_SEGMENTATION_MASK, lightDirection=[1, 1, 1])

    def setup_bird_view_visualizer(self):
        self.viz_view_matrix = self.con.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.3, -0.06, 1.3],
                                                                          distance=1.06,
                                                                          yaw=-120.3, pitch=-12.48, roll=0,
                                                                          upAxisIndex=2)
        self.viz_proj_matrix = self.con.computeProjectionMatrixFOV(fov=60, aspect=float(self.cam_width / self.cam_height), nearVal=0.1,
                                                                   farVal=100.0)

    @staticmethod
    def seperate_rgbd_rgb_d(rgbd: List, height: int, width: int) -> Tuple[NDArray, NDArray]:
        """Seperate rgb and depth from the rgbd image, return RGB and depth"""
        rgb = np.array(rgbd[2]).reshape(height, width, 4) / 255
        rgb = rgb[:, :, :3]
        depth = np.array(rgbd[3]).reshape(height, width)
        return rgb, depth

    @staticmethod
    def linearize_depth(depth: NDArray, far_val: float, near_val: float):
        """OpenGL returns contracted depth, linearize it"""
        depth_linearized = near_val / (far_val - (far_val - near_val) * depth + 0.00000001)
        return depth_linearized

    def get_rgbd_at_cur_pose(self, type, view_matrix) -> Tuple[NDArray, NDArray]:
        """Get RGBD image at current pose"""
        # cur_p = self.ur5.get_current_pose(self.camera_link_index)
        rgbd = self.get_image_at_curr_pose(type, view_matrix)
        rgb, depth = self.seperate_rgbd_rgb_d(rgbd, height=self.cam_height, width=self.cam_width)
        depth = depth.astype(np.float32)
        depth = self.linearize_depth(depth, self.far_val, self.near_val) - 0.5
        return rgb, depth

    def visualize_points(self, points: List, type: str) -> None:
        dx = 0.1
        for point in points:
            if type == "curriculum":
                loc = point[1][0]
                branch = point[1][1]
                normal_vec = point[1][2]
            elif type == "reachable":
                loc = point[0]
                branch = point[1]
                normal_vec = point[2]
            #         self.debug_branch = self.pyb.con.addUserDebugLine(point,
            #                                                       point + 5 * normal_vec / np.linalg.norm(normal_vec),
            #                                                       [1, 0, 0], 2)
            self.add_debug_item('sphere', 'step', lineFromXYZ=[loc[0],
                                                               loc[1], loc[2]],
                                lineToXYZ=[loc[0] + 0.005, loc[1] + 0.005, \
                                           loc[2] + 0.005],
                                lineColorRGB=[1, 0, 0],
                                lineWidth=200)
            self.add_debug_item('sphere', 'step', lineFromXYZ=[loc[0] - dx * branch[0],
                                                               loc[1] - dx * branch[1],
                                                               loc[2] - dx * branch[2]],
                                lineToXYZ=[loc[0] + dx * branch[0], loc[1] + dx * branch[1], \
                                           loc[2] + dx * branch[2]],
                                lineColorRGB=[1, 1, 0],
                                lineWidth=200)

    def visualize_rot_mat(self, rot_mat: List, pos):
        dx = 0.1
        colors = np.eye(3)
        for i in range(3):
            self.add_debug_item('sphere', 'step', lineFromXYZ=[pos[0], pos[1], pos[2]],
                                lineToXYZ=[pos[0] + rot_mat[i][0] * dx, pos[1] + rot_mat[i][1] * dx,\
                                           pos[2] + rot_mat[i][2] * dx],
                                lineColorRGB=colors[i],
                                lineWidth=200)
