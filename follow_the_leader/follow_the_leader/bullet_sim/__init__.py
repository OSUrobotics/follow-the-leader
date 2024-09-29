import os

# Global URDF path pointing to robot and supports URDF

MESHES_AND_URDF_PATH = os.path.abspath("/home/roosh/pruning_sb3/meshes_and_urdf")
ROBOT_URDF_PATH = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'ur5e', 'ur5e_cutter_new_calibrated_precise_level.urdf')
SUPPORT_AND_POST_PATH = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'supports_and_post.urdf')

