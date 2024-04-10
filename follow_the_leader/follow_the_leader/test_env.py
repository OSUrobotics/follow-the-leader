#!/usr/bin/env python3

import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from bullet_sim import MESHES_AND_URDF_PATH, active_scan_env

if __name__ == "__main__":
    print(f"looking {os.path.join(MESHES_AND_URDF_PATH, 'urdf/trees/envy/train')}")
    env = active_scan_env.ActiveScanEnv(tree_urdf_path=os.path.join(MESHES_AND_URDF_PATH, "urdf/trees/envy/train"), tree_obj_path=os.path.join(MESHES_AND_URDF_PATH, "meshes/trees/envy/train"), renders=True)
    for i in range(200):
        env.render(mode="human")