import sqlite3

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from follow_the_leader.utils.ros_utils import PinholeCameraModelNP, TFNode
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from itertools import product
from collections import defaultdict
import pandas as pd
from scipy.stats import ttest_ind
from sensor_msgs.msg import CameraInfo, RegionOfInterest
import pyvista as pv
import cv2
from PIL import Image


sample_cam_info = CameraInfo(
    height=480,
    width=848,
    distortion_model="plumb_bob",
    binning_x=0,
    binning_y=0,
    d=[-0.05469128489494324, 0.05773274227976799, 7.857435412006453e-05, 0.0003967129159718752, -0.018736450001597404],
    k=[437.00222778, 0.0, 418.9420166, 0.0, 439.22055054, 240.41038513, 0.0, 0.0, 1.0],
    r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    p=[437.00222778, 0.0, 418.9420166, 0.0, 0.0, 439.22055054, 240.41038513, 0.0, 0.0, 0.0, 1.0, 0.0],
    roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
)

cam = PinholeCameraModelNP()
cam.fromCameraInfo(sample_cam_info)

"""
Uses PyVista to plot the results
The ground truth can be either a probes.csv file (real data) or a ground_truth.pickle file (simulated data)
"""


def reconstruct_probe_list(vals, probe_len=0.128, radius_unit=1e-3):
    branch_id = -1
    leader_pts = []
    leader_radii = []
    side_branches = []
    side_branches_radii = []
    for row in vals:
        if np.abs(row).sum() == 0:
            branch_id += 1
            side_branches.append([])
            side_branches_radii.append([])
            continue

        pos = row[:3]
        quat = row[3:7]
        radius = radius_unit * row[7] / 2

        tf = np.identity(4)
        tf[:3, 3] = pos
        tf[:3, :3] = Rotation.from_quat(quat).as_matrix()
        pt = TFNode.mul_homog(tf, [0, 0, probe_len + radius])

        if branch_id < 0:
            leader_pts.append(pt)
            leader_radii.append(radius)
        else:
            side_branches[branch_id].append(pt)
            side_branches_radii[branch_id].append(radius)

    return {
        "leader": np.array(leader_pts),
        "leader_radii": np.array(leader_radii),
        "side_branches": [np.array(pts) for pts in side_branches if len(pts)],
        "side_branches_radii": [np.array(radii) for radii in side_branches_radii],
    }


def process_file(results_file):

    is_gt = "ground_truth" in results_file
    is_probe = "probe" in results_file

    radii = []
    if is_gt:

        with open(results_file, "rb") as fh:
            info = pickle.load(fh)

        branches = [info["leader"]] + info["side_branches"]
        for i, branch in enumerate(branches):
            if i == 0:
                radius = info["leader_radius"]
            else:
                radius = info["side_branch_radius"]

            radii.append([radius] * len(branch))

    elif is_probe:
        vals = np.genfromtxt(results_file, delimiter=",")
        info = reconstruct_probe_list(vals, probe_len=0.1072)
        branches = [info["leader"]] + info["side_branches"]
        radii = [info["leader_radii"]] + info["side_branches_radii"]

    else:

        with open(results_file, "rb") as fh:
            info = pickle.load(fh)

        branches = []

        raw_models = [info["leader_raw"]] + info["side_branches_raw"]
        for i, raw_branch in enumerate(raw_models):

            cur_branch = []
            cur_radii = []

            for point_hist in raw_branch.model:
                pt = point_hist.as_point(np.identity(4))
                if pt is None:
                    continue
                radius = point_hist.radius
                if radius is None:
                    continue

                cur_branch.append(pt)
                cur_radii.append(radius)

            # Check if too short
            if i != 0 and np.linalg.norm((cur_branch[-1] - cur_branch[0]) * (1, 1, 0)) <= 0.025:
                continue

            branches.append(np.array(cur_branch))
            radii.append(np.array(cur_radii))

    return {
        "branches": [np.array(branch) for branch in branches],
        "radii": [np.array(rs) for rs in radii],
    }


def pose_array_to_matrix(vec):
    tf = np.identity(4)
    tf[:3, 3] = vec[:3]
    tf[:3, :3] = Rotation.from_quat(vec[3:]).as_matrix()
    return tf


def single_plot(tree_info, target_z_val=0.50):

    plotter = pv.Plotter()
    branches = tree_info["branches"]
    radii = tree_info["radii"]

    branch_cyls = []
    interp_dist = 0.04
    selected_cyl = None
    cam_pos = None
    reinterp_leader = None

    for i, (branch, branch_radii) in enumerate(zip(branches, radii)):

        dists = np.zeros(len(branch))
        dists[1:] = np.linalg.norm(branch[1:] - branch[:-1], axis=1).cumsum()

        branch_interp = interp1d(dists, branch.T)
        radius_interp = interp1d(dists, branch_radii)
        new_ds = np.arange(0, dists[-1], interp_dist)

        branch = branch_interp(new_ds).T
        branch_radii = radius_interp(new_ds)

        if i == 0:
            reinterp_leader = branch

        cur_branch_cyl = []
        for j in range(len(branch) - 1):
            pt_1 = branch[j]
            pt_2 = branch[j + 1]

            if target_z_val is not None:
                if i == 0 and (pt_1[2] < target_z_val < pt_2[2]):
                    selected_cyl = j
                    target_pt = pt_1 + (pt_2 - pt_1) * (target_z_val - pt_1[2]) / (pt_2[2] - pt_1[2])

                    cam_pos = np.identity(4)
                    cam_pos[:3, 3] = target_pt - [0.26, 0.01, 0]
                    cam_pos[:3, :3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T

                elif pt_2[2] > target_z_val + 0.06:
                    break

            radius = (branch_radii[j] + branch_radii[j + 1]) / 2

            cyl_center = (pt_1 + pt_2) / 2
            cyl = pv.Cylinder(
                cyl_center,
                direction=pt_2 - pt_1,
                radius=radius,
                height=np.linalg.norm(pt_2 - pt_1),
                resolution=50,
                capping=False,
            )
            cur_branch_cyl.append(cyl)
        branch_cyls.append(cur_branch_cyl)

    for i, cyl_collection in enumerate(branch_cyls):
        for j, cyl in enumerate(cyl_collection):
            if i == 0 and j == selected_cyl:
                color = "yellow"
            elif i != 0:
                color = "green"
            elif i == 0 and selected_cyl is None and j > len(cyl_collection) - 6:
                color = "yellow"
            else:
                color = "blue"

            plotter.add_mesh(cyl, color=color, smooth_shading=True, opacity=1)

    if cam_pos is not None:

        cone = pv.Cone(cam_pos[:3, 3] + [0.10, 0, 0], [-1, 0, 0], height=0.20, radius=0.02, resolution=100)
        plotter.add_mesh(cone, color="grey", smooth_shading=True, opacity=0.3)
        cylinder = pv.Cylinder(cam_pos[:3, 3] + [0.23, 0, 0], [1, 0, 0], radius=0.001, height=0.06)
        plotter.add_mesh(cylinder, color="red", smooth_shading=True, opacity=0.2)

        fake_img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

        branch_pxs = cam.project3dToPixel(TFNode.mul_homog(np.linalg.inv(cam_pos), reinterp_leader))
        cv2.polylines(fake_img, [branch_pxs.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 255), 2)

        cv2.line(
            fake_img,
            branch_pxs[selected_cyl].astype(int),
            branch_pxs[selected_cyl + 1].astype(int),
            (255, 255, 0),
            thickness=4,
        )

        output_path = "/home/main/Documents/Dissertation and ICRA 2024 Paper"
        Image.fromarray(fake_img).save(os.path.join(output_path, "reprojected_model.png"))

    plotter.show()


def plot(gt_info, estimated_info, interp_dist=0.02):
    plotter = pv.Plotter(shape=(1, 3))

    all_cylinders = []

    for i, tree_info in enumerate([gt_info, estimated_info]):

        plotter.subplot(0, i)

        branches = tree_info["branches"]
        radii = tree_info["radii"]

        current_tree_cyls = []

        for _, (branch, branch_radii) in enumerate(zip(branches, radii)):

            dists = np.zeros(len(branch))
            dists[1:] = np.linalg.norm(branch[1:] - branch[:-1], axis=1).cumsum()

            branch_interp = interp1d(dists, branch.T)
            radius_interp = interp1d(dists, branch_radii)
            new_ds = np.arange(0, dists[-1], interp_dist)

            branch = branch_interp(new_ds).T
            branch_radii = radius_interp(new_ds)

            color = "blue" if i == 0 else "green"
            cyls = []
            for j in range(len(branch) - 1):
                pt_1 = branch[j]
                pt_2 = branch[j + 1]
                radius = (branch_radii[j] + branch_radii[j + 1]) / 2

                cyl_center = (pt_1 + pt_2) / 2
                cyl = pv.Cylinder(
                    cyl_center,
                    direction=pt_2 - pt_1,
                    radius=radius,
                    height=np.linalg.norm(pt_2 - pt_1),
                    resolution=50,
                    capping=False,
                )

                plotter.add_mesh(cyl, color=color, smooth_shading=True, opacity=1)
                cyls.append(cyl)

            current_tree_cyls.extend(cyls)

        if i == 0:
            plotter.add_text("Ground Truth")
        elif i == 1:
            plotter.add_text("Estimated")
        all_cylinders.append(current_tree_cyls)

    plotter.subplot(0, 2)
    for cyl in all_cylinders[0]:  # GT
        plotter.add_mesh(cyl, color="grey", smooth_shading=True, opacity=0.2)

    for cyl in all_cylinders[1]:  # EST
        plotter.add_mesh(cyl, color="red", smooth_shading=True, opacity=1.0)

    plotter.show()


if __name__ == "__main__":

    root = "/home/main/data/model_the_leader/real_data"
    tree_id = 0
    # run_id = '009'
    run_id = "006"  # 3/006 is good illustration
    gt_file = os.path.join(root, str(tree_id), "probes.csv")
    eval_file = os.path.join(root, str(tree_id), run_id, "0_results.pickle")

    gt_info = process_file(gt_file)
    eval_info = process_file(eval_file)
    plot(gt_info, eval_info)

    exit()

    identifiers = ["{}_3".format(i) for i in [0, 5, 10, 15, 20, 25]]
    for identifier in identifiers:
        gt_file = f"/home/main/data/model_the_leader/{identifier}_ground_truth.pickle"
        eval_file = f"/home/main/data/model_the_leader/{identifier}_results.pickle"

        gt_info = process_file(gt_file)
        eval_info = process_file(eval_file)

        # single_plot(eval_info, target_z_val=None)
        plot(gt_info, eval_info)
