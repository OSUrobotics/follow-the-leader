import sqlite3

import cv_bridge
import matplotlib.pyplot as plt
import pickle
import os
import sys
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
from copy import deepcopy
import open3d.core as o3c
import open3d as o3d
import open3d.t.pipelines.registration as treg
from transforms3d.quaternions import quat2mat, mat2quat

matrix_data = [[9.95941040e-01, 9.00034445e-02, -9.08589981e-04, -5.66533854e-03],
               [-8.97696700e-02, 9.92518806e-01, -8.27515970e-02, 2.48555284e-01],
               [-6.54613613e-03, 8.24972754e-02, 9.96569791e-01, 1.25147467e-02],
               [0., 0., 0., 1.]]

"""
Runs analysis on the results of the real experiments and outputs the stats used in the paper.
"""

def draw_registration_result(source, target, transformation):
    print(transformation.shape)
    source.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source.to_legacy(),
         target.to_legacy()])
    
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_xlabel("x")
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_xlabel("y")
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.set_zlabel("z")

def avg_matrix(matrices):
    quaternions = [mat2quat(matrix[:3, :3]) for matrix in matrices]
    translations = [matrix[:3, 3] for matrix in matrices]
    # print(translations)
    avg_t = np.average(translations, axis=0)
    # print(avg_t)
    
    # Number of quaternions to average
    samples = len(quaternions)
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)

    for i in range(0, samples):
        quat = quaternions[i]
        # multiply quat with its transposed version quat' and add mat_a
        mat_a = np.outer(quat, quat) + mat_a

    # scale
    mat_a = (1.0/ samples)*mat_a
    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)
    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    avg_quat = np.real(np.ravel(eigen_vectors[:, 0]))
    avg_rot = quat2mat(avg_quat)
    avg_homog = np.vstack([np.column_stack([avg_rot, avg_t]), [0, 0, 0, 1]])
    print(avg_homog)
    return avg_homog

def process_final_data(input_path, trial, run, pickle_id):
    import yaml
    run_path = os.path.join(input_path, trial, run)
    data = {}

    probed_data = []
    print(f"{os.path.split(run_path)[0]}")
    real_data_path = os.path.split(run_path)[0]

    # # trying to reverse engineer from probes data
    # for branch_id in sorted(os.listdir(real_data_path)):
    #     if not os.path.isdir(os.path.join(real_data_path, branch_id)):
    #         continue
    probes_file = os.path.join(real_data_path, "probes.csv")
    probed_data = np.genfromtxt(probes_file, delimiter=",")
        # # indicate a new branch by adding a row of infs
        # marker_to_switch = np.zeros((probed_branch.shape[1]))
        # # marker_to_switch.fill(np.Inf)
        # probed_data.append(marker_to_switch)
        # probed_data.extend([i for i in probed_branch if not np.isin(i, probed_data).all()])
        # break
    # print(probed_data)
    gt_data = reconstruct_probe_list(probed_data, probe_len=0.1079)
    # print(gt_data)
    # print(len(gt_data["side_branches"]))
    # exit(0)
    with open(os.path.join(run_path, pickle_id), "rb") as fh:
        eval_data = pickle.load(fh)

    bag_file_db = os.path.join(run_path, "bag_data", "bag_data_0.db3")
    reader = BagReader(bag_file_db)
    camera_info = list(reader.query("/camera/color/camera_info"))[0][1]
    poses = np.array([pose_to_tf(pose) for _, pose in reader.query("/camera_pose")])
    camera = PinholeCameraModelNP()
    camera.fromCameraInfo(camera_info)

    try:
        query = list(reader.query("/controller_params"))
        controller_params = query[-1][1]
        global config
        config = {"ee_speed": controller_params.ee_speed, "pan_frequency": int(controller_params.pan_frequency),
                   "pan_magnitude_deg": controller_params.pan_magnitude_deg, "z_desired": controller_params.z_desired}
        if int(controller_params.pan_frequency) != 0:
            raise("Doing wihout rotation tests")   
        data.update(config)
    
    except Exception as e:
        try:
            import yaml
            with open(os.path.join(run_path, "config.yaml"), "r") as fh:
                config = yaml.safe_load(fh)
                config = {k: config[k] for k in ["ee_speed", "pan_frequency", "pan_magnitude_deg", "z_desired"]}
                data.update(config)
        except Exception as ee:
            print(f"USING OLD CONFIG {ee}")
            data.update(config)


    # # TEMP
    # bla = '/home/main/test'
    # bridge = cv_bridge.CvBridge()
    # from PIL import Image
    # for i, (ts, msg) in enumerate(reader.query('/model_diagnostic')):
    #
    #     img_msg = bridge.imgmsg_to_cv2(msg, 'rgb8')
    #     Image.fromarray(img_msg).save(os.path.join(bla, 'diag_imgs', f'{ts}.png'))
    #
    # for ts, msg in reader.query('/camera/color/image_raw'):
    #     img_msg = bridge.imgmsg_to_cv2(msg, 'rgb8')
    #     Image.fromarray(img_msg).save(os.path.join(bla, 'rgb', f'{ts}.png'))

    w = camera.width
    h = camera.height

    # callback_after_iteration = lambda updated_result_dict : print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    # updated_result_dict["iteration_index"].item(),
    # updated_result_dict["fitness"].item(),
    # updated_result_dict["inlier_rmse"].item()))

    leader_pts = reinterp_point_list(gt_data["leader"], by_n=20000)[0]
    est_leader_pts = eval_data["leader"]


    min_z = max(leader_pts[:, 2].min(), est_leader_pts[:, 2].min())
    max_z = min(leader_pts[:, 2].max(), est_leader_pts[:, 2].max())

    zs = leader_pts[:, 2]
    leader_eval = leader_pts[(zs >= min_z) & (zs <= max_z)]
    tree = KDTree(leader_eval)

    zs = est_leader_pts[:, 2]
    est_leader_pts = est_leader_pts[(zs >= min_z) & (zs <= max_z)]

    dists = tree.query(est_leader_pts)[0]
    data["Average Distance"] = dists.mean()

    # Radius analysis
    leader_radii_raw = gt_data["leader_radii"]
    z_vals = gt_data["leader"][:, 2]
    radius_interp = interp1d(z_vals, leader_radii_raw)

    raw_leader = eval_data["leader_raw"]
    raw_info = [(pt.as_point(np.identity(4)), pt.radius) for pt in raw_leader.model]
    radii = np.array([radius for pt, radius in raw_info if pt is not None and (min_z <= pt[2] <= max_z)])
    corresponding_z_vals = [pt[2] for pt, _ in raw_info if pt is not None and (min_z <= pt[2] <= max_z)]
    interp_radii = radius_interp(corresponding_z_vals)
    data["Leader Radius Error"] = radii - interp_radii
    # data["Leader Radius Error %"] = np.mean(radii) / interp_radii - 1

    # Regarding the positioning of the robot with respect to the GT

    ee_pos = np.array([pose[:3, 3] for pose in poses])
    data["Completed"] = ee_pos[:, 2].max() > 0.75 * leader_pts[:, 2].max()

    # Side branch analysis
    sb_results, branch_data = analyze_side_branch_data(
        gt_data, eval_data, initial_pose=poses[0], max_z=max_z, visualize=True, save_fig=os.path.join(run_path, pickle_id)
    )

    for branch in branch_data:
        branch.update(config)
    data.update(sb_results)

    # z_errs = []
    # centering_errs = []
    #
    # for pose_1, pose_2 in zip(poses[:-1], poses[1:]):
    #
    #     if pose_1[2,3] < 0.325 or pose_2[2,3] > 0.75:
    #         continue
    #
    #     rot = Rotation.from_matrix(pose_1[:3,:3].T @ pose_2[:3,:3]).as_euler('XYZ')
    #     if np.linalg.norm(rot) > 0.005:
    #         continue
    #
    #     cam_pts = TFNode.mul_homog(np.linalg.inv(pose_1), leader_eval)
    #     leader_pxs = camera.project3dToPixel(cam_pts)
    #     center_closest = np.argmin(np.abs(leader_pxs[:,1] - h/2))
    #     center_px = leader_pxs[center_closest]
    #     px_error = np.abs(center_px[0] - w / 2) / (w / 2)
    #     z_error = np.abs(0.20 - cam_pts[center_closest][2])
    #
    #     z_errs.append(z_error)
    #     centering_errs.append(px_error)
    #
    # data['Average Z-Error'] = np.mean(z_errs)
    # data['Average Abs Z-Error'] = np.abs(z_errs).mean()
    # data['Max Z-Error'] = np.max(z_errs)
    # data['Average Centering Error'] = np.mean(centering_errs)
    # data['Average Abs Centering Error'] = np.abs(centering_errs).mean()
    # data['Max Centering Error'] = np.max(centering_errs)

    unaggregated_data = {
        "PB Radius Error": radii - interp_radii,
        "PB Residuals": dists,
    }

    return data, branch_data, unaggregated_data


def analyze_side_branch_data(gt_data, eval_data, initial_pose, max_z=1.0, visualize=True, save_fig=None):
    overall_sb = {}

    sbs_gt = [reinterp_point_list(sb, by_dist=0.001)[0] for sb in gt_data["side_branches"]]
    sbs_eval = [reinterp_point_list(sb, by_dist=0.001)[0] for sb in eval_data["side_branches"]]

    # # If it didn't complete, don't add branches that it didn't reach
    # is_gt = [i for i, sb in enumerate(sbs_gt) if sb[0, 2] <= max_z + 0.02]
    # sbs_gt = [sbs_gt[i] for i in is_gt]
    # # sbs_raw_gt = [gt_data['side_branches_raw'][i] for i in is_gt]

    # # Filter out too short SBs
    # is_eval = [i for i, sb in enumerate(sbs_eval) if np.linalg.norm((sb[-1] - sb[0]) * [1, 1, 0]) > 0.025]
    # sbs_eval = [sbs_eval[i] for i in is_eval]
    # sbs_raw_eval = [eval_data["side_branches_raw"][i] for i in is_eval]

    kdtrees_gt = [KDTree(pts) for pts in sbs_gt]
    matches = defaultdict(list)

    for i_eval, sb_eval in enumerate(sbs_eval):
        vec_eval = normalize(sb_eval[-1] - sb_eval[0])

        for i_gt, sb_gt in enumerate(sbs_gt):
            vec_gt = normalize(sb_gt[-1] - sb_gt[0])
            if np.arccos(vec_eval @ vec_gt) > np.radians(60):
                continue

            max_idx = min(len(sb_gt), len(sb_eval), 50)
            dists, _ = kdtrees_gt[i_gt].query(sb_eval[:max_idx])

            if np.median(dists) < 0.05:
                # print('Match found (median dist was {:.5f})'.format(np.median(dists)))
                matches[i_gt].append(i_eval)

    matched_status_gt = {}
    matched_status_eval = {}

    for i_gt, candidate_sb_is in matches.items():
        pts_gt = sbs_gt[i_gt]
        # len_gt = np.linalg.norm(pts_gt[1:] - pts_gt[:-1], axis=1).sum()

        best_i_sb = None
        best_i_sb_diff = np.inf
        for i_sb in candidate_sb_is:
            pts_sb = sbs_eval[i_sb]
            # len_sb = np.linalg.norm(pts_sb[1:] - pts_sb[:-1], axis=1).sum()

            idx_max = min(len(pts_sb), len(pts_gt))
            diff = np.linalg.norm(pts_sb[:idx_max] - pts_gt[:idx_max], axis=1).mean()

            # diff = abs(len_sb - len_gt)
            if diff < best_i_sb_diff:
                best_i_sb = i_sb
                best_i_sb_diff = diff

        if best_i_sb is not None:
            matched_status_gt[i_gt] = best_i_sb
            matched_status_eval[best_i_sb] = i_gt

    matched_ids = list(matched_status_gt.values())
    if len(matched_ids) != len(set(matched_ids)):
        print("A single side branch got matched to more than 1 GT branch! Figure out why")
        import pdb

        # pdb.set_trace()

    overall_sb["GT Branches"] = len(sbs_gt)
    overall_sb["Missed Branches"] = len([i for i in range(len(sbs_gt)) if matched_status_gt.get(i) is None])
    overall_sb["Overdetected Side Branches"] = 0
    overall_sb["Spurious Side Branches"] = 0

    for i in range(len(sbs_eval)):
        if i in matched_status_eval:
            continue
        for candidates in matches.values():
            if i in candidates:
                overall_sb["Overdetected Side Branches"] += 1
                break
        else:
            overall_sb["Spurious Side Branches"] += 1
    overall_sb['False positive rate'] = (overall_sb["Spurious Side Branches"]) / overall_sb["GT Branches"]
    overall_sb['False negative rate'] = overall_sb["Missed Branches"] / overall_sb["GT Branches"]
    missed_data = False
    if overall_sb["Spurious Side Branches"] or overall_sb["Overdetected Side Branches"] or overall_sb["Missed Branches"]:
        missed_data = True
        # print(data)

    branch_statistics = []

    for i_match_gt, i_match_eval in matched_status_gt.items():
        branch_data = {}

        pts_gt = sbs_gt[i_match_gt]
        pts_eval = sbs_eval[i_match_eval]

        len_gt = get_len(pts_gt)
        len_eval = get_len(pts_eval)
        branch_data["Length Error"] = len_eval - len_gt
        branch_data["Length Error %"] = len_eval / len_gt - 1

        idx_to_use = min(30, min(len(pts_gt), len(pts_eval)) - 1)
        vec_gt_init = normalize(pts_gt[idx_to_use] - pts_gt[0])
        vec_eval_init = normalize(pts_eval[idx_to_use] - pts_eval[0])
        branch_data["Angle Error"] = np.arccos(vec_gt_init @ vec_eval_init)

        initial_pose_vec = initial_pose[:3, :3] @ np.array([0, 0, 1])
        planar_rotation = np.arccos(initial_pose_vec @ normalize(vec_gt_init * [1, 1, 0]))
        branch_data["GT Planar Rotation"] = planar_rotation

        min_z = max(pts_gt[:, 2].min(), pts_eval[:, 2].min())
        max_z = min(pts_gt[:, 2].max(), pts_eval[:, 2].max())

        zs = pts_gt[:, 2]
        branch_eval = pts_gt[(zs >= min_z) & (zs <= max_z)]

        # Radius analysis
        sb_radii = gt_data["side_branches_radii"][i_match_gt]
        z_vals = gt_data["side_branches"][i_match_gt][:, 2]
        radius_interp = interp1d(z_vals, sb_radii)

        raw_sb = eval_data["side_branches_raw"][i_match_eval]
        raw_info = [(pt.as_point(np.identity(4)), pt.radius) for pt in raw_sb.model]
        eval_radii = np.array([radius for pt, radius in raw_info if pt is not None and (min_z <= pt[2] <= max_z)])
        corresponding_z_vals = [pt[2] for pt, _ in raw_info if pt is not None and (min_z <= pt[2] <= max_z)]
        interp_radii = radius_interp(corresponding_z_vals)

        print(f"SB radius {sb_radii} \n  Interpolated  \n{interp_radii}")
        branch_data["SB Radius Error"] = eval_radii - interp_radii


        tree = KDTree(branch_eval)

        zs = pts_eval[:, 2]
        est_sb_pts = pts_eval[(zs >= min_z) & (zs <= max_z)]

        dists = tree.query(est_sb_pts)[0]
        branch_data["SB Residuals"] = dists

        branch_statistics.append(branch_data)

    if visualize:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        ax.plot(*gt_data["leader"].T, color="orange", linestyle="dashed")
        ax.plot(*eval_data["leader"].T, color="brown")

        for i_gt, sb_gt in enumerate(sbs_gt):
            color = "green"
            match_status = matched_status_gt.get(i_gt, None)
            if match_status is None:
                color = "red"

            ax.plot(*sb_gt.T, color=color, linestyle="dashed")

        for i_eval, sb_eval in enumerate(sbs_eval):
            color = "green"
            match_status = matched_status_eval.get(i_eval, None)
            if match_status is None:
                color = "pink"

            ax.plot(*sb_eval.T, color=color)

        set_axes_equal(ax)
        plt.show()
        fig.savefig(f"{save_fig}_plot.png")
        # plt.close()

    return overall_sb, branch_statistics


class BagReader:
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()
        topics_data = self.conn.execute("SELECT id, name, type FROM topics").fetchall()

        self.topics = [r[1] for r in topics_data]
        self.topic_name_to_id = {r[1]: r[0] for r in topics_data}
        self.topic_name_to_type = {r[1]: get_message(r[2]) for r in topics_data}

    def query(self, topic_name):
        topic_id = self.topic_name_to_id[topic_name]
        topic_msg_type = self.topic_name_to_type[topic_name]

        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id))
        for ts, data in rows:
            yield ts, deserialize_message(data, topic_msg_type)


def pose_to_tf(pose_stamped: PoseStamped):
    tl = pose_stamped.pose.position
    q = pose_stamped.pose.orientation

    tl = np.array([tl.x, tl.y, tl.z])
    q = np.array([q.x, q.y, q.z, q.w])

    tf = np.identity(4)
    tf[:3, 3] = tl
    tf[:3, :3] = Rotation.from_quat(q).as_matrix()

    return tf


def reinterp_point_list(pts, by_dist=None, by_n=None):
    if by_dist is None and by_n is None:
        raise ValueError("Please specify either one of by_dist or by_n")

    cum_dists = np.zeros(len(pts))
    cum_dists[1:] = np.linalg.norm(pts[1:] - pts[:-1], axis=1).cumsum()

    max_dist = cum_dists[-1]

    interp = interp1d(cum_dists, pts.T)
    if by_dist is not None:
        ds = np.arange(0, max_dist, by_dist)
    else:
        ds = np.linspace(0, max_dist, by_n)

    return interp(ds).T, max_dist


def reconstruct_probe_list(vals, probe_len=0.128, radius_unit=1e-3):
    branch_id = -1
    leader_pts = []
    leader_radii = []
    side_branches = []
    side_branches_radii = []
    for row in vals:
        if np.isclose(np.abs(row).sum(), 0.): # our hack to switch branches
            branch_id += 1
            side_branches.append([])
            side_branches_radii.append([])
            continue
        # if np.abs(row).sum() == 0: # null probe
        #     continue

        pos = row[:3]
        quat = row[3:7]
        radius = radius_unit * row[7] / 2

        tf = np.identity(4)
        tf[:3, 3] = pos
        tf[:3, :3] = Rotation.from_quat(quat).as_matrix()
        pt = TFNode.mul_homog(tf, [0, 0, probe_len + radius])

        # offset_matrix = np.array(matrix_data)
        # pt = TFNode.mul_homog(offset_matrix, pt)

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
        "side_branches_radii": [np.array(radii) for radii in side_branches_radii if len(radii)],
    }


def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_len(pts):
    return np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()

def calculate_difference(matrix1, matrix2):
    r1 = Rotation.from_matrix(matrix1[:3, :3])
    r2 = Rotation.from_matrix(matrix2[:3, :3])
    rotation_difference = r1.inv() * r2
    translation_difference = matrix1[:3, 3] - matrix2[:3, 3]
    return rotation_difference, translation_difference

if __name__ == "__main__":
    global matrix_tf
    matrix_tf = []
    global config 
    config = {"ee_speed": 0.4, "pan_frequency": 0, "pan_magnitude_deg": 0., "z_desired": 0.4}
    try:
        input_path = str(sys.argv[1]) # has multiple runs organised by folder
    except:
        input_path = os.path.join(os.path.expanduser("~"), "data", "model_the_leader", "real_data")

    trees = []
    runs = []
    pickles = []
    for folder_id in sorted(os.listdir(input_path)):
        subfolder = os.path.join(input_path, str(folder_id))
        if not os.path.isdir(subfolder):
            continue
        trees.append(folder_id)
        for run in sorted(os.listdir(subfolder)):
            run_path = os.path.join(subfolder, run)
            if not os.path.isdir(run_path):
                continue
            runs.append((folder_id, run))
            for picklefile in sorted(os.listdir(run_path), reverse=True):
                result_path = os.path.join(run_path, picklefile)
                if not os.path.isfile(f"{result_path}") or not result_path.endswith(".pickle"):
                    continue  
                pickles.append((folder_id, run, picklefile))
                break # if you only want to analyse one pickle

    all_unaggregated = defaultdict(lambda: defaultdict(list))

    run_id_prev = None
    i = 0
    curr_tree_id = None
    for (tree_id, run_id, pickle_id) in pickles:
        # if not tree_id == "tree5":  # if you want to analyse one tree at atime
        #     continue
        if curr_tree_id != tree_id:
            config = {"ee_speed": 0.4, "pan_frequency": 0, "pan_magnitude_deg": 0., "z_desired": 0.4}
            curr_tree_id = tree_id
        print("Tree {}, run {}, pickle {}".format(tree_id, run_id, pickle_id))
        try:
            data, sb_data, unagg_data = process_final_data(input_path, tree_id, run_id, pickle_id)
            print(f"data: {data}\n, sb_data: {sb_data}\n, unagg_data: {unagg_data}\n")
        except Exception as e:
            print(f"Failure! {e}")
            continue

        identifier = "Rot" if data["pan_frequency"] != 0 else "Speed {:.2f}".format(data["ee_speed"])
        print(identifier)
        all_unaggregated[identifier]["Leader Residuals"].append(data["Average Distance"])
        all_unaggregated[identifier]["Leader Radius Error"].extend(data["Leader Radius Error"])

        for datum in sb_data:
            all_unaggregated[identifier]["SB Radius Error"].extend(datum["SB Radius Error"])
            all_unaggregated[identifier]["SB Residuals"].extend(datum["SB Residuals"])

        for key, val in unagg_data.items():
            all_unaggregated[identifier][key].extend(val)
        all_unaggregated[identifier]["z_desired"].append(data["z_desired"])
        all_unaggregated[identifier]['False positive rate'].append(data['False positive rate'])
        all_unaggregated[identifier]['False negative rate'].append(data['False negative rate'])

    unagg_summary = {}
    for param_set, unagg_data in all_unaggregated.items():
        rez = {}
        runs = len(unagg_data['False positive rate'])
        rez['runs'] = runs
        for stat, raw_data in unagg_data.items():
            print(f"raw data {raw_data} stat {stat}")
            raw_data = np.array(raw_data)
            rez[f"{stat} Mean"] = np.mean(raw_data)
            rez[f"{stat} RMSE"] = np.sqrt(np.mean(np.square(raw_data)))
            rez[f"{stat} Median"] = np.median(raw_data)
            rez[f"{stat} Stdev"] = np.std(raw_data)
        unagg_summary[param_set] = rez
    unagg_summary_df = pd.DataFrame(unagg_summary)
    unagg_summary_df[sorted(unagg_summary_df.columns)].to_csv(os.path.join(input_path, "without_icp.csv"))



    exit()
