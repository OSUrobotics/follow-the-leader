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
import pyvista as pv

"""
Processes the bag files for the simulated experiments and outputs statistics files.
"""


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
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def process_final_data(identifier, root):

    data = {}

    gt_file = os.path.join(root, f"{identifier}_ground_truth.pickle")
    eval_file = os.path.join(root, f"{identifier}_results.pickle")

    with open(gt_file, "rb") as fh:
        gt_data = pickle.load(fh)

    with open(eval_file, "rb") as fh:
        eval_data = pickle.load(fh)

    bag_file_db = os.path.join(root, f"{identifier}_data", f"{identifier}_data_0.db3")
    reader = BagReader(bag_file_db)
    camera_info = list(reader.query("/camera/color/camera_info"))[0][1]
    poses = np.array([pose_to_tf(pose) for _, pose in reader.query("/camera_pose")])
    camera = PinholeCameraModelNP()
    camera.fromCameraInfo(camera_info)

    w = camera.width
    h = camera.height

    # Compare leaders
    leader_pts = gt_data["leader"]
    leader_eval, _ = reinterp_point_list(leader_pts, by_n=10000)
    tree = KDTree(leader_eval)

    est_leader_pts = eval_data["leader"]
    zs = est_leader_pts[:, 2]
    est_leader_pts = est_leader_pts[(zs > 0.325) & (zs < 0.75)]

    dists = tree.query(est_leader_pts)[0]
    data["Average Distance"] = dists.mean()
    data["Median Distance"] = np.median(dists)

    # Radius analysis

    raw_leader = eval_data["leader_raw"]
    raw_info = [(pt.as_point(np.identity(4)), pt.radius) for pt in raw_leader.model]
    radii = np.array([radius for pt, radius in raw_info if pt is not None and (0.325 < pt[2] < 0.75)])
    data["Leader Radius Error"] = np.mean(radii) - gt_data["leader_radius"]
    data["Leader Radius Error %"] = np.mean(radii) / gt_data["leader_radius"] - 1
    data["Leader Radius Median Error"] = np.median(radii) - gt_data["leader_radius"]
    data["Leader Radius Median Error %"] = np.median(radii) / gt_data["leader_radius"] - 1

    # Regarding the positioning of the robot with respect to the GT

    ee_pos = np.array([pose[:3, 3] for pose in poses])
    data["Completed"] = ee_pos[:, 2].max() > 0.73

    # Side branch analysis
    sb_results, branch_data = analyze_side_branch_data(
        gt_data,
        eval_data,
        initial_pose=poses[0],
        max_z=1.0 if data["Completed"] else ee_pos[:, 2].max(),
        visualize=False,
    )
    data.update(sb_results)

    z_errs = []
    centering_errs = []

    for pose_1, pose_2 in zip(poses[:-1], poses[1:]):

        if pose_1[2, 3] < 0.325 or pose_2[2, 3] > 0.75:
            continue

        rot = Rotation.from_matrix(pose_1[:3, :3].T @ pose_2[:3, :3]).as_euler("XYZ")
        if np.linalg.norm(rot) > 0.005:
            continue

        cam_pts = TFNode.mul_homog(np.linalg.inv(pose_1), leader_eval)
        leader_pxs = camera.project3dToPixel(cam_pts)
        center_closest = np.argmin(np.abs(leader_pxs[:, 1] - h / 2))
        center_px = leader_pxs[center_closest]
        px_error = np.abs(center_px[0] - w / 2) / (w / 2)
        z_error = np.abs(0.20 - cam_pts[center_closest][2])

        z_errs.append(z_error)
        centering_errs.append(px_error)

    data["Average Z-Error"] = np.mean(z_errs)
    data["Average Abs Z-Error"] = np.abs(z_errs).mean()
    data["Average Z-Error"] = np.median(z_errs)
    data["Median Abs Z-Error"] = np.median(np.abs(z_errs))
    data["Max Z-Error"] = np.max(z_errs)
    data["Average Centering Error"] = np.mean(centering_errs)
    data["Average Abs Centering Error"] = np.abs(centering_errs).mean()
    data["Median Centering Error"] = np.median(centering_errs)
    data["Median Abs Centering Error"] = np.median(np.abs(centering_errs))
    data["Max Centering Error"] = np.max(centering_errs)

    unaggregated_data = {
        "Centering Error": centering_errs,
        "Z-Error": z_errs,
        "Radius Error": radii - gt_data["leader_radius"],
        "Radius Error %": radii / gt_data["leader_radius"] - 1,
        "Residuals": dists,
    }

    return data, branch_data, unaggregated_data


def analyze_side_branch_data(gt_data, eval_data, initial_pose, max_z=1.0, visualize=True):

    data = {}

    sbs_gt = [reinterp_point_list(sb, by_dist=0.001)[0] for sb in gt_data["side_branches"]]
    sbs_eval = [reinterp_point_list(sb, by_dist=0.001)[0] for sb in eval_data["side_branches"]]

    # If it didn't complete, don't add branches that it didn't reach
    is_gt = [i for i, sb in enumerate(sbs_gt) if sb[0, 2] <= max_z + 0.02]
    sbs_gt = [sbs_gt[i] for i in is_gt]
    # sbs_raw_gt = [gt_data['side_branches_raw'][i] for i in is_gt]

    # Filter out too short SBs
    is_eval = [i for i, sb in enumerate(sbs_eval) if np.linalg.norm((sb[-1] - sb[0]) * [1, 1, 0]) > 0.025]
    sbs_eval = [sbs_eval[i] for i in is_eval]
    sbs_raw_eval = [eval_data["side_branches_raw"][i] for i in is_eval]

    kdtrees_gt = [KDTree(pts) for pts in sbs_gt]
    matches = defaultdict(list)

    # Attempt to match the detected side branches against the ground truth side branches
    for i_eval, sb_eval in enumerate(sbs_eval):

        vec_eval = normalize(sb_eval[-1] - sb_eval[0])
        for i_gt, sb_gt in enumerate(sbs_gt):

            vec_gt = normalize(sb_gt[-1] - sb_gt[0])
            if np.arccos(vec_eval @ vec_gt) > np.radians(60):
                continue

            max_idx = min(len(sb_gt), len(sb_eval), 50)
            dists, _ = kdtrees_gt[i_gt].query(sb_eval[:max_idx])

            if np.median(dists) < 0.02:
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

        pdb.set_trace()

    data["GT Branches"] = len(sbs_gt)
    data["Missed Branches"] = len([i for i in range(len(sbs_gt)) if matched_status_gt.get(i) is None])
    data["Overdetected Side Branches"] = 0
    data["Spurious Side Branches"] = 0

    for i in range(len(sbs_eval)):
        if i in matched_status_eval:
            continue
        for candidates in matches.values():
            if i in candidates:
                data["Overdetected Side Branches"] += 1
                break
        else:
            data["Spurious Side Branches"] += 1

    missed_data = False
    if data["Spurious Side Branches"] or data["Overdetected Side Branches"] or data["Missed Branches"]:
        missed_data = True
        # print(data)

    branch_statistics = []

    # Quick analysis on unmatched branches
    for i_unmatched_gt in range(len(sbs_gt)):
        if i_unmatched_gt in matched_status_gt:
            continue

        branch_data = {"Matched": False}
        initial_pose_vec = initial_pose[:3, :3] @ np.array([0, 0, 1])
        pts_gt = sbs_gt[i_unmatched_gt]
        vec_gt_init = normalize(pts_gt[1] - pts_gt[0])
        planar_rotation = np.arccos(initial_pose_vec @ normalize(vec_gt_init * [1, 1, 0]))
        branch_data["GT Planar Rotation"] = planar_rotation
        branch_statistics.append(branch_data)

    for i_match_gt, i_match_eval in matched_status_gt.items():

        branch_data = {"Matched": True}

        pts_gt = sbs_gt[i_match_gt]
        pts_eval = sbs_eval[i_match_eval]

        len_gt = get_len(pts_gt)
        len_eval = get_len(pts_eval)
        branch_data["Length Error"] = len_eval - len_gt
        branch_data["Length Error %"] = len_eval / len_gt - 1

        vec_gt_init = normalize(pts_gt[1] - pts_gt[0])
        vec_eval_init = normalize(pts_eval[1] - pts_eval[0])
        branch_data["Angle Error"] = np.arccos(vec_gt_init @ vec_eval_init)

        initial_pose_vec = initial_pose[:3, :3] @ np.array([0, 0, 1])
        planar_rotation = np.arccos(initial_pose_vec @ normalize(vec_gt_init * [1, 1, 0]))
        branch_data["GT Planar Rotation"] = planar_rotation

        raw_sb = sbs_raw_eval[i_match_eval]
        radii = np.array(list(filter(lambda x: x is not None, [pt.radius for pt in raw_sb.model])))
        radius_error = radii.mean() - gt_data["side_branch_radius"]
        branch_data["Radius Error"] = radius_error
        branch_data["Radius Error %"] = radii.mean() / gt_data["side_branch_radius"] - 1
        branch_data["Radius Error Median"] = np.median(radii) - gt_data["side_branch_radius"]
        branch_data["Radius Error Median %"] = np.median(radii) / gt_data["side_branch_radius"] - 1

        branch_statistics.append(branch_data)

    if visualize and missed_data:
        ax = plt.figure().add_subplot(projection="3d")

        ax.plot(*gt_data["leader"].T, color="orange", linestyle="dashed")
        ax.plot(*eval_data["leader"].T, color="orange")

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

    return data, branch_statistics


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


def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_len(pts):
    return np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()


if __name__ == "__main__":
    folder = os.path.join(os.path.expanduser("~"), "data", "model_the_leader")
    total_experiments = 50
    max_branches = 5

    param_set_rotation = {0: 0, 1: 1.5, 2: 1.5, 3: 2.5, 4: 2.5}
    param_set_angle = {0: 0, 1: 22.5, 2: 45, 3: 22.5, 4: 45}

    all_data = []
    all_branch_data = []
    all_unaggregated = defaultdict(lambda: defaultdict(list))

    for experiment_id, branches in product(np.arange(total_experiments), np.arange(max_branches + 1)):
        identifier = f"{experiment_id}_{branches}"
        tree_id, param_set_id = divmod(experiment_id, 5)
        cache_file = os.path.join(folder, f"{identifier}_stats.pickle")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fh:
                info = pickle.load(fh)

        else:
            data = {
                "rotation": param_set_rotation[param_set_id],
                "angle": param_set_angle[param_set_id],
            }
            print(identifier)
            exp_data, branch_data, unaggregated_data = process_final_data(identifier, folder)
            data.update(exp_data)
            all_branch_data.extend(branch_data)

            info = {"exp_data": data, "branch_data": branch_data, "unaggregated_data": unaggregated_data}

            with open(cache_file, "wb") as fh:
                pickle.dump(info, fh)

        info["exp_data"]["Rotation Frequency"] = param_set_rotation[param_set_id]
        info["exp_data"]["Lookat Angle"] = param_set_angle[param_set_id]

        all_data.append(info["exp_data"])
        for branch_data in info["branch_data"]:
            branch_data["Rotation Frequency"] = param_set_rotation[param_set_id]
            branch_data["Lookat Angle"] = param_set_angle[param_set_id]

        for key, val in info["unaggregated_data"].items():
            all_unaggregated[param_set_id][key].extend(val)

        all_branch_data.extend(info["branch_data"])

    experiments_df = pd.DataFrame(all_data)
    experiments_df["Abs Leader Radius Error %"] = experiments_df["Leader Radius Error %"].abs()

    branches_df = pd.DataFrame(all_branch_data)
    branches_df["Abs Length Error"] = branches_df["Length Error"].abs()
    branches_df["Abs Length Error %"] = branches_df["Length Error %"].abs()
    branches_df["Abs Radius Error"] = branches_df["Radius Error"].abs()
    branches_df["Abs Radius Error %"] = branches_df["Radius Error %"].abs()

    # # Warning: Will now have NaNs due to the presence of unmatched branches
    # no_rot = branches_df['Rotation Frequency'] == 0
    # unmatched = branches_df[no_rot & (branches_df['Matched'] == False)]['GT Planar Rotation']
    # matched = branches_df[no_rot & (branches_df['Matched'] == True)]['GT Planar Rotation']
    # plt.scatter(unmatched, [1] * len(unmatched), color='red', marker='^')
    # plt.scatter(matched, [0] * len(matched), color='green', marker='o')
    # plt.xlabel('Rotation relative to initial camera pose')
    # plt.show()
    #

    branches_df = branches_df[branches_df["Matched"] == True]

    stats_exp = experiments_df.groupby(["Rotation Frequency", "Lookat Angle"]).mean()
    stats_branches = branches_df.groupby(["Rotation Frequency", "Lookat Angle"]).mean()

    experiments_df_baseline = experiments_df.query("`Rotation Frequency` == 0 & `Lookat Angle` == 0")
    param_sets = [[1.5, 22.5], [1.5, 45.0], [2.5, 22.5], [2.5, 45.0]]
    stats_to_test_exp = ["Average Distance", "Abs Leader Radius Error %", "Average Abs Z-Error"]

    for rot_freq, lookat in param_sets:
        sub_df = experiments_df.query("`Rotation Frequency` == {} & `Lookat Angle` == {}".format(rot_freq, lookat))
        for stat in stats_to_test_exp:

            pval_col = "{} P".format(stat)
            if stat not in stats_exp.columns:
                stats_exp[pval_col] = np.nan

            stats_base = experiments_df_baseline[stat]
            stats_comp = sub_df[stat]

            rez = ttest_ind(stats_base, stats_comp)
            stats_exp.loc[(rot_freq, lookat), pval_col] = rez.pvalue

        # Separate analysis for missed branches
        missed_total_baseline = experiments_df_baseline["Missed Branches"].sum()
        total_baseline = experiments_df_baseline["GT Branches"].sum()
        obs_baseline = np.zeros(int(total_baseline))
        obs_baseline[: int(missed_total_baseline)] = 1

        missed_total_comp = sub_df["Missed Branches"].sum()
        total_comp = sub_df["GT Branches"].sum()
        obs_comp = np.zeros(int(total_comp))
        obs_comp[: int(missed_total_comp)] = 1

        pval_col = "Missed Branches P"
        if pval_col not in stats_exp.columns:
            stats_exp[pval_col] = np.nan

        stats_exp.loc[(rot_freq, lookat), "Missed Branches"] = missed_total_comp / total_comp
        stats_exp.loc[(0, 0), "Missed Branches"] = missed_total_baseline / total_baseline
        stats_exp.loc[(rot_freq, lookat), pval_col] = ttest_ind(obs_baseline, obs_comp).pvalue

    stats_exp = stats_exp[sorted(stats_exp.columns)]
    stats_exp.to_csv(os.path.join(folder, "stats_experiments.csv"))

    branches_df_baseline = branches_df.query("`Rotation Frequency` == 0 & `Lookat Angle` == 0")
    param_sets = [[0, 0], [1.5, 22.5], [1.5, 45.0], [2.5, 22.5], [2.5, 45.0]]
    # stats_to_test_branch = ['Abs Length Error %', 'Abs Radius Error %', 'Angle Error']
    stats_to_test_branch = ["Length Error %", "Radius Error %", "Angle Error"]

    for rot_freq, lookat in param_sets:
        sub_df = branches_df.query("`Rotation Frequency` == {} & `Lookat Angle` == {}".format(rot_freq, lookat))
        for stat in stats_to_test_branch:

            pval_col = "{} P".format(stat)
            stdev_col = f"{stat} Stdev"
            if stat not in stats_exp.columns:
                stats_exp[pval_col] = np.nan
                stats_exp[stdev_col] = np.nan

            stats_comp = sub_df[stat]
            stats_branches.loc[(rot_freq, lookat), stdev_col] = np.std(stats_comp)

            if rot_freq != 0:
                rez = ttest_ind(stats_base, stats_comp)
                stats_base = branches_df_baseline[stat]
                stats_branches.loc[(rot_freq, lookat), pval_col] = rez.pvalue

    stats_branches = stats_branches[sorted(stats_branches.columns)]
    stats_branches.to_csv(os.path.join(folder, "stats_branches.csv"))

    unagg_summary = {}
    for param_set, unagg_data in all_unaggregated.items():
        rez = {}
        for stat, raw_data in unagg_data.items():
            raw_data = np.array(raw_data)
            rez[f"{stat} Mean"] = np.mean(raw_data)
            rez[f"{stat} Median"] = np.median(raw_data)
            rez[f"{stat} Stdev"] = np.std(raw_data)

        unagg_summary[param_set] = rez

    unagg_summary_df = pd.DataFrame(unagg_summary)
    unagg_summary_df[sorted(unagg_summary_df.columns)].to_csv(os.path.join(folder, "stats_experiments_collective.csv"))
