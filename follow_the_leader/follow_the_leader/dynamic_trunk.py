""" Implements a dynamic program to select longest sequence of centered 3D points while remaining within constraints. """

import bisect
import time
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c

from kneed import KneeLocator
from open3d.t.geometry import PointCloud
from follow_the_leader.utils.params import Params
from follow_the_leader.utils.ros_utils import log_entry_exit
from follow_the_leader.utils.robust_pca import R_pca


def cost_function(params, points, trunk, frame_dict):
    """
    Calculate the cost of adding a point to the trunk. depends on:
    - minimise curvature of the trunk measured as change in angle from existing vector
    - maximise radius when fitting a cylinder to segment
    - maximise length of trunk

    Args:
        points (np.array): The remaining points to consider
        trunk (np.array): The current trunk
        curvature_threshold (float): The maximum curvature allowed

    Returns:
        float: The cost of adding the point to the trunk
    """


def extract_o3d_pcd(frame_dict):
    pcdlist = []
    for mode in frame_dict["depth_dicts"].keys():
        info = frame_dict["depth_dicts"][mode]
        pcd_map = {}
        try:
            positions = np.array(info["pts"])
            pcd_map["positions"] = o3c.Tensor(positions, o3c.float32)
            pcd_map["pxs"] = o3c.Tensor(info["pts_2d"], o3c.int32)
            pcd_map["error"] = o3c.Tensor(info["error"], o3c.float32)
            # pcd_map["radius"] = o3c.Tensor(info["radius"], o3c.float32)
            pcd = o3d.t.geometry.PointCloud(pcd_map)
            pcdlist.append(pcd)
        except Exception as e:
            print(f"Exception {e}")
    # sum all pointclouds, they're in the same frame!
    if len(pcdlist) == 0:
        return None
    pcd = pcdlist[0]
    for i in range(1, len(pcdlist)):
        pcd += pcdlist[i]
    return pcd


def extract_minimum_projection_point(points, trunk_vector):
    proj = points @ trunk_vector
    min_idx = np.argmin(proj)
    min_point = points[min_idx]
    return min_point, proj


@log_entry_exit
def init_by_density(pcd, params):
    """
    guess initial trunk growth direction and root point using density clustering
    """
    if not isinstance(pcd, o3d.t.geometry.PointCloud):
        raise TypeError("Input must be an open3d.t.geometry.PointCloud object")
    debug_mode = params.get_param("debug_mode")
    resolution = params.get_param("min_resolution")
    max_feature_size = params.get_param("max_feature_size")
    start_time = time.time()

    pcd_calc: o3d.geometry.PointCloud = pcd.to_legacy().voxel_down_sample(
        voxel_size=resolution
    )
    # get k-distances elbow point
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_calc)
    # dists = [resolution]
    nns = []
    for i in range(0, len(pcd_calc.points)):
        [k, idx, dd] = pcd_tree.search_radius_vector_3d(
            pcd_calc.points[i], radius=resolution
        )  # ndim * 2
        nns.append(k)
        # bisect_insert(dists, d, resolution)
        # bisect_insert(nns, k, 1)
    min_nn = int(np.floor(np.mean(nns)))
    print(f"Ks: {np.mean(nns)}")

    # x, y = np.arange(len(dists)), np.array(dists)
    # x_k, y_k = np.arange(len(min_nn)), np.array(min_nn)
    # knee_d = KneeLocator(x, y, curve="convex", direction="increasing")
    # knee_k = KneeLocator(x_k, y_k, curve="convex", direction="increasing")
    # knee_k.plot_knee()
    # plt.show()
    # pcd_calc.points["neighbours"] = o3d.utility.DoubleVector(all_k)
    # # create mask and label if neighbors > knee_k.knee_y
    # mask = np.array(all_k) > knee_k.knee_y
    # pcd_calc.points = o3d.utility.Vector3dVector(np.asarray(pcd_calc.points)[mask])

    labels = np.array(
        pcd_calc.cluster_dbscan(eps=3 * resolution, min_points=min_nn, print_progress=False)
    )
    max_label = labels.max()
    num_clusters = max_label + 1
    # if num_clusters < 1 or num_clusters > 50:
    #     raise ValueError(
    #         f"Number of clusters {num_clusters} is not within expected range"
    #     )
    if debug_mode:
        print(f"point cloud has {num_clusters} clusters")
        # knee.plot_knee()
        # plt.show()
        colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd_calc.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw([pcd_calc])

    # get cluster with most points
    cluster_sizes = np.bincount(labels[labels >= 0])
    largest_cluster_id = np.argmax(cluster_sizes)
    largest_cluster_idxes = np.argwhere(labels == largest_cluster_id).flatten()
    pcd_largest_cluster = pcd_calc.select_by_index(largest_cluster_idxes)
    if pcd_largest_cluster.is_empty():
        pcd_largest_cluster = pcd_calc

    # plane fitting
    plane_model, inliers = pcd_largest_cluster.segment_plane(
        distance_threshold= 3 * resolution, ransac_n=max(min_nn, 3), num_iterations=1000
    )
    inlier_cloud = pcd_largest_cluster.select_by_index(inliers)
    trunk_points = np.asarray(inlier_cloud.points)
    if trunk_points.shape[0] < 3:
        trunk_points = np.asarray(pcd_largest_cluster.points)

    # Perform PCA on the largest cluster to get the initial trunk vector
    centered_points, noise = R_pca(trunk_points).fit(iter_print=100000)
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    max_eig_idx = np.argmax(eigenvalues)
    trunk_vector = eigenvectors[:, max_eig_idx]
    trunk_vector = trunk_vector / np.linalg.norm(trunk_vector)
    # get minimum point in cluster if projected on the trunk vector
    min_point, proj = extract_minimum_projection_point(
        trunk_points, trunk_vector
    )
    max_point = trunk_points[np.argmax(proj)]

    if debug_mode:
        print(f"Initial trunk vector: {trunk_vector}")
        print(f"Min {min_point}")
        pts = []
        lines = []
        colors = []
        for i in range(0, len(eigenvalues)):
            if i == max_eig_idx:
                continue
            eg = eigenvectors[:, i]
            proj = trunk_points @ eg
            min_idx = np.argmin(proj)
            max_idx = np.argmax(proj)
            pts.append(trunk_points[min_idx])
            pts.append(trunk_points[max_idx])
            lines.append([len(pts) - 2, len(pts) - 1])
            colors.append([0, 0, 0])
        # largest eigenvalue in red
        pts.append(min_point)
        pts.append(max_point)
        lines.append([len(pts) - 2, len(pts) - 1])
        colors.append([1, 0, 0])
        lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        lineset.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw([pcd_calc, lineset])

    return min_point, trunk_vector


def bisect_insert(list_to_insert, element, min_check=None):
    idx = bisect.bisect_left(list_to_insert, element)
    if idx == len(list_to_insert) and not np.isclose(list_to_insert[-1], element):
        list_to_insert.append(element)
    if not np.isclose(list_to_insert[idx], element):
        if min_check is not None and element <= min_check:
            return
        else:
            list_to_insert.insert(idx, element)


def dynamic_trunk(params, frame_dict, pcd, test_trunk_vec=None, existing_model=None):
    """determine trunk using a dynamic programming approach

    :param params: parameters for modelling
    :param frame_dict: all frame data
    :param pcd: o3d point cloud
    :param existing_model: previous model, defaults to None
    """
    move_vec = frame_dict["move_vec"]
    if existing_model is None:
        if test_trunk_vec is None:
            root_pt, trunk_vec = init_by_density(pcd, params)
        if test_trunk_vec is not None:
            trunk_vec = test_trunk_vec
            root_pt, _ = extract_minimum_projection_point(np.asarray(pcd.points), trunk_vec)


def gen_model():
    raise NotImplementedError


def mpl_debug(points, trunk):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    # Plot the trunk
    ax.plot(trunk[:, 0], trunk[:, 1], trunk[:, 2], color="red")

    plt.show()


if __name__ == "__main__":
    # read pickle data
    import pickle
    import os

    parent_folder = "/home/roosh/bagfiles/ftl_18Dec2024_11:55:15"
    model = None
    params = Params()
    params.add_param("min_resolution", 1e-3)
    params.add_param("debug_mode", False)
    params.add_param("max_feature_size", 0.2)
    # iterate through all the pickle files
    for file in sorted(os.listdir(parent_folder)):
        print(f"file {file}")
        if file.endswith(".pickle"):
            file = os.path.join(parent_folder, file)
            frame_dict: dict = pickle.load(open(file, "rb"))
            pcd = extract_o3d_pcd(frame_dict)
            # o3d.visualization.draw([pcd])
            if frame_dict.get("move_vec") is None:
                frame_dict["move_vec"] = init_by_density(pcd)

            tr = dynamic_trunk(params, frame_dict, pcd, model)
            # model = gen_model(frame_dict, pcd, tr, model)
