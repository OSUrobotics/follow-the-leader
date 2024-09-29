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


def cost_function(points, trunk, curvature_threshold):
    """
    Calculate the cost of adding a point to the trunk

    Args:
        points (np.array): The remaining points to consider
        trunk (np.array): The current trunk
        curvature_threshold (float): The maximum curvature allowed

    Returns:
        float: The cost of adding the point to the trunk
    """
    # Calculate the curvature of the trunk
    trunk_curvature = calculate_curvature(trunk)
    # Calculate the curvature of the trunk with the new point
    new_trunk = np.vstack([trunk, points[0]])
    new_trunk_curvature = calculate_curvature(new_trunk)
    # Calculate the cost of adding the point
    cost = new_trunk_curvature - trunk_curvature
    return cost

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

def init_by_density(pcd, params):
    """ guess initial trunk vector and root point using density clustering ie is tree growing (up/down or left/right)"""
    if not isinstance(pcd, o3d.t.geometry.PointCloud):
        raise TypeError("Input must be an open3d.t.geometry.PointCloud object")
    debug_mode = params.get_param("debug_mode")
    resolution = params.get_param("min_resolution")
    max_feature_size = params.get_param("max_feature_size")
    start_time = time.time()

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pcd_calc: o3d.geometry.PointCloud = pcd.to_legacy().voxel_down_sample(voxel_size=resolution)
        # get k-distances elbow point
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_calc)
        dists = [resolution]
        for i in range(0, len(pcd_calc.points)):
            [k, idx, dd] = pcd_tree.search_hybrid_vector_3d(
                pcd_calc.points[i], radius = max_feature_size, max_nn = 6
            )  # ndim * 2
            max_d = round(max(dd), 4)
            insertion_point = bisect.bisect_left(dists, max_d)
            if not np.isclose(dists[insertion_point - 1], max_d) and max_d > resolution:
                dists.insert(insertion_point, max_d)
        x = np.arange(len(dists))
        y = np.array(dists)
        knee = KneeLocator(x, y, curve="convex", direction="increasing")

        labels = np.array(pcd_calc.cluster_dbscan(eps=knee.knee_y, min_points=6, print_progress=False)) # 2 * dim min_points
        max_label = labels.max()
        num_clusters = max_label + 1
        if num_clusters < 1 or num_clusters > 20:
            raise ValueError(f"Number of clusters {num_clusters} is not within expected range")

        if debug_mode:
            print(f"knee point {knee.knee, knee.knee_y}")
            print(f"point cloud has {max_label + 1} clusters")
            # knee.plot_knee()
            # plt.show()
            colors =plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            pcd_calc.colors = o3d.utility.Vector3dVector(colors[:, :3])
            # o3d.visualization.draw([pcd_calc])

    # get pca for cluster with most points
    cluster_sizes = np.bincount(labels[labels >= 0])
    largest_cluster_idx = np.argmax(cluster_sizes)
    largest_cluster_points = np.asarray(pcd_calc.points)[labels == largest_cluster_idx]

    # Perform PCA on the largest cluster to get the initial trunk vector
    mean = np.mean(largest_cluster_points, axis=0)
    centered_points = largest_cluster_points - mean
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    trunk_vector = eigenvectors[:, np.argmax(eigenvalues)]
    trunk_vector = trunk_vector / np.linalg.norm(trunk_vector)
    # get minimum point in cluster if projected on the trunk vector
    proj = largest_cluster_points @ trunk_vector
    min_idx = np.argmin(proj)
    min_point = largest_cluster_points[min_idx]
    max_idx = np.argmax(proj)
    max_point = largest_cluster_points[max_idx]

    if debug_mode:
        print(f"Initial trunk vector: {trunk_vector}")
        print(f"Min {min_point}")
        pts = []
        lines = []
        colors = []
        for i in range(0, len(eigenvalues)):
            if i == np.argmax(eigenvalues):
                continue
            eg = eigenvectors[:, i]
            proj = largest_cluster_points @ eg
            min_idx = np.argmin(proj)
            max_idx = np.argmax(proj)
            pts.append(largest_cluster_points[min_idx])
            pts.append(largest_cluster_points[max_idx])
            lines.append([len(pts) - 2, len(pts) - 1])
            colors.append([0, 0, 0])
        # largest eigenvalue in red
        pts.append(min_point)
        pts.append(max_point)
        lines.append([len(pts) - 2, len(pts) - 1])
        colors.append([1, 0, 0])
        print(f"pts {pts} lines {lines} colors {colors}")
        lineset = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts), lines=o3d.utility.Vector2iVector(lines))
        lineset.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw([pcd_calc, lineset])

    end_time = time.time()
    print(f"init_by_density execution time: {end_time - start_time} seconds")
    return min_point, trunk_vector

def dynamic_trunk(params, frame_dict, pcd, existing_model=None):
    """ determine trunk using a dynamic programming approach
    
    :param params: parameters for modelling
    :param frame_dict: all frame data
    :param pcd: o3d point cloud
    :param existing_model: previous model, defaults to None
    """
    move_vec = frame_dict["move_vec"]
    if existing_model is None:
        root_pt, trunk_vec = init_by_density(pcd, params)


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
    params.add_param("debug_mode", True)
    params.add_param("max_feature_size", 2.0)
    # iterate through all the pickle files
    for file in sorted(os.listdir(parent_folder)):
        print(f"file {file}")
        if file.endswith(".pickle"):
            file = os.path.join(parent_folder, file)
            frame_dict : dict = pickle.load(open(file, "rb"))
            pcd = extract_o3d_pcd(frame_dict)
            # o3d.visualization.draw([pcd])
            if frame_dict.get("move_vec") is None:
                frame_dict["move_vec"] = init_by_density(pcd)

            tr = dynamic_trunk(params, frame_dict, pcd, model)
            # model = gen_model(frame_dict, pcd, tr, model)
