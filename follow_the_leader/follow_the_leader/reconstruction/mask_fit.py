import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.special import comb
from skimage.morphology import skeletonize, medial_axis
import networkx as nx
from scipy.spatial import KDTree
from scipy.interpolate import interp1d

from b_spline_fit import BSplineFit
from image_utils import get_contiguous_distance

class BSplineBasedDetection:
    def __init__(self, mask, outlier_threshold=4, use_medial_axis=False, use_vec_weighted_metric=False):
        self.mask = mask
        self.skel = None
        self.dists = None
        self.stats = {}
        self.outlier_threshold = outlier_threshold
        self.use_medial_axis = use_medial_axis
        self.use_vec_weighted_metric = use_vec_weighted_metric

        self.subsampled_graph = None
        self.selected_path = None
        self.selected_curve = None

    def construct_skeletal_graph(self, trim=0):
        graph = nx.Graph()
        if self.use_medial_axis:
            skel, dists = medial_axis(self.mask, return_distance=True)
        else:
            skel = skeletonize(self.mask)
            dists = None
        
        if trim:
            skel[:trim] = 0
            skel[-trim:] = 0
            skel[:, :trim] = 0
            skel[:, -trim:] = 0

        self.skel = skel
        self.dists = dists
        pxs = np.array(np.where(skel)).T[:, [1, 0]]
        for px in pxs:
            graph.add_node(tuple(px))
            for dir in np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [-1, -1], [-1, 1], [1, -1], [1, 1]]):
                new_px = dir + px
                if (
                    (0 <= new_px[0] < self.mask.shape[1])
                    and (0 <= new_px[1] < self.mask.shape[0])
                    and skel[new_px[1], new_px[0]]
                ):
                    graph.add_node(tuple(new_px))
                    graph.add_edge(tuple(px), tuple(new_px), distance=np.linalg.norm(dir))

        subgraph = nx.minimum_spanning_tree(graph.subgraph(max(nx.connected_components(graph), key=len)))
        deg_1_nodes = [n for n, deg in subgraph.degree if deg == 1]

        downsampled = nx.Graph()
        # Process the edges to form a downsampled graph

        node = deg_1_nodes[0]
        downsampled.add_node(node)
        path = [node]
        for edge in nx.dfs_edges(subgraph):
            node = path[-1]
            if edge[0] == node:
                path.append(edge[1])
            else:
                path = [edge[0], edge[1]]

            if subgraph.degree(edge[1]) != 2:
                downsampled.add_node(edge[1])
                downsampled.add_edge(path[0], path[-1], path=path)
                path = [edge[1]]

        return downsampled

    def do_curve_search(self, graph, start_nodes, vec, min_len=0, filter_mask=None, return_stats=False):
        best_score = 0
        best_curve = None
        best_path = None
        stats = {}

        for node in start_nodes:
            path_dict = {node: None}

            def retrieve_path_pts(n):
                edges = []
                path = [n]
                while path_dict[n] is not None:
                    edges.append((path_dict[n], n))
                    n = path_dict[n]
                    path.append(n)
                edges = edges[::-1]
                path = path[::-1]

                px_list = []
                for edge in edges:
                    px_path = graph.edges[edge]["path"]
                    if px_path[0] != edge[0]:
                        px_path = px_path[::-1]
                    px_list.extend(px_path)

                pts = np.array(px_list)
                return path, pts

            for edge in nx.dfs_edges(graph, source=node):
                path_dict[edge[1]] = edge[0]
                node_path, pts = retrieve_path_pts(edge[1])
                if filter_mask is not None:
                    pts = pts[~filter_mask[pts[:, 1], pts[:, 0]]]
                if not len(pts):
                    continue

                curve = BSplineFit(degree="cubic", dim=2, data_pts=pts)
                ctrl_pts, points_in_t, residuals = curve.bezier_fit(pts)
                self.ctrl_pts = [
                    ctrl_pts[i]
                    for i in range(ctrl_pts.shape[0])
                ]
                self.ts = points_in_t
                self.residuals = residuals
                matched_pts = residuals < self.outlier_threshold
                if self.use_vec_weighted_metric and vec is not None:
                    # For each contiguous section of matches, accumulate total distance in the vec direction
                    # Will accumulate negatively if it goes against the vector
                    score = get_contiguous_distance(pts, matched_pts, vec)
                else:
                    score = matched_pts.sum()

                if score > best_score:
                    curve.ctrl_pts = ctrl_pts
                    curve.ts = points_in_t
                    curve.residuals = residuals
                    best_score = score
                    best_curve = curve
                    best_path = node_path
                    stats = {
                        "pts": pts,
                        "matched_idx": matched_pts,
                        "score": score,
                        "consistency": matched_pts.mean(),
                    }

        if return_stats:
            return best_curve, best_path, stats

        return best_curve, best_path

    def fit(self, vec=None, trim=0):
        if vec is None:
            # Run SVD to find the most significant direction
            pxs = np.fliplr(np.array(np.where(self.mask)).T)
            pxs = pxs - pxs.mean(axis=0)
            # SVD takes a lot of memory - we subsample points as we only need an estimate
            pxs = pxs[np.random.choice(len(pxs), 100, replace=False)]
            u, s, v = np.linalg.svd(pxs, full_matrices=True)
            vec = v[0]

        vec = np.array(vec) / np.linalg.norm(vec)
        # Iterate through the edges and use the vec to determine the orientation

        graph = self.construct_skeletal_graph(trim=trim)
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(graph.nodes)

        for p1, p2 in graph.edges:
            path = graph.edges[p1, p2]["path"]
            if np.dot(np.array(p2) - p1, vec) < 0:
                p1, p2 = p2, p1
            if path[0] != p1:
                path = path[::-1]
            directed_graph.add_edge(p1, p2, path=path)

        start_nodes = [n for n in graph.nodes if directed_graph.out_degree(n) == 1 and directed_graph.in_degree(n) == 0]
        best_curve, best_path, stats = self.do_curve_search(directed_graph, start_nodes, vec=vec, return_stats=True)

        self.subsampled_graph = graph
        self.selected_path = best_path
        self.selected_curve = best_curve
        self.stats = stats

        return best_curve

    def run_side_branch_search(self, min_len=80, filter_mask=None, visualize=""):
        if self.selected_path is None:
            raise Exception("Please run the fit function first")

        graph = self.subsampled_graph
        assert isinstance(graph, nx.Graph)

        # Subgraph pre-processing
        main_path = self.selected_path
        for edge in zip(main_path[:-1], main_path[1:]):
            graph.remove_edge(*edge)

        candidate_edges = []
        to_remove = []
        for i, node in enumerate(main_path):
            for neighbor in graph[node]:
                edge = (node, neighbor)
                path = graph.edges[edge]["path"]
                if path[0] != node:
                    path = path[::-1]  # Orient the path from the main branch outwards

                to_remove.append(edge)
                if 0 < i < len(main_path) - 1:
                    candidate_edges.append((edge, path))

        graph.remove_edges_from(to_remove)

        side_branches = []
        stats = []
        for candidate_edge, path in candidate_edges:
            graph.add_edge(*candidate_edge, path=path)

            best_curve, best_path, match_stats = self.do_curve_search(
                graph,
                start_nodes=[candidate_edge[0]],
                vec=None,
                min_len=min_len,
                filter_mask=filter_mask,
                return_stats=True,
            )
            if best_curve is not None:
                info = {"curve": best_curve, "path": best_path, "stats": match_stats}
                side_branches.append(info)
                if visualize:
                    stats.append(match_stats)

            graph.remove_edge(*candidate_edge)

        if visualize:
            import cv2
            from PIL import Image

            base_img = np.dstack([self.mask * 255] * 3).astype(np.uint8)
            ts = np.linspace(0, 1, 201)
            eval_bezier = self.selected_curve.eval_crv(ts)

            cv2.polylines(base_img, [eval_bezier.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 255), 4)
            for info, stat in zip(side_branches, stats):
                curve = info["curve"]
                eval_bezier = curve.eval_crv(ts)
                msg = "Scores: {}, {:.1f}%".format(stat["score"], stat["consistency"] * 100)
                cv2.polylines(base_img, [eval_bezier.reshape((-1, 1, 2)).astype(int)], False, (0, 128, 0), 4)
                draw_pt = eval_bezier[len(eval_bezier) // 2].astype(int)
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                if draw_pt[0] + text_size[0] > base_img.shape[1]:
                    draw_pt[0] -= text_size[1]

                cv2.putText(base_img, msg, draw_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            Image.fromarray(base_img).save(visualize)
            Image.fromarray(self.skel).save(f"{visualize}_skel.png")

        return side_branches

    def get_radius_interpolator_on_path(self, path=None, min_quant=0.25, max_quant=0.75):
        if self.dists is None:
            raise Exception("Please run the medial axis transform before running this function")

        if path is None:
            path = self.stats["pts"]

        radii = self.dists[path[:, 1], path[:, 0]].copy()
        if 0 in radii:
            raise Exception("Some pixels in the specified path were not part of the skeleton!")
        q_min = np.quantile(radii, min_quant)
        q_max = np.quantile(radii, max_quant)
        min_bound = q_min - min_quant * (q_max - q_min) / (max_quant - min_quant)
        max_bound = q_max + max_quant * (q_max - q_min) / (max_quant - min_quant)
        radii[(radii < min_bound) | (radii > max_bound)] = np.median(radii)

        cum_dists = np.zeros(len(path))
        cum_dists[1:] = np.linalg.norm(path[1:] - path[:-1], axis=1).cumsum()
        cum_dists /= cum_dists[-1]

        return interp1d(cum_dists, radii)

# TESTS
def side_branch_test():
    from PIL import Image
    import cv2

    proc_dir = "/home/main/training_data/"
    output_dir = os.path.join(proc_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    files = [x for x in os.listdir(proc_dir) if x.endswith(".png") and x.startswith("mask")]

    for file in files:
        input_file = os.path.join(proc_dir, file)
        output_file = os.path.join(output_dir, file)
        mask = np.array(Image.open(input_file)).copy()

        detection = BSplineBasedDetection(mask, outlier_threshold=6, use_medial_axis=True)
        curve = detection.fit(vec=(0, -1))
        if curve is None:
            continue

        radius_interpolator = detection.get_radius_interpolator_on_path()
        detection.run_side_branch_search(visualize=output_file, min_len=40)

# def ransac_fit_test():
#     import matplotlib.pyplot as plt
#     import time

#     for _ in range(100):
#         rand_pts = np.random.uniform(-1, 1, (4, 3))
#         curve = Bezier(rand_pts)
#         num_ts = np.random.randint(10, 50)
#         ts = np.random.uniform(0, 1, num_ts)
#         ts.sort()

#         vals = curve(ts) + np.random.uniform(-0.01, 0.01, (num_ts, 3))

#         super_noisy_pts = int(np.random.uniform(0.1, 0.2) * num_ts)
#         idxs_to_modify = (
#             np.random.choice(num_ts - 2, super_noisy_pts, replace=False) + 1
#         )  # Don't modify the start/end points
#         vals[idxs_to_modify] += np.random.uniform(-2.0, 2.0, (super_noisy_pts, 3))

#         start = time.time()
#         fit_curve, stats = Bezier.iterative_fit(vals, max_iters=100)
#         end = time.time()

#         print("Fit of {} points took {:.3f}s ({} iters)".format(num_ts, end - start, stats["iters"]))
#         print("Percent inliers: {:.2f}% (init {:.2f}%)".format(stats["inliers"] * 100, stats["init_inliers"] * 100))

#         ax = plt.figure().add_subplot(projection="3d")

#         ts = np.linspace(0, 1, 51)
#         real_pts = curve(ts)
#         est_pts = fit_curve(ts)
#         naive_pts = Bezier.fit(vals)(ts)

#         ax.plot(*real_pts.T, color="green", linestyle="dashed")
#         ax.plot(*est_pts.T, color="blue")
#         ax.plot(*naive_pts.T, color="red", linestyle="dotted")
#         ax.scatter(*vals[stats["inlier_idx"]].T, color="green")
#         ax.scatter(*vals[~stats["inlier_idx"]].T, color="red")

#         plt.show()

if __name__ == "__main__":
    side_branch_test()