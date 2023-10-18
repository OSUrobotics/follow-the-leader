#!/usr/bin/env python3
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.special import comb
from skimage.morphology import skeletonize, medial_axis
import networkx as nx
from scipy.spatial import KDTree
from scipy.interpolate import interp1d


class BezierBasedDetection:
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
                cum_dists = np.zeros(pts.shape[0])
                offsets = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                cum_dists[1:] = np.cumsum(offsets)
                if cum_dists[-1] < min_len:
                    continue

                curve = Bezier.fit(pts, 3)
                new_dists = curve.arclen * cum_dists / cum_dists[-1]
                new_dists[-1] = curve.arclen
                eval_bezier, _ = curve.eval_by_arclen(new_dists, normalized=False)
                matched_pts = np.linalg.norm(pts - eval_bezier, axis=1) < self.outlier_threshold
                if self.use_vec_weighted_metric and vec is not None:
                    # For each contiguous section of matches, accumulate total distance in the vec direction
                    # Will accumulate negatively if it goes against the vector
                    score = get_contiguous_distance(pts, matched_pts, vec)
                else:
                    score = matched_pts.sum()

                if score > best_score:
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

    def fit(self, vec=None, trim=30):
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
            eval_bezier = self.selected_curve(ts)

            cv2.polylines(base_img, [eval_bezier.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 255), 4)
            for info, stat in zip(side_branches, stats):
                curve = info["curve"]
                eval_bezier = curve(ts)
                msg = "Scores: {}, {:.1f}%".format(stat["score"], stat["consistency"] * 100)
                cv2.polylines(base_img, [eval_bezier.reshape((-1, 1, 2)).astype(int)], False, (0, 128, 0), 4)
                draw_pt = eval_bezier[len(eval_bezier) // 2].astype(int)
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                if draw_pt[0] + text_size[0] > base_img.shape[1]:
                    draw_pt[0] -= text_size[1]

                cv2.putText(base_img, msg, draw_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            Image.fromarray(base_img).save(visualize)

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


class Bezier:
    # https://stackoverflow.com/questions/12643079/b%c3%a9zier-curve-fitting-with-scipy

    def __init__(self, ctrl_pts, approx_eval=201):
        self.pts = ctrl_pts
        self.approx_eval = approx_eval
        self._kd_tree = None
        self._ts = None

        # Arc length-based interpolation tools
        self._len_approx = None
        self._dist_t_interpolator = None
        self._t_dist_interpolator = None

    @property
    def n(self):
        return len(self.pts)

    @property
    def deg(self):
        return len(self.pts) - 1

    @staticmethod
    def bpoly(i, n, t):
        return comb(n, i) * t ** (n - i) * (1 - t) ** i

    def __call__(self, t):
        t = np.array(t)
        polys = [self.bpoly(i, self.deg, t)[..., np.newaxis] * pt for i, pt in enumerate(self.pts)]
        return np.sum(polys, axis=0)

    def _load_arclen_tools(self):
        if self._len_approx is not None:
            return
        ts_approx = np.linspace(0, 1, self.approx_eval)
        eval_pts = self(ts_approx)
        cum_dists = np.zeros(self.approx_eval)
        cum_dists[1:] = np.linalg.norm(eval_pts[:-1] - eval_pts[1:], axis=1).cumsum()
        self._len_approx = cum_dists[-1]

        self._dist_t_interpolator = interp1d(cum_dists, ts_approx)
        self._t_dist_interpolator = interp1d(ts_approx, cum_dists)

    @property
    def arclen(self):
        self._load_arclen_tools()
        return self._len_approx

    def eval_by_arclen(self, dists, normalized=False):
        # Evaluates the curve, but with the ts parametrized by the distance along the curve (using an approximation)
        self._load_arclen_tools()
        if normalized:
            dists = np.array(dists) * self.arclen
        ts = self._dist_t_interpolator(dists)
        return self(ts), ts

    def t_to_curve_dist(self, ts):
        self._load_arclen_tools()
        return self._t_dist_interpolator(ts)

    def tangent(self, t):
        t = np.array(t)
        polys = [
            self.deg * self.bpoly(i, self.deg - 1, t)[..., np.newaxis] * (p2 - p1)
            for i, (p1, p2) in enumerate(zip(self.pts[:-1], self.pts[1:]))
        ]
        return -np.sum(polys, axis=0)

    def visualize(self, other_pts=None):
        eval_pts = self(np.linspace(0, 1, 100))
        if other_pts is not None:
            plt.scatter(*other_pts.T, color="grey", marker="x", s=3)
        plt.scatter(*self.pts.T, color="red", marker="*", s=10)
        plt.scatter(*eval_pts.T, color="blue", s=5)
        plt.show()

    @classmethod
    def fit(cls, pts, degree=3):
        # Assumes pts[0] and pts[-1] represent the respective endpoints of the Bezier curve
        n = len(pts)
        t = np.linspace(0, 1, n)
        b_mat = np.array([cls.bpoly(i, degree, t) for i in range(degree + 1)]).T
        fit = np.linalg.pinv(b_mat) @ pts
        return cls(fit)

    @classmethod
    def iterative_fit(
        cls,
        pts,
        degree=3,
        inlier_threshold=0.05,
        resample_p=0.5,
        exclude_furthest=0.10,
        max_iters=10,
        stop_threshold=0.75,
    ):
        """
        Attempts to fit a Bezier curve to noisy data by with a RANSAC-like approach.
        Iteratively estimates a model, identifies inliers, and reestimates the model until the desired number of
        inliers has been met.
        inlier_threshold: Distance from point to curve to be considered an inlier
        resample_p: Proportion of points that are resampled to fit a new model
        exclude_furthest: This proportion of points furthest from the current model will not be considered for model fitting
        stop_threshold: Process terminates once this proportion of inliers has been met
        """

        stats = {
            "success": False,
            "iters": 0,
            "inliers": 0,
            "inlier_idx": [],
            "init_inliers": 0,
        }

        current_model = cls.fit(pts, degree=degree)
        best_model = None
        best_inlier_p = 0.0

        for i in range(max_iters):
            stats["iters"] += 1
            dists, _ = current_model.query_pt_distance(pts)
            inliers = dists < inlier_threshold
            inlier_p = inliers.mean()
            stats["inliers"] = inlier_p
            if i == 0:
                stats["init_inliers"] = inlier_p

            if inlier_p > best_inlier_p:
                best_model = current_model
                best_inlier_p = inlier_p
                stats["inlier_idx"] = inliers

            if inlier_p >= stop_threshold:
                stats["success"] = True
                return current_model, stats

            # Using the current model, exclude the furthest points and subsample a new selection of points
            eligible = np.argsort(dists)[: int((1 - exclude_furthest) * len(pts))]
            to_sample = np.random.choice(eligible, int(resample_p * len(pts)), replace=False)
            to_use = np.zeros(len(pts), dtype=bool)
            to_use[0] = to_use[-1] = True
            to_use[to_sample] = True
            current_model = cls.fit(pts[to_use])

        print("Warning: Iterative fit maxed out!")
        return best_model, stats

    def query_pt_distance(self, pts):
        if self._kd_tree is None:
            self._ts = np.linspace(0, 1, self.approx_eval, endpoint=True)
            curve_eval = self(self._ts)
            self._kd_tree = KDTree(curve_eval)

        dists, idxs = self._kd_tree.query(pts)
        return dists, self._ts[idxs]


def get_contiguous_distance(pts, matches, vec):
    current_start = None
    dist = 0
    for i, match in enumerate(matches):
        if current_start is None and match:
            current_start = i
        elif current_start is not None and not match:
            offsets = pts[current_start + 1 : i] - pts[current_start : i - 1]
            dist += (offsets * vec).sum()
            current_start = None
    if current_start is not None:
        i = len(matches)
        offsets = pts[current_start + 1 : i] - pts[current_start : i - 1]
        dist += (offsets * vec).sum()

    return dist


# TESTS
def side_branch_test():
    from PIL import Image
    import cv2

    proc_dir = os.path.join(os.path.expanduser("~"), "Pictures", "masks")
    output_dir = os.path.join(proc_dir, "outputs")
    files = [x for x in os.listdir(proc_dir) if x.endswith(".png")]

    for file in files:
        input_file = os.path.join(proc_dir, file)
        output_file = os.path.join(output_dir, file)
        img = np.array(Image.open(input_file))[:, :, :3].copy()
        mask = img.mean(axis=2) > 128

        detection = BezierBasedDetection(mask, outlier_threshold=6, use_medial_axis=True)
        curve = detection.fit(vec=(0, -1))
        if curve is None:
            continue

        radius_interpolator = detection.get_radius_interpolator_on_path()
        detection.run_side_branch_search(visualize=output_file, min_len=40, radius_interpolator=radius_interpolator)


def ransac_fit_test():
    import matplotlib.pyplot as plt
    import time

    for _ in range(100):
        rand_pts = np.random.uniform(-1, 1, (4, 3))
        curve = Bezier(rand_pts)
        num_ts = np.random.randint(10, 50)
        ts = np.random.uniform(0, 1, num_ts)
        ts.sort()

        vals = curve(ts) + np.random.uniform(-0.01, 0.01, (num_ts, 3))

        super_noisy_pts = int(np.random.uniform(0.1, 0.2) * num_ts)
        idxs_to_modify = (
            np.random.choice(num_ts - 2, super_noisy_pts, replace=False) + 1
        )  # Don't modify the start/end points
        vals[idxs_to_modify] += np.random.uniform(-2.0, 2.0, (super_noisy_pts, 3))

        start = time.time()
        fit_curve, stats = Bezier.iterative_fit(vals, max_iters=100)
        end = time.time()

        print("Fit of {} points took {:.3f}s ({} iters)".format(num_ts, end - start, stats["iters"]))
        print("Percent inliers: {:.2f}% (init {:.2f}%)".format(stats["inliers"] * 100, stats["init_inliers"] * 100))

        ax = plt.figure().add_subplot(projection="3d")

        ts = np.linspace(0, 1, 51)
        real_pts = curve(ts)
        est_pts = fit_curve(ts)
        naive_pts = Bezier.fit(vals)(ts)

        ax.plot(*real_pts.T, color="green", linestyle="dashed")
        ax.plot(*est_pts.T, color="blue")
        ax.plot(*naive_pts.T, color="red", linestyle="dotted")
        ax.scatter(*vals[stats["inlier_idx"]].T, color="green")
        ax.scatter(*vals[~stats["inlier_idx"]].T, color="red")

        plt.show()


if __name__ == "__main__":
    side_branch_test()
