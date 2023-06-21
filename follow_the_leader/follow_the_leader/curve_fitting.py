import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.special import comb
from skimage.morphology import skeletonize, medial_axis
import networkx as nx
from scipy.spatial import KDTree
from scipy.interpolate import interp1d

class BezierBasedDetection:
    def __init__(self, mask, outlier_threshold=4):
        self.mask = mask
        self.skel = None
        self.outlier_threshold = outlier_threshold

        self.subsampled_graph = None
        self.selected_path = None
        self.selected_curve = None


    def construct_skeletal_graph(self, trim=0):
        graph = nx.Graph()
        skel = skeletonize(self.mask)
        if trim:
            skel[:trim] = 0
            skel[-trim:] = 0
            skel[:,:trim] = 0
            skel[:,-trim:] = 0

        self.skel = skel
        pxs = np.array(np.where(skel)).T[:, [1, 0]]
        for px in pxs:
            graph.add_node(tuple(px))
            for dir in np.array([[-1,0],[1,0],[0,1],[0,-1],[-1,-1],[-1,1],[1,-1],[1,1]]):
                new_px = dir + px
                if (0 <= new_px[0] < self.mask.shape[1]) and (0 <= new_px[1] < self.mask.shape[0]) and skel[new_px[1], new_px[0]]:
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

    def do_curve_search(self, graph, start_nodes, min_len=0, return_stats=False):
        most_matches = 0
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
                    px_path = graph.edges[edge]['path']
                    if px_path[0] != edge[0]:
                        px_path = px_path[::-1]
                    px_list.extend(px_path)

                pts = np.array(px_list)
                return path, pts

            for edge in nx.dfs_edges(graph, source=node):
                path_dict[edge[1]] = edge[0]
                path, pts = retrieve_path_pts(edge[1])
                cum_dists = np.zeros(pts.shape[0])
                offsets = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                cum_dists[1:] = np.cumsum(offsets)
                if cum_dists[-1] < min_len:
                    continue

                cum_dists /= cum_dists[-1]
                curve = Bezier.fit(pts, 3)
                eval_bezier, _ = curve.eval_by_arclen(cum_dists)
                matched_pts = np.linalg.norm(pts - eval_bezier, axis=1) < self.outlier_threshold
                matches = matched_pts.sum()

                if matches > most_matches:
                    most_matches = matches
                    best_curve = curve
                    best_path = path
                    stats = {
                        'pts': pts,
                        'matched_idx': matched_pts,
                        'matches': matches,
                        'consistency': matched_pts.mean(),
                    }

        if return_stats:
            return best_curve, best_path, stats

        return best_curve, best_path

    def fit(self, vec=(0, 1), trim=30):

        vec = np.array(vec) / np.linalg.norm(vec)
        # Iterate through the edges and use the vec to determine the orientation

        graph = self.construct_skeletal_graph(trim=trim)
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(graph.nodes)

        for p1, p2 in graph.edges:
            path = graph.edges[p1, p2]['path']
            if np.dot(np.array(p2) - p1, vec) < 0:
                p1, p2 = p2, p1
            if path[0] != p1:
                path = path[::-1]
            directed_graph.add_edge(p1, p2, path=path)

        start_nodes = [n for n in graph.nodes if directed_graph.out_degree(n) == 1 and directed_graph.in_degree(n) == 0]
        best_curve, best_path = self.do_curve_search(directed_graph, start_nodes)

        self.subsampled_graph = graph
        self.selected_path = best_path
        self.selected_curve = best_curve

        return best_curve

    def run_side_branch_search(self, min_len=80, visualize=''):
        if self.selected_path is None:
            raise Exception('Please run the fit function first')

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
                path = graph.edges[edge]['path']
                to_remove.append(edge)
                if 0 < i < len(main_path) - 1:
                    candidate_edges.append((edge, path))

        graph.remove_edges_from(to_remove)

        side_branches = []
        stats = []
        for candidate_edge, path in candidate_edges:
            graph.add_edge(*candidate_edge, path=path)

            best_curve, best_path, match_stats = self.do_curve_search(graph, start_nodes=[candidate_edge[0]],
                                                                      min_len=min_len, return_stats=True)
            if best_curve is not None:
                side_branches.append(best_curve)
                if visualize:
                    stats.append(match_stats)

            graph.remove_edge(*candidate_edge)

        if visualize:
            from PIL import Image

            base_img = np.dstack([self.mask * 255] * 3).astype(np.uint8)
            ts = np.linspace(0, 1, 201)
            eval_bezier = self.selected_curve(ts)

            cv2.polylines(base_img, [eval_bezier.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 255), 4)
            for curve, stat in zip(side_branches, stats):

                eval_bezier = curve(ts)
                msg = 'Matches: {}, {:.1f}%'.format(stat['matches'], stat['consistency'] * 100)
                cv2.polylines(base_img, [eval_bezier.reshape((-1, 1, 2)).astype(int)], False, (0, 128, 0), 4)
                draw_pt = eval_bezier[len(eval_bezier) // 2].astype(int)
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                if draw_pt[0] + text_size[0] > base_img.shape[1]:
                    draw_pt[0] -= text_size[1]

                cv2.putText(base_img, msg, draw_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            Image.fromarray(base_img).save(visualize)

        return side_branches


class Bezier:
    # https://stackoverflow.com/questions/12643079/b%c3%a9zier-curve-fitting-with-scipy

    def __init__(self, ctrl_pts, approx_eval=201):
        self.pts = ctrl_pts
        self.approx_eval = approx_eval
        self._kd_tree = None
        self._ts = None

    @property
    def n(self):
        return len(self.pts)

    @property
    def deg(self):
        return len(self.pts) - 1

    @staticmethod
    def bpoly(i, n, t):
        return comb(n, i) * t ** (n-i) * (1 - t) ** i


    def __call__(self, t):
        t = np.array(t)
        polys = [self.bpoly(i, self.deg, t)[..., np.newaxis] * pt for i, pt in enumerate(self.pts)]
        return np.sum(polys, axis=0)

    def eval_by_arclen(self, ts):
        # Evaluates the curve, but with the ts parametrized by the distance along the curve (using an approximation)

        ts_approx = np.linspace(0, 1, self.approx_eval)
        eval_pts = self(ts_approx)
        cum_dists = np.zeros(self.approx_eval)
        cum_dists[1:] = np.linalg.norm(eval_pts[:-1] - eval_pts[1:], axis=1).cumsum()
        cum_dists /= cum_dists[-1]

        interp = interp1d(cum_dists, ts_approx)
        remapped_ts = interp(ts)
        return self(remapped_ts), remapped_ts


    def tangent(self, t):
        t = np.array(t)
        polys = [self.deg * self.bpoly(i, self.deg - 1, t)[..., np.newaxis] * (p2 - p1)
                 for i, (p1, p2) in enumerate(zip(self.pts[:-1], self.pts[1:]))]
        return -np.sum(polys, axis=0)

    def visualize(self, other_pts=None):
        eval_pts = self(np.linspace(0, 1, 100))
        if other_pts is not None:
            plt.scatter(*other_pts.T, color='grey', marker='x', s=3)
        plt.scatter(*self.pts.T, color='red', marker='*', s=10)
        plt.scatter(*eval_pts.T, color='blue', s=5)
        plt.show()

    @classmethod
    def fit(cls, pts, degree=3):
        # Assumes pts[0] and pts[-1] represent the respective endpoints of the Bezier curve
        n = len(pts)
        t = np.linspace(0, 1, n)
        b_mat = np.array([cls.bpoly(i, degree, t) for i in range(degree+1)]).T
        fit = np.linalg.pinv(b_mat) @ pts
        return cls(fit)

    def query_pt_distance(self, pts):

        if self._kd_tree is None:
            self._ts = np.linspace(0, 1, self.approx_eval, endpoint=True)
            curve_eval = self(self._ts)
            self._kd_tree = KDTree(curve_eval)

        dists, idxs = self._kd_tree.query(pts)
        return dists, self._ts[idxs]


if __name__ == '__main__':

    from PIL import Image
    import cv2

    proc_dir = os.path.join(os.path.expanduser('~'), 'Pictures', 'masks')
    output_dir = os.path.join(proc_dir, 'outputs')
    files = [x for x in os.listdir(proc_dir) if x.endswith('.png')]

    for file in files:
        input_file = os.path.join(proc_dir, file)
        output_file = os.path.join(output_dir, file)
        img = np.array(Image.open(input_file))[:,:,:3].copy()
        mask = img.mean(axis=2) > 128

        detection = BezierBasedDetection(mask, outlier_threshold=6)
        curve = detection.fit(vec=(0,-1))
        if curve is None:
            continue

        detection.run_side_branch_search(visualize=output_file, min_len=40)
