import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.special import comb
from skimage.morphology import skeletonize
import networkx as nx
from scipy.spatial import KDTree

class BezierBasedDetection:
    def __init__(self, mask):
        self.mask = mask

        self.skel = None

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

    def fit(self, vec=(0, 1), trim=30, outlier_px=4):

        DEBUG_IMGS = []

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
        most_matches = 0
        best_curve = None

        for node in start_nodes:

            path_dict = {node: None}
            def retrieve_path_pts(n):
                edges = []
                while path_dict[n] is not None:
                    edges.append((path_dict[n], n))
                    n = path_dict[n]
                edges = edges[::-1]
                pts = np.concatenate([directed_graph.edges[edge]['path'][:-1] for edge in edges], axis=0)
                return pts

            for edge in nx.dfs_edges(directed_graph, source=node):
                path_dict[edge[1]] = edge[0]
                pts = retrieve_path_pts(edge[1])
                cum_dists = np.zeros(pts.shape[0])
                offsets = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                cum_dists[1:] = np.cumsum(offsets)
                cum_dists /= cum_dists[-1]
                curve = Bezier.fit(pts, 3)
                eval_bezier = curve(cum_dists)
                matches = (np.linalg.norm(pts - eval_bezier, axis=1) < outlier_px).sum()

                if matches > most_matches:
                    most_matches = matches
                    best_curve = curve

                # # # DEBUG
                # mask_img = np.dstack([(self.mask * 255).astype(np.uint8)] * 3)
                # cv2.polylines(mask_img, [eval_bezier.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 255), 2)
                #
                # diag = 'Matches: {}({:.2f}%, Best: {})'.format(matches, (matches / len(pts)) * 100, most_matches)
                # mask_img = cv2.putText(mask_img, diag, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                # DEBUG_IMGS.append(mask_img)
                # plt.imshow(mask_img)
                # plt.title('Number of matches: {} ({:.2f}%) (Best: {})'.format(matches, (matches / len(pts)) * 100, most_matches))
                # plt.show()
        #
        # if DEBUG_IMGS:
        #     import imageio
        #     imageio.mimsave('curve_fit.gif', DEBUG_IMGS, duration=25 * float(len(DEBUG_IMGS)))

        return best_curve


class LeaderDetector:

    def process_mask(self, mask):
        # Mask should be a np.uint8 image
        submasks = self.split_mask(mask)
        return [BezierBasedDetection(submask) for submask in submasks]

    def split_mask(self, mask):
        """Split the mask image up into connected components, discarding anything really small
        @param mask - the mask image
        @return a list of boolean indices for each component"""
        output = cv2.connectedComponentsWithStats(mask)
        labels = output[1]
        stats = output[2]

        ret_masks = []
        i_area = 0
        for i, stat in enumerate(stats):
            if np.sum(mask[labels == i]) == 0:
                continue

            if stat[cv2.CC_STAT_WIDTH] < 5:
                continue
            if stat[cv2.CC_STAT_HEIGHT] < 0.5 * mask.shape[1]:
                continue
            if i_area < stat[cv2.CC_STAT_AREA]:
                i_area = stat[cv2.CC_STAT_AREA]
            ret_masks.append(labels == i)

        return ret_masks


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
    import time

    img = np.array(Image.open(os.path.join(os.path.expanduser('~'), 'sample_mask_2.png')))[:,:,:3].copy()
    mask = img.mean(axis=2) > 128
    start = time.time()

    curve = BezierBasedDetection(mask).fit(vec=(0,-1))
    print('Took {:.2f} seconds'.format(time.time() - start))
    eval_pts = curve(np.linspace(0,1,100)).astype(int)

    cv2.polylines(img, [eval_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)
    for t in np.linspace(0.2, 0.8, 4):
        mid_pt = curve(t)
        mid_tan = curve.tangent(t)
        mid_tan = 30 * mid_tan / np.linalg.norm(mid_tan)
        cv2.line(img, mid_pt.astype(int), (mid_pt + mid_tan).astype(int), (0,255,0), 2)
    plt.imshow(img)
    plt.show()
