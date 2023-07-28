import numpy as np
from follow_the_leader.utils.ros_utils import TFNode
from collections import defaultdict
import cv2

class PointHistory:
    def __init__(self, max_error=4.0):
        self.points = []
        self.errors = []
        self.radii = []
        self.max_error = max_error
        self.base_tf = None
        self.base_tf_inv = None

    def add_point(self, point, error, tf, radius):
        self.errors.append(error)
        self.radii.append(radius)
        if self.base_tf_inv is None:
            self.base_tf = tf
            self.base_tf_inv = np.linalg.inv(tf)
            self.points.append(point)
        else:
            self.points.append(TFNode.mul_homog(self.base_tf_inv @ tf, point))

    def as_point(self, inv_tf):

        errors = np.array(self.errors)
        idx = errors < self.max_error
        if np.any(idx):
            pts = np.array(self.points)[idx]
            errs = errors[idx]
            weights = 1 - np.array(errs) / self.max_error
            weights /= weights.sum()
            pt = (pts.T * weights).sum(axis=1)
            return TFNode.mul_homog(inv_tf @ self.base_tf, pt)

        return None

    @property
    def radius(self):
        if not self.radii:
            return None

        errors = np.array(self.errors)
        idx = errors < self.max_error
        if np.any(idx):
            try:
                radii = np.array(self.radii)[idx]
            except IndexError:
                import pdb
                pdb.set_trace()
            errs = errors[idx]
            weights = 1 - np.array(errs) / self.max_error
            weights /= weights.sum()

            return radii.dot(weights)


    def clear(self):
        self.points = []
        self.errors = []
        self.base_tf = None
        self.base_tf_inv = None

class BranchModel:

    def __init__(self, n=0, cam=None):
        self.model = [PointHistory() for _ in range(n)]
        self.inv_tf = None
        self.cam = cam
        self.trust = {}
        self.redo_render = True
        self._render = None

    def set_inv_tf(self, inv_tf):
        # inv_tf is a 4x4 transform matrix that relates the position of the base with respect to the camera (T_cam_base)
        self.inv_tf = inv_tf
        self.redo_render = True

    def set_camera(self, cam):
        self.cam = cam
        self.redo_render = True

    def retrieve_points(self, inv_tf=None, filter_none=False, trust_threshold=0):
        if inv_tf is None:
            inv_tf = self.inv_tf

        all_pts = [pt.as_point(inv_tf) for pt in self.model]
        if filter_none:
            all_pts = np.array([pt for i, pt in enumerate(all_pts) if pt is not None]).reshape(-1,3)
            # all_pts = np.array([pt for i, pt in enumerate(all_pts) if pt is not None and self.trust[i] > trust_threshold]).reshape(-1,3)

        return all_pts

    def point(self, i):
        return self.model[i].as_point(self.inv_tf)

    def update_point(self, tf, i, pt, err, radius):
        self.redo_render = True
        self.model[i].add_point(pt, err, tf, radius)

    @property
    def branch_mask(self):
        if self.redo_render:
            pts = self.retrieve_points(filter_none=True)
            radii = [pt.radius for pt in self.model]
            radii = np.array([r for r in radii if r is not None])

            pxs = self.cam.project3dToPixel(pts)
            px_radii = self.cam.getDeltaU(radii, pts[:,2])

            self._render = self.render_mask(self.cam.width, self.cam.height, pxs, px_radii)
            self.redo_render = False

        return self._render

    @staticmethod
    def render_mask(w, h, pxs, px_radii):
        mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(len(pxs) - 1):
            px_0 = pxs[i].astype(int)
            px_1 = pxs[i + 1].astype(int)
            radius = (px_radii[i] + px_radii[i + 1]) / 2
            thickness = max(int(radius * 2), 1)
            mask = cv2.line(mask, px_0, px_1, color=255, thickness=thickness)

        return mask > 128

    def update_trust(self, idx, val, reset=False):
        if reset:
            del self.trust[idx]
        else:
            if idx not in self.trust:
                self.trust[idx] = 0
            self.trust[idx] += val

    def clear(self, idxs=None):

        if idxs is None:
            self.model = []
        else:
            for idx in idxs:
                self.model[idx].clear()

    def extend_by(self, n):
        for _ in range(n):
            self.model.append(PointHistory())

    def chop_at(self, i):

        for idx in range(i+1, len(self.model)):
            if idx in self.trust:
                self.trust[idx] = 0
        self.model = self.model[:i+1]
        self.redo_render = True

    def __bool__(self):
        return bool(self.model)

    def __len__(self):
        return len(self.model)

    def __getitem__(self, item):
        return self.model[item]



