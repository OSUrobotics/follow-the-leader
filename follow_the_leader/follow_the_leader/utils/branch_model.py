import numpy as np
from follow_the_leader.utils.ros_utils import TFNode
from collections import defaultdict

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
        self.counters = defaultdict(lambda: 0)
        self.redo_render = True

    def set_inv_tf(self, inv_tf):
        # inv_tf is a 4x4 transform matrix that relates the position of the base with respect to the camera (T_cam_base)
        self.inv_tf = inv_tf
        self.redo_render = True

    def set_camera(self, cam):
        self.cam = cam
        self.redo_render = True

    def retrieve_points(self, inv_tf=None, filter_none=False):
        if inv_tf is None:
            inv_tf = self.inv_tf

        all_pts = [pt.as_point(inv_tf) for pt in self.model]
        if filter_none:
            all_pts = np.array([pt for pt in all_pts if pt is not None]).reshape(-1,3)

        return all_pts

    def point(self, i):
        return self.model[i].as_point(self.inv_tf)

    def update_point(self, tf, i, pt, err, radius):
        self.redo_render = True
        self.model[i].add_point(pt, err, tf, radius)

    def increment_counter(self, key):
        self.counters[key] += 1

    def retrieve_counter(self, key):
        return self.counters[key]

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
        self.redo_render = True
        self.model = self.model[:i+1]

    def __bool__(self):
        return bool(self.model)

    def __len__(self):
        return len(self.model)

    def __getitem__(self, item):
        return self.model[item]



