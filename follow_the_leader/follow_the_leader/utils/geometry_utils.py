import numpy as np
from scipy.interpolate import interp1d


def get_max_bend(pts):
    """
    Computes the maximum bending angle along a set of points consisting of [start, pt, end].
    Used as a crude estimate of how sharply a set of points bends.
    """

    if len(pts) <= 2:
        return None

    intermediate_points = pts[1:-1]
    vec_start = pts[0] - intermediate_points
    vec_end = intermediate_points - pts[-1]
    dps = (vec_start * vec_end).sum(axis=1) / (np.linalg.norm(vec_start, axis=1) * np.linalg.norm(vec_end, axis=1))
    dps[dps < -1] = -1
    dps[dps > 1] = 1

    return np.max(np.arccos(dps))


def get_max_pt_distance(pts_1, pts_2):
    """
    Finds the maximum distance of corresponding points between two curves (represented as sequences of points).
    Does so by interpolating the longer curve and then getting the points corresponding to the distances along the shorter curve
    """

    cumul_1 = convert_to_cumul_dists(pts_1)
    cumul_2 = convert_to_cumul_dists(pts_2)

    # make pts_1 refer to the shorter set of points
    if cumul_1[-1] > cumul_2[-1]:
        pts_2, pts_1 = pts_1, pts_2
        cumul_2, cumul_1 = cumul_1, cumul_2

    interp = interp1d(cumul_2, pts_2.T)
    return np.max(np.linalg.norm(pts_1 - interp(cumul_1).T, axis=1))


def convert_to_cumul_dists(pts):
    """
    Helper utility for converting a set of N points to an N-array corresponding to the cumulative distances,
    starting at 0.0 for the first point
    """

    dists = np.zeros(len(pts))
    dists[1:] = np.linalg.norm(pts[:-1] - pts[1:], axis=1).cumsum()
    return dists


def get_pt_line_dist_and_orientation(pts, origin, ray):
    """
    Given a set of "base" points, as well as a ray with a given origin in space,
    determine the distance of each point from the ray, as well as orientations corresponding to
    whether the point projects onto the positive or negative portion of the ray
    """

    diff = pts - origin
    ray = ray / np.linalg.norm(ray)
    proj_comp = diff.dot(ray)
    dists = np.linalg.norm(diff - proj_comp[..., np.newaxis] * ray, axis=1)
    return dists, np.sign(proj_comp)
