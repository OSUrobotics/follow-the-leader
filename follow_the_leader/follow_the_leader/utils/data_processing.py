from collections import defaultdict
from copy import deepcopy

import numpy as np
from follow_the_leader.utils.image_utils import (fill_holes_and_dilate,
                                                 mask_from_uv)

autodict = defaultdict(lambda: defaultdict(list))
def serialise_autodict(d):
    return {k: dict(v) for k, v in d.items()}

def subtract_missing_depth(mask_pxs, depth_dict, size, hole_fill_size):
    """
    From pixel mask, subtract points missing depth data
    :param mask_pxs: mask pixel coordinates Nx2
    :param depth_dict: dictionary of depth data
    :param size: size of the mask
    :param hole_fill_size: size of the hole fill kernel
    :return: mask of particles with missing depth data
    """
    depth_dict = deepcopy(depth_dict)
    missed_idxs = filter_depth_dict_by_z(depth_dict, z_min=1e-3)
    pxs_particle = mask_pxs[missed_idxs]
    m1 = fill_holes_and_dilate(mask_from_uv(depth_dict["pts_2d"], size), hole_fill_size)
    m2 = mask_from_uv(pxs_particle, size)
    particle_mask = np.logical_and(m1 < 128, m2 > 128)
    return particle_mask

def filter_depth_dict_by_z(depths_dict, z_min = None, z_max = None):
    """Edits dictionary to filter out points with. Returns idxs of filtered points."""
    pts = depths_dict["pts"]
    if z_min is not None and z_max is not None:
        valid_idxs = np.bitwise_and(pts[:, -1] > z_min, pts[:, -1] < z_max)
    elif z_min is not None:
        valid_idxs = pts[:, -1] > z_min
    elif z_max is not None:
        valid_idxs = pts[:, -1] < z_max
    depths_dict["pts"] = depths_dict["pts"][valid_idxs]
    depths_dict["pts_2d"] = depths_dict["pts_2d"][valid_idxs]
    depths_dict["error"] = depths_dict["error"][valid_idxs]
    # invert the valid_idxs to get the points filtered out
    return np.logical_not(valid_idxs)

def filter_depth_dict_by_error(depths_dict, error_max):
    """Edits dictionary to filter out points with. Returns idxs of filtered points."""
    valid_idxs = depths_dict["error"] < error_max
    depths_dict["pts"] = depths_dict["pts"][valid_idxs]
    depths_dict["pts_2d"] = depths_dict["pts_2d"][valid_idxs]
    depths_dict["error"] = depths_dict["error"][valid_idxs]
    # invert the valid_idxs to get the points filtered out
    return np.logical_not(valid_idxs)