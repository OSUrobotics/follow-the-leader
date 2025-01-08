"""
File contains utility functions for calculating maximum distance at which a feature of a given size can be detected by a camera, as well as for calculating constrained distances for achieving a particular fov.

Implements the following formula
    fov_length = 2 * working distance * tan(angular_fov/2)
    feature_size = size_in_m / pixels_needed
    ifov = afov / pixels_in_direction
Other useful formulas:
    resolution_needed = pixels_needed * (fov_length / feature_size_m)
"""

from math import tan, atan2, degrees, radians
import numpy as np
from follow_the_leader.utils.image_utils import PinholeCameraModelNP
from follow_the_leader.utils.ros_utils import TFNode


def max_distance_for_feature_gen(ifov, px = 5, size_in_m = 0.003):
    feature_size = size_in_m / float(px)
    d = feature_size / (2. * tan(ifov / 2.))
    return d

def constrained_dist_gen(min_angular_fov, resolution, desired_fov, feature_px, size_in_m):
    max_dist = max_distance_for_feature_gen(min_angular_fov / resolution, feature_px, size_in_m)
    dist = desired_fov / (2 * tan(min_angular_fov / 2))
    if dist > max_dist:
        raise ValueError(f"Max distance supporting feature size is {max_dist}, for params {dist} needed reduce desired_fov or increase feature size")
    return dist

def max_distance_for_feature(camera: PinholeCameraModelNP, feature_px: int = 5, size_in_m: float = 2e-3):
    afov = min(camera.getAFOV())
    return max_distance_for_feature_gen(afov, feature_px, size_in_m)

def constrained_dist(camera: PinholeCameraModelNP, desired_fov, feature_px: int = 5, size_in_m: float = 2e-3):
    afov = camera.getAFOV()
    idx = np.argmin(afov)
    try:
        dist = constrained_dist_gen(afov[idx], camera.resolution[idx], desired_fov, feature_px, size_in_m)
        return dist
    except Exception as e:
        return e

if __name__ == "__main__":
    # realsense d405 calculations
    camera = PinholeCameraModelNP()
    camera.fromCameraInfo(TFNode.define_dummy_camera())
    print(f"afov: {degrees(min(camera.getAFOV()))} resolution {camera.resolution[np.argmin(camera.getAFOV())]}")
    print(constrained_dist(camera, 3., 5, 2.5e-2))
    print(max_distance_for_feature(camera))
    # orbec femto bolt calculations
    print(constrained_dist_gen(radians(51), 960, 3., 5, 2.5e-2))
    print(constrained_dist_gen(radians(51), 960, 0.3, 5, 5e-3))
    print(max_distance_for_feature_gen(radians(51)/960, 5, 5e-3))
