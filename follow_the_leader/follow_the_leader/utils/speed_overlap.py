from math import tan, atan2, degrees, radians
import numpy as np
from follow_the_leader.utils.image_utils import PinholeCameraModelNP
from follow_the_leader.utils.ros_utils import TFNode


def max_dist_for_px_overlap(
    camera: PinholeCameraModelNP, z_distance: float, max_overlap: float
):
    """
    Calculate the maximum distance the camera can move while maintaining a certain overlap.
    """
    max_px_travel = min(camera.width, camera.height) * (1 - max_overlap)
    max_dist_travel = min(
        camera.getDeltaX(max_px_travel, z_distance),
        camera.getDeltaY(max_px_travel, z_distance),
    )
    return max_dist_travel


def max_speed(camera: PinholeCameraModelNP, z_distance: float, max_overlap: float, fps=30, processing_time=1.0):
    """
    Calculate the maximum speed at which the camera can move while maintaining a certain overlap.
    """
    shutter_time = 1. / fps
    t = shutter_time + processing_time
    max_dist_travel = max_dist_for_px_overlap(camera, z_distance, max_overlap)
    return max_dist_travel / t


if __name__ == "__main__":
    camera = PinholeCameraModelNP()
    camera.fromCameraInfo(TFNode.define_dummy_camera())
    print(max_dist_for_px_overlap(camera, 3.0, 0.95))
    print(max_speed(camera, 3.0, 0.95))
