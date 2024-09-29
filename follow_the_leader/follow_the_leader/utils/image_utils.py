import numpy as np
from scipy.ndimage import label, binary_fill_holes
import cv2
from image_geometry import PinholeCameraModel
from copy import deepcopy

def fill_holes_and_dilate(cv_mask: cv2.Mat, fill_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_size, fill_size))
    res = cv2.morphologyEx(cv_mask, cv2.MORPH_CLOSE, kernel)
    res = cv2.dilate(res, kernel, iterations=2)
    return res

def convex_hull(cv_mask: cv2.Mat, fill_size):
    res = fill_holes_and_dilate(cv_mask, fill_size)
    contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, contours, -1, 255, -1)
    return res

def fill_holes_sk(img):
    mask = img > 128
    filled = binary_fill_holes(mask)
    # make opencv compatible
    return filled.astype(np.uint8) * 255

def erosion(img, fill_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_size, fill_size))
    return cv2.erode(img, kernel, iterations=1)

def uv_from_mask(mask):
    return np.argwhere(mask)[:, [1, 0]]

def uv_for_value(image, value):
    return np.argwhere(image == value)[:, [1, 0]]

def value_at_uv(image, uv):
    return image[uv[:, 1], uv[:, 0]]

def mask_from_uv(uv, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[uv[:, 1], uv[:, 0]] = 255
    return cv2.Mat(mask)

def euclidean_ray_length_to_z_coordinate(depth_image, camera_model):
        """ From https://github.com/ethz-asl/scenenet_ros_tools/blob/master/nodes/scenenet_to_rosbag.py
        """
        center_x = camera_model.cx()
        center_y = camera_model.cy()

        constant_x = 1 / camera_model.fx()
        constant_y = 1 / camera_model.fy()

        vs = np.array(
            [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
        us = np.array(
            [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

        return (np.sqrt(
            np.square(depth_image / 1000.0) /
            (1 + np.square(vs[np.newaxis, :]) + np.square(us[:, np.newaxis]))) *
                1000.0).astype(np.uint16)

def mask_point_selection(
    mask, strategy="uniform_sample", num_points=100
):
    mask_pxs = uv_from_mask(mask)
    if len(mask_pxs) > num_points:
        if strategy == "random_sample":
            mask_pxs = mask_pxs[
                np.random.choice(len(mask_pxs), num_points, replace=False)
            ]
        elif strategy == "uniform_sample":
            step = max(1, len(mask_pxs) // num_points)
            mask_pxs = mask_pxs[::step]
        elif strategy == "none":
            pass
        elif strategy == "fixed_spacing":
            raise NotImplementedError
        else:
            raise ValueError("Unknown strategy: {}".format(strategy))
    return mask_pxs


class PinholeCameraModelNP(PinholeCameraModel):
    """
    Modifications to the PinholeCameraModel class to make them operate with Numpy.
    """

    def __init__(self):
        super().__init__()
        self.total_pixels = None

    def project3dToPixel(self, pts):
        pts = np.array(pts)
        pts_homog = np.ones((*pts.shape[:-1], pts.shape[-1] + 1))
        pts_homog[..., :3] = pts

        x, y, w = np.array(self.P) @ pts_homog.T
        return np.array([x / w, y / w]).T

    def getDeltaU(self, deltaX, Z):
        fx = self.P[0, 0]
        return fx * deltaX / Z

    def pixelTo3D(self, pxs, depth_image):
        """
        Given a set of pixel coordinates uv and a depth image, return the 3D coordinates of the pixels.
        pxs: Nx2 array of pixel coordinates
        depth_image: HxW depth image
        Adapted from
        https://github.com/ethz-asl/scenenet_ros_tools/blob/master/nodes/scenenet_to_rosbag.py
        """
        center_x = self.cx()
        center_y = self.cy()

        constant_x = 1. / self.fx()
        constant_y = 1. / self.fy()

        us = (pxs[:, 0] - center_x) * constant_x
        vs = (pxs[:, 1] - center_y) * constant_y

        # # Find z coordinate of each pixel given the depth in ray length.
        # z = value_at_uv(euclidean_ray_length_to_z_coordinate(depth_image, self), pxs)
        # Where the depth data is stored as the z coordinate, change the above line to:
        z = value_at_uv(depth_image, pxs)
        np.nan_to_num(z, copy=False)

        # Convert the z coordinate from mm to m.
        z = z / 1000.0
        x = np.multiply(z, us)
        y = np.multiply(z, vs)

        # stacked = np.ma.dstack((x, y, z, pxs))
        # compressed = stacked.compressed()
        # pointcloud = compressed.reshape((int(compressed.shape[0] / 6), 6))
        return np.array([x, y, z]).T.reshape(-1, 3)

    def getAFOV(self):
        fov_x = 2 * np.arctan2(self.width, 2 * self.fx())
        fov_y = 2 * np.arctan2(self.height, 2 * self.fy())
        return fov_x, fov_y

    def getIFOV(self):
        afov_x, afov_y = self.getAFOV()
        return afov_x / self.width, afov_y / self.height
    

if __name__ == "__main__":
    import os
    hole_fill_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'test', 'resources', 'hole_fill_example.jpg'))
    hole_fill_img = cv2.imread(hole_fill_img_path, cv2.IMREAD_GRAYSCALE)
    filled = fill_holes_and_dilate(hole_fill_img, 5)
    filled_sk = fill_holes_sk(hole_fill_img)
    convex_hull_im = convex_hull(hole_fill_img, 10)
    if (filled == hole_fill_img).all():
        print('fill_holes failed')
    # show both images
    cv2.imshow('original', hole_fill_img)
    cv2.imshow('filled', filled)
    cv2.imshow('filled_sk', filled_sk)
    cv2.imshow('convex_hull', convex_hull_im)
    cv2.waitKey(0)
