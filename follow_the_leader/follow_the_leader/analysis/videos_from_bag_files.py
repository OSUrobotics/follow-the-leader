import sqlite3

import cv_bridge
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from follow_the_leader.utils.ros_utils import PinholeCameraModelNP, TFNode
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from itertools import product
from collections import defaultdict
import imageio
import pyvista as pv

"""
For a given bag file, outputs GIFs/videos showing the RGB feed, masks,
skeletonization diagnostics, and the rendered model
"""


class BagReader:
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()
        topics_data = self.conn.execute("SELECT id, name, type FROM topics").fetchall()

        self.topics = [r[1] for r in topics_data]
        self.topic_name_to_id = {r[1]: r[0] for r in topics_data}
        self.topic_name_to_type = {r[1]: get_message(r[2]) for r in topics_data}

    def query(self, topic_name):
        topic_id = self.topic_name_to_id[topic_name]
        topic_msg_type = self.topic_name_to_type[topic_name]

        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id))
        for ts, data in rows:
            yield ts, deserialize_message(data, topic_msg_type)


def backfill_source_against_timestamps(reader, timestamps, topic, use_raw_ts=True):

    msgs = []

    last_processed_ts = 0

    for raw_ts, msg in reader.query(topic):
        if use_raw_ts:
            ts = raw_ts * 1e-9
        else:
            stamp = msg.header.stamp
            ts = stamp.sec + stamp.nanosec * 1e-9

        while True:

            if last_processed_ts >= len(timestamps):
                break

            baseline_ts = timestamps[last_processed_ts]
            if baseline_ts <= ts:
                msgs.append(msg)
                last_processed_ts += 1
            else:
                break

    msgs.extend([msgs[-1]] * (len(timestamps) - len(msgs)))
    return msgs


def stamp_to_ts(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def models_to_renders(models, pose_interp, main_rad=0.01, side_rad=0.01):

    pt_lists = []
    base_pose = pose_interp(stamp_to_ts(models[0].header.stamp))
    tf_base_world = np.linalg.inv(pose_array_to_matrix(base_pose))

    for model in models:

        pose = pose_interp(stamp_to_ts(model.header.stamp))
        tf_world_cam = pose_array_to_matrix(pose)
        tf_base_cam = tf_base_world @ tf_world_cam

        pt_list = []
        last_id = -1
        for id, point in zip(model.ids, model.points):
            if last_id != id:
                last_id = id
                pt_list.append([])
            pt = np.array([point.x, point.y, point.z])
            pt_list[-1].append(TFNode.mul_homog(tf_base_cam, pt))
        pt_lists.append(pt_list)

    cyl_lists = []
    for pt_list in pt_lists:
        all_cyls = []
        for i, branch_pts in enumerate(pt_list):
            for pt_1, pt_2 in zip(branch_pts[:-1], branch_pts[1:]):

                cyl_center = (pt_1 + pt_2) / 2
                color = "blue" if i == 0 else "green"
                rad = main_rad if i == 0 else side_rad
                cyl = pv.Cylinder(
                    cyl_center,
                    direction=pt_2 - pt_1,
                    radius=rad,
                    height=np.linalg.norm(pt_2 - pt_1),
                    resolution=50,
                    capping=False,
                )
                all_cyls.append((cyl, color))

        cyl_lists.append(all_cyls)

    # Get the camera for the last cyl list
    plotter = pv.Plotter(off_screen=True)
    for cyl, color in cyl_lists[-1]:
        plotter.add_mesh(cyl, color=color, smooth_shading=True, opacity=1)

    plotter.camera.tight(view="xy", negative=False, padding=0.02)
    cam = plotter.camera
    props = (cam.position, cam.roll, cam.elevation, cam.azimuth)
    plotter.clear()

    return_imgs = []
    last_ts = None
    for i, cyl_list in enumerate(cyl_lists):

        if not i % 10:
            print(f"\t{i}/{len(cyl_lists)}")

        current_ts = stamp_to_ts(models[i].header.stamp)
        if current_ts == last_ts:
            return_imgs.append(return_imgs[-1])
            continue

        last_ts = current_ts
        for cyl, color in cyl_list:
            plotter.add_mesh(cyl, color=color, smooth_shading=True, opacity=1)

        plotter.camera.position = props[0]
        plotter.camera.roll = props[1]
        plotter.camera.elevation = props[2]
        plotter.camera.azimuth = props[3]
        img = plotter.screenshot(return_img=True, transparent_background=False)

        # Issue here is that the transform frame is wrong! Need to figure out how to get transforms

        plotter.clear()

        return_imgs.append(img)

    return return_imgs


def pose_to_tf(pose_stamped: PoseStamped, as_matrix=True):
    tl = pose_stamped.pose.position
    q = pose_stamped.pose.orientation

    tl = np.array([tl.x, tl.y, tl.z])
    q = np.array([q.x, q.y, q.z, q.w])

    if not as_matrix:
        return np.concatenate([tl, q])

    tf = np.identity(4)
    tf[:3, 3] = tl
    tf[:3, :3] = Rotation.from_quat(q).as_matrix()

    return tf


def pose_array_to_matrix(vec):
    tf = np.identity(4)
    tf[:3, 3] = vec[:3]
    tf[:3, :3] = Rotation.from_quat(vec[3:]).as_matrix()
    return tf


def process_bag_file(file, video_output=None, gif_output=None, leader_rad=0.01, sb_rad=0.01):
    reader = BagReader(file)
    bridge = cv_bridge.CvBridge()

    # Get all the RGB images and the corresponding timestamps
    timestamps = []
    all_images = []
    for _, msg in reader.query("/camera/color/image_rect_raw"):
        stamp = msg.header.stamp
        ts = stamp.sec + stamp.nanosec * 1e-9

        timestamps.append(ts)
        all_images.append(bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"))

    all_ts = np.array(timestamps)
    fps_float = np.round((len(timestamps) - 1) / (all_ts[-1] - all_ts[0]))
    fps = int(fps_float)

    # Retrieve the poses of the camera, which will be used for rendering the 3D model
    poses = []
    poses_ts = []
    for _, pose in reader.query("/camera_pose"):
        stamp = pose.header.stamp
        poses_ts.append(stamp.sec + stamp.nanosec * 1e-9)
        poses.append(pose_to_tf(pose, as_matrix=False))
    poses = np.array(poses)
    pose_interp = interp1d(poses_ts, poses.T)

    # Get all the images for the masks, diagnostics, and model renders, and sync them against the RGB timestamps
    model_renders = models_to_renders(
        backfill_source_against_timestamps(reader, all_ts, topic="/tree_model", use_raw_ts=False),
        pose_interp,
        main_rad=leader_rad,
        side_rad=sb_rad,
    )
    diagnostics = [
        bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        for msg in backfill_source_against_timestamps(reader, all_ts, "/model_diagnostic", use_raw_ts=True)
    ]
    masks = [
        bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        for msg in backfill_source_against_timestamps(reader, all_ts, topic="/image_mask", use_raw_ts=False)
    ]

    # Output AVIs
    if video_output is not None:
        import cv2

        writer = None
        format = cv2.VideoWriter_fourcc(*"MP4V")

        for rgb, diag, mask in zip(all_images, diagnostics, masks):
            mask = np.dstack([mask] * 3)
            img = np.concatenate([rgb, diag, mask], axis=0).astype(np.uint8)
            if writer is None:
                writer = cv2.VideoWriter(video_output, format, fps_float, (img.shape[1], img.shape[0]))

            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()

        writer = None
        for img in model_renders:
            if writer is None:
                writer = cv2.VideoWriter(
                    video_output.replace(".mp4", "_render.mp4"), format, fps_float, (img.shape[1], img.shape[0])
                )

            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()

    # Output GIFs
    for label, imgs in [("rgb", all_images), ("diagnostic", diagnostics), ("mask", masks), ("render", model_renders)]:
        if gif_output is not None:
            imageio.mimsave(gif_output.replace(".gif", f"_{label}.gif"), imgs, fps=fps)


if __name__ == "__main__":
    root = "/home/main/data/model_the_leader/real_data"
    to_proc = [10, 11, 12, 13]

    for folder_id in to_proc:
        subfolder = os.path.join(root, str(folder_id))
        for run in os.listdir(subfolder):
            if not os.path.isdir(os.path.join(subfolder, run)):
                continue

            results_file = os.path.join(subfolder, run, "0_results.pickle")
            with open(results_file, "rb") as fh:
                rez = pickle.load(fh)

            leader_radius = np.mean([x.radius for x in rez["leader_raw"].model if x.radius is not None])
            sb_radii = []
            for sb in rez["side_branches_raw"]:
                sb_radii.append(np.mean([x.radius for x in sb.model if x.radius is not None]))
            sb_radius = np.mean(sb_radii)

            bag_file = os.path.join(subfolder, run, "bag_data", "bag_data_0.db3")
            print("Proc {}".format(bag_file))
            process_bag_file(
                bag_file,
                video_output=os.path.join(root, "videos", "bag_gifs", f"{folder_id}_{run}.mp4"),
                gif_output=os.path.join(root, "videos", "bag_gifs", "{}_{}.gif".format(folder_id, run)),
                leader_rad=leader_radius,
                sb_rad=sb_radius,
            )
