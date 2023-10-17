import sqlite3
import matplotlib.pyplot as plt
import os
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from follow_the_leader.utils.ros_utils import PinholeCameraModelNP, TFNode
from follow_the_leader.curve_fitting import BezierBasedDetection, Bezier
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from cv_bridge import CvBridge
from follow_the_leader.utils.branch_model import BranchModel
import cv2
from skimage.morphology import binary_dilation
from PIL import Image, ImageDraw

bridge = CvBridge()

"""
Processes a bag file and shows the detected branches in the skeleton (show_skel_results()),
as well as the results of using PIPs to track points across 8 frames (show_pips_tracking())

"""


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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

    def query_closest(self, topic_name, ts):
        topic_id = self.topic_name_to_id[topic_name]
        topic_msg_type = self.topic_name_to_type[topic_name]

        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {} ORDER BY ABS(timestamp - {}) ASC LIMIT 1".format(
                topic_id, ts
            )
        )
        for ts, data in rows:
            yield ts, deserialize_message(data, topic_msg_type)


def show_skel_results(bag_file):
    bag_file_db = os.path.join(bag_file)
    reader = BagReader(bag_file_db)
    camera_info = list(reader.query("/camera/color/camera_info"))[0][1]
    for _, msg in reader.query("/image_mask"):
        mask = bridge.imgmsg_to_cv2(msg) > 128
        h, w = mask.shape

        detection = BezierBasedDetection(mask, use_medial_axis=True, use_vec_weighted_metric=True)
        curve = detection.fit((0, -1))

        skel = detection.skel
        radius_interpolator = detection.get_radius_interpolator_on_path()

        ds = np.linspace(0, 1, 11)
        pts, _ = curve.eval_by_arclen(ds, normalized=True)
        radii = radius_interpolator(ds)
        leader_mask_estimate = BranchModel.render_mask(w, h, pts, radii)

        sbs = detection.run_side_branch_search(min_len=20, filter_mask=leader_mask_estimate)

        if len(sbs):
            output_path = "/home/main/Documents/Dissertation and ICRA 2024 Paper"

            sbs_drawn = np.zeros((h, w, 3), dtype=np.uint8)
            for sb in sbs:
                curve = sb["curve"]
                pxs, _ = curve.eval_by_arclen(np.linspace(0, 1, 51), normalized=True)
                cv2.polylines(sbs_drawn, [pxs.reshape((-1, 1, 2)).astype(int)], False, (0, 255, 0), 5)

            skel_inflated = binary_dilation(skel)

            zeros = np.zeros((h, w))
            # Base mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(output_path, "mask.png"))

            # Mask showing skeleton
            node_r = 3
            skel_inflated_rgb = Image.fromarray(np.dstack([skel_inflated * 255, zeros, zeros]).astype(np.uint8))
            skel_inflated_rgb.putalpha(skel_inflated_rgb.getchannel("R").point(lambda x: 255 if x else 0))
            skel_img = Image.fromarray(np.dstack([mask * 255] * 3).astype(np.uint8))
            skel_img.paste(skel_inflated_rgb, mask=skel_inflated_rgb)
            node_px = list(detection.subsampled_graph.nodes)
            draw = ImageDraw.Draw(skel_img)
            for px in node_px:
                px = np.array(px)
                ul = px - (node_r, node_r)
                br = px + (node_r, node_r)
                draw.ellipse((*ul, *br), fill="blue", outline="black")
            skel_img.save(os.path.join(output_path, "skel.png"))

            # Mask showing skeleton with chosen path
            stats = detection.stats
            chosen_pxs = stats["pts"]
            chosen_pxs_set = {tuple(px) for px in chosen_pxs}

            skel_inflated_rgb.putalpha(skel_inflated_rgb.getchannel("R").point(lambda x: 128 if x else 0))

            chosen_skel_img = Image.fromarray(np.dstack([mask * 255] * 3).astype(np.uint8))
            chosen_skel_img.paste(skel_inflated_rgb, mask=skel_inflated_rgb)

            chosen_skel = np.zeros((h, w), dtype=bool)
            chosen_skel[chosen_pxs[:, 1], chosen_pxs[:, 0]] = True
            chosen_skel_inflated = binary_dilation(chosen_skel)
            chosen_skel_inflated = Image.fromarray(
                np.dstack([zeros, chosen_skel_inflated * 255, zeros]).astype(np.uint8)
            )
            chosen_skel_inflated.putalpha(chosen_skel_inflated.getchannel("G").point(lambda x: 255 if x else 0))
            chosen_skel_img.paste(chosen_skel_inflated, mask=chosen_skel_inflated)

            draw = ImageDraw.Draw(chosen_skel_img)
            for px in node_px:
                if px in chosen_pxs_set:
                    continue
                px = np.array(px)
                r = 3
                ul = px - (r, r)
                br = px + (r, r)
                draw.ellipse((*ul, *br), fill="red", outline="black", width=1)

            for px in node_px:
                if px not in chosen_pxs_set:
                    continue
                px = np.array(px)
                r = 5
                ul = px - (r, r)
                br = px + (r, r)
                draw.ellipse((*ul, *br), fill="green", outline="blue", width=2)

            chosen_skel_img.save(os.path.join(output_path, "chosen_skeleton.png"))

            # Showing side branches that have been found
            leader_est_flat = Image.fromarray(np.dstack([leader_mask_estimate * 200] * 3).astype(np.uint8))
            leader_est_flat.putalpha(leader_est_flat.getchannel("R").point(lambda x: 255 if x else 0))
            sb_img = Image.fromarray(np.dstack([mask * 255] * 3).astype(np.uint8))
            sb_img.paste(skel_inflated_rgb, mask=skel_inflated_rgb)
            sb_img.paste(leader_est_flat, mask=leader_est_flat)

            leader_curve_drawn = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.polylines(leader_curve_drawn, [pts.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 255), 3)
            leader_curve_img = Image.fromarray(leader_curve_drawn)
            leader_curve_img.putalpha(leader_curve_img.getchannel("B").point(lambda x: 255 if x else 0))
            sb_img.paste(leader_curve_img, mask=leader_curve_img)

            sbs_drawn = Image.fromarray(sbs_drawn)
            sbs_drawn.putalpha(sbs_drawn.getchannel("G").point(lambda x: 128 if x else 0))
            sb_img.paste(sbs_drawn, mask=sbs_drawn)
            sb_img.save(os.path.join(output_path, "side_branches.png"))

            plt.imshow(np.array(sb_img))
            plt.show()

    camera = PinholeCameraModelNP()
    camera.fromCameraInfo(camera_info)

    w = camera.width
    h = camera.height


def show_pips_tracking(bag_file):
    output_path = "/home/main/Documents/Dissertation and ICRA 2024 Paper"

    bag_file_db = os.path.join(bag_file)
    reader = BagReader(bag_file_db)
    camera_info = list(reader.query("/camera/color/camera_info"))[0][1]

    from follow_the_leader.point_tracker import PointTriangulator

    cam = PinholeCameraModelNP()
    cam.fromCameraInfo(camera_info)
    triangulator = PointTriangulator(cam)

    queue = []

    for _, msg in reader.query("/image_mask"):
        queue.append({"mask": bridge.imgmsg_to_cv2(msg), "ts": stamp_to_float(msg.header.stamp)})

    current_target = 0
    for _, msg in reader.query("/camera/color/image_rect_raw"):
        stamp = stamp_to_float(msg.header.stamp)
        ts_diff = abs(stamp - queue[current_target]["ts"])
        if ts_diff < 1e-5:
            print("Match (TS Diff: {:.6f}s)".format(ts_diff))
            queue[current_target]["rgb"] = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            current_target += 1

        if current_target >= len(queue):
            break

    poses = []
    all_ts = []
    for _, pose in reader.query("/camera_pose"):
        all_ts.append(stamp_to_float(pose.header.stamp))
        poses.append(pose_to_tf(pose, as_matrix=False))
    poses = np.array(poses)

    pose_interp = interp1d(all_ts, poses.T)
    move_thres = 0.02 / 8

    from follow_the_leader.point_tracker import RotatingQueue
    from follow_the_leader.networks.pips_model import PipsTracker

    img_queue = RotatingQueue(size=7)
    tracker = PipsTracker(
        model_dir=os.path.join(os.path.expanduser("~"), "follow-the-leader-deps", "pips", "pips", "reference_model")
    )

    last_pose = np.zeros(7)
    current_target = 0

    all_imgs = list(reader.query("/camera/color/image_rect_raw"))
    for raw_ts, msg in all_imgs:
        ts = stamp_to_float(msg.header.stamp)
        try:
            pose = pose_interp(ts)
        except ValueError:
            print("Error getting pose")
            continue

        print(current_target)

        if img_queue.is_full and ts >= queue[current_target]["ts"]:
            all_infos = img_queue.as_list()[::-1]
            imgs = [queue[current_target]["rgb"]] + [info["rgb"] for info in all_infos]
            img_poses = [pose_interp(queue[current_target]["ts"])] + [info["pose"] for info in all_infos]
            pose_matrices = [pose_array_to_matrix(pose) for pose in img_poses]
            mask = queue[current_target]["mask"]

            detection = BezierBasedDetection(mask, use_medial_axis=True, use_vec_weighted_metric=True)
            curve = detection.fit((0, -1))
            sb_curves = [sb_info["curve"] for sb_info in detection.run_side_branch_search(min_len=40)]

            # Output some utils for curve fitting
            rgb = queue[current_target]["rgb"]
            Image.fromarray(rgb).save(os.path.join(output_path, "tracking_raw_rgb.png"))
            Image.fromarray(mask).save(os.path.join(output_path, "tracking_raw_mask.png"))
            curve_img = np.dstack([mask] * 3).astype(np.uint8)
            curve_pxs = curve(np.linspace(0, 1, 101))
            cv2.polylines(curve_img, [curve_pxs.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 255), 5)
            Image.fromarray(curve_img).save(os.path.join(output_path, "tracking_mask_with_curve.png"))
            # Reprojection
            closest_curve = list(reader.query_closest("/tree_model", raw_ts))[0][1]
            last_curve_pts = np.array(
                [[pt.x, pt.y, pt.z] for pt, pid in zip(closest_curve.points, closest_curve.ids) if pid == 0]
            )
            last_curve_pts += np.random.uniform(-0.005, 0.005, last_curve_pts.shape)
            last_curve_pts[-2] += np.array([0.025, 0, 0])
            reproj_pxs = cam.project3dToPixel(last_curve_pts)
            cv2.polylines(curve_img, [reproj_pxs.reshape((-1, 1, 2)).astype(int)], False, (0, 255, 0), 3)
            reproj_model_img = Image.fromarray(curve_img)
            draw = ImageDraw.Draw(reproj_model_img)
            for px in reproj_pxs:
                if not (0 < px[0] < cam.width and 0 < px[1] < cam.height):
                    continue
                is_close = curve.query_pt_distance(px)[0] < 5.0
                color = "green" if is_close else "red"
                outline = "black" if is_close else (200, 0, 0)
                draw.ellipse((*px - (6, 6), *px + (6, 6)), fill=color, outline=outline, width=3)
            reproj_model_img.save(os.path.join(output_path, "tracking_mask_with_reproj_model.png"))

            # Choose closest points as query points
            _, refit_ts = curve.query_pt_distance(reproj_pxs)
            refit_pxs = curve(refit_ts)
            query_pts_img = np.dstack([mask] * 3).astype(np.uint8)
            cv2.polylines(query_pts_img, [refit_pxs.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 255), 5)
            query_pts_img = Image.fromarray(query_pts_img)
            draw = ImageDraw.Draw(query_pts_img)
            for px in refit_pxs:
                draw.ellipse((*px - (6, 6), *px + (6, 6)), fill="red", outline="black", width=3)
            query_pts_img.save(os.path.join(output_path, "tracking_new_query_pts.png"))
            trajs = tracker.track_points(refit_pxs, imgs)
            refit_pts = triangulator.compute_3d_points(pose_matrices, np.transpose(trajs, (1, 0, 2)))

            # plot_3d(refit_pts, [[None] * len(refit_pts)])

            all_curves = [curve] + sb_curves
            to_eval = []
            for curve in all_curves:
                arclen = curve.arclen
                pts, _ = curve.eval_by_arclen(np.arange(0, arclen, 30))
                to_eval.append(pts)

            trajs = tracker.track_points(np.concatenate(to_eval), imgs)
            for i, (rgb, traj) in enumerate(zip(imgs, trajs)):
                if i == 0:
                    mask_rgb = np.dstack([mask] * 3).astype(np.uint8)
                    img_array = (0.5 * rgb + 0.5 * mask_rgb).astype(np.uint8)
                    img = draw_pts_on_image(img_array, traj, to_eval)
                    img.save(os.path.join(output_path, f"tracking_mask.png"))

                img_array = rgb
                img = draw_pts_on_image(img_array, traj, to_eval)
                img.save(os.path.join(output_path, f"tracking_{i}.png"))

            plt.imshow(np.array(img))
            plt.show()

            # # TEMP
            # refit_pts[len(refit_pts) // 2] += np.array([0.05, 0.05, 0])
            #
            # refit_curve, stats = Bezier.iterative_fit(refit_pts, inlier_threshold=0.01, max_iters=100,
            #                                           stop_threshold=0.6)
            # inlier_idx = stats['inlier_idx']
            #
            # ax = plt.figure().add_subplot(projection='3d')
            # ax.scatter(*refit_pts[inlier_idx].T, color='green', marker='o', s=40)
            # ax.scatter(*refit_pts[~inlier_idx].T, color='red', marker='x', s=40)
            # fit_curve_pts = refit_curve(np.linspace(0, 1, 1000))
            # ax.plot(*fit_curve_pts.T, color='blue', marker='None', markersize=40, linestyle='solid')
            # set_axes_equal(ax)
            # plt.show()

            current_target += 1
            if current_target >= len(queue):
                return

        if np.linalg.norm(pose[:3] - last_pose[:3]) > move_thres:
            info = {"rgb": bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"), "pose": pose, "ts": ts}
            img_queue.append(info)
            last_pose = pose


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


def stamp_to_float(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def draw_pts_on_image(img_array, all_pts, line_info):
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Draw connector lines
    cur_idx = 0
    for pts in line_info:
        start_idx = cur_idx
        end_idx = cur_idx + len(pts)
        traj_ss = all_pts[start_idx:end_idx]

        color = "blue" if start_idx == 0 else "green"
        for pt1, pt2 in zip(traj_ss[:-1], traj_ss[1:]):
            draw.line((*pt1, *pt2), fill=color, width=5)

        cur_idx += len(pts)

    node_r = 6
    for px in all_pts:
        ul = px - (node_r, node_r)
        br = px + (node_r, node_r)
        draw.ellipse((*ul, *br), fill="red", outline="black", width=2)

    return img


def plot_3d(pts_3d, line_info):
    ax = plt.figure().add_subplot(projection="3d")
    cur_idx = 0
    for pts in line_info:
        start_idx = cur_idx
        end_idx = cur_idx + len(pts)

        pts_ss = pts_3d[start_idx:end_idx]
        color = "blue" if start_idx == 0 else "green"

        ax.plot(*pts_ss.T, color=color, marker="o", markersize=8)

        # projed = pts_ss.copy()
        # projed[:,2] = pts_ss[:,2].min()
        # ax.plot(*projed.T, marker='o', markersize=8, path_effects=[SimpleLineShadow(shadow_color='black', linewidth=5), Normal()])

        cur_idx += len(pts)

    set_axes_equal(ax)
    plt.show()


def pose_array_to_matrix(vec):
    tf = np.identity(4)
    tf[:3, 3] = vec[:3]
    tf[:3, :3] = Rotation.from_quat(vec[3:]).as_matrix()
    return tf


if __name__ == "__main__":
    bag_file = "/home/main/data/model_the_leader/real_data/10/001/bag_data/bag_data_0.db3"
    show_skel_results(bag_file)
    show_pips_tracking(bag_file)
