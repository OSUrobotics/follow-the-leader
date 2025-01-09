#!/usr/bin/env python3
import os.path
from collections import defaultdict
from threading import Lock

import numpy as np
import rclpy
from cv_bridge import CvBridge
from follow_the_leader.networks.pips_model import PipsTracker
from follow_the_leader.utils.ros_utils import (SharedData, TFNode,
                                               process_list_as_dict)
from follow_the_leader_msgs.action import ImageProcessor
from follow_the_leader_msgs.msg import (Point2D, StateTransition,
                                        Tracked3DPointGroup,
                                        Tracked3DPointResponse,
                                        TrackedPointGroup, TrackedPointRequest)
from follow_the_leader_msgs.srv import Query3DPoints

from geometry_msgs.msg import Point
import rclpy.action
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header

bridge = CvBridge()


class RotatingQueue:
    def __init__(self, size=8) -> None:
        self.queue = [None] * size
        self.idx = 0
        self.mutex = Lock()
        return

    def empty(self) -> None:
        self.queue = [None] * len(self.queue)
        self.idx = 0
        return

    @property
    def is_full(self) -> bool:
        return not None in self.queue

    def append(self, item) -> None:
        self.queue[self.idx] = item
        self.idx = (self.idx + 1) % len(self.queue)
        return

    def as_list(self):
        if not self.is_full:
            return self.queue[: self.idx]
        else:
            return self.queue[self.idx :] + self.queue[: self.idx]

    def __len__(self) -> int:
        if self.is_full:
            return len(self.queue)
        return self.idx

    def __enter__(self):
        self.mutex.__enter__()
        return

    def __exit__(self, *args, **kwargs):
        self.mutex.__exit__(*args, **kwargs)
        return

class PointTracker(TFNode):
    def __init__(self):
        super().__init__("point_tracker_node", cam_info_topic="/camera/color/camera_info")
        # State variables
        self.current_request = SharedData()
        self.image_queue = RotatingQueue(size=8)
        self.back_image_queue = RotatingQueue(size=16)
        self.mask_image_queue = RotatingQueue(size=16)
        self.tracker = PipsTracker(
            model_dir=os.path.join(os.path.expanduser("~"), "follow-the-leader-deps", "pips", "pips", "reference_model")
        )
        self.last_pos = None

        # Config
        default_params = {
            "movement_threshold": 0.0075 / 8,
            "base_frame": "base_link",
            "min_points": 4,
            "do_3d_point_estimation": True,
            "camera_topic_name": "/blank",
            "log_path": "/tmp",
        }
        self.declare_parameter_dict(**default_params)
        self.base_frame = self.get_param_val("base_frame")
        self.movement_threshold = self.get_param_val("movement_threshold")

        # ROS Utils
        self.cb = MutuallyExclusiveCallbackGroup()
        self.cb_reentrant = ReentrantCallbackGroup()

        while True:
            self.get_logger().info("Waiting for camera tf...", throttle_duration_sec=1.0)
            tf = self.lookup_transform(self.base_frame, self.camera.tf_frame, sync=True, timeout=Duration(seconds=1))
            if tf is not None:
                break
        self.query_srv = self.create_service(Query3DPoints, "/query_3d_points", callback=self.handle_query_request)
        self.image_sub = self.create_subscription(
            Image,
            self.get_param_val('camera_topic_name'),
            self.handle_image_callback,
            1,
            callback_group=self.cb,
        )
        self.tracking_request_sub = self.create_subscription(
            TrackedPointRequest,
            "/point_tracking_request",
            self.handle_tracking_request,
            10,
            callback_group=self.cb_reentrant,
        )
        self.tracked_3d_pub = self.create_publisher(Tracked3DPointResponse, "/point_tracking_response", 1)
        # self.pc_pub = self.create_publisher(PointCloud2, "/point_tracking_response_pc", 1)z
        self.transition_sub = self.create_subscription(
            StateTransition, "state_transition", self.handle_state_transition, 1, callback_group=self.cb_reentrant
        )
        # Image processor action server
        self.action_cb_reentrant = ReentrantCallbackGroup()
        self._action_server = rclpy.action.ActionServer(
            self,
            ImageProcessor,
            "generate_mask",
            self.generate_mask_callback,
            callback_group=self.action_cb_reentrant,
        )
        # self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        return

    def handle_state_transition(self, msg: StateTransition):
        action = process_list_as_dict(msg.actions, "node", "action").get(self.get_name())
        if not action:
            return

        self.back_image_queue.empty()

        if action == "activate":
            pass
        elif action == "reset":
            self.reset()
        else:
            raise ValueError("Unknown action {} for node {}".format(action, self.get_name()))
        return

    def handle_query_request(self, req: Query3DPoints.Request, resp: Query3DPoints.Response):
        with self.back_image_queue:
            if len(self.back_image_queue) < 8:
                self.get_logger().info("less than 8 images in queue")
                resp.success = False
                return resp
            queue = self.back_image_queue.as_list()[::-1]

        track = req.track
        req_msg = req.request
        req_time = Time.from_msg(req_msg.image.header.stamp)
        to_proc = []
        for i in range(len(queue) - 6):
            if req_time > queue[i]["stamp"]:
                image_info = self.process_image_info(req_msg.image)
                if image_info is not None:
                    to_proc.append(image_info)
                    to_proc.extend(queue[i: i + 7]) 
                break
        else:
            self.get_logger().info("requested time too far in past")
            resp.success = False
            return resp
        if len(to_proc) < 8:
            self.get_logger().info("not enough images to process")
            resp.success = False
            return resp
        grouped_pts = {}
        for group in req_msg.groups:
            grouped_pts[group.name] = np.array([[px.x, px.y] for px in group.points])
        resp.response, _, _ = self.run_point_tracking(to_proc, grouped_pts, ref_idx=0)
        resp.success = True

        if track:
            self.get_logger().warn("tracking")  
            self.handle_tracking_request(req_msg)
        return resp

    def handle_tracking_request(self, msg: TrackedPointRequest) -> None:
        with self.current_request:
            groups = msg.groups
            if msg.action == TrackedPointRequest.ACTION_REMOVE:
                for group in groups:
                    self.current_request.pop(group.name, None)
                return

            self.current_request.clear()
            for group in groups:
                self.get_logger().info("Pips new request received {}".format(group.name))
                self.current_request[group.name] = np.array([[pt.x, pt.y] for pt in group.points])

            self.image_queue.empty()
            if msg.image.data:
                self.image_queue.append(self.process_image_info(msg.image))
        return

    def flatten_groups(self, grouped_pts):
        all_pts = []
        all_names = []
        for name, points in grouped_pts.items():
            all_pts.append(points)
            all_names.extend([name] * len(points))
        return np.concatenate(all_pts, axis=0), all_names

    def process_image_info(self, img_msg: Image):
        stamp = Time.from_msg(img_msg.header.stamp)
        pose = None
        if self.get_param_val("do_3d_point_estimation"):
            pose = self.lookup_transform(
                self.base_frame,
                self.camera.tf_frame,
                time=stamp,
                sync=True,
                as_matrix=True,
            )
        if pose is None:
            return None
        self.get_logger().debug("found!")

        info = {
            "stamp": stamp,
            "frame_id": img_msg.header.frame_id,
            "image": bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8"),
            "pose": pose,
        }
        return info

    def handle_image_callback(self, msg):
        self.get_logger().debug("handle_image_callback")
        if self.camera.tf_frame is None:
            self.get_logger().info("camera unavailable")
            return

        if self.movement_threshold is None:
            self.get_logger().fatal("movement threshold undeclared")
            return

        img_info = self.process_image_info(img_msg=msg)
        self.mask_image_queue.append(img_info)
        if img_info is None or not isinstance(img_info["pose"], np.ndarray):
            self.get_logger().debug("lookup failure")
            return
        current_pos = img_info["pose"][:3, 3]
        if (
            self.last_pos is not None
            and np.linalg.norm(current_pos - self.last_pos)
            < self.movement_threshold
        ):
            self.get_logger().debug(
                f"camera moved {np.linalg.norm(current_pos - self.last_pos)} while threshold: {self.movement_threshold}"
            )
            return
        self.get_logger().debug("adding image to queue")
        self.back_image_queue.append(img_info)

        if self.current_request:
            self.update_tracker()

        self.last_pos = current_pos
        return
    
    def generate_mask_callback(self, goal_handle):
        self.get_logger().info("generate_mask_callback")
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return ImageProcessor.Result()
            if goal_handle.is_active:
                break

        # Process the image
        img_msg = wait_for_message(self, Image, self.get_param_val('camera_topic_name'))
        img_info = self.process_image_info(img_msg)
        if img_info is None:
            goal_handle.abort()
            return ImageProcessor.Result()

        image = img_info["image"]
        mask = self.tracker.generate_mask(image)
        mask_msg = bridge.cv2_to_imgmsg(mask)
        mask_msg.header = img_msg.header
        goal_handle.succeed(mask_msg)
        return ImageProcessor

    def run_point_tracking(self, image_info, grouped_pts, ref_idx=0, z_threshold=10.0):
        images = [info["image"] for info in image_info]
        targets, groups = self.flatten_groups(grouped_pts)
        trajs = self.tracker.track_points(targets, images)
        trajs = np.transpose(trajs, (1, 0, 2))  # Point, frame, coordinate

        pts_3d = None
        if self.get_param_val("do_3d_point_estimation"):
            ref_pose = np.linalg.inv(image_info[ref_idx]["pose"]) # base to camera
            camera_frame_tf_matrices = [(ref_pose @ info["pose"]) for info in image_info] # camera_idx to base to camera
            triangulator = PointTriangulator(self.camera, min_points=self.get_param_val("min_points"))
            pts_3d = triangulator.compute_3d_points(camera_frame_tf_matrices, trajs)
            mask = np.bitwise_and(pts_3d[:, 2] > 0, pts_3d[:, 2] < z_threshold)
            reprojs = triangulator.get_reprojs(pts_3d, camera_frame_tf_matrices, trajs)
            error = np.linalg.norm(trajs - reprojs, axis=2)
            avg_error = error.mean(axis=1)
            max_error = error.max(axis=1)
            # self.get_logger().info('Average pix error:\n')
            # self.get_logger().info(', '.join('{:.3f}'.format(x) for x in avg_error))
            # self.get_logger().info('Max pix error:\n')
            # self.get_logger().info(', '.join('{:.3f}'.format(x) for x in max_error))

        trajs = np.transpose(trajs, (1, 0, 2))


        frame_id = image_info[ref_idx]["frame_id"]
        stamp = image_info[ref_idx]["stamp"].to_msg()

        response = Tracked3DPointResponse(header=Header(frame_id=frame_id, stamp=stamp))
        if pts_3d is not None:
            # pc = create_cloud_xyz32(Header(frame_id=frame_id, stamp=stamp), points=pts_3d)
            # self.pc_pub.publish(pc)
            for group, pts_and_errs in self.unflatten_tracked_points(zip(pts_3d, max_error), groups).items():
                points, errors = zip(*pts_and_errs)
                response.groups.append(
                    Tracked3DPointGroup(name=group, points=[Point(x=x, y=y, z=z) for x, y, z in points], errors=errors)
                )

        for group, points_2d in self.unflatten_tracked_points(trajs[ref_idx].astype(np.float64), groups).items():
            response.groups_2d.append(TrackedPointGroup(name=group, points=[Point2D(x=x, y=y) for x, y in points_2d]))

        response.image = bridge.cv2_to_imgmsg(image_info[ref_idx]["image"])
        response.image.header.frame_id = image_info[ref_idx]["frame_id"]
        response.image.header.stamp = image_info[ref_idx]["stamp"].to_msg()

        return response, trajs, groups

    def update_tracker(self):
        if not self.image_queue.is_full:
            return

        self.get_logger().info("Updating tracker")
        with self.current_request:
            response, trajs, groups = self.run_point_tracking(
                self.image_queue.as_list(), self.current_request, ref_idx=-1
            )
            self.tracked_3d_pub.publish(response)
            self.update_request_from_trajectory(trajs, groups)
            return response

    def unflatten_tracked_points(self, points, groups):
        rez = defaultdict(list)
        for point, group in zip(points, groups):
            rez[group].append(point)

        return rez

    def update_request_from_trajectory(self, trajs, groups):
        w = self.camera.width
        h = self.camera.height
        final_locs = trajs[-1]
        is_outside = (final_locs[:, 0] < 0) | (final_locs[:, 0] >= w) | (final_locs[:, 1] < 0) | (final_locs[:, 1] >= h)
        idx_to_stay = np.where(~is_outside)[0]
        new_req = {}

        update_locs = trajs[1]
        for idx in idx_to_stay:
            group = groups[idx]
            if group not in new_req:
                new_req[group] = []
            new_req[group].append(update_locs[idx])

        self.current_request.clear()
        for group, points in new_req.items():
            self.current_request[group] = np.array(points)
        return

    def reset(self, *_, **__):
        self.current_request.clear()
        self.image_queue.empty()
        return

    def debug_tracking(self, images, trajs, reprojs=None, output=None):
        import cv2
        from PIL import Image

        final_imgs = []
        # trajs is point x frame x dim
        for i, img in enumerate(images):
            img = img.copy()
            pts = trajs[:, i]
            for j, pt in enumerate(pts):
                img = cv2.circle(img, pt.astype(int), 4, (0, 0, 255), -1)
                if reprojs is not None and not np.any(np.isnan(reprojs[j, i])):
                    img = cv2.circle(img, reprojs[j, i].astype(int), 4, (0, 255, 255), -1)

            final_imgs.append(img)
            if output is not None:
                stamp = self.get_clock().now().seconds_nanoseconds()[0]
                Image.fromarray(img).save(os.path.join(output, f"{stamp}_{i+1}.png"))
                self.get_logger().info("output image")

        return final_imgs


class PointTriangulator:
    def __init__(self, camera, min_points=2):
        self.camera = camera
        self.min_points = min_points
        return

    @property
    def k(self):
        return self.camera.K

    def run_triangulation(self, pose_matrices, point_traj):
        """
        pose_matrices: List of N 4x4 matrices
        trajs: N x 2 array of point trajectories in typical image XY format
        """

        D = np.zeros((len(pose_matrices) * 2, 4))
        for i, (pose_mat, point) in enumerate(zip(pose_matrices, point_traj)):
            proj_mat = self.k @ np.linalg.inv(pose_mat)[:3]
            D[2 * i] = proj_mat[2] * point[0] - proj_mat[0]
            D[2 * i + 1] = proj_mat[2] * point[1] - proj_mat[1]

        _, _, v = np.linalg.svd(D, full_matrices=True)
        pts_3d = v[-1, :3] / v[-1, 3]
        return pts_3d

    def compute_3d_points(self, pose_matrices, point_trajs):
        """
        pose_matrices: List of N 4x4 matrices
        trajs: T x N x 2 array of point trajectories in typical image XY format
        """
        all_rez = []
        for traj in point_trajs:
            interior = (
                (traj[:, 0] >= 0)
                & (traj[:, 0] <= self.camera.width)
                & (traj[:, 1] >= 0)
                & (traj[:, 1] <= self.camera.height)
            )
            traj = traj[interior]
            if len(traj) < self.min_points:
                all_rez.append(np.zeros(3))
            else:
                all_rez.append(self.run_triangulation(pose_matrices, traj))

        return np.array(all_rez)

    def get_reprojs(self, points_3d, pose_matrices, point_trajs):
        """
        points_3d: K x 3 matrix of 3D points
        pose_matrices: N x 2 array of point trajectories in typical XY format
        point_traj: K x N x 2 array
        """

        rez = np.zeros(point_trajs.shape)

        for j, pose_mat in enumerate(pose_matrices):
            pose_t_base = np.linalg.inv(pose_mat)
            for i, (pt_3d, traj) in enumerate(zip(points_3d, point_trajs)):
                if not np.abs(pt_3d).sum():
                    rez[i, j] = np.nan
                    continue

                pt_3d_h = np.ones(4)
                pt_3d_h[:3] = pt_3d
                pt_3d_t = (pose_t_base @ pt_3d_h)[:3]

                reproj = self.camera.project3dToPixel(pt_3d_t)
                rez[i, j] = reproj

        return rez


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = PointTracker()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.dump_params(node.get_parameter("log_path").value)
        # do custom cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
