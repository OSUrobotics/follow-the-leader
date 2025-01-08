#!/usr/bin/env python3
import os
import pickle
from datetime import datetime
from threading import Lock

import cv2
import numpy as np
import rclpy
import rclpy.logging
from cv_bridge import CvBridge
from follow_the_leader.curve_fitting import Bezier, BezierBasedDetection
from follow_the_leader.utils import geometry_utils as geom
from follow_the_leader.utils.branch_model import BranchModel
from follow_the_leader.utils.data_processing import *
from follow_the_leader.utils.image_utils import (
    fill_holes_and_dilate,
    mask_point_selection,
)
from follow_the_leader.utils.ros_utils import TFNode, process_list_as_dict, log_entry_exit
from follow_the_leader.utils.viz_utils import gen_points_marker
from follow_the_leader_msgs.msg import (
    ControllerParams,
    ImageMaskPair,
    Point2D,
    States,
    StateTransition,
    Tracked3DPointGroup,
    Tracked3DPointResponse,
    TrackedPointGroup,
    TrackedPointRequest,
    TreeModel,
)
from follow_the_leader_msgs.srv import Query3DPoints
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.time import Time
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from skimage.measure import label
from std_msgs.msg import ColorRGBA, Empty, Header
from visualization_msgs.msg import Marker, MarkerArray

bridge = CvBridge()

class Curve3DModeler(TFNode):
    def __init__(self):
        super().__init__(
            "curve_3d_model_node", cam_info_topic="/camera/color/camera_info"
        )

        # ROS parameters
        params = {
            "base_frame": "base_link",
            "log_path": "/tmp",
            "logging": False,
            "debug_mode": True,
            "camera_topic_name": "/camera/rgb_example_topic",
            "px_ratio_rgbd": 1e-2,
            "px_ratio_particle": 3e-4,
            "px_mins": 30,
            "reconstruction_err_threshold": 4.0,  # TODO: Is this actually being used? I thought it was...
            "image_padding": 10.0,
            "mask_update_dist": 0.01,
            "curve_spacing": 30.0,
            "consistency_threshold": 0.6,
            "curve_2d_inlier_threshold": 25.0,
            "all_bg_retries": 4,
            "curve_3d_inlier_threshold": 0.03,
            "curve_3d_ransac_iters": 50,
            "mask_hole_fill": 5,
            "min_side_branch_length": 0.03,
            "min_side_branch_px_length": 20,
            "z_filter_threshold": 1.0,
            "overlap_ratio": 0.3,
        }
        self.declare_parameter_dict(**params)
        self.logging = self.get_param_val("logging")
        self.debug_mode = self.get_parameter("debug_mode")
        self.base_frame_name = self.get_param_val("base_frame")
        self.camera_topic_name = self.get_param_val("camera_topic_name")
        if self.debug_mode:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # Internal state
        self.active = False
        self.paused = False
        self.received_first_mask = False
        self.current_model = BranchModel(cam=self.camera)
        self.current_side_branches = []
        self.start_pose = None
        self.last_pose = None
        self.last_mask_msg = None  # set to None when last mask was invalid
        self.last_depth_msg = None
        self.all_bg_counter = 0

        self.identifier = None  # from run expts
        self.save_folder = None  # from run expts
        self.log_folder = None  # session specific from launch

        if self.logging:
            self.log_folder = self.get_parameter("log_path").value
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)

        self.update_info = {}
        self.lock = Lock()  # lock for last mask and depth
        self.processing_lock = Lock()  # lock for model updates

        # ROS Utils
        self.cb_mutex = MutuallyExclusiveCallbackGroup()
        self.cb_mutex_data = MutuallyExclusiveCallbackGroup()
        self.cb_reentrant = ReentrantCallbackGroup()
        while True:
            self.get_logger().info(
                "Waiting for camera tf...", throttle_duration_sec=1.0
            )
            tf = self.lookup_transform(
                self.base_frame_name,
                self.camera.tf_frame,
                sync=True,
                timeout=Duration(seconds=1),
            )
            if tf is not None:
                break

        self.state_announce_pub = self.create_publisher(States, "state_announcement", 1)
        self.tree_model_pub = self.create_publisher(TreeModel, "/tree_model", 1)
        self.rviz_model_pub = self.create_publisher(
            MarkerArray, "/curve_3d_rviz_array", 1
        )
        if self.debug_mode:
            self.img_sub = self.create_subscription(
            Image,
            self.camera_topic_name,
            self.image_model_reproject,
            1,
            callback_group=self.cb_reentrant,
            )
            self.depth_debug_pcd = self.create_publisher(PointCloud2, "/depth_debug_pcd", 1)
            self.particle_debug_pcd = self.create_publisher(
                PointCloud2, "/particle_debug_pcd", 1
            )
            self.diag_image_pub = self.create_publisher(Image, "model_diagnostic", 1)
        self.img_mask_sub = self.create_subscription(
            ImageMaskPair,
            "/image_mask_pair",
            self.process_mask,
            1,
            callback_group=self.cb_mutex_data,
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth_registered/image_rect",
            self.process_depth,
            20,
            callback_group=self.cb_mutex_data,
        )

        self.reset_sub = self.create_subscription(
            Empty, "/reset_model", self.reset, 1, callback_group=self.cb_reentrant
        )
        self.params_sub = self.create_subscription(
            ControllerParams,
            "/controller_params",
            self.handle_params_update,
            1,
            callback_group=self.cb_reentrant,
        )
        self.transition_sub = self.create_subscription(
            StateTransition,
            "state_transition",
            self.handle_state_transition,
            1,
            callback_group=self.cb_reentrant,
        )
        self.point_query_client = self.create_client(
            Query3DPoints, "/query_3d_points", callback_group=self.cb_reentrant
        )
        while not self.point_query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "point_query_client service not available, waiting..."
            )

        self.create_timer(0.01, self.update_model_at_rate, callback_group=self.cb_mutex)
        return

    def handle_state_transition(self, msg: StateTransition):
        action = process_list_as_dict(msg.actions, "node", "action").get(
            self.get_name()
        )
        if not action:
            return

        if action == "activate":
            self.start_modeling()
        elif action == "reset":
            self.stop_modeling()
        elif action == "pause":
            self.pause()
        elif action == "resume":
            self.resume()

        else:
            raise ValueError(
                "Unknown action {} for node {}".format(action, self.get_name())
            )

        return

    def handle_params_update(self, msg: ControllerParams):
        self.save_folder = msg.save_folder
        self.identifier = msg.identifier
        return

    def reset(self, *_, **__):
        with self.processing_lock and self.lock:
            self.active = False
            self.paused = False
            self.current_model = BranchModel(cam=self.camera)
            self.current_side_branches = []
            self.last_pose = None
            self.last_mask_msg = None
            self.all_bg_counter = 0
            self.update_info = {}
            self.get_logger().info("Model reset!", throttle_duration_sec=1.0)
        return

    def start_modeling(self, *_, **__):
        self.get_logger().info(
            "start_modeling with camera: " + self.camera_topic_name, once=True
        )
        self.reset()
        if self.camera.tf_frame is None:
            self.get_logger().info(
                "Camera TF frame is not set! Cannot start modeling",
                throttle_duration_sec=1.0,
            )
            self.active = False
            return
        self.last_pose = self.get_camera_frame_pose(
            position_only=False, timeout=Duration(seconds=5.0)
        )  # wait for initial pose
        self.start_pose = self.last_pose
        self.active = True
        return

    def stop_modeling(self, *_, **__):
        self.active = False
        # self.process_final_model()
        return

    def pause(self):
        self.paused = True
        return

    def resume(self):
        self.paused = False
        return

    def process_final_model(self):
        if self.logging or (self.save_folder and self.identifier):
            time = self.update_info.get("stamp", None)
            time = self.get_clock().now().to_msg()
            if self.save_folder and self.identifier:
                file = os.path.join(
                    self.save_folder, f"{self.identifier}_results.pickle"
                )
                self.get_logger().info("Saving constructed model to {}".format(file))
            else:
                file = os.path.join(
                    self.log_folder, f"model_{time.sec}_{time.nanosec}.pickle"
                )
                self.get_logger().info(
                    "Saving constructed model to {}".format(file),
                    throttle_duration_sec=5.0,
                )
            data = {
                "leader": self.current_model.retrieve_points(
                    inv_tf=np.identity(4), filter_none=True
                ),
                "side_branches": [
                    sb.retrieve_points(inv_tf=np.identity(4), filter_none=True)
                    for sb in self.current_side_branches
                ],
                "leader_raw": self.current_model,
                "side_branches_raw": self.current_side_branches,
                "start_pose": self.start_pose,
            }

            with open(file, "wb") as fh:
                pickle.dump(data, fh)

            self.identifier = None
            self.save_folder = None

    def process_mask(self, msg: ImageMaskPair):
        # Hack to guard against bad optical flow masks when initially moving
        if not self.received_first_mask:
            self.received_first_mask = True

            if not self.has_parameter("num_particle_points") or not self.has_parameter(
                "num_rgbd_points"
            ):
                total_pixels = msg.mask.height * msg.mask.width
                max_rgbd = max(
                    int(self.get_param_val("px_ratio_rgbd") * total_pixels),
                    self.get_param_val("px_mins"),
                )
                max_particles = int(
                    self.get_param_val("px_ratio_particle") * total_pixels
                )
                self.declare_parameter_dict(
                    **{
                        "num_particle_points": max_particles,
                        "num_rgbd_points": max_rgbd,
                    }
                )
                self.get_logger().debug(
                    f"from {total_pixels} {max_particles} particles {max_rgbd} rgbd pxs"
                )
            return

        with self.lock:
            self.last_mask_msg = msg

    def process_depth(self, msg: Image):
        with self.lock:
            if self.last_depth_msg is None:
                self.last_depth_msg = msg
            if self.last_mask_msg is not None:
                age = Time.from_msg(
                    self.last_mask_msg.mask.header.stamp
                ) - Time.from_msg(self.last_depth_msg.header.stamp)
                if age > Duration(nanoseconds=1):  # get depth only if outdated
                    self.last_depth_msg = msg

    # @log_entry_exit
    def populate_update_info(self) -> bool:
        with self.lock:
            if self.last_mask_msg is None or self.last_depth_msg is None:
                return False

            rgb_msg = self.last_mask_msg.rgb
            depth_msg = self.last_depth_msg
            stamp = self.last_mask_msg.mask.header.stamp

        self.update_info["stamp"] = stamp
        mask_raw = bridge.imgmsg_to_cv2(
            self.last_mask_msg.mask, desired_encoding="mono8"
        )
        self.update_info["mask_raw"] = mask_raw
        self.update_info["mask"] = fill_holes_and_dilate(
                mask_raw, fill_size=self.get_param_val("mask_hole_fill")
            ) > 128
        
        self.update_info["rgb"] = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        self.update_info["rgb_msg"] = rgb_msg
        self.update_info["tf"] = self.get_camera_frame_pose(time=stamp)
        self.update_info["inv_tf"] = np.linalg.inv(self.update_info["tf"])
        self.update_info["depth_msg"] = depth_msg
        self.update_info["depth"] = bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )
        self.current_model.set_inv_tf(self.update_info["inv_tf"])
        for side_branch in self.current_side_branches:
            side_branch.set_inv_tf(self.update_info["inv_tf"])
        return True

    def update_model_at_rate(self):
        # TODO: should attempt premption?
        if self.paused:
            self.get_logger().debug("paused", throttle_duration_sec=1.0)
            return
        if not self.active:
            self.get_logger().debug("not active", throttle_duration_sec=1.0)
            return

        with self.processing_lock:
            self.update_info = {}
            if not self.populate_update_info():  # if last mask was None
                return False

            steps = [
                self.get_primary_movement_direction,
                self.get_mask_3d,
                # self.run_mask_curve_detection,
                # self.update_side_branches,
                # self.reconcile_2d_3d_curves,
                # self.process_side_branches,
                self.publish_curve,
            ]

            success = False
            for step in steps:
                success = step()
                if not success:
                    break

            # self.process_final_model()
            self.publish_diagnostic_image()

        if self.active and self.update_info.get("reinitialize"):
            self.get_logger().info("reinit")
            self.reset()
            self.active = True
        elif self.active and self.update_info.get("terminate"):
            self.get_logger().info("terminate")
            self.active = False
            self.state_announce_pub.publish(States(state=States.IDLE))

        return success

    def convert_tracking_response(self, msg: Tracked3DPointResponse):
        info = autodict
        for group in msg.groups:
            name = group.name
            info[name]["pts"] = np.array(
                [np.array((p.x, p.y, p.z)) for p in group.points]
            )
            info[name]["error"] = np.array(group.errors)

        for group in msg.groups_2d:
            name = group.name
            info[name]["pts_2d"] = np.array([(p.x, p.y) for p in group.points])

        return info

    @log_entry_exit
    def query_particle_point_estimates(
        self, name_px_dict, img_msg, tf_to_base, track=False
    ):
        """query point estimates from the point tracker

        :param name_px_dict: {"group": pxs}
        :param img_msg: image message pixels are in
        :param track: not implemented, defaults to False
        :param tf_to_base: transform from camera to base
        :return: tracking response
        """
        req = Query3DPoints.Request()
        req.track = track
        req.request.image = img_msg
        req.request.z_filter_max = self.get_param_val("z_filter_threshold")
        start = datetime.now()
        for name, pxs in name_px_dict.items():
            pts = [Point2D(x=p[0], y=p[1]) for p in pxs.astype(float)]
            group = TrackedPointGroup(name=name, points=pts)
            req.request.groups.append(group)

        resp = self.point_query_client.call(req)
        if resp is None:
            self.get_logger().warn("point tracking query return None")
            return None
        if not resp.success:
            self.get_logger().warn("point tracking query returned failure")
            return None
        end = datetime.now()
        self.get_logger().debug(
            f"point tracking query for {len(pxs)} in {(end - start).total_seconds()}s"
        )
        return self.convert_tracking_response(resp.response)

        # TODO: async

    def query_depth_image_estimates(self, name_px_dict, img_msg, tf_to_base):
        info = autodict
        # TODO: depends on camera model, implement as composable service in image_proc
        depth_mat = np.asarray(
            bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        )
        for name, pxs in name_px_dict.items():
            pxs = pxs.astype(int)
            pts = self.camera.pixelTo3D(pxs, depth_image=depth_mat)
            # pts_in_base = self.mul_homog(tf_to_base, pts)
            info[name]["pts"] = pts
            info[name]["pts_2d"] = pxs
            # using realsense conditioning on error
            # https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
            info[name]["error"] = np.where(
                pts[:, -1] > 0.2, 0.2, np.multiply(pts[:, -1], pts[:, -1]) * 0.2
            )
        return info

    @log_entry_exit
    def get_mask_3d(self, imgdepthmask=None, tf_to_base=None):
        if imgdepthmask is None:
            img_msg = self.update_info["rgb_msg"]
            mask = self.update_info["mask"]
            depth_msg = self.update_info["depth_msg"]
            tf_to_base = self.update_info["tf"]
        else:
            img_msg, depth_msg, mask = imgdepthmask
            tf_to_base = tf_to_base
        pxs_rgbd = mask_point_selection(
            mask,
            strategy="none",
            num_points=self.get_param_val("num_rgbd_points"),
        )
        rgbd_depths = self.query_depth_image_estimates(
            {"rgbd": pxs_rgbd}, depth_msg, tf_to_base
        )

        particle_mask = subtract_missing_depth(
            pxs_rgbd,
            rgbd_depths["rgbd"],
            mask.shape,
            self.get_param_val("mask_hole_fill"),
        )
        if (
            self.get_param_val("num_particle_points") / np.count_nonzero(particle_mask)
            < 0.8
        ):
            self.get_logger().warn(
                f"Too many pxs missing depth data: {np.count_nonzero(particle_mask)}, downsampling more than 20%"
            )
        pxs_downsample = mask_point_selection(
            particle_mask,
            strategy="uniform_sample",
            num_points=self.get_param_val("num_particle_points"),
        )
        particle_depth_flag = False
        idx = filter_depth_dict_by_z(
            rgbd_depths["rgbd"],
            z_min=1e-3,
            z_max=self.get_param_val("z_filter_threshold"),
        )
        try:
            particle_depths = self.query_particle_point_estimates(
                {"particle": pxs_downsample}, img_msg, tf_to_base, False
            )
            if particle_depths is not None:
                filter_depth_dict_by_z(
                    particle_depths["particle"],
                    z_min=1e-3,
                    z_max=self.get_param_val("z_filter_threshold"),
                )
                particle_depth_flag = True
        except Exception as e:
            self.get_logger().error(f"Error querying particle points: {e}")

        if self.logging and self.debug_mode:
            self.depth_debug_pcd.publish(
                create_cloud_xyz32(
                    Header(frame_id=self.camera.tf_frame, stamp=img_msg.header.stamp),
                    points=rgbd_depths["rgbd"]["pts"],
                )
            )
            if particle_depth_flag:
                self.particle_debug_pcd.publish(
                    create_cloud_xyz32(
                        Header(
                            frame_id=self.camera.tf_frame, stamp=img_msg.header.stamp
                        ),
                        points=particle_depths["particle"]["pts"],
                    )
                )
            frame_data = {}
            frame_data["depth_dicts"] = serialise_autodict(rgbd_depths)
            if particle_depth_flag:
                frame_data["depth_dicts"].update(serialise_autodict(
                    particle_depths
                ))
            frame_data.update(self.update_info)
            file = os.path.join(
                self.log_folder,
                f"{Time.from_msg(img_msg.header.stamp).nanoseconds}_depths.pickle",
            )
            with open(file, "wb") as fh:
                self.get_logger().debug(f"Saving depths to {file}")
                pickle.dump(frame_data, fh)

    def get_primary_movement_direction(self) -> bool:
        self.get_logger().debug("get_primary_movement_direction")
        if not self.current_model:
            vec_msg = self.last_mask_msg.image_frame_offset
            move_vec = np.array([vec_msg.x, vec_msg.y])
        else:
            # Determine the primary direction of movement based on the existing model
            all_pts = self.current_model.retrieve_points(filter_none=False)
            self.update_info["all_pts"] = all_pts

            valid_idxs = [i for i, pt in enumerate(all_pts) if pt is not None]
            if len(valid_idxs) < 2:
                self.get_logger().info(
                    "Not enough valid pixels in the model, reinitializing..."
                )
                self.update_info["reinitialize"] = True
                return False

            first_pt = all_pts[min(valid_idxs)]
            last_pt = all_pts[max(valid_idxs)]
            first_px = np.array(self.camera.project3dToPixel(first_pt))
            last_px = np.array(self.camera.project3dToPixel(last_pt))

            if (
                np.linalg.norm(first_pt - last_pt) < 0.025
                or np.linalg.norm(first_px - last_px) < 30
            ):
                self.get_logger().info("Model looks too squished in! Reinitializing")
                self.update_info["reinitialize"] = True
                return False

            move_vec = last_px - first_px
            move_vec = move_vec / np.linalg.norm(move_vec)
        self.update_info["move_vec"] = move_vec

        return True

    def publish_curve(self) -> bool:
        self.get_logger().debug("publish_curve")
        if not self.current_model:
            return True

        time = self.update_info.get("stamp", None)
        if time is None:
            time = self.get_clock().now().to_msg()

        msg = TreeModel()
        msg.header.frame_id = self.camera.tf_frame
        msg.header.stamp = time

        main_points = [
            Point(x=p[0], y=p[1], z=p[2])
            for p in self.current_model.retrieve_points(filter_none=True)
        ]
        msg.points.extend(main_points)
        msg.ids.extend([0] * len(main_points))

        for i, branch in enumerate(self.current_side_branches, start=1):
            points = [
                Point(x=p[0], y=p[1], z=p[2])
                for p in branch.retrieve_points(filter_none=True)
            ]
            msg.points.extend(points)
            msg.ids.extend([i] * len(points))
        self.tree_model_pub.publish(msg)

        # Publish the 3D model of the main branches and the side branches
        markers = MarkerArray()

        marker = Marker()
        marker.ns = self.get_name()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)

        marker = Marker()
        marker.header.frame_id = self.camera.tf_frame
        marker.header.stamp = time
        marker.ns = self.get_name()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.points = main_points
        marker.scale.x = 0.02
        marker.color = ColorRGBA(r=0.5, g=1.0, b=0.5, a=1.0)
        markers.markers.append(marker)

        for i, side_branch in enumerate(self.current_side_branches, start=1):
            pts = side_branch.retrieve_points(filter_none=True)

            marker = Marker()
            marker.header.frame_id = self.camera.tf_frame
            marker.header.stamp = time
            marker.ns = self.get_name()
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.scale.x = 0.02
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in pts]
            markers.markers.append(marker)

        self.rviz_model_pub.publish(markers)

        return True

    def model_updater(self):
        self.get_logger().info("starting updates", once=True)
        if self.paused:
            self.get_logger().info("paused", throttle_duration_sec=1.0)
            return
        if not self.active:
            self.get_logger().info("not active", throttle_duration_sec=1.0)
            self.start_modeling()
            return

        pose = self.get_camera_frame_pose(position_only=False)
        self.get_logger().debug("Got first camera pose", once=True)
        if self.last_pose is None:
            self.last_pose = pose

        if np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3]) > self.get_param_val(
            "mask_update_dist"
        ):
            # Ignore if rotation is too much
            rotation = Rotation.from_matrix(
                self.last_pose[:3, :3].T @ pose[:3, :3]
            ).as_euler("XYZ")
            if np.linalg.norm(rotation) > np.radians(0.5):
                self.last_pose = pose
                return
            if self.update_tracking_request():
                self.last_pose = pose
        else:
            self.get_logger().info("Not enough movement", throttle_duration_sec=1.0)

    def is_in_padding_region(self, px):
        pad = self.get_param_val("image_padding")
        w = self.camera.width
        h = self.camera.height

        return px[0] < pad or px[0] > (w - pad) or px[1] < pad or px[1] > (h - pad)

    def publish_diagnostic_image(self):
        if self.update_info.get("mask") is None:
            return

        mask_img = np.dstack([self.update_info["mask"] * 255] * 3)
        # submask_img = np.zeros(mask_img.shape)
        # submask = self.update_info.get("submask", None)
        # if submask is not None:
        #     submask_img[submask] = [0, 255, 0]

        # leader_est = self.update_info.get("leader_mask_estimate", None)
        # if leader_est is not None:
        #     leader_est_img = np.zeros(mask_img.shape)
        #     leader_est_img[leader_est] = [255, 0, 255]
        #     submask_img = 0.5 * submask_img + 0.5 * leader_est_img

        diag_img = 0.3 * self.update_info["rgb"] + 0.35 * mask_img
        #
        # if self.current_model:
        #     reconstructed_mask = self.current_model.branch_mask
        #     for sb in self.current_side_branches:
        #         reconstructed_mask[sb.branch_mask] = True

        #     alpha = reconstructed_mask * 0.5
        #     alpha = np.dstack([alpha] * 3)
        #     zeros = np.zeros_like(reconstructed_mask)
        #     overlay = np.dstack([zeros, zeros, reconstructed_mask * 255])
        #     diag_img = overlay * alpha + diag_img * (1 - alpha)

        # pxs = self.camera.project3dToPixel(self.current_model.retrieve_points(filter_none=True)).astype(int)
        # cv2.polylines(diag_img, [pxs.reshape((-1, 1, 2))], False, (255, 0, 0), 5)
        # for px in pxs:
        #     diag_img = cv2.circle(diag_img, px, 7, (0, 0, 255), -1)

        # curve = self.update_info.get("curve", None)
        # if curve is not None:
        #     eval_pts = curve(np.linspace(0, 1, 200)).astype(int)
        #     cv2.polylines(diag_img, [eval_pts.reshape((-1, 1, 2))], False, (0, 0, 200), 3)

        # for sb_info in self.update_info.get("side_branches", []):
        #     curve = sb_info["curve"]
        #     pxs = curve(np.linspace(0, 1, 20)).astype(int)
        #     cv2.polylines(diag_img, [pxs.reshape((-1, 1, 2))], False, (200, 0, 0), 3)

        # detection = self.update_info.get("detection", None)
        # if detection is not None:
        #     diag_img[detection.skel] = [255, 255, 0]

        img_msg = bridge.cv2_to_imgmsg(diag_img.astype(np.uint8), encoding="rgb8")
        self.diag_image_pub.publish(img_msg)

    def image_model_reproject(self, msg: Image):
        if self.active or not self.current_model:
            return
        header = msg.header
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8") // 2
        cam_frame = self.get_camera_frame_pose(header.stamp, position_only=False)
        inv_tf = np.linalg.inv(cam_frame)

        draw_px = self.camera.project3dToPixel(
            self.current_model.retrieve_points(inv_tf, filter_none=True)
        ).astype(int)
        if not draw_px.size:
            return

        cv2.polylines(img, [draw_px.reshape((-1, 1, 2))], False, (0, 0, 255), 3)

        for side_branch in self.current_side_branches:
            pxs = self.camera.project3dToPixel(
                side_branch.retrieve_points(filter_none=True)
            ).astype(int)
            if not len(pxs):
                continue
            cv2.polylines(img, [pxs.reshape((-1, 1, 2))], False, (0, 255, 255), 3)

        new_img_msg = bridge.cv2_to_imgmsg(
            img.astype(np.uint8), encoding="rgb8", header=header
        )
        self.diag_image_pub.publish(new_img_msg)

    def get_camera_frame_pose(
        self, time=None, position_only=False, timeout=rclpy.time.Duration(seconds=0.1)
    ):
        """Get the pose of the camera frame in the base frame
        :param time: time to get the transform at, None for latest
        :param position_only: return only the 3x1 position matrix
        :param timeout: timeout for the transform lookup
        :return: 4x4 transform matrix
        """
        tf_mat = self.lookup_transform(
            self.base_frame_name,
            self.camera.tf_frame,
            time,
            as_matrix=True,
            sync=True,
            timeout=timeout,
        )
        if tf_mat is None:
            raise ValueError("Failed to get camera frame pose")
        if position_only:
            return tf_mat[:3, 3]
        return tf_mat

    def px_in_img(self, px):
        return (0 <= px[0] < self.camera.width) and (0 <= px[1] < self.camera.height)

    def filter_px_to_img(self, px, convert_int=True):
        if convert_int:
            px = px.astype(int)
        return px[
            (px[:, 0] >= 0)
            & (px[:, 0] < self.camera.width)
            & (px[:, 1] >= 0)
            & (px[:, 1] < self.camera.height)
        ]


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = Curve3DModeler()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.dump_params(node.get_param_val("log_path"))
        # do custom cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
