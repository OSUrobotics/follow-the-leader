#!/usr/bin/env python3
import os
import rclpy
import numpy as np
from std_msgs.msg import Header, Empty, ColorRGBA
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from skimage.measure import label
from follow_the_leader_msgs.msg import (
    Point2D,
    TreeModel,
    ImageMaskPair,
    States,
    TrackedPointRequest,
    TrackedPointGroup,
    Tracked3DPointGroup,
    Tracked3DPointResponse,
    StateTransition,
    ControllerParams,
)
from follow_the_leader_msgs.srv import Query3DPoints
from collections import defaultdict
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.parameter import Parameter
from follow_the_leader.curve_fitting import BezierBasedDetection, Bezier
from follow_the_leader.utils.ros_utils import TFNode, process_list_as_dict
from follow_the_leader.utils import geometry_utils as geom
from follow_the_leader.utils.branch_model import BranchModel
from threading import Lock
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import pickle
import cv2

bridge = CvBridge()


class Curve3DModeler(TFNode):
    def __init__(self):
        super().__init__("curve_3d_model_node", cam_info_topic="/camera/color/camera_info")

        # ROS parameters
        params = {
            "base_frame": "base_link",
            "reconstruction_err_threshold": 4.0,  # TODO: Is this actually being used? I thought it was...
            "image_padding": 10.0,
            "mask_update_dist": 0.01,
            "curve_spacing": 30.0,
            "consistency_threshold": 0.6,
            "curve_2d_inlier_threshold": 25.0,
            "all_bg_retries": 4,
            "curve_3d_inlier_threshold": 0.03,
            "curve_3d_ransac_iters": 50,
            "mask_hole_fill": 300,
            "min_side_branch_length": 0.03,
            "min_side_branch_px_length": 20,
            "z_filter_threshold": 1.0,
        }
        self.declare_parameter_dict(**params)
        self.camera_topic_name = self.declare_parameter("camera_topic_name", Parameter.Type.STRING)
        self.tracking_name = "model"

        # State variables
        self.active = False
        self.paused = False
        self.received_first_mask = False
        self.current_model = BranchModel(cam=self.camera)
        self.current_side_branches = []
        self.start_pose = None
        self.last_pose = None
        self.last_mask_info = None
        self.all_bg_counter = 0

        self.identifier = None
        self.save_folder = None

        self.update_info = {}

        # ROS Utils
        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.cb_reentrant = ReentrantCallbackGroup()
        self.state_announce_pub = self.create_publisher(States, "state_announcement", 1)
        self.tree_model_pub = self.create_publisher(TreeModel, "/tree_model", 1)
        self.rviz_model_pub = self.create_publisher(MarkerArray, "/curve_3d_rviz_array", 1)
        self.diag_image_pub = self.create_publisher(Image, "model_diagnostic", 1)
        self.img_mask_sub = self.create_subscription(ImageMaskPair, "/image_mask_pair", self.process_mask, 1)
        self.img_sub = self.create_subscription(
            Image,
            self.camera_topic_name.get_parameter_value().string_value,
            self.image_model_reproject,
            1,
            callback_group=self.cb_reentrant,
        )
        self.reset_sub = self.create_subscription(
            Empty, "/reset_model", self.reset, 1, callback_group=self.cb_reentrant
        )
        self.params_sub = self.create_subscription(
            ControllerParams, "/controller_params", self.handle_params_update, 1, callback_group=self.cb_reentrant
        )
        self.transition_sub = self.create_subscription(
            StateTransition, "state_transition", self.handle_state_transition, 1, callback_group=self.cb_reentrant
        )
        self.point_query_client = self.create_client(Query3DPoints, "/query_3d_points")
        self.lock = Lock()
        self.processing_lock = Lock()
        self.create_timer(0.01, self.update, callback_group=self.cb_group)
        return

    def handle_state_transition(self, msg: StateTransition):
        action = process_list_as_dict(msg.actions, "node", "action").get(self.get_name())
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
            raise ValueError("Unknown action {} for node {}".format(action, self.get_name()))
        
        return

    def handle_params_update(self, msg: ControllerParams):
        self.save_folder = msg.save_folder
        self.identifier = msg.identifier
        return

    def reset(self, *_, **__):
        with self.processing_lock:
            self.active = False
            self.paused = False
            self.received_first_mask = False
            self.current_model = BranchModel(cam=self.camera)
            self.current_side_branches = []
            self.last_pose = None
            self.last_mask_info = None
            self.all_bg_counter = 0
            self.update_info = {}
            print("Model reset!")
        return

    def start_modeling(self, *_, **__):
        self.reset()
        self.last_pose = self.get_camera_frame_pose(position_only=False)
        self.start_pose = self.last_pose
        self.active = True
        return

    def stop_modeling(self, *_, **__):
        self.active = False
        self.process_final_model()
        return

    def pause(self):
        self.paused = True
        return

    def resume(self):
        self.paused = False
        return

    def process_final_model(self):
        if self.identifier and self.save_folder:
            file = os.path.join(self.save_folder, f"{self.identifier}_results.pickle")
            data = {
                "leader": self.current_model.retrieve_points(inv_tf=np.identity(4), filter_none=True),
                "side_branches": [
                    sb.retrieve_points(inv_tf=np.identity(4), filter_none=True) for sb in self.current_side_branches
                ],
                "leader_raw": self.current_model,
                "side_branches_raw": self.current_side_branches,
                "start_pose": self.start_pose,
            }

            with open(file, "wb") as fh:
                pickle.dump(data, fh)

            print("Saved constructed model to {}".format(file))

            self.identifier = None
            self.save_folder = None

    def process_mask(self, msg: ImageMaskPair):
        # Hack to guard against bad optical flow masks when initially moving
        if not self.received_first_mask:
            self.received_first_mask = True
            return

        with self.lock:
            self.last_mask_info = msg

    def query_point_estimates(self, name_px_dict, img_msg, track=False):
        req = Query3DPoints.Request()
        req.track = track
        req.request.image = img_msg

        for name, pxs in name_px_dict.items():
            pts = [Point2D(x=p[0], y=p[1]) for p in pxs]
            group = TrackedPointGroup(name=name, points=pts)
            req.request.groups.append(group)

        resp = self.point_query_client.call(req)
        if not resp.success:
            return None
        return self.convert_tracking_response(resp.response)

    def update_tracking_request(self) -> bool:
        with self.processing_lock:
            self.update_info = {}
            if not self.process_last_mask_info():
                return False

            steps = [
                self.get_primary_movement_direction,
                self.run_mask_curve_detection,
                self.update_side_branches,
                self.reconcile_2d_3d_curves,
                self.process_side_branches,
                self.publish_curve,
            ]

            success = False
            for step in steps:
                success = step()
                if not success:
                    break

            self.publish_diagnostic_image()

        if self.active and self.update_info.get("reinitialize"):
            self.reset()
            self.active = True
        elif self.active and self.update_info.get("terminate"):
            self.active = False
            self.state_announce_pub.publish(States(state=States.IDLE))

        return success

    def process_last_mask_info(self) -> bool:
        with self.lock:
            if self.last_mask_info is None:
                return False

            rgb_msg = self.last_mask_info.rgb
            mask = bridge.imgmsg_to_cv2(self.last_mask_info.mask, desired_encoding="mono8") > 128
            stamp = self.last_mask_info.mask.header.stamp

        self.update_info["stamp"] = stamp
        self.update_info["mask"] = mask
        self.update_info["rgb"] = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        self.update_info["rgb_msg"] = rgb_msg
        self.update_info["tf"] = self.get_camera_frame_pose(time=stamp)
        self.update_info["inv_tf"] = np.linalg.inv(self.update_info["tf"])
        self.current_model.set_inv_tf(self.update_info["inv_tf"])
        for side_branch in self.current_side_branches:
            side_branch.set_inv_tf(self.update_info["inv_tf"])
        return True

    def get_primary_movement_direction(self) -> bool:
        if not self.current_model:
            vec_msg = self.last_mask_info.image_frame_offset
            move_vec = np.array([vec_msg.x, vec_msg.y])
        else:
            # Determine the primary direction of movement based on the existing model
            all_pts = self.current_model.retrieve_points(filter_none=False)
            self.update_info["all_pts"] = all_pts

            valid_idxs = [i for i, pt in enumerate(all_pts) if pt is not None]
            if len(valid_idxs) < 2:
                print("Not enough valid pixels in the model, reinitializing...")
                self.update_info["reinitialize"] = True
                return False

            first_pt = all_pts[min(valid_idxs)]
            last_pt = all_pts[max(valid_idxs)]
            first_px = np.array(self.camera.project3dToPixel(first_pt))
            last_px = np.array(self.camera.project3dToPixel(last_pt))

            if np.linalg.norm(first_pt - last_pt) < 0.025 or np.linalg.norm(first_px - last_px) < 30:
                print("Model looks too squished in! Reinitializing")
                self.update_info["reinitialize"] = True
                return False

            move_vec = last_px - first_px
            move_vec = move_vec / np.linalg.norm(move_vec)
        self.update_info["move_vec"] = move_vec

        return True

    def update_side_branches(self) -> bool:
        """
        Updates the existing side branches' trust levels depending on if they are in the mask
        """

        to_delete = []

        for i, sb in enumerate(self.current_side_branches):
            for idx, pt in enumerate(sb.retrieve_points(filter_none=False)):
                if pt is None:
                    continue
                px = self.camera.project3dToPixel(pt).astype(int)
                if self.px_in_img(px):
                    if self.update_info["mask"][px[1], px[0]]:
                        sb.update_trust(idx, 1)
                    else:
                        sb.update_trust(idx, -1)

            avg_trust = sb.get_average_trust()
            if avg_trust is not None and avg_trust < -1:  # TODO: Hardcoded
                print("Side branch trust was too low! Deleting")
                to_delete.append(i)

        for i in to_delete[::-1]:
            del self.current_side_branches[i]

        return True

    def run_mask_curve_detection(self) -> bool:
        """
        Fits a curve to the mask. Utilizes the currently existing model by projecting the model into the image and
        taking the submask component with the most matches.
        """

        submask = self.update_info["mask"]
        if self.current_model:
            # Determine parts of the original 3D model that are still in frame
            in_frame_idxs = []
            in_frame_pxs = []
            for i in range(len(self.current_model)):
                idx = len(self.current_model) - i - 1
                pt = self.current_model.point(idx)
                if pt is None:
                    continue
                px = self.camera.project3dToPixel(pt)
                if 0 <= int(px[0]) < self.camera.width and 0 <= int(px[1]) < self.camera.height:
                    in_frame_idxs.append(idx)
                    in_frame_pxs.append(px)
                else:
                    break

            if not in_frame_pxs:
                print("All pxs were outside the image!")
                self.last_mask_info = None
                return False

            in_frame_pxs = np.array(in_frame_pxs)
            in_frame_idxs = np.array(in_frame_idxs)

            # Split the mask into connected subcomponents and identify which one has the most matches
            pxs_int = in_frame_pxs.astype(int)
            labels = label(self.update_info["mask"])
            label_list, counts = np.unique(labels[pxs_int[:, 1], pxs_int[:, 0]], return_counts=True)
            most_freq_label = label_list[np.argmax(counts)]

            if most_freq_label == 0:
                print("Most points were projected into the BG! Not processing")
                self.all_bg_counter += 1
                if self.all_bg_counter >= self.get_param_val("all_bg_retries"):
                    print("It looks like the model is lost! Resetting the model...")
                    self.current_model.chop_at(min(in_frame_idxs) - 1)
                    in_frame_pxs = []
                    in_frame_idxs = []
                    # self.update_info['terminate'] = True
                else:
                    self.last_mask_info = None
                    return False

            else:
                submask = labels == most_freq_label

            self.update_info["valid_pxs"] = in_frame_pxs
            self.update_info["valid_idxs"] = in_frame_idxs

        self.all_bg_counter = 0
        self.update_info["submask"] = submask

        # Fill in small holes in the mask to avoid skeletonization issues
        fill_holes(submask, fill_size=self.get_param_val("mask_hole_fill"))

        # Use the chosen submask to run the Bezier curve fit
        detection = BezierBasedDetection(submask, use_medial_axis=True, use_vec_weighted_metric=True)
        curve = detection.fit(vec=self.update_info["move_vec"], trim=int(self.get_param_val("image_padding")))
        self.update_info["detection"] = detection
        if curve is None:
            print("No good curve was found!")
            return False

        # Compute a masked image of the leader using the fit curve and the estimated radii
        radius_interpolator = detection.get_radius_interpolator_on_path()
        self.update_info["radius_interpolator"] = radius_interpolator

        ts = np.linspace(0, 1, 11)
        eval_pxs = curve.eval_by_arclen(ts, normalized=True)[0]
        px_radii = radius_interpolator(ts)
        leader_mask_estimate = BranchModel.render_mask(self.camera.width, self.camera.height, eval_pxs, px_radii)

        # Find 3D side branches
        side_branch_info = detection.run_side_branch_search(
            min_len=self.get_param_val("min_side_branch_px_length"), filter_mask=leader_mask_estimate
        )

        self.update_info["curve"] = curve
        self.update_info["side_branches"] = side_branch_info
        self.update_info["leader_mask_estimate"] = leader_mask_estimate

        return True

    def reconcile_2d_3d_curves(self) -> bool:
        # Takes the fit 2D curve, obtains 3D estimates, and construct a 3D curve. Then check to see which ones agree

        curve = self.update_info["curve"]
        pixel_spacing = self.get_param_val("curve_spacing")

        if self.current_model and len(self.update_info.get("valid_idxs", [])):
            pxs = self.update_info["valid_pxs"]
            idxs = self.update_info["valid_idxs"]

            px_thres = self.get_param_val("curve_2d_inlier_threshold")
            dists, ts = curve.query_pt_distance(pxs)
            px_consistent_idx = dists < px_thres

            if px_consistent_idx.sum() < 2 or px_consistent_idx.mean() < self.get_param_val("consistency_threshold"):
                print("The current 3D model does not seem to be consistent with the extracted 2D model. Skipping")
                self.last_mask_info = None
                return False

            self.current_model.clear(idxs[~px_consistent_idx])

            # Chop off the model beyond any inconsistent pixels, and reinterpolate any inconsistent pixels in between
            consistent_idx = idxs[px_consistent_idx]
            min_consistent_idx = consistent_idx.min()
            max_consistent_idx = consistent_idx.max()
            self.current_model.chop_at(max_consistent_idx)

            current_model_idxs = np.arange(min_consistent_idx, max_consistent_idx + 1)
            consistent_ds = curve.t_to_curve_dist(ts[px_consistent_idx])
            current_ds = interp1d(consistent_idx, consistent_ds)(current_model_idxs)
            current_pxs, _ = curve.eval_by_arclen(current_ds)

            start_d = current_ds[-1] + pixel_spacing

        else:
            consistent_idx = []
            current_pxs = np.zeros((0, 2))
            current_model_idxs = []
            max_consistent_idx = len(self.current_model) - 1
            current_ds = []
            start_d = pixel_spacing / 2

        # Add new 2D curve points to the model

        new_ds = np.arange(start_d, curve.arclen, pixel_spacing)
        new_pxs, _ = curve.eval_by_arclen(new_ds)

        # Make sure that the new pixel isn't too close to the edge, because points that are too close to the edge
        # will not be in the past images
        move_vec = self.update_info["move_vec"]
        padding = self.get_param_val("image_padding")
        i = 0
        for new_px in new_pxs:
            px_adj = new_px - padding * move_vec
            if not (0 < px_adj[0] < self.camera.width and 0 < px_adj[1] < self.camera.height):
                break
            i += 1
            self.current_model.extend_by(1)
        new_pxs = new_pxs[:i]

        all_pxs = np.concatenate([current_pxs, new_pxs])
        all_idxs = np.concatenate([current_model_idxs, np.arange(len(new_pxs)) + max_consistent_idx + 1]).astype(int)
        all_ds = np.concatenate([current_ds, new_ds])

        to_req = {"main": all_pxs}
        ts = np.linspace(0.1, 1, 10)  # TODO: HARDCODED
        for i, info in enumerate(self.update_info.get("side_branches", [])):
            to_req[f"sb_{i}"] = info["curve"](ts)

        pt_est_info = self.query_point_estimates(to_req, self.update_info["rgb_msg"], track=False)
        if not pt_est_info:
            # Point tracker hasn't accumulated enough history, need to wait
            self.last_mask_info = None
            return False
        pt_main_info = pt_est_info.pop("main")
        pts = pt_main_info["pts"]

        # Invalid points (too close to the edge) will return as [0,0,0] - filter these out
        # Also filter out points that are behind the camera or too far

        bad_z_value = (pts[:, 2] < 0) | (pts[:, 2] > self.get_param_val("z_filter_threshold"))
        valid_idx = (np.abs(pts).sum(axis=1) > 0) & (~bad_z_value)
        pts = pts[valid_idx]
        all_idxs = all_idxs[valid_idx]
        all_ds = all_ds[valid_idx]

        # Because the 3D estimate of the curve corresponds to the surface, extend each estimate by the computed radius
        all_ds_normalized = all_ds / curve.arclen
        radii_px = self.update_info["radius_interpolator"](all_ds_normalized)
        radii_d = self.camera.getDeltaX(radii_px, pts[:, 2])
        pts[:, 2] += radii_d
        self.update_info["radii_d"] = dict(zip(all_idxs, radii_d))
        self.update_info["radii_px"] = dict(zip(all_idxs, radii_d))

        if len(pts) < 3:
            print("Too few points")
            return False
        curve_3d, stats = Bezier.iterative_fit(
            pts,
            inlier_threshold=self.get_param_val("curve_3d_inlier_threshold"),
            max_iters=self.get_param_val("curve_3d_ransac_iters"),
            stop_threshold=self.get_param_val("consistency_threshold"),
        )
        if not stats["success"]:
            print("Couldn't find a fit on the 3D curve!")
            self.last_mask_info = None
            return False

        # If we have a model from before, make sure that the existing points match up with the computed Bezier curve
        # (Sometimes the estimated Bezier curve is very poor despite having few outliers)
        if len(consistent_idx):
            prev_points = np.array([self.update_info["all_pts"][i] for i in consistent_idx])
            existing_model_dists, _ = curve_3d.query_pt_distance(prev_points)
            if np.any(existing_model_dists > 2 * self.get_param_val("curve_3d_inlier_threshold")):  # TODO: Hardcoded
                print("The fit 3D curve does not match up well with the previous model!")
                self.last_mask_info = None
                return False

        inliers = stats["inlier_idx"]
        _, ts = curve_3d.query_pt_distance(pts[inliers])
        radii_px = self.update_info["radius_interpolator"](ts)
        radii = self.camera.getDeltaX(radii_px, pts[inliers][:, 2])

        for params in zip(all_idxs[inliers], curve_3d(ts), pt_main_info["error"][valid_idx][inliers], radii):
            self.current_model.update_point(self.update_info["tf"], *params)

        self.update_info["curve_3d"] = curve_3d
        self.update_info["3d_point_estimates"] = pt_est_info

        return True

    def process_side_branches(self) -> bool:
        """
        Uses the 3D estimates of the side branches to update the 3D branch model.
        Uses the following logic:

        [Existing side branches]
        - Preprocess the existing branches by checking their 2D projections
            - If it falls in the main leader, don't attempt to update this branch
            - If it falls into the BG, don't attempt to update this branch, and add 1 to the BG counter
            - Otherwise, move to the next steps

        [Newly detected side branches]
        - Attempt to fit a 3D curve from the 3d estimates
        - If it projects into the main leader model, ignore it

        [Matching]
        - Match each newly detected side branch with any side branch where the bases are within some distance of each other
        - Take the shorter branch and subsample some points, and check the max distance from the newly detected branch
        - If they are sufficiently close, assume these are the same branch and update the 3D model
        - Otherwise assume that it is new and add it to the side branch models
        """

        # Construct a pixel mask with labels for each modeled object
        # 0 = BG, -1 = leader, positive integer = corresponding side branch (subtract 1 for index)
        leader_mask_estimate = self.update_info["leader_mask_estimate"]
        label_mask = np.zeros_like(leader_mask_estimate, dtype=int)
        for i, sb in enumerate(self.current_side_branches, start=1):
            label_mask[sb.branch_mask] = i
        label_mask[leader_mask_estimate] = -1

        # Process the results of the side branches by fitting 3D curves to them
        curve_3d = self.update_info["curve_3d"]
        curve_3d_eval_pts = curve_3d(np.linspace(0, 1, 101))
        detected_side_branches = self.update_info["side_branches"]
        side_branch_pt_info = self.update_info["3d_point_estimates"]

        for i, detected_side_branch in enumerate(detected_side_branches):
            # Take the pixels associated with the detection and check which label they fall into
            skel_pxs = detected_side_branch["stats"]["pts"]
            label_list, counts = np.unique(label_mask[skel_pxs[:, 1], skel_pxs[:, 0]], return_counts=True)
            most_freq_label = label_list[np.argmax(counts)]

            if most_freq_label == 0:
                # This is a new branch
                sb_index = None
            elif most_freq_label == -1:
                # This detection mostly falls inside the leader - don't process it
                continue
            else:
                sb_index = most_freq_label - 1

            # Fit 3D curves to the detected side branch
            est_3d_pts = side_branch_pt_info[f"sb_{i}"]["pts"]
            bad_z_value = (est_3d_pts[:, 2] < 0) | (est_3d_pts[:, 2] > self.get_param_val("z_filter_threshold"))
            est_3d_pts = est_3d_pts[~bad_z_value]
            if len(est_3d_pts) < 6:  # TODO: HARDCODED
                continue

            sb_3d, sb_stats = Bezier.iterative_fit(
                est_3d_pts,
                inlier_threshold=self.get_param_val("curve_3d_inlier_threshold"),
                max_iters=self.get_param_val("curve_3d_ransac_iters"),
                stop_threshold=self.get_param_val("consistency_threshold"),
            )
            if not sb_stats["success"]:
                continue

            # Find the point of intersection with the main leader - Make sure it's not too far!
            sb_origin = sb_3d(0)
            sb_tangent = sb_3d.tangent(0)
            dists, orientations = geom.get_pt_line_dist_and_orientation(curve_3d_eval_pts, sb_origin, sb_tangent)

            idx = (dists < 0.03) & (orientations < 0)  # TODO: Hardcoded
            if not np.any(idx):
                continue

            dists = dists[idx]
            pt_match = curve_3d_eval_pts[idx][np.argmin(dists)]

            # Compute the points to subsample as well as the corresponding radii
            ds = np.arange(0, sb_3d.arclen, 0.01)
            sb_pts_3d, _ = sb_3d.eval_by_arclen(ds)
            radius_interpolator = self.update_info["detection"].get_radius_interpolator_on_path(
                detected_side_branch["stats"]["pts"]
            )
            radii = self.camera.getDeltaX(radius_interpolator(ds / sb_3d.arclen), sb_pts_3d[:, 2])

            sb_pts_3d = np.concatenate([[pt_match], sb_pts_3d])
            radii = np.concatenate([[radii[0]], radii])
            cumul_dists = geom.convert_to_cumul_dists(sb_pts_3d)

            # Check if the branch looks too bendy - Usually a sign of a bad estimate
            if len(sb_pts_3d) < 3 or geom.get_max_bend(sb_pts_3d) > np.radians(75):  # TODO: Hardcoded
                continue

            # At this point, the branch has passed all checks
            # If the branch is new, add it to the model
            # Otherwise merge the current side branch estimate with the new one

            # TODO: This can be redone, there is no need to create two interp1d objects
            agg_interp = interp1d(cumul_dists, np.concatenate([sb_pts_3d.T, radii.reshape(1, -1)], axis=0))
            ds = np.arange(0, cumul_dists[-1], 0.01)
            interped = agg_interp(ds)
            pts = interped[:3].T
            radii = interped[3]

            if sb_index is None:
                sb = BranchModel(n=len(ds), cam=self.camera)
                sb.set_inv_tf(self.update_info["inv_tf"])
                self.current_side_branches.append(sb)
            else:
                # Update all the current points - do we want to characterize them by their interpolated distance?
                # Also extend the length of the model if necessary
                sb = self.current_side_branches[sb_index]
                if len(ds) > len(sb):
                    sb.extend_by(len(ds) - len(sb))
                elif len(ds) < len(sb):
                    for i in range(len(ds), len(sb)):
                        pt = sb.point(i)
                        if pt is None:
                            continue

                        px = self.camera.project3dToPixel(pt).astype(int)
                        if self.is_in_padding_region(px):
                            break

                        mask_val = self.update_info["mask"][px[1], px[0]]
                        if not mask_val:
                            sb.chop_at(i - 1)
                            break

            for i, (pt, radius) in enumerate(zip(pts, radii)):
                sb.update_point(self.update_info["tf"], i, pt, 1.0, radius)

        return True

    def publish_curve(self) -> bool:
        if not self.current_model:
            return True

        time = self.update_info.get("stamp", None)
        if time is None:
            time = self.get_clock().now().to_msg()

        msg = TreeModel()
        msg.header.frame_id = self.camera.tf_frame
        msg.header.stamp = time

        main_points = [Point(x=p[0], y=p[1], z=p[2]) for p in self.current_model.retrieve_points(filter_none=True)]
        msg.points.extend(main_points)
        msg.ids.extend([0] * len(main_points))

        for i, branch in enumerate(self.current_side_branches, start=1):
            points = [Point(x=p[0], y=p[1], z=p[2]) for p in branch.retrieve_points(filter_none=True)]
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

    def update(self):
        if self.paused or not self.active:
            return

        if not self.camera.tf_frame:
            return

        pose = self.get_camera_frame_pose(position_only=False)
        if self.last_pose is None:
            self.last_pose = pose

        if np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3]) > self.get_param_val("mask_update_dist"):
            # Ignore if rotation is too much
            rotation = Rotation.from_matrix(self.last_pose[:3, :3].T @ pose[:3, :3]).as_euler("XYZ")
            if np.linalg.norm(rotation) > np.radians(0.5):
                self.last_pose = pose
                return

            if self.update_tracking_request():
                self.last_pose = pose

    def is_in_padding_region(self, px):
        pad = self.get_param_val("image_padding")
        w = self.camera.width
        h = self.camera.height

        return px[0] < pad or px[0] > (w - pad) or px[1] < pad or px[1] > (h - pad)

    def convert_tracking_response(self, msg: Tracked3DPointResponse):
        info = defaultdict(lambda: defaultdict(list))
        for group in msg.groups:
            name = group.name
            info[name]["pts"] = np.array([(p.x, p.y, p.z) for p in group.points])
            info[name]["error"] = np.array(group.errors)

        for group in msg.groups_2d:
            name = group.name
            info[name]["pts_2d"] = np.array([(p.x, p.y) for p in group.points])

        return info

    def publish_diagnostic_image(self):
        if self.update_info.get("mask") is None:
            return

        mask_img = np.dstack([self.update_info["mask"] * 255] * 3)
        submask_img = np.zeros(mask_img.shape)
        submask = self.update_info.get("submask", None)
        if submask is not None:
            submask_img[submask] = [0, 255, 0]

        leader_est = self.update_info.get("leader_mask_estimate", None)
        if leader_est is not None:
            leader_est_img = np.zeros(mask_img.shape)
            leader_est_img[leader_est] = [255, 0, 255]
            submask_img = 0.5 * submask_img + 0.5 * leader_est_img

        diag_img = 0.3 * self.update_info["rgb"] + 0.35 * mask_img + 0.35 * submask_img
        if self.current_model:
            reconstructed_mask = self.current_model.branch_mask
            for sb in self.current_side_branches:
                reconstructed_mask[sb.branch_mask] = True

            alpha = reconstructed_mask * 0.5
            alpha = np.dstack([alpha] * 3)
            zeros = np.zeros_like(reconstructed_mask)
            overlay = np.dstack([zeros, zeros, reconstructed_mask * 255])
            diag_img = overlay * alpha + diag_img * (1 - alpha)

        pxs = self.camera.project3dToPixel(self.current_model.retrieve_points(filter_none=True)).astype(int)
        cv2.polylines(diag_img, [pxs.reshape((-1, 1, 2))], False, (255, 0, 0), 5)
        for px in pxs:
            diag_img = cv2.circle(diag_img, px, 7, (0, 0, 255), -1)

        curve = self.update_info.get("curve", None)
        if curve is not None:
            eval_pts = curve(np.linspace(0, 1, 200)).astype(int)
            cv2.polylines(diag_img, [eval_pts.reshape((-1, 1, 2))], False, (0, 0, 200), 3)

        for sb_info in self.update_info.get("side_branches", []):
            curve = sb_info["curve"]
            pxs = curve(np.linspace(0, 1, 20)).astype(int)
            cv2.polylines(diag_img, [pxs.reshape((-1, 1, 2))], False, (200, 0, 0), 3)

        detection = self.update_info.get("detection", None)
        if detection is not None:
            diag_img[detection.skel] = [255, 255, 0]

        img_msg = bridge.cv2_to_imgmsg(diag_img.astype(np.uint8), encoding="rgb8")
        self.diag_image_pub.publish(img_msg)

    def image_model_reproject(self, msg: Image):
        if self.active or not self.current_model:
            return
        if self.camera.tf_frame is None:
            return
        header = msg.header
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8") // 2
        cam_frame = self.get_camera_frame_pose(header.stamp, position_only=False)
        inv_tf = np.linalg.inv(cam_frame)

        draw_px = self.camera.project3dToPixel(self.current_model.retrieve_points(inv_tf, filter_none=True)).astype(int)
        if not draw_px.size:
            return

        cv2.polylines(img, [draw_px.reshape((-1, 1, 2))], False, (0, 0, 255), 3)

        for side_branch in self.current_side_branches:
            pxs = self.camera.project3dToPixel(side_branch.retrieve_points(filter_none=True)).astype(int)
            if not len(pxs):
                continue
            cv2.polylines(img, [pxs.reshape((-1, 1, 2))], False, (0, 255, 255), 3)

        new_img_msg = bridge.cv2_to_imgmsg(img.astype(np.uint8), encoding="rgb8", header=header)
        self.diag_image_pub.publish(new_img_msg)

    def get_camera_frame_pose(self, time=None, position_only=False):
        tf_mat = self.lookup_transform(self.get_param_val("base_frame"), self.camera.tf_frame, time, as_matrix=True)
        if position_only:
            return tf_mat[:3, 3]
        return tf_mat

    def px_in_img(self, px):
        return (0 <= px[0] < self.camera.width) and (0 <= px[1] < self.camera.height)

    def filter_px_to_img(self, px, convert_int=True):
        if convert_int:
            px = px.astype(int)
        return px[(px[:, 0] >= 0) & (px[:, 0] < self.camera.width) & (px[:, 1] >= 0) & (px[:, 1] < self.camera.height)]


def fill_holes(mask, fill_size):
    if not fill_size:
        return

    holes = label(~mask)
    start_label = 1
    while True:
        hole_mask = holes == start_label

        # Don't fill holes on the edge of the image, only internal ones
        if np.any(hole_mask[[0, -1]]) or np.any(hole_mask[:, [0, -1]]):
            start_label += 1
            continue
        hole_size = hole_mask.sum()
        if not hole_size:
            break
        elif hole_size < fill_size:
            mask[hole_mask] = True
        start_label += 1


def main(args=None):
    rclpy.init(args=args)
    node = Curve3DModeler()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)


if __name__ == "__main__":
    main()
