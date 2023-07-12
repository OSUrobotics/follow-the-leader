import os
import rclpy
import numpy as np
from std_msgs.msg import Header, Empty, ColorRGBA
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from skimage.measure import label
from follow_the_leader_msgs.msg import Point2D, PointList, ImageMaskPair, TrackedPointRequest, TrackedPointGroup, Tracked3DPointGroup, Tracked3DPointResponse, StateTransition
from follow_the_leader_msgs.srv import Query3DPoints
from collections import defaultdict
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from follow_the_leader.curve_fitting import BezierBasedDetection, Bezier
from follow_the_leader.utils.ros_utils import TFNode, process_list_as_dict
from threading import Lock
from scipy.interpolate import interp1d
import cv2
bridge = CvBridge()


class PointHistory:
    def __init__(self, max_error=4.0):
        self.points = []
        self.errors = []
        self.max_error = max_error
        self.base_tf = None
        self.base_tf_inv = None

    def add_point(self, point, error, tf):
        self.errors.append(error)
        if self.base_tf_inv is None:
            self.base_tf = tf
            self.base_tf_inv = np.linalg.inv(tf)
            self.points.append(point)
        else:
            self.points.append(TFNode.mul_homog(self.base_tf_inv @ tf, point))

    def as_point(self, inv_tf):

        errors = np.array(self.errors)
        idx = errors < self.max_error
        if np.any(idx):
            pts = np.array(self.points)[idx]
            errs = errors[idx]
            weights = 1 - np.array(errs) / self.max_error
            weights /= weights.sum()
            pt = (pts.T * weights).sum(axis=1)
            return TFNode.mul_homog(inv_tf @ self.base_tf, pt)
        return None

    def clear(self):
        self.points = []
        self.errors = []
        self.base_tf = None
        self.base_tf_inv = None


class Curve3DModeler(TFNode):
    def __init__(self):
        super().__init__('curve_3d_model_node', cam_info_topic='/camera/color/camera_info')

        # ROS parameters
        self.base_frame = self.declare_parameter('base_frame', 'base_link')
        self.padding = self.declare_parameter('image_padding', 10.0)
        self.recon_err_thres = self.declare_parameter('reconstruction_err_threshold', 4.0)

        self.mask_update_threshold = self.declare_parameter('mask_update_dist', 0.02)
        self.curve_spacing = self.declare_parameter('curve_spacing', 30.0)
        self.consistency_threshold = self.declare_parameter('consistency_threshold', 0.6)
        self.curve_2d_inlier_threshold = self.declare_parameter('curve_2d_inlier_threshold', 25.0)
        self.all_bg_retries = self.declare_parameter('all_bg_retries', 4)
        self.bg_z_allowance = self.declare_parameter('bg_z_allowance', 0.05)
        self.curve_3d_inlier_threshold = self.declare_parameter('curve_3d_inlier_threshold', 0.03)
        self.curve_3d_ransac_iters = self.declare_parameter('curve_3d_ransac_iters', 50)
        self.mask_hole_fill = self.declare_parameter('mask_hole_fill', 300)
        self.min_side_branch_length = self.declare_parameter('min_side_branch_length', 0.03)
        self.tracking_name = 'model'

        # State variables
        self.active = False
        self.paused = False
        self.received_first_mask = False
        self.current_model = []
        self.current_side_branches = []
        self.current_side_branches_bg_count = []
        self.last_pos = None
        self.last_mask_info = None
        self.all_bg_counter = 0

        self.update_info = {}

        # ROS Utils
        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.cb_reentrant = ReentrantCallbackGroup()
        self.curve_pub = self.create_publisher(PointList, 'curve_3d', 1)
        self.rviz_model_pub = self.create_publisher(MarkerArray, '/curve_3d_rviz_array', 1)
        self.diag_image_pub = self.create_publisher(Image, 'model_diagnostic', 1)
        self.img_mask_sub = self.create_subscription(ImageMaskPair, '/image_mask_pair', self.process_mask, 1)
        self.img_sub = self.create_subscription(Image, '/camera/color/image_rect_raw', self.image_model_reproject, 1, callback_group=self.cb_reentrant)
        self.reset_sub = self.create_subscription(Empty, '/reset_model', self.reset, 1, callback_group=self.cb_reentrant)
        self.transition_sub = self.create_subscription(StateTransition, 'state_transition',
                                                       self.handle_state_transition, 1, callback_group=self.cb_reentrant)
        self.point_query_client = self.create_client(Query3DPoints, '/query_3d_points')
        self.lock = Lock()
        self.processing_lock = Lock()
        self.create_timer(0.01, self.update, callback_group=self.cb_group)

    def handle_state_transition(self, msg: StateTransition):
        action = process_list_as_dict(msg.actions, 'node', 'action').get(self.get_name())
        if not action:
            return

        if action == 'activate':
            self.start_modeling()
        elif action == 'reset':
            self.stop_modeling()
        elif action == 'pause':
            self.pause()
        elif action == 'resume':
            self.resume()

        else:
            raise ValueError('Unknown action {} for node {}'.format(action, self.get_name()))

    def reset(self, *_, **__):
        with self.processing_lock:
            self.active = False
            self.paused = False
            self.received_first_mask = False
            self.current_model = []
            self.current_side_branches = []
            self.current_side_branches_bg_count = []
            self.last_pos = None
            self.last_mask_info = None
            self.all_bg_counter = 0
            self.update_info = {}
            print('Model reset!')

    def start_modeling(self, *_, **__):
        self.reset()
        self.last_pos = self.get_camera_frame_pose(position_only=True)
        self.active = True

    def stop_modeling(self, *_, **__):
        self.active = False
        self.process_final_model()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def process_final_model(self):
        ...

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

        if self.active and self.update_info.get('reinitialize'):
            self.reset()
            self.active = True
        return success

    def process_last_mask_info(self) -> bool:
        with self.lock:

            if self.last_mask_info is None:
                return False

            rgb_msg = self.last_mask_info.rgb
            mask = bridge.imgmsg_to_cv2(self.last_mask_info.mask, desired_encoding='mono8') > 128
            stamp = self.last_mask_info.mask.header.stamp

        self.update_info['stamp'] = stamp
        self.update_info['mask'] = mask
        self.update_info['rgb'] = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        self.update_info['rgb_msg'] = rgb_msg
        self.update_info['tf'] = self.get_camera_frame_pose(time=stamp)
        self.update_info['inv_tf'] = np.linalg.inv(self.update_info['tf'])
        return True

    def get_primary_movement_direction(self) -> bool:

        if not self.current_model:
            vec_msg = self.last_mask_info.image_frame_offset
            move_vec = np.array([vec_msg.x, vec_msg.y])
        else:
            inv_tf = self.update_info['inv_tf']

            # Determine the primary direction of movement based on the existing model
            all_pts = [pt.as_point(inv_tf) for pt in self.current_model]
            self.update_info['all_pts'] = all_pts

            valid_idxs = [i for i, pt in enumerate(all_pts) if pt is not None]
            if len(valid_idxs) < 2:
                print('Not enough valid pixels in the model, reinitializing...')
                self.update_info['reinitialize'] = True
                return False

            first_pt = all_pts[min(valid_idxs)]
            last_pt = all_pts[max(valid_idxs)]
            first_px = np.array(self.camera.project3dToPixel(first_pt))
            last_px = np.array(self.camera.project3dToPixel(last_pt))

            if np.linalg.norm(first_pt - last_pt) < 0.025 or np.linalg.norm(first_px - last_px) < 30:
                print('Model looks too squished in! Reinitializing')
                self.update_info['reinitialize'] = True
                return False

            move_vec = last_px - first_px
            move_vec = move_vec / np.linalg.norm(move_vec)
        self.update_info['move_vec'] = move_vec

        return True

    def run_mask_curve_detection(self) -> bool:
        """
        Fits a curve to the mask. Utilizes the currently existing model by projecting the model into the image and
        taking the submask component with the most matches.
        """

        submask = self.update_info['mask']
        if self.current_model:

            # Determine parts of the original 3D model that are still in frame
            in_frame_idxs = []
            in_frame_pxs = []
            for i in range(len(self.current_model)):
                idx = len(self.current_model) - i - 1
                pt = self.current_model[idx].as_point(self.update_info['inv_tf'])
                if pt is None:
                    continue
                px = self.camera.project3dToPixel(pt)
                if 0 <= int(px[0]) < self.camera.width and 0 <= int(px[1]) < self.camera.height:
                    in_frame_idxs.append(idx)
                    in_frame_pxs.append(px)
                else:
                    break

            if not in_frame_pxs:
                print('All pxs were outside the image!')
                self.last_mask_info = None
                return False

            in_frame_pxs = np.array(in_frame_pxs)
            in_frame_idxs = np.array(in_frame_idxs)
            self.update_info['valid_pxs'] = in_frame_pxs
            self.update_info['valid_idxs'] = in_frame_idxs

            # Split the mask into connected subcomponents and identify which one has the most matches
            pxs_int = in_frame_pxs.astype(int)
            labels = label(self.update_info['mask'])
            label_list, counts = np.unique(labels[pxs_int[:, 1], pxs_int[:, 0]], return_counts=True)
            most_freq_label = label_list[np.argmax(counts)]

            if most_freq_label == 0:
                print('Most points were projected into the BG! Not processing')
                self.all_bg_counter += 1
                if self.all_bg_counter >= self.all_bg_retries.value:
                    print('It looks like the model is lost! Resetting...')
                    self.update_info['reinitialize'] = True
                self.last_mask_info = None
                return False

            submask = labels == most_freq_label

        self.all_bg_counter = 0
        self.update_info['submask'] = submask

        # Fill in small holes in the mask to avoid skeletonization issues
        holes = label(~submask)
        start_label = 1
        while True:
            hole_mask = holes == start_label

            # Don't fill holes on the edge of the image, only internal ones
            if np.any(hole_mask[[0,-1]]) or np.any(hole_mask[:,[0,-1]]):
                start_label += 1
                continue
            hole_size = hole_mask.sum()
            if not hole_size:
                break
            elif hole_size < self.mask_hole_fill.value:
                submask[hole_mask] = True
            start_label += 1

        # Use the chosen submask to run the Bezier curve fit
        detection = BezierBasedDetection(submask, use_medial_axis=True)
        curve = detection.fit(vec=self.update_info['move_vec'], trim=int(self.padding.value))
        self.update_info['detection'] = detection
        if curve is None:
            print('No good curve was found!')
            return False

        # Compute a masked image of the leader using the fit curve and the estimated radii
        radius_interpolator = detection.get_radius_interpolator_on_path()
        self.update_info['radius_interpolator'] = radius_interpolator
        leader_mask_estimate = np.zeros(self.update_info['mask'].shape, dtype=np.uint8)
        ts = np.linspace(0, 1, 11)
        for t_s, t_e in zip(ts[:-1], ts[1:]):
            curve_start, curve_end = curve.eval_by_arclen([t_s, t_e], normalized=True)[0]
            curve_start = curve_start.astype(int)
            curve_end = curve_end.astype(int)
            r = radius_interpolator((t_s + t_e) / 2)
            leader_mask_estimate = cv2.line(leader_mask_estimate, curve_start, curve_end, color=255,
                                            thickness=int(r * 2))
        leader_mask_estimate = leader_mask_estimate > 128

        # Find 3D side branches
        side_branch_info = detection.run_side_branch_search(min_len=20, filter_mask=leader_mask_estimate)  # TODO: MAGIC NUMBER

        self.update_info['curve'] = curve
        self.update_info['side_branches'] = side_branch_info
        self.update_info['leader_mask_estimate'] = leader_mask_estimate > 128

        return True

    def reconcile_2d_3d_curves(self) -> bool:
        # Takes the fit 2D curve, obtains 3D estimates, and construct a 3D curve. Then check to see which ones agree

        curve = self.update_info['curve']
        pixel_spacing = self.curve_spacing.value

        if self.current_model:
            pxs = self.update_info['valid_pxs']
            idxs = self.update_info['valid_idxs']

            px_thres = self.curve_2d_inlier_threshold.value
            dists, ts = curve.query_pt_distance(pxs)
            pt_consistent = dists < px_thres

            if pt_consistent.sum() < 2 or pt_consistent.mean() < self.consistency_threshold.value:
                print('The current 3D model does not seem to be consistent with the extracted 2D model. Skipping')
                self.last_mask_info = None
                return False

            for inconsistent_idx in idxs[~pt_consistent]:
                self.current_model[inconsistent_idx].clear()

            # Chop off the model beyond any inconsistent pixels, and reinterpolate any inconsistent pixels in between
            consistent_idx = idxs[pt_consistent]
            min_consistent_idx = consistent_idx.min()
            max_consistent_idx = consistent_idx.max()
            self.current_model = self.current_model[:max_consistent_idx + 1]        # Chopping

            current_model_idxs = np.arange(min_consistent_idx, max_consistent_idx + 1)
            consistent_ds = curve.t_to_curve_dist(ts[pt_consistent])
            current_ds = interp1d(consistent_idx, consistent_ds)(current_model_idxs)
            current_pxs, _ = curve.eval_by_arclen(current_ds)

            start_d = current_ds[-1] + pixel_spacing

        else:
            current_pxs = np.zeros((0,2))
            current_model_idxs = []
            max_consistent_idx = -1
            current_ds = []
            start_d = pixel_spacing / 2

        # Add new 2D curve points to the model

        new_ds = np.arange(start_d, curve.arclen, pixel_spacing)
        new_pxs, _ = curve.eval_by_arclen(new_ds)

        # Make sure that the new pixel isn't too close to the edge, because points that are too close to the edge
        # will not be in the past images
        move_vec = self.update_info['move_vec']
        padding = self.padding.value
        i = 0
        for new_px in new_pxs:
            px_adj = new_px - padding * move_vec
            if not (0 < px_adj[0] < self.camera.width and 0 < px_adj[1] < self.camera.height):
                break
            i += 1
            self.current_model.append(PointHistory())
        new_pxs = new_pxs[:i]

        all_pxs = np.concatenate([current_pxs, new_pxs])
        all_idxs = np.concatenate([current_model_idxs, np.arange(len(new_pxs)) + max_consistent_idx + 1]).astype(int)
        all_ds = np.concatenate([current_ds, new_ds])

        to_req = {'main': all_pxs}
        ts = np.linspace(0.1, 1, 10)      # TODO: HARDCODED
        for i, info in enumerate(self.update_info.get('side_branches', [])):
            to_req[f'sb_{i}'] = info['curve'](ts)

        pt_est_info = self.query_point_estimates(to_req, self.update_info['rgb_msg'], track=False)
        if not pt_est_info:
            # Point tracker hasn't accumulated enough history, need to wait
            self.last_mask_info = None
            return False
        pt_main_info = pt_est_info.pop('main')
        pts = pt_main_info['pts']

        # Invalid points (too close to the edge) will return as [0,0,0] - filter these out
        valid_idx = np.abs(pts).sum(axis=1) > 0
        pts = pts[valid_idx]
        all_idxs = all_idxs[valid_idx]
        all_ds = all_ds[valid_idx]

        # Because the 3D estimate of the curve corresponds to the surface, extend each estimate by the computed radius
        all_ds_normalized = all_ds / curve.arclen
        radii_px = self.update_info['radius_interpolator'](all_ds_normalized)
        radii_d = self.camera.getDeltaX(radii_px, pts[:,2])
        pts[:,2] += radii_d
        self.update_info['radii_d'] = dict(zip(all_idxs, radii_d))
        self.update_info['radii_px'] = dict(zip(all_idxs, radii_d))

        curve_3d, stats = Bezier.iterative_fit(pts,
                                               inlier_threshold=self.curve_3d_inlier_threshold.value,
                                               max_iters=self.curve_3d_ransac_iters.value,
                                               stop_threshold=self.consistency_threshold.value)
        if not stats['success']:
            print("Couldn't find a fit on the 3D curve!")
            self.last_mask_info = None
            return False

        inliers = stats['inlier_idx']
        _, ts = curve_3d.query_pt_distance(pts[inliers])
        for model_idx, pt, err in zip(all_idxs[inliers], curve_3d(ts), pt_main_info['error'][valid_idx][inliers]):
            self.current_model[model_idx].add_point(pt, err, self.update_info['tf'])

        self.update_info['curve_3d'] = curve_3d
        self.update_info['3d_point_estimates'] = pt_est_info

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

        leader_mask_estimate = self.update_info['leader_mask_estimate']
        curve_3d = self.update_info['curve_3d']
        pt_est_info = self.update_info['3d_point_estimates']
        label_mask = self.update_info['mask'].copy().astype(int)
        label_mask[leader_mask_estimate] = 2        # 0 = BG, 1 = Non-leader pixel, 2 = Leader pixel

        # Preprocessing existing branches
        eligible_branches = {}
        to_delete = []
        for i, side_branch in enumerate(self.current_side_branches):
            pts = self.convert_pt_hist_list_to_pts(side_branch, self.update_info['inv_tf'])
            pxs = self.filter_px_to_img(self.camera.project3dToPixel(pts), convert_int=True)
            if not len(pxs):
                continue

            label_list, counts = np.unique(label_mask[pxs[:, 1], pxs[:, 0]], return_counts=True)
            most_freq_label = label_list[np.argmax(counts)]
            if most_freq_label == 1:
                self.current_side_branches_bg_count[i] = 0
                eligible_branches[i] = pts
            elif most_freq_label == 0:
                self.current_side_branches_bg_count[i] += 1
                if self.current_side_branches_bg_count[i] > self.all_bg_retries.value:
                    to_delete.append(i)

        # Process the results of the side branches by fitting 3D curves to them
        curve_3d_eval_pts = curve_3d(np.linspace(0, 1, 101))

        for _, side_pt_info in pt_est_info.items():
            sb_3d, sb_stats = Bezier.iterative_fit(side_pt_info['pts'],
                                                   inlier_threshold=self.curve_3d_inlier_threshold.value,
                                                   max_iters=self.curve_3d_ransac_iters.value,
                                                   stop_threshold=self.consistency_threshold.value)
            if not sb_stats['success']:
                continue

            # Find the point of intersection with the main leader - Make sure it's not too far!
            sb_origin = sb_3d(0)
            sb_tangent = sb_3d.tangent(0)
            dists, orientations = get_pt_line_dist_and_orientation(curve_3d_eval_pts, sb_origin, sb_tangent)
            pt_match = curve_3d_eval_pts[np.argmin(dists)]
            idx = (dists < 0.05) & (orientations < 0)  # TODO: Hardcoded
            if not np.any(idx):
                continue
            # Terminal branch point should be sufficiently far from leader
            if np.linalg.norm(sb_3d(1.0) - pt_match) < self.min_side_branch_length.value:
                continue
            sb_pts_3d = np.concatenate([[pt_match], sb_3d(np.linspace(0.1, 1, 10))])

            # Finally, check the existing branches to make sure you don't have an overlap
            matches_existing_branch = False
            for idx, pts_3d in eligible_branches.items():
                if get_max_pt_distance(pts_3d, sb_pts_3d) < self.curve_3d_inlier_threshold.value:
                    matches_existing_branch = True
                    break

            if matches_existing_branch:
                # TODO: Maybe want to merge the two estimates?
                continue

            # Side branch has been identified as a new branch - Add it to the current model
            sb = [PointHistory() for _ in range(len(sb_pts_3d))]
            for pt_hist, pt in zip(sb, sb_pts_3d):
                pt_hist.add_point(pt, pt_hist.max_error / 2, self.update_info['tf'])

            self.current_side_branches.append(sb)
            self.current_side_branches_bg_count.append(0)

        for idx in to_delete:
            del self.current_side_branches[idx]
            del self.current_side_branches_bg_count[idx]

        return True

    def publish_curve(self) -> bool:

        if not self.current_model:
            return True

        time = self.update_info.get('stamp', None)
        if time is None:
            time = self.get_clock().now().to_msg()
        tf = self.get_camera_frame_pose(time=time)
        inv_tf = np.linalg.inv(tf)
        pts = [pt.as_point(inv_tf) for pt in self.current_model]

        msg = PointList()
        msg.header.frame_id = self.camera.tf_frame
        msg.header.stamp = time
        msg.points = [Point(x=p[0], y=p[1], z=p[2]) for p in pts if p is not None]
        self.curve_pub.publish(msg)

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
        marker.points = msg.points
        marker.scale.x = 0.02
        marker.color = ColorRGBA(r=0.5, g=1.0, b=0.5, a=1.0)
        markers.markers.append(marker)

        for i, side_branch in enumerate(self.current_side_branches, start=1):
            pts = self.convert_pt_hist_list_to_pts(side_branch, inv_tf)

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

        pos = self.get_camera_frame_pose(position_only=True)
        if self.last_pos is None or np.linalg.norm(pos - self.last_pos) > self.mask_update_threshold.value:
            if self.update_tracking_request():
                self.last_pos = pos

    def is_in_padding_region(self, px):
        pad = self.padding.value
        w = self.camera.width
        h = self.camera.height

        return px[0] < pad or px[0] > (w - pad) or px[1] < pad or px[1] > (h - pad)

    def convert_tracking_response(self, msg: Tracked3DPointResponse):
        info = defaultdict(lambda: defaultdict(list))
        for group in msg.groups:
            name = group.name
            info[name]['pts'] = np.array([(p.x, p.y, p.z) for p in group.points])
            info[name]['error'] = np.array(group.errors)

        for group in msg.groups_2d:
            name = group.name
            info[name]['pts_2d'] = np.array([(p.x, p.y) for p in group.points])

        return info

    def publish_diagnostic_image(self):

        if self.update_info.get('mask') is None:
            return

        mask_img = np.dstack([self.update_info['mask'] * 255] * 3)
        submask_img = np.zeros(mask_img.shape)
        submask = self.update_info.get('submask', None)
        if submask is not None:
            submask_img[submask] = [0, 255, 0]

        leader_est = self.update_info.get('leader_mask_estimate', None)
        if leader_est is not None:
            leader_est_img = np.zeros(mask_img.shape)
            leader_est_img[leader_est] = [255, 0, 255]
            submask_img = 0.5 * submask_img + 0.5 * leader_est_img

        diag_img = 0.3 * self.update_info['rgb'] + 0.35 * mask_img + 0.35 * submask_img

        pxs = []
        for pt_hist in self.current_model:
            pt = pt_hist.as_point(self.update_info['inv_tf'])
            if pt is not None:
                pxs.append(self.camera.project3dToPixel(pt))

        pxs = np.array(pxs).astype(int)
        cv2.polylines(diag_img, [pxs.reshape((-1, 1, 2))], False, (255, 0, 0), 5)
        for px in pxs:
            diag_img = cv2.circle(diag_img, px, 7, (0, 0, 255), -1)

        curve = self.update_info.get('curve', None)
        if curve is not None:
            eval_pts = curve(np.linspace(0, 1, 200)).astype(int)
            cv2.polylines(diag_img, [eval_pts.reshape((-1, 1, 2))], False, (0, 0, 200), 3)

        detection = self.update_info.get('detection', None)
        if detection is not None:
            diag_img[detection.skel] = [255, 255, 0]

        if self.current_model:
            draw_px = []

            radii_px = self.update_info.get('radii_px', {})
            radii_d = self.update_info.get('radii_d', {})

            for i in range(len(self.current_model)):
                idx = len(self.current_model) - i - 1
                pt = self.current_model[idx].as_point(self.update_info['inv_tf'])
                if pt is None:
                    continue
                px = self.camera.project3dToPixel(pt)
                draw_px.append((idx, px))
                if not (0 < px[0] < self.camera.width and 0 < px[1] < self.camera.height):
                    break

            for i, px in draw_px:

                draw_left = None
                draw_right = None
                width = None
                px_rad = radii_px.get(i)
                if px_rad is not None:
                    draw_left = tuple((px - [px_rad, 0]).astype(int))
                    draw_right = tuple((px + [px_rad, 0]).astype(int))
                    width = radii_d.get(i)

                px = (int(px[0]), int(px[1]))

                if draw_left is not None:
                    diag_img = cv2.line(diag_img, draw_left, draw_right, color=(0,0,0), thickness=2)
                    text = '{} (r={:.1f})'.format(i, width * 100)
                else:
                    text = str(i)

                diag_img = cv2.circle(diag_img, px, 6, (0, 255, 0), -1)
                diag_img = cv2.putText(diag_img, text, px, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
                diag_img = cv2.putText(diag_img, text, px, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            for sb_info in self.update_info.get('side_branches', []):
                curve = sb_info['curve']
                pxs = curve(np.linspace(0, 1, 20)).astype(int)
                cv2.polylines(diag_img, [pxs.reshape((-1, 1, 2))], False, (200, 0, 0), 3)

        img_msg = bridge.cv2_to_imgmsg(diag_img.astype(np.uint8), encoding='rgb8')
        self.diag_image_pub.publish(img_msg)

    def image_model_reproject(self, msg: Image):

        if self.active or not self.current_model:
            return

        header = msg.header
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') // 2
        cam_frame = self.get_camera_frame_pose(header.stamp, position_only=False)
        inv_tf = np.linalg.inv(cam_frame)

        draw_px = []
        for i in range(len(self.current_model)):
            idx = len(self.current_model) - i - 1
            pt = self.current_model[idx].as_point(inv_tf)
            if pt is None:
                continue
            px = self.camera.project3dToPixel(pt)
            draw_px.append(px.astype(int))

        if not draw_px:
            return

        draw_px = np.array(draw_px)
        cv2.polylines(img, [draw_px.reshape((-1, 1, 2))], False, (0, 0, 255), 3)

        for side_branch in self.current_side_branches:
            pxs = self.camera.project3dToPixel(self.convert_pt_hist_list_to_pts(side_branch, inv_tf)).astype(int)
            if not len(pxs):
                continue
            cv2.polylines(img, [pxs.reshape((-1, 1, 2))], False, (0, 255, 255), 3)

        new_img_msg = bridge.cv2_to_imgmsg(img.astype(np.uint8), encoding='rgb8', header=header)
        self.diag_image_pub.publish(new_img_msg)


    def get_camera_frame_pose(self, time=None, position_only=False):
        tf_mat = self.lookup_transform(self.base_frame.value, self.camera.tf_frame, time, as_matrix=True)
        if position_only:
            return tf_mat[:3,3]
        return tf_mat

    @staticmethod
    def convert_pt_hist_list_to_pts(pt_hists, inv_tf):
        pts = [pt_hist.as_point(inv_tf) for pt_hist in pt_hists]
        pts = np.array([pt for pt in pts if pt is not None])
        return pts

    def filter_px_to_img(self, px, convert_int=True):
        if convert_int:
            px = px.astype(int)
        return px[(px[:, 0] >= 0) & (px[:, 0] < self.camera.width) & (px[:, 1] >= 0) & (px[:, 1] < self.camera.height)]


def get_pt_line_dist_and_orientation(pts, origin, ray):
    diff = pts - origin
    proj_comp = diff.dot(ray)
    dists = np.linalg.norm(diff - proj_comp[...,np.newaxis] * ray, axis=1)
    return dists, np.sign(proj_comp)


def convert_to_cumul_dists(pts):
    dists = np.zeros(len(pts))
    dists[1:] = np.linalg.norm(pts[:-1] - pts[1:], axis=1).cumsum()
    return dists


def get_max_pt_distance(pts_1, pts_2):
    cumul_1 = convert_to_cumul_dists(pts_1)
    cumul_2 = convert_to_cumul_dists(pts_2)

    # make pts_1 refer to the shorter set of points
    if cumul_1[-1] > cumul_2[-1]:
        pts_2, pts_1 = pts_1, pts_2
        cumul_2, cumul_1 = cumul_1, cumul_2

    interp = interp1d(cumul_2, pts_2.T)
    return np.max(np.linalg.norm(pts_1 - interp(cumul_1).T, axis=1))


def main(args=None):
    rclpy.init(args=args)
    node = Curve3DModeler()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)


if __name__ == '__main__':
    main()
