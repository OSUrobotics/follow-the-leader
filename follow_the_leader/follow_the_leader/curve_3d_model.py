import os
import rclpy
import numpy as np
from std_msgs.msg import Header, Empty
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from skimage.measure import label
from follow_the_leader_msgs.msg import Point2D, PointList, ImageMaskPair, TrackedPointRequest, TrackedPointGroup, Tracked3DPointGroup, Tracked3DPointResponse, StateTransition
from follow_the_leader_msgs.srv import Query3DPoints
from collections import defaultdict
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from follow_the_leader.curve_fitting import BezierBasedDetection, Bezier
from follow_the_leader.utils.ros_utils import TFNode, process_list_as_dict, wait_for_future_synced
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
        self.tracking_name = 'model'

        # State variables
        self.active = False
        self.paused = False
        self.received_first_mask = False
        self.current_model = []
        self.last_pos = None
        self.last_mask_info = None
        self.all_bg_counter = 0

        self.update_info = {}

        # ROS Utils
        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.cb_reentrant = ReentrantCallbackGroup()
        self.curve_pub = self.create_publisher(PointList, 'curve_3d', 1)
        self.rviz_model_pub = self.create_publisher(Marker, 'curve_3d_rviz', 1)
        self.diag_image_pub = self.create_publisher(Image, 'model_diagnostic', 1)
        self.img_mask_sub = self.create_subscription(ImageMaskPair, '/image_mask_pair', self.process_mask, 1)
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
        self.reset()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def process_final_model(self):
        self.current_model = []

    def process_mask(self, msg: ImageMaskPair):

        # Hack to guard against bad optical flow masks when initially moving
        if not self.received_first_mask:
            self.received_first_mask = True
            return

        with self.lock:
            self.last_mask_info = msg

    def query_point_estimates(self, pxs, img_msg, name='dummy', track=False):
        pts = [Point2D(x=p[0], y=p[1]) for p in pxs]
        req = Query3DPoints.Request()
        req.track = track
        req.request.image = img_msg
        req.request.groups.append(TrackedPointGroup(name=name, points=pts))
        resp = self.point_query_client.call(req)
        if not resp.success:
            return None
        return self.convert_tracking_response(resp.response)[name]

    def update_tracking_request(self) -> bool:
        with self.processing_lock:
            self.update_info = {}
            if not self.process_last_mask_info():
                return False

            steps = [
                self.get_primary_movement_direction,
                self.run_mask_curve_detection,
                self.reconcile_2d_3d_curves,
                self.publish_curve,
            ]

            success = False
            for step in steps:
                success = step()
                if not success:
                    break

            self.publish_diagnostic_image()

        if self.update_info.get('reinitialize'):
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
            hole_size = hole_mask.sum()
            if not hole_size:
                break
            elif hole_size < 200:
                submask[hole_mask] = True
            start_label += 1

        # Use the chosen submask to run the Bezier curve fit
        detection = BezierBasedDetection(submask, use_medial_axis=True)
        curve = detection.fit(vec=self.update_info['move_vec'], trim=int(self.padding.value))
        self.update_info['detection'] = detection
        if curve is None:
            print('No good curve was found!')
            return False

        self.update_info['curve'] = curve
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

            if pt_consistent.mean() < self.consistency_threshold.value:
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

        pt_est_info = self.query_point_estimates(all_pxs, self.update_info['rgb_msg'], track=False)
        if not pt_est_info:
            # Point tracker hasn't accumulated enough history, need to wait
            self.last_mask_info = None
            return False
        pts = pt_est_info['pts']

        # Because the 3D estimate of the curve corresponds to the surface, extend each estimate by the computed radius
        radius_interpolator = self.update_info['detection'].get_radius_interpolator_on_path()
        all_ds_normalized = np.concatenate([current_ds, new_ds]) / curve.arclen
        radii_px = radius_interpolator(all_ds_normalized)
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
        for model_idx, pt, err in zip(all_idxs[inliers], pt_est_info['pts'][inliers], pt_est_info['error'][inliers]):
            self.current_model[model_idx].add_point(pt, err, self.update_info['tf'])

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

        marker = Marker()
        marker.header.frame_id = self.camera.tf_frame
        marker.header.stamp = time
        marker.type = Marker.LINE_STRIP
        marker.points = msg.points
        marker.scale.x = 0.02
        marker.color.r = 0.5
        marker.color.g = 1.0
        marker.color.b = 0.5
        marker.color.a = 1.0
        self.rviz_model_pub.publish(marker)

        return True

    def update(self):
        if self.paused or not self.active:
            return

        if not self.camera.tf_frame:
            return

        pos = self.get_camera_frame_pose(position_only=True)
        if not self.current_model or np.linalg.norm(pos - self.last_pos) > self.mask_update_threshold.value:
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

        img_msg = bridge.cv2_to_imgmsg(diag_img.astype(np.uint8), encoding='rgb8')
        self.diag_image_pub.publish(img_msg)

    def get_camera_frame_pose(self, time=None, position_only=False):
        tf_mat = self.lookup_transform(self.base_frame.value, self.camera.tf_frame, time, as_matrix=True)
        if position_only:
            return tf_mat[:3,3]
        return tf_mat


def main(args=None):
    rclpy.init(args=args)
    node = Curve3DModeler()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)


if __name__ == '__main__':
    main()
