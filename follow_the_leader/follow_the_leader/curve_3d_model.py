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
from collections import defaultdict
from threading import Event
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
        self.curve_spacing = self.declare_parameter('curve_spacing', 0.10)
        self.consistency_threshold = self.declare_parameter('consistency_threshold', 0.6)
        self.px_consistency_threshold = self.declare_parameter('px_consistency_threshold', 25.0)
        self.all_bg_retries = self.declare_parameter('all_bg_retries', 4)
        self.bg_z_allowance = self.declare_parameter('bg_z_allowance', 0.05)
        self.tracking_name = 'model'

        # State variables
        self.active = False
        self.current_model = []
        self.current_tracking_point_index = 0
        self.last_pos = None
        self.last_mask_info = None
        self.points_received = False
        self.last_sent_request = None
        self.all_bg_counter = 0

        # ROS Utils
        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.cb_reentrant = ReentrantCallbackGroup()
        self.curve_pub = self.create_publisher(PointList, 'curve_3d', 1)
        self.rviz_model_pub = self.create_publisher(Marker, 'curve_3d_rviz', 1)
        self.diag_image_pub = self.create_publisher(Image, 'model_diagnostic', 1)
        self.point_tracking_pub = self.create_publisher(TrackedPointRequest, '/point_tracking_request', 1)
        self.img_mask_sub = self.create_subscription(ImageMaskPair, '/image_mask_pair', self.process_mask, 1)
        self.point_tracking_sub = self.create_subscription(Tracked3DPointResponse, '/point_tracking_response', self.process_3d_points, 1, callback_group=self.cb_reentrant)
        self.reset_sub = self.create_subscription(Empty, '/reset_model', self.reset, 1, callback_group=self.cb_reentrant)
        self.transition_sub = self.create_subscription(StateTransition, 'state_transition',
                                                       self.handle_state_transition, 1, callback_group=self.cb_reentrant)
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
        else:
            raise ValueError('Unknown action {} for node {}'.format(action, self.get_name()))

    def reset(self, *_, **__):
        self.current_model = []
        self.current_tracking_point_index = 0
        self.last_mask_info = None
        self.last_pos = None
        self.active = False
        self.points_received = False
        self.last_sent_request = None
        self.all_bg_counter = 0
        print('Model reset!')

    def start_modeling(self, *_, **__):
        self.reset()
        self.last_pos = self.get_camera_frame_pose(position_only=True)
        self.active = True

    def stop_modeling(self, *_, **__):
        self.active = False
        self.process_final_model()
        self.reset()

    def process_final_model(self):
        self.current_model = []

    def process_mask(self, msg: ImageMaskPair):

        with self.lock:
            self.last_mask_info = msg

    def process_3d_points(self, msg: Tracked3DPointResponse):

        with self.processing_lock:
            resp_dict = self.convert_tracking_response(msg)
            print('Received 3D tracking response with keys: {}'.format(', '.join(sorted(resp_dict))))
            info = resp_dict.get(self.last_sent_request, None)
            if info is None:
                return

            # Get the current TF of the optical frame
            stamp = msg.header.stamp
            tf = self.get_camera_frame_pose(time=stamp)

            if self.current_model:
                # Add points to the existing model
                for i, (pt, err) in enumerate(zip(info['pts'], info['error'])):
                    self.current_model[self.current_tracking_point_index + i].add_point(pt, err, tf)

                self.points_received = True

    def update_tracking_request(self) -> bool:
        # Run curve fitting on the mask skeleton - Cut out the regions of the skeleton in the padding zone
        with self.lock:

            if self.last_mask_info is None:
                return False

            rgb_msg = self.last_mask_info.rgb
            mask = bridge.imgmsg_to_cv2(self.last_mask_info.mask, desired_encoding='mono8') > 128
            vec_msg = self.last_mask_info.image_frame_offset
            mask_vec = np.array([vec_msg.x, vec_msg.y])
            stamp = self.last_mask_info.mask.header.stamp

        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        pad = int(self.padding.value)
        track_px = []
        tf = self.get_camera_frame_pose(time=stamp)
        inv_tf = np.linalg.inv(tf)


        if not self.current_model:

            submask = mask
            detection = BezierBasedDetection(submask)
            curve = detection.fit(vec=mask_vec, trim=pad)

            # Use the curve parametrization to subsample points that are not in the padding region, and send the tracking request
            ts = np.arange(self.curve_spacing.value / 2, 1, self.curve_spacing.value)
            pxs_int = curve(ts)
            for px, t in zip(pxs_int, ts):
                if t < 0.5 and self.is_in_padding_region(px):
                    continue
                track_px.append(Point2D(x=px[0], y=px[1]))
                self.current_model.append(PointHistory(max_error=self.recon_err_thres.value))

        else:
            # In the event that you already have a current model, check to see if it is consistent with the mask
            # If inconsistent, simply ignore the current mask and wait for a new one
            # If it's consistent, we use the curve parametrization to add new points to the tracking based on the latest consistent point
            if not self.points_received:
                return False

            # Determine the primary direction of movement based on the existing model
            all_pts = [pt.as_point(inv_tf) for pt in self.current_model]
            current_pts = all_pts[self.current_tracking_point_index:]

            valid_idxs = [i for i, pt in enumerate(all_pts) if pt is not None]
            if len(valid_idxs) < 2:
                print('Not enough valid pixels in the model, reinitializing...')
                self.reset()
                return False

            first_pt = all_pts[min(valid_idxs)]
            last_pt = all_pts[max(valid_idxs)]
            first_px = np.array(self.camera.project3dToPixel(first_pt))
            last_px = np.array(self.camera.project3dToPixel(last_pt))

            if np.linalg.norm(first_pt - last_pt) < 0.025 or np.linalg.norm(first_px - last_px) < 30:
                print('Model looks too squished in! Reinitializing')
                self.reset()
                return False

            move_vec = last_px - first_px
            move_vec = move_vec / np.linalg.norm(move_vec)
            print('Movement vector: {:.2f}, {:.2f} ({}, {})'.format(*move_vec, 'RIGHT' if move_vec[0] > 0 else 'LEFT', 'DOWN' if move_vec[1] > 0 else 'UP'))


            # Background analysis - we look at the range of Z values for existing points and filter out points with Z values that deviate too much
            zs = np.array([pt[2] for pt in current_pts if pt is not None])
            if not zs.size:
                print('No valid pixels based on Z value, reinitializing model...')
                self.reset()
                return False

            q_25 = np.quantile(zs, 0.25)
            q_75 = np.quantile(zs, 0.75)
            iqr = q_75 - q_25
            exp_z_low = q_25 - 0.5 * iqr - self.bg_z_allowance.value
            exp_z_high = q_75 + 0.5 * iqr + self.bg_z_allowance.value

            abs_idxs = []
            valid_pxs = []
            valid_3d_pts = []

            last_px = None
            for i, pt in enumerate(current_pts):
                if pt is not None and exp_z_low < pt[2] < exp_z_high:
                    px = self.camera.project3dToPixel(pt)
                    # Make sure the pixel path does not reverse against the image movement
                    if last_px is not None and move_vec.dot(px - last_px) < 0:
                        continue
                    abs_idxs.append(i + self.current_tracking_point_index)
                    valid_3d_pts.append(pt)
                    valid_pxs.append(px)

            if not valid_pxs:
                print('No valid pixels, reinitializing model...')
                self.reset()
                return False

            abs_idxs = np.array(abs_idxs)
            valid_pxs = np.array(valid_pxs)
            valid_3d_pts = np.array(valid_3d_pts)

            # Figure out which submask has the most existing points projected into it
            labels = label(mask)
            pxs_int = valid_pxs.astype(int)
            pxs_int = pxs_int[(pxs_int[:,0] >= 0) & (pxs_int[:,0] < self.camera.width) & (pxs_int[:,1] >= 0) & (pxs_int[:,1] < self.camera.height)]

            if not pxs_int.size:
                print('All pxs were outside the image')
                self.last_mask_info = None
                self.publish_diagnostic_image(rgb, mask, inv_tf)
                return False

            label_list, counts = np.unique(labels[pxs_int[:,1], pxs_int[:,0]], return_counts=True)
            most_freq_label = label_list[np.argmax(counts)]

            if most_freq_label == 0:
                print('Most points were projected into the BG! Not processing')
                self.all_bg_counter += 1
                if self.all_bg_counter >= self.all_bg_retries.value:
                    print('It looks like the model is lost! Resetting...')
                    self.reset()
                self.last_mask_info = None
                self.publish_diagnostic_image(rgb, mask, inv_tf)
                return False

            self.all_bg_counter = 0
            submask = labels == most_freq_label
            detection = BezierBasedDetection(submask)
            curve = detection.fit(vec=move_vec, trim=pad)

            px_thres = self.px_consistency_threshold.value
            consistent_thres = self.consistency_threshold.value

            # Compute the consistency of the current model against the curve
            # Consistency is simply how many of the reprojected points lie close to the predicted curve
            dists, ts = curve.query_pt_distance(valid_pxs)
            pt_consistent = dists < px_thres
            consistent = pt_consistent.mean() > consistent_thres

            if not consistent:
                print('Mask is inconsistent with current model. Not updating curve')
                self.last_mask_info = None
                self.publish_diagnostic_image(rgb, mask, inv_tf, submask, detection.skel, curve)
                return False

            print('Model was deemed to be consistent! Updating curve')
            # Check which pixels at the start of the model are in the padding region
            # Remove them from tracking by updating the tracking point index

            if valid_pxs.size < 2:
                # All tracked pixels are invalid - Very unusual?
                print('All tracked pixels are invalid - This is highly unlikely')
                self.current_tracking_point_index = len(self.current_model)
            else:
                abs_idx = self.current_tracking_point_index
                chop_front = 0
                for chop_front, (px, abs_idx) in enumerate(zip(valid_pxs, abs_idxs)):
                    if not self.is_in_padding_region(px):
                        break
                self.current_tracking_point_index = abs_idx

            # No valid pixels to track - Essentially, has finished scanning the existing branch
            if len(self.current_model) == self.current_tracking_point_index:
                print("Branch scan appears to be done!")
                self.last_mask_info = None
                self.publish_diagnostic_image(rgb, mask, inv_tf, submask, detection.skel, curve)
                return True

            abs_idxs = abs_idxs[chop_front:]
            valid_pxs = valid_pxs[chop_front:]
            valid_3d_pts = valid_3d_pts[chop_front:]
            ts = ts[chop_front:]
            pt_consistent = pt_consistent[chop_front:]

            # Find the index of the last consistent pixel and chop the model to that value
            rel_idx_max = np.max(np.where(pt_consistent))
            abs_idx_max = abs_idxs[rel_idx_max]
            self.current_model = self.current_model[:abs_idx_max + 1]
            max_consistent_t = ts[rel_idx_max]

            # Initialize the new set of points to track
            n_existing = len(self.current_model) - self.current_tracking_point_index
            t_interp = interp1d(abs_idxs, ts)
            for rel_idx in range(n_existing):

                abs_idx = rel_idx + self.current_tracking_point_index
                if abs_idx not in abs_idxs:
                    # This point was deemed to be invalid and in need of reinitialization
                    assert rel_idx != 0 and rel_idx != n_existing - 1
                    self.current_model[abs_idx].clear()
                    px = curve(float(t_interp(abs_idx)))
                else:
                    sub_idx = (abs_idxs == abs_idx).argmax()
                    px = valid_pxs[sub_idx]

                track_px.append(Point2D(x=px[0], y=px[1]))

            # Add new points by looking ahead on the computed curve from the most recent consistent point
            start_t = max_consistent_t + self.curve_spacing.value
            for i_new, t in enumerate(np.arange(start_t, 1.0, self.curve_spacing.value), start=1):
                px = curve(t)
                px_int = px.astype(int)
                if 0 <= px_int[0] < self.camera.width and 0 <= px_int[1] < self.camera.height and submask[px_int[1], px_int[0]]:
                    self.current_model.append(PointHistory(max_error=self.recon_err_thres.value))
                    track_px.append(Point2D(x=px[0], y=px[1]))
                else:
                    break

        print('Finished processing!')
        self.publish_diagnostic_image(rgb, mask, inv_tf, submask, detection.skel, curve, track_px)
        self.publish_curve(stamp)

        s, ns = self.get_clock().now().seconds_nanoseconds()
        self.last_sent_request = '{}_{}_{}'.format(self.tracking_name, s, ns)
        print('Creating new request {}'.format(self.last_sent_request))
        req = TrackedPointRequest()
        req.header.stamp = stamp
        req.action = TrackedPointRequest.ACTION_REPLACE
        req.groups.append(TrackedPointGroup(name=self.last_sent_request, points=track_px))
        req.image = rgb_msg
        self.point_tracking_pub.publish(req)

        self.last_mask_info = None
        self.points_received = False
        return True


    def publish_curve(self, time=None):

        if not self.current_model:
            return

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

    def update(self):
        if not self.active:
            return

        if not self.camera.tf_frame:
            return

        pos = self.get_camera_frame_pose(position_only=True)
        if not self.current_model or np.linalg.norm(pos - self.last_pos) > self.mask_update_threshold.value:
            with self.processing_lock:
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

    def publish_diagnostic_image(self, rgb, mask, inv_tf, submask=None, skel=None, curve=None, track_2d=None):

        mask_img = np.dstack([mask * 255] * 3)
        submask_img = np.zeros(rgb.shape)
        if submask is not None:
            submask_img[submask] = [0, 255, 0]

        diag_img = 0.3 * rgb + 0.35 * mask_img + 0.35 * submask_img

        pxs = []
        for pt_hist in self.current_model:
            pt = pt_hist.as_point(inv_tf)
            if pt is not None:
                pxs.append(self.camera.project3dToPixel(pt))

        pxs = np.array(pxs).astype(int)
        cv2.polylines(diag_img, [pxs.reshape((-1, 1, 2))], False, (255, 0, 0), 5)
        for px in pxs:
            diag_img = cv2.circle(diag_img, px, 7, (0, 0, 255), -1)

        if curve is not None:
            eval_pts = curve(np.linspace(0, 1, 200)).astype(int)
            cv2.polylines(diag_img, [eval_pts.reshape((-1, 1, 2))], False, (0, 0, 200), 3)

        if skel is not None:
            diag_img[skel] = [255, 255, 0]

        if track_2d is not None:
            if isinstance(track_2d[0], Point2D):
                track_2d = [[p.x, p.y] for p in track_2d]

            track_2d = np.array(track_2d).astype(int)
            for i, pt in enumerate(track_2d, start=1 + self.current_tracking_point_index):
                diag_img = cv2.circle(diag_img, pt, 6, (0, 255, 0), -1)
                diag_img = cv2.putText(diag_img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
                diag_img = cv2.putText(diag_img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        img_msg = bridge.cv2_to_imgmsg(diag_img.astype(np.uint8), encoding='rgb8')
        print('Published diag!')
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
