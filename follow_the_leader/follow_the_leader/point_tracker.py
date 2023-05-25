import os.path

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from geometry_msgs.msg import Point
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from pyba.pyba import CameraNetwork
from follow_the_leader.networks.pips_model import PipsTracker
from follow_the_leader_msgs.msg import Point2D, TrackedPointGroup, TrackedPointRequest, Tracked3DPointGroup, Tracked3DPointResponse
from collections import defaultdict
from threading import Event
import asyncio
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
bridge = CvBridge()


def wait_for_future_synced(future):
    event = Event()
    def done_callback(_):
        nonlocal event
        event.set()
    future.add_done_callback(done_callback)
    event.wait()
    resp = future.result()
    return resp

class RotatingQueue:
    def __init__(self, size=8):
        self.queue = [None] * size
        self.idx = 0

    def empty(self):
        self.queue = [None] * len(self.queue)
        self.idx = 0

    @property
    def is_full(self):
        return not None in self.queue

    def append(self, item):
        self.queue[self.idx] = item
        self.idx = (self.idx + 1) % len(self.queue)

    def as_list(self):
        if not self.is_full:
            return self.queue[:self.idx]
        else:
            return self.queue[self.idx:] + self.queue[:self.idx]


class PointTracker(Node):
    def __init__(self):
        super().__init__('point_tracker_node')

        # State variables
        self.current_request = {}
        self.image_queue = RotatingQueue(size=8)
        self.tracker = PipsTracker(
            model_dir=os.path.join(os.path.expanduser('~'), 'repos', 'pips', 'pips', 'reference_model'))
        self.last_pos = None

        # Config
        # TODO: Retrieve configuration params properly
        self.movement_threshold = 0
        self.movement_threshold = 0.05 / 8
        self.base_frame = 'base_link'
        self.tool_frame = 'tool0'
        # self.do_3d_point_estimation = False
        self.do_3d_point_estimation = True

        # ROS Utils
        self.custom_callback = MutuallyExclusiveCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera = PinholeCameraModel()
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.set_camera_info, 1)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_rect_raw', self.handle_image_callback, 1, callback_group=self.custom_callback)
        self.tracking_request_sub = self.create_subscription(TrackedPointRequest, '/point_tracking_request', self.handle_tracking_request, 10)
        self.tracked_3d_pub = self.create_publisher(Tracked3DPointResponse, '/point_tracking_response', 1)
        self.pc_pub = self.create_publisher(PointCloud2, '/point_tracking_response_pc', 1)

    def set_camera_info(self, msg):
        self.camera.fromCameraInfo(msg)
        # self.cam_info_sub.destroy()

    def handle_tracking_request(self, msg: TrackedPointRequest):
        groups = msg.groups
        if msg.action == TrackedPointRequest.ACTION_REMOVE:
            for group in groups:
                self.current_request.pop(group.name, None)
            return

        self.current_request = {}
        for group in groups:
            self.current_request[group.name] = np.array([[pt.x, pt.y] for pt in group.points])

        self.image_queue.empty()
        print('Received request')

    def flatten_current_request(self):
        all_pts = []
        all_names = []
        for name, points in self.current_request.items():
            all_pts.append(points)
            all_names.extend([name] * len(points))
        return np.concatenate(all_pts, axis=0), all_names

    def handle_image_callback(self, msg):
        if self.camera.tf_frame is None or not self.current_request:
            return

        current_pos = self.get_camera_frame_pose(position_only=True)
        if self.movement_threshold and (not (self.last_pos is None or np.linalg.norm(current_pos - self.last_pos) > self.movement_threshold)):
            return
        print('Movement threshold passed')

        self.last_pos = current_pos
        image_time_pose = None
        if self.do_3d_point_estimation:
            image_time_pose = self.get_camera_frame_pose(time=msg.header.stamp, position_only=False)

        info = {}
        info['stamp'] = msg.header.stamp
        info['frame_id'] = msg.header.frame_id
        info['image'] = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        info['pose'] = image_time_pose
        self.image_queue.append(info)
        self.update_tracker(last_image=msg)

    def update_tracker(self, last_image=None):
        if not self.image_queue.is_full:
            return

        print('Updating tracker')

        all_info = self.image_queue.as_list()
        images = [info['image'] for info in all_info]
        targets, groups = self.flatten_current_request()
        trajs = self.tracker.track_points(targets, images)

        pts_3d = None
        if self.do_3d_point_estimation:
            pose_t_w = np.linalg.inv(all_info[-1]['pose'])
            # TODO: The poses now are using the Tool TF, which they should be using the camera frame TF
            # But for some reason the camera TF frame is yielding bad 3D estimates; figure out why
            camera_frame_tf_matrices = [(pose_t_w @ info['pose']) for info in all_info]
            distort = np.array(self.camera.distortionCoeffs())[:,0]
            cams = {i: {
                'R': tf[:3,:3],
                'tvec': tf[:3,3],
                'intr': self.camera.K,
                'distort': distort,

            } for i, tf in enumerate(camera_frame_tf_matrices)}

            cam_net = CameraNetwork(points2d=trajs.reshape((trajs.shape[0], 1, *trajs.shape[1:])).copy(), calib=cams)
            pts_3d = cam_net.points3d[0]

        import pdb
        pdb.set_trace()

        frame_id = all_info[-1]['frame_id']
        stamp = all_info[-1]['stamp']

        response = Tracked3DPointResponse(header=Header(frame_id=frame_id, stamp=stamp))
        if pts_3d is not None:
            pc = create_cloud_xyz32(Header(frame_id=frame_id, stamp=stamp), points=pts_3d)
            self.pc_pub.publish(pc)
            for group, points in self.unflatten_tracked_points(pts_3d, groups).items():
                response.groups.append(Tracked3DPointGroup(name=group, points=[Point(x=x, y=y, z=z) for x, y, z in points]))

        for group, points_2d in self.unflatten_tracked_points(trajs[-1].astype(np.float), groups).items():
            response.groups_2d.append(TrackedPointGroup(name=group, points=[Point2D(x=x, y=y) for x,y in points_2d]))
        if last_image is not None:
            response.image = last_image


        self.tracked_3d_pub.publish(response)
        self.update_request_from_trajectory(trajs, groups)


    def unflatten_tracked_points(self, points, groups):

        rez = defaultdict(list)
        for point, group in zip(points, groups):
            rez[group].append(point)

        return rez

    def update_request_from_trajectory(self, trajs, groups):
        w = self.camera.width
        h = self.camera.height
        final_locs = trajs[-1]
        is_outside = (final_locs[:,0] < 0) | (final_locs[:,0] >= w) | (final_locs[:,1] < 0) | (final_locs[:,1] >= h)
        idx_to_stay = np.where(~is_outside)[0]
        new_req = {}

        update_locs = trajs[1]
        for idx in idx_to_stay:
            group = groups[idx]
            if group not in new_req:
                new_req[group] = []
            new_req[group].append(update_locs[idx])

        self.current_request = {group: np.array(points) for group, points in new_req.items()}


    def reset(self, *_, **__):
        self.current_request = {}
        self.image_queue.empty()

    def get_camera_frame_pose(self, time=None, position_only=False):

        time = time or rclpy.time.Time()
        # future = self.tf_buffer.wait_for_transform_async(self.base_frame, self.camera.tf_frame, time)
        future = self.tf_buffer.wait_for_transform_async(self.base_frame, self.tool_frame, time)
        wait_for_future_synced(future)

        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, time)
        except TransformException as ex:
            self.get_logger().warn('Received TF Exception: {}'.format(ex))
            return

        tl = tf.transform.translation
        if position_only:
            return np.array([tl.x, tl.y, tl.z])
        q = tf.transform.rotation
        mat = np.identity(4)
        mat[:3,3] = [tl.x, tl.y, tl.z]
        mat[:3,:3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        return mat


def main(args=None):
    try:
        rclpy.init(args=args)
        executor = MultiThreadedExecutor()
        node = PointTracker()
        rclpy.spin(node, executor=executor)
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()