import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from pyba.pyba import CameraNetwork
from follow_the_leader.networks.pips_model import PipsTracker
bridge = CvBridge()


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
        self.camera = PinholeCameraModel()
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.set_camera_info, 1)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_rect_raw', self.handle_image_callback, 1)
        self.current_request = None
        self.image_queue = RotatingQueue(size=8)
        self.tracker = PipsTracker()

        # TODO: Retrieve this param properly
        self.movement_threshold = 0.05 / 8
        self.base_frame = 'base_link'
        self.tool_frame = 'tool0'
        self.last_pos = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def set_camera_info(self, msg):
        self.camera.fromCameraInfo(msg)
        self.cam_info_sub.destroy()

    def handle_image_callback(self, msg):
        if self.camera.tf_frame is None or self.current_request is None:
            return

        current_pos = self.get_tool_pose(position_only=True)
        if not (self.last_pos is None or np.linalg.norm(current_pos - self.last_pos) < self.movement_threshold):
            return

        info = {}
        info['image'] = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        info['pose'] = self.get_tool_pose(time=msg.header.stamp, position_only=False)
        self.image_queue.append(info)
        self.update_tracker(stamp=msg.header.stamp)

    def update_tracker(self, stamp=None):
        if not self.image_queue.is_full:
            return

        all_info = self.image_queue.as_list()

        # TODO: Organize and feed in images

        trajs = self.tracker.track_points(targets, imgs)
        trajs = trajs.reshape((trajs.shape[0], 1, *trajs.shape[1:]))



        pose_t_w = np.linalg.inv(all_info[-1]['pose'])

        # TODO: Check if this is correct
        camera_frame_tf_matrices = [(pose_t_w @ info['pose']) for info in all_info]
        distort = self.camera.distortionCoeffs()
        cams = {i: {
            'R': tf[:3,:3],
            'tvec': tf[:,3],
            'intr': self.camera.K,
            'distort': distort,

        } for i, tf in enumerate(camera_frame_tf_matrices)}

        cam_net = CameraNetwork(points2d=trajs, calib=cams)
        pts_3d = cam_net.points3d

        # TODO: Publish 3D points

        # TODO: Remove any tracked points that are now outside of the image

        # TODO: If we are using fixed frame tracking, update the request to use the starting positions

    def reset(self, *_, **__):
        self.current_request = None
        self.image_queue.empty()


    def get_tool_pose(self, time=None, position_only=False):
        try:
            tf = self.tf_buffer.lookup_transform(self.tool_frame, self.base_frame, time or rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn('Received TF Exception: {}'.format(ex))
            return

        tl = tf.transform.translation
        if position_only:
            return np.array([tl.x, tl.y, tl.z])
        q = tf.transform.rotation
        mat = np.ones(4)
        mat[:3,3] = [tl.x, tl.y, tl.z]
        mat[:3,3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        return mat


def main(args=None):
    rclpy.init(args=args)
    node = PointTracker()
    rclpy.spin(node)


if __name__ == '__main__':
    main()