import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from follow_the_leader.networks.flowgan import FlowGAN
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge
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

        current_pos = self.get_tool_position()
        if not (self.last_pos is None or np.linalg.norm(current_pos - self.last_pos) < self.movement_threshold):
            return

        img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.image_queue.append(img)

        # Proc img with PIPs




    def get_tool_position(self, time=None):
        try:
            tf = self.tf_buffer.lookup_transform(self.tool_frame, self.base_frame, time or rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn('Received TF Exception: {}'.format(ex))
            return

        tl = tf.transform.translation
        return np.array([tl.x, tl.y, tl.z])


def main(args=None):
    rclpy.init(args=args)
    node = PointTracker()
    rclpy.spin(node)


if __name__ == '__main__':
    main()