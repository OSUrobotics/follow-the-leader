import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3
from follow_the_leader_msgs.msg import ImageMaskPair
from follow_the_leader.networks.flowgan import FlowGAN
from cv_bridge import CvBridge
from follow_the_leader.utils.ros_utils import TFNode
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Lock

bridge = CvBridge()


class ImageProcessorNode(TFNode):
    def __init__(self):
        super().__init__('image_processor_node', cam_info_topic='/camera/color/camera_info')

        self.cb = MutuallyExclusiveCallbackGroup()
        self.pub = self.create_publisher(Image, 'image_mask', 10)
        self.image_mask_pub = self.create_publisher(ImageMaskPair, 'image_mask_pair', 10)
        self.sub = self.sub = self.create_subscription(Image, '/camera/color/image_rect_raw', self.image_callback, 1,
                                                       callback_group=self.cb)
        self.image_processor = None
        self.just_activated = False
        self.last_image = None
        self.lock = Lock()

        # TODO: Retrieve this param properly
        self.movement_threshold = 0.0075
        self.base_frame = 'base_link'
        self.last_pos = None
        if self.movement_threshold:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

    def load_image_processor(self, force_size=None):
        with self.lock:
            if self.image_processor is None and (self.camera.tf_frame or force_size):
                if self.camera.tf_frame:
                    size = (self.camera.width, self.camera.height)
                else:
                    size = force_size
                self.image_processor = FlowGAN(size, size, use_flow=True, gan_name='synthetic_flow_pix2pix',
                                               gan_input_channels=6, gan_output_channels=1)

    def _handle_cam_info(self, msg: CameraInfo):
        super()._handle_cam_info(msg)
        self.load_image_processor()


    def image_callback(self, msg: Image):

        self.last_image = msg
        if self.image_processor is None:
            self.load_image_processor(msg.width, msg.height)

        vec = Vector3()
        if self.movement_threshold:
            tf_mat = self.lookup_transform(self.base_frame, self.camera.tf_frame, rclpy.time.Time(), as_matrix=True)
            pos = tf_mat[:3,3]
            if self.last_pos is None:
                self.last_pos = pos
            else:
                diff = pos - self.last_pos
                if np.linalg.norm(pos - self.last_pos) < self.movement_threshold:
                    return

                movement = np.linalg.inv(tf_mat[:3,:3]) @ diff
                movement /= np.linalg.norm(movement)
                vec = Vector3(x=movement[0], y=movement[1], z=movement[2])
                self.last_pos = pos

        img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        mask = self.image_processor.process(img).mean(axis=2).astype(np.uint8)
        if self.just_activated:
            self.just_activated = False
            return

        mask_msg = bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header.stamp = msg.header.stamp
        image_mask_pair = ImageMaskPair(rgb=msg, mask=mask_msg, image_frame_offset=vec)

        self.pub.publish(mask_msg)
        self.image_mask_pub.publish(image_mask_pair)


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = ImageProcessorNode()
    rclpy.spin(node, executor=executor)


if __name__ == '__main__':
    main()
