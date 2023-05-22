import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from sensor_msgs.msg import Image
from follow_the_leader.networks.flowgan import FlowGAN
from cv_bridge import CvBridge
bridge = CvBridge()


class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        self.pub = self.create_publisher(Image, 'image_mask', 10)
        self.sub = self.sub = self.create_subscription(Image, '/camera/color/image_rect_raw', self.image_callback, 1)
        self.image_processor = None
        self.just_activated = False
        self.last_image = None

        # TODO: Retrieve this param properly
        self.movement_threshold = 0.0075
        self.base_frame = 'base_link'
        self.tool_frame = 'tool0'
        self.tf_buffer = None
        self.tf_listener = None
        self.last_pos = None
        if self.movement_threshold:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

    def load_image_processor(self, w, h):

        size = (w, h)
        self.image_processor = FlowGAN(size, size, use_flow=True, gan_name='synthetic_flow_pix2pix', gan_input_channels=6,
                                       gan_output_channels=1)

    def image_callback(self, msg):

        self.last_image = msg
        if self.image_processor is None:
            self.load_image_processor(msg.width, msg.height)

        if self.movement_threshold:
            try:
                tf = self.tf_buffer.lookup_transform(self.tool_frame, self.base_frame, rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().warn('Received TF Exception: {}'.format(ex))
                return

            t = tf.transform.translation
            pos = np.array([t.x, t.y, t.z])
            if self.last_pos is None or np.linalg.norm(pos - self.last_pos) > self.movement_threshold:
                self.last_pos = pos
            else:
                return

        img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        mask = self.image_processor.process(img).mean(axis=2).astype(np.uint8)
        if self.just_activated:
            self.just_activated = False
            return

        mask_msg = bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header.stamp = self.get_clock().now().to_msg()
        print('Should have published!')
        self.pub.publish(mask_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()