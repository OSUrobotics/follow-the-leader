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
from follow_the_leader.networks.pips_model import PipsTracker
from follow_the_leader_msgs.msg import Point2D, TrackedPointGroup, TrackedPointRequest, Tracked3DPointGroup, Tracked3DPointResponse
from collections import defaultdict
from threading import Event
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
bridge = CvBridge()

class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')

        # Point tracking params
        self.fwd_speed = 0.10                   # Forward Z-component of visual servoing
        self.max_speed = 0.20                   # Maximum magnitude of the final velocity vector
        self.k_img = 1.0                        # How quickly should the camera try to align the image target against the prediction?
        self.max_reproj_ignore_threshold = 2.0  # If the max reprojection error exceeds this, we ignore the estimate
        self.stop_dist = 0.01                   # Terminate servoing when within the distance


        # Point tracking state variables
        self.active = False
        self.image_target_pixel = None
        self.current_px_estimate = None
        self.current_3d_estimate = None

        # ROS2 utils
        self.cb_group = ReentrantCallbackGroup()
        self.point_tracking_name = 'vs'
        self.point_tracking_sub = self.create_subscription(Tracked3DPointResponse, '/point_tracking_response',
                                                           self.handle_point_tracking_response, 1,
                                                           callback_group=self.cb_group)
        self.timer = self.create_timer(0.01, self.send_servo_command)


    def handle_point_tracking_response(self, msg: Tracked3DPointResponse):
        pass

    def send_servo_command(self):
        pass



def main(args=None):
    try:
        rclpy.init(args=args)
        executor = MultiThreadedExecutor()
        node = VisualServoingNode()
        rclpy.spin(node, executor=executor)
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()