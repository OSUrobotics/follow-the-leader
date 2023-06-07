import os.path

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_vector3
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from geometry_msgs.msg import Point, TwistStamped, Vector3, Vector3Stamped
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from follow_the_leader.networks.pips_model import PipsTracker
from follow_the_leader_msgs.msg import Point2D, TrackedPointGroup, TrackedPointRequest, Tracked3DPointGroup, Tracked3DPointResponse, VisualServoingRequest
from collections import defaultdict
from threading import Event
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
bridge = CvBridge()

class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')

        # Point tracking params
        self.fwd_speed = self.declare_parameter('forward_speed', 0.15)
        self.max_speed = self.declare_parameter('max_speed', 0.20)
        self.k_img = self.declare_parameter('k_img', 1.0)
        self.max_reproj_ignore_threshold = self.declare_parameter('reprojection_error_ignore', 2.0)
        self.stop_dist = self.declare_parameter('stop_dist', 0.15)
        self.cam_info_topic = self.declare_parameter('cam_info_topic', '/camera/color/camera_info')
        self.tool_frame = self.declare_parameter('tool_frame', 'tool0')
        self.target_frame = self.declare_parameter('target_frame', None)
        self.no_3d_est_scale = self.declare_parameter('no_3d_est_scale', 0.2)

        # Point tracking state variables
        self.active = False
        self.image_target = None
        self.current_px_estimate = None
        self.current_3d_estimate = None

        # ROS2 utils
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera = PinholeCameraModel()
        self.cb_group = ReentrantCallbackGroup()
        self.camera_sub = self.create_subscription(CameraInfo, self.cam_info_topic.value, self.set_cam_info, 1,
                                                   callback_group=self.cb_group)
        self.point_tracking_name = 'vs'
        self.servoing_sub = self.create_subscription(VisualServoingRequest, '/visual_servoing_request',
                                                     self.handle_servoing_request, 1,
                                                     callback_group=self.cb_group)
        self.point_tracking_pub = self.create_publisher(TrackedPointRequest, 'point_tracking_request', 1)
        self.point_response_sub = self.create_subscription(Tracked3DPointResponse, 'point_tracking_response', self.handle_3d_point_tracking_response, 1)
        self.servo_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.timer = self.create_timer(0.01, self.send_servo_command)

    def set_cam_info(self, msg):
        self.camera.fromCameraInfo(msg)

    def handle_servoing_request(self, msg: VisualServoingRequest):

        print('Received servoing request!')

        self.image_target = np.array([msg.image_target.x, msg.image_target.y])
        self.h = msg.start_image.height
        self.w = msg.start_image.width

        req = TrackedPointRequest()
        req.action = TrackedPointRequest.ACTION_REPLACE
        req.groups.append(TrackedPointGroup(name=self.point_tracking_name, points=msg.points))
        self.point_tracking_pub.publish(req)

        self.active = True
        self.current_px_estimate = np.array([[msg.points[0].x, msg.points[0].y]])

    def reset(self):
        self.active = False
        self.image_target = None
        self.current_px_estimate = None
        self.current_3d_estimate = None

    def handle_3d_point_tracking_response(self, msg: Tracked3DPointResponse):
        for group in msg.groups_2d:
            if group.name == self.point_tracking_name:
                self.current_px_estimate = np.array([[p.x, p.y] for p in group.points])

        for group in msg.groups:
            if group.name == self.point_tracking_name:
                self.current_3d_estimate = np.array([[p.x, p.y, p.z] for p in group.points])
                print('Updated 3D est! Now {:.3f}, {:.3f}, {:.3f}'.format(*self.current_3d_estimate[0]))

    def send_servo_command(self):
        if not self.active:
            return

        if self.current_px_estimate is None:
            print('Warning! Visual servoing is active but there is no 2D estimate? This shouldn\'t be the case')
            return

        est_2d = self.current_px_estimate[0]
        offscreen = est_2d[0] < 0 or est_2d[1] < 0 or est_2d[0] > self.camera.width or est_2d[1] > self.camera.height

        if self.current_3d_estimate is None:
            print('Warning! No 3D estimate, and no safeguards to stop excessive servoing')
        else:
            est_3d = self.current_3d_estimate[0]
            if est_3d[2] <= self.stop_dist.value:
                print('Done')
                self.reset()
                return

        if self.target_frame.value is None:
            # Pure image-space visual servoing
            if offscreen:
                print('Target is offscreen, ending visual servoing!')
                self.reset()
                return

            base_vec = np.array(self.camera.projectPixelTo3dRay(est_2d))
            base_vec *= self.fwd_speed.value / base_vec[2]

            if self.current_3d_estimate is None:
                base_vec *= self.no_3d_est_scale.value

        else:
            # Base vector is derived from the 3D estimate and the target frame
            raise NotImplementedError()

        tracked_px = self.current_px_estimate[0]
        diff = (tracked_px - self.image_target) / np.array(
            [self.camera.width, self.camera.height])  # Error in each dim is in [-1, 1]

        img_diff_vec = np.array([diff[0], diff[1], 0]) * self.k_img.value

        final_vec = base_vec + img_diff_vec
        norm = np.linalg.norm(final_vec)
        if norm > self.max_speed.value:
            final_vec *= self.max_speed.value / norm

        vec = Vector3Stamped()
        vec.header.frame_id = self.camera.tf_frame
        vec.vector = Vector3(x=final_vec[0], y=final_vec[1], z=final_vec[2])
        tool_tf = self.tf_buffer.lookup_transform(target_frame=self.tool_frame.value,
                                                  source_frame=self.camera.tf_frame, time=rclpy.time.Time())
        tool_frame_vec = do_transform_vector3(vec, tool_tf)

        twist = TwistStamped()
        twist.header.frame_id = self.tool_frame.value
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.twist.linear = tool_frame_vec.vector
        self.servo_pub.publish(twist)


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