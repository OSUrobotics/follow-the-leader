#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import (
    TwistStamped,
    Vector3,
    Vector3Stamped,
    Transform,
    TransformStamped,
    Point,
    Pose,
    PoseStamped,
)
import numpy as np
from follow_the_leader.curve_fitting import BezierBasedDetection
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from image_geometry import PinholeCameraModel
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_vector3
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import SwitchController


class FollowTheLeaderController_ROS(Node):
    def __init__(self):
        super().__init__("ftl_controller")
        # Config

        self.base_frame = self.declare_parameter("base_frame")
        self.tool_frame = self.declare_parameter("tool_frame")
        self.base_ctrl = self.declare_parameter("base_controller")
        self.servo_ctrl = self.declare_parameter("servo_controller")
        self.min_height = self.declare_parameter("min_height")
        self.max_height = self.declare_parameter("max_height")
        self.ee_speed = self.declare_parameter("ee_speed")
        self.k_centering = self.declare_parameter("k_centering")
        self.cam_info_topic = self.declare_parameter("cam_info_topic")
        self.publish_diagnostic = True

        # ROS2-based utility setup
        self.service_handler_group = ReentrantCallbackGroup()
        self.img_subscriber_group = ReentrantCallbackGroup()

        self.pinhole_camera = PinholeCameraModel()
        self.cam_info_sub = self.create_subscription(
            CameraInfo, self.cam_info_topic.value, self.update_pinhole_camera, 1
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.srv_start = self.create_service(
            Trigger, "servo_start", self.start, callback_group=self.service_handler_group
        )
        self.srv_stop = self.create_service(Trigger, "servo_stop", self.stop, callback_group=self.service_handler_group)

        self.enable_servo = self.create_client(
            Trigger, "/servo_node/start_servo", callback_group=self.service_handler_group
        )
        self.disable_servo = self.create_client(
            Trigger, "/servo_node/stop_servo", callback_group=self.service_handler_group
        )
        self.switch_ctrl = self.create_client(
            SwitchController, "/controller_manager/switch_controller", callback_group=self.service_handler_group
        )

        # State vars
        self.active = False
        self.up = False
        self.default_action = None
        self.last_action = None

        self.mask_sub = self.create_subscription(
            Image, "/image_mask", self.process_mask, 1, callback_group=self.img_subscriber_group
        )
        self.pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.diagnostic_pub = self.create_publisher(Image, "/diagnostic", 10)
        self.timer = self.create_timer(0.01, self.twist_callback)
        self.reset_state()
        print("Done loading")
        return
    

    def update_pinhole_camera(self, msg):
        self.pinhole_camera.fromCameraInfo(msg)
        self.destroy_subscription(self.cam_info_sub)
        return
    

    def reset_state(self):
        self.active = False
        self.default_action = None
        self.up = False
        self.last_action = None
        return
    

    def start(self, _, resp):
        if self.active:
            resp.success = False
            resp.message = "Servoing is already active!"
            return resp

        # Initialize movement based on the current location of the arm
        pos = self.get_tool_pose(as_array=True)[:3]
        z = pos[2]
        lower_dist = z - self.min_height.value
        upper_dist = self.max_height.value - z

        self.up = upper_dist > lower_dist
        self.default_action = np.array([0, -1]) if self.up else np.array([0, 1])

        switch_ctrl_req = SwitchController.Request(
            start_controllers=[self.servo_ctrl.value], stop_controllers=[self.base_ctrl.value]
        )
        self.enable_servo.call(Trigger.Request())
        self.switch_ctrl.call(switch_ctrl_req)
        self.active = True

        resp.success = True
        resp.message = "Servoing started!"
        print(resp)
        return resp
    

    def stop(self, *args):
        self.reset_state()
        switch_ctrl_req = SwitchController.Request(
            start_controllers=[self.base_ctrl.value], stop_controllers=[self.servo_ctrl.value]
        )
        self.disable_servo.call(Trigger.Request())
        self.switch_ctrl.call(switch_ctrl_req)
        msg = "Servoing stopped!"
        print(msg)

        if args:
            _, resp = args
            resp.success = True
            resp.message = msg
            return resp
        return
    

    def process_mask(self, msg):
        if not self.active:
            return

        img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        x_mid = img.shape[1] / 2
        y_mid = img.shape[0] / 2

        detection = BezierBasedDetection(img > 128)
        curve = detection.fit(vec=self.default_action, trim=30)
        ts = np.linspace(0, 1, 50)
        pts = curve(ts)
        center_idx = np.argmin(np.abs(pts[:, 1] - y_mid))
        center_pt = pts[center_idx]
        diff = (center_pt[0] - x_mid) / center_pt[0]

        tangent = curve.tangent(ts[center_idx])
        tangent = tangent / np.linalg.norm(tangent)
        centering = self.k_centering.value * diff * np.array([1, 0])

        self.last_action = tangent + centering

        if self.publish_diagnostic:
            arrow_base_len = 50
            diag = np.dstack([img] * 3).astype(np.uint8)
            diag[:, int(x_mid - 1) : int(x_mid + 2)] = (128, 128, 128)
            cv2.polylines(diag, [pts.reshape((-1, 1, 2)).astype(int)], False, (0, 0, 128), 2)
            diag[detection.skel] = (0, 0, 255)
            cint = center_pt.astype(int)
            diag = cv2.arrowedLine(diag, cint, (center_pt + tangent * arrow_base_len).astype(int), (0, 255, 0), 8)
            diag = cv2.arrowedLine(diag, cint, (center_pt + centering * arrow_base_len).astype(int), (255, 0, 0), 8)
            diag = cv2.arrowedLine(
                diag,
                cint,
                (center_pt + arrow_base_len * self.last_action / np.linalg.norm(self.last_action)).astype(int),
                (255, 255, 0),
                8,
            )
            diag = cv2.circle(diag, cint, 4, (128, 128, 128), thickness=3)

            img_msg = bridge.cv2_to_imgmsg(diag, encoding="rgb8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.diagnostic_pub.publish(img_msg)

        print("[DEBUG] Processed mask!")
        return
    

    def twist_callback(self):
        if not self.active:
            return

        if not self.pinhole_camera.tf_frame:
            print("No camera frame has been received!")
            return

        # Check for termination
        pos = self.get_tool_pose(as_array=True)[:3]
        if self.up:
            print("[DEBUG] Moving up, cur Z = {:.2f}, max Z = {:.2f}".format(pos[2], self.max_height.value))
        else:
            print("[DEBUG] Moving down, cur Z = {:.2f}, max Z = {:.2f}".format(pos[2], self.min_height.value))

        if (self.up and pos[2] >= self.max_height.value) or (not self.up and pos[2] <= self.min_height.value):
            self.stop()
            return

        # Convert the pixel action to a corresponding EE-frame action
        action = self.last_action if self.last_action is not None else self.default_action
        vel = self.ee_speed.value * action / np.linalg.norm(action)

        vec = Vector3Stamped()
        vec.header.frame_id = self.pinhole_camera.tf_frame
        vec.vector = Vector3(x=vel[0], y=vel[1], z=0.0)
        tool_tf = self.tf_buffer.lookup_transform(
            target_frame=self.tool_frame.value, source_frame=self.pinhole_camera.tf_frame, time=rclpy.time.Time()
        )
        # tool_tf = self.tf_buffer.lookup_transform(target_frame=self.pinhole_camera.tf_frame, source_frame=self.tool_frame, time=rclpy.time.Time())
        tool_frame_vec = do_transform_vector3(vec, tool_tf)

        cmd = TwistStamped()
        cmd.header.frame_id = self.tool_frame.value
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear = tool_frame_vec.vector

        self.pub.publish(cmd)
        return

        # print('[DEBUG] Sent vel command: {:.3f}, {:.3f}'.format(*vel))
        # t = tool_frame_vec.vector
        # print('[DEBUG] TFed command in tool frame: {:.3f}, {:.3f}, {:.3f}'.format(t.x, t.y, t.z))

    def get_tool_pose(self, time=None, as_array=True):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame.value, self.tool_frame.value, time or rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn("Received TF Exception: {}".format(ex))
            return
        pose = convert_tf_to_pose(tf)
        if as_array:
            p = pose.pose.position
            o = pose.pose.orientation
            return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w])
        else:
            return pose


def convert_tf_to_pose(tf: TransformStamped):
    pose = PoseStamped()
    pose.header = tf.header
    tl = tf.transform.translation
    pose.pose.position = Point(x=tl.x, y=tl.y, z=tl.z)
    pose.pose.orientation = tf.transform.rotation

    return pose


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    ctrl = FollowTheLeaderController_ROS()
    rclpy.spin(ctrl, executor)
    return

if __name__ == "__main__":
    main()

# ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 robot_ip:=169.254.57.122 launch_rviz:=true use_fake_hardware:=true initial_joint_controller:=joint_trajectory_controller
# ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_fake_hardware:=true
