#!/usr/bin/env python

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import TwistStamped, Vector3, Vector3Stamped, Transform, TransformStamped, Point, Pose, PoseStamped
import numpy as np
from follow_the_leader.curve_fitting import BezierBasedDetection, Bezier
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()

from std_msgs.msg import Empty
from follow_the_leader.utils.ros_utils import TFNode
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_vector3, do_transform_point
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import SwitchController
from follow_the_leader_msgs.msg import PointList


class FollowTheLeaderController_3D_ROS(TFNode):

    def __init__(self):
        super().__init__('ftl_controller_3d', cam_info_topic='/camera/color/camera_info')
        # Config

        self.base_frame = self.declare_parameter('base_frame', 'base_link')
        self.tool_frame = self.declare_parameter('tool_frame', 'tool0')
        self.base_ctrl = self.declare_parameter('base_controller', 'scaled_joint_trajectory_controller')
        self.servo_ctrl = self.declare_parameter('servo_controller', 'forward_position_controller')
        self.min_height = self.declare_parameter('min_height', 0.325)
        self.max_height = self.declare_parameter('max_height', 0.55)
        self.ee_speed = self.declare_parameter('ee_speed', 0.15)
        self.k_centering = self.declare_parameter('k_centering', 1.0)
        self.k_z = self.declare_parameter('k_z', 1.0)
        self.z_desired = self.declare_parameter('z_desired', 0.20)
        self.publish_diagnostic = True

        # State variables
        self.active = False
        self.up = False
        self.default_action = None
        self.last_curve_pts = None

        # ROS2 setup
        self.service_handler_group = ReentrantCallbackGroup()
        self.curve_subscriber_group = ReentrantCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

        self.srv_start = self.create_service(Trigger, 'servo_3d_start', self.start, callback_group=self.service_handler_group)
        self.srv_stop = self.create_service(Trigger, 'servo_3d_stop', self.stop, callback_group=self.service_handler_group)
        self.enable_servo = self.create_client(Trigger, '/servo_node/start_servo', callback_group=self.service_handler_group)
        self.disable_servo = self.create_client(Trigger, '/servo_node/stop_servo', callback_group=self.service_handler_group)
        self.switch_ctrl = self.create_client(SwitchController, '/controller_manager/switch_controller', callback_group=self.service_handler_group)

        self.curve_sub = self.create_subscription(PointList, '/curve_3d', self.process_curve, 1, callback_group=self.curve_subscriber_group)
        self.pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.reset_model_pub = self.create_publisher(Empty, '/reset_model', 1)
        self.timer = self.create_timer(0.01, self.twist_callback)
        self.reset_state()
        print('Done loading')

    def reset_state(self):
        self.active = False
        self.default_action = None
        self.up = False
        self.last_curve_pts = None
        self.reset_model_pub.publish(Empty())

    def start(self, _, resp):

        print('Got request')
        if self.active:
            resp.success = False
            resp.message = 'Servoing is already active!'
            return resp

        # Initialize movement based on the current location of the arm
        pos = self.get_tool_pose(as_array=True)[:3]
        z = pos[2]
        lower_dist = z - self.min_height.value
        upper_dist = self.max_height.value - z

        self.up = upper_dist > lower_dist
        self.default_action = np.array([0, -1, 0]) if self.up else np.array([0, 1, 0])

        switch_ctrl_req = SwitchController.Request(start_controllers=[self.servo_ctrl.value], stop_controllers=[self.base_ctrl.value])
        self.enable_servo.call(Trigger.Request())
        self.switch_ctrl.call(switch_ctrl_req)
        self.active = True

        resp.success = True
        resp.message = 'Servoing started!'
        print(resp)
        return resp

    def stop(self, *args):
        self.reset_state()
        switch_ctrl_req = SwitchController.Request(start_controllers=[self.base_ctrl.value], stop_controllers=[self.servo_ctrl.value])
        self.disable_servo.call(Trigger.Request())
        self.switch_ctrl.call(switch_ctrl_req)
        msg = 'Servoing stopped!'
        print(msg)

        if args:
            _, resp = args
            resp.success = True
            resp.message = msg
            return resp

    def process_curve(self, msg: PointList):
        if not self.active:
            return

        if msg.header.frame_id != self.camera.tf_frame:
            print('Warning! The frame IDs of the 3D curve and camera did not match')
            return

        stamp = msg.header.stamp
        tf = self.lookup_transform(self.base_frame.value, msg.header.frame_id, time=stamp, as_matrix=True)
        curve_pts = np.array([[p.x, p.y, p.z] for p in msg.points])
        if not curve_pts.size:
            return

        curve_pts_base = self.mul_homog(tf, curve_pts)
        self.last_curve_pts = curve_pts_base
        print('Curve points processed!')

    def get_vel_from_curve(self):
        if self.last_curve_pts is None:
            return self.default_action

        time = self.get_clock().now()
        tf_mat = self.lookup_transform(self.camera.tf_frame, self.base_frame.value, time=time, as_matrix=True)
        curve_pts_optical = self.mul_homog(tf_mat, self.last_curve_pts)

        cx = self.camera.width / 2
        cy = self.camera.height / 2
        c = np.array([cx, cy])

        pts_to_consider = []
        for pt in curve_pts_optical:
            px = self.camera.project3dToPixel(pt)
            if 0 <= px[0] < self.camera.width and 0 <= px[1] < self.camera.height:
                pts_to_consider.append(pt)
        if not pts_to_consider:
            print('Received curve is entirely out of range of the image!')
            return None

        curve = Bezier.fit(pts_to_consider, 3)
        ts = np.linspace(0, 1, 101)
        eval_pts = curve(ts)
        eval_pxs = np.array([self.camera.project3dToPixel(pt) for pt in eval_pts])

        idx_c = np.argmin(np.abs(eval_pxs[:,1] - cy))
        t_c = ts[idx_c]
        pt_c = eval_pts[idx_c]
        px_c = eval_pxs[idx_c]

        print('[SELECTION]\n\tPx: {:.3f}, {:.3f}\n\tPt: {:.3f}, {:.3f}, {:.3f}'.format(*px_c, *pt_c))
        grad = curve.tangent(t_c)
        z_c = pt_c[2]     # in 3D space
        px_x_c = px_c[0]  # in pixel space

        grad = grad / np.linalg.norm(grad) * self.ee_speed.value
        x_diff = np.array([(px_x_c - cx) / cx, 0, 0])
        z_diff = np.array([0, 0, z_c - self.z_desired.value])
        final_vec = grad + x_diff * self.k_centering.value + z_diff * self.k_z.value
        final_vec = final_vec / np.linalg.norm(final_vec) * self.ee_speed.value

        return final_vec


    def twist_callback(self):
        if not self.active:
            return

        if not self.camera.tf_frame:
            print('No camera frame has been received!')
            return

        # Check for termination
        pos = self.get_tool_pose(as_array=True)[:3]
        if self.up:
            print('[DEBUG] Moving up, cur Z = {:.2f}, max Z = {:.2f}'.format(pos[2], self.max_height.value))
        else:
            print('[DEBUG] Moving down, cur Z = {:.2f}, max Z = {:.2f}'.format(pos[2], self.min_height.value))

        if (self.up and pos[2] >= self.max_height.value) or (not self.up and pos[2] <= self.min_height.value):
            self.stop()
            return

        # Convert the pixel action to a corresponding EE-frame action
        vel = self.get_vel_from_curve()
        if vel is None:
            self.stop()
            return

        vel = vel / np.linalg.norm(vel) * self.ee_speed.value

        vec = Vector3Stamped()
        vec.header.frame_id = self.camera.tf_frame
        vec.vector = Vector3(x=vel[0], y=vel[1], z=vel[2])
        tool_tf = self.lookup_transform(self.tool_frame.value, self.camera.tf_frame, time=rclpy.time.Time(), as_matrix=False)
        tool_frame_vec = do_transform_vector3(vec, tool_tf)

        cmd = TwistStamped()
        cmd.header.frame_id = self.tool_frame.value
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear = tool_frame_vec.vector

        self.pub.publish(cmd)

        print('[DEBUG] Sent vel command: {:.3f}, {:.3f}, {:.3f}'.format(*vel))
        t = tool_frame_vec.vector
        print('[DEBUG] TFed command in tool frame: {:.3f}, {:.3f}, {:.3f}'.format(t.x, t.y, t.z))

    def get_tool_pose(self, time=None, as_array=True):
        tf = self.lookup_transform(self.base_frame.value, self.tool_frame.value, time or rclpy.time.Time(), as_matrix=False)
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
    ctrl = FollowTheLeaderController_3D_ROS()
    rclpy.spin(ctrl, executor)




if __name__ == '__main__':
    main()

# ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 robot_ip:=169.254.57.122 launch_rviz:=true use_fake_hardware:=true initial_joint_controller:=joint_trajectory_controller
# ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_fake_hardware:=true
