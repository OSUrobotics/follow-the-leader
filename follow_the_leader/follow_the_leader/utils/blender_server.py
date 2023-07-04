import bpy
from mathutils import Matrix, Vector, Quaternion
from mathutils.geometry import interpolate_bezier

import os
import numpy as np
import cv2
import random
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, RegionOfInterest, Image
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
from follow_the_leader_msgs.msg import StateTransition, States
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class ImageServer(Node):
    def __init__(self):
        super().__init__('image_server')

        # ROS parameters
        self.base_frame = self.declare_parameter('base_frame', 'base_link')
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.render_size = self.declare_parameter('render_size', [320, 240])
        self.textures_location = self.declare_parameter('textures_location',
                                                        os.path.join(os.path.expanduser('~'), 'Pictures', 'textures'))
        self.spindle_dist = self.declare_parameter('spindle_dist', 0.20)
        self.num_side_branches = self.declare_parameter('num_side_branches', 2)
        self.side_branch_range = self.declare_parameter('side_branch_range', [0.0, 1.0])


        # State variables
        self.active = False
        self.last_loc = None

        # Blender environment init
        self.main_spindle = None
        self.main_spindle_eval = []
        self.side_branches = []
        self.initialize_environment()

        # ROS utils
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cb = ReentrantCallbackGroup()
        self.mutex_cb = MutuallyExclusiveCallbackGroup()
        self.cam_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 1)
        self.image_pub = self.create_publisher(Image, '/camera/color/image_rect_raw', 1)
        self.diagnostic_pub = self.create_publisher(MarkerArray, 'controller_diagnostic', 1)
        self.transition_sub = self.create_subscription(StateTransition, 'state_transition',
                                                       self.handle_state_transition, 1,
                                                       callback_group=self.cb)
        self.init_tree_spindle_srv = self.create_service(Trigger, '/initialize_tree_spindle',
                                                         self.randomize_tree_spindle)

        self.image_timer = self.create_timer(0.01, self.image_timer_callback, callback_group=self.mutex_cb)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def handle_state_transition(self, msg):
        self.active = msg.state_end != States.IDLE

    def timer_callback(self):

        now = self.get_clock().now().to_msg()
        self.cam_info.header.stamp = now
        self.cam_info_pub.publish(self.cam_info)

        markers = MarkerArray()
        if self.main_spindle is not None:

            # Main spindle
            marker = Marker()
            marker.header.frame_id = self.base_frame.value
            marker.header.stamp = now
            marker.ns = self.get_name()

            marker.id = 1
            marker.type = Marker.LINE_STRIP
            marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in self.main_spindle_eval]
            marker.scale.x = 0.01
            marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.5)
            markers.markers.append(marker)

            # Side branches
            all_pts = []
            for branch in self.side_branches:
                branch.rotation_mode = 'XYZ'
                loc = branch.location
                rot = branch.rotation_euler

                all_pts.append(Point(x=loc[0], y=loc[1], z=loc[2]))
                tip = rot.to_matrix() @ Vector([0,0,0.10]) + loc
                all_pts.append(Point(x=tip[0], y=tip[1], z=tip[2]))

            marker = Marker()
            marker.header.frame_id = self.base_frame.value
            marker.header.stamp = now
            marker.type = Marker.LINE_LIST
            marker.id = 2
            marker.points = all_pts
            marker.scale.x = 0.01
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5)
            markers.markers.append(marker)

        self.diagnostic_pub.publish(markers)

    def image_timer_callback(self):

        if not self.active:
            return

        tf = self.tf_buffer.lookup_transform(self.base_frame.value, self.camera_frame.value, rclpy.time.Time())
        stamp = tf.header.stamp
        tl = tf.transform.translation
        q = tf.transform.rotation
        pos = np.array([tl.x, tl.y, tl.z])
        if self.last_loc is not None and np.linalg.norm(pos - self.last_loc) < 1e-6:
            return
        self.last_loc = pos

        # Rotation needs to be inverted because Blender's up reference is flipped relative to CV conventions
        base_rot = Quaternion([q.w, q.x, q.y, q.z]).to_matrix()
        z_rot = Matrix([[-1,0,0],[0,-1,0],[0,0,1]])
        self.camera_obj.location = [tl.x, tl.y, tl.z]
        self.camera_obj.rotation_quaternion = (z_rot @ base_rot @ z_rot).to_quaternion()

        # There doesn't seem to be a way to retrieve image pixels directly, so it must be saved to a file first
        bpy.ops.render.render(write_still=True)
        img_array = cv2.imread(self.scene.render.filepath)
        img_msg = self.bridge.cv2_to_imgmsg(img_array, encoding='bgr8')
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.camera_frame.value

        self.image_pub.publish(img_msg)

    def initialize_environment(self):
        self.scene = bpy.data.scenes.new("Scene")
        bpy.context.window.scene = self.scene
        world = bpy.data.worlds.new("Base World")
        world.use_nodes = True
        self.scene.world = world

        self.camera_data = bpy.data.cameras.new("Camera")
        self.camera_data.lens_unit = 'FOV'
        self.camera_data.angle = np.radians(42)
        self.camera_obj = bpy.data.objects.new("Camera", self.camera_data)
        self.camera_obj.location = (0, -1.0, 1.0)
        self.camera_obj.rotation_mode = 'QUATERNION'

        self.scene.collection.objects.link(self.camera_obj)
        self.scene.camera = self.camera_obj

        render_size = self.render_size.value
        self.scene.render.resolution_x = render_size[0]
        self.scene.render.resolution_y = render_size[1]
        self.scene.render.filepath = os.path.join(os.path.expanduser('~'), '_temp.png')

        K = get_calibration_matrix_K_from_blender(self.camera_data, self.scene)
        P = np.zeros((3,4))
        P[:3,:3] = K

        self.cam_info = CameraInfo(
            height=render_size[1], width=render_size[0], distortion_model='plumb_bob', binning_x=0, binning_y=0,
            d=np.zeros(5), r=np.identity(3).flatten(), k=K.flatten(), p=P.flatten(),
            roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
        )
        self.cam_info.header.frame_id = self.camera_frame.value

        self.light = create_light()
        self.randomize_tree_spindle()

    def randomize_tree_spindle(self, *args):

        random_image = self.load_random_image()
        material = None
        if random_image is not None:
            material = create_tiled_material(random_image, repeat_factor=10)

        cam_pose = np.identity(4)

        # Determine ctrl points for Bezier curve
        pts = np.array([
            [0,0,0],
            [np.random.uniform(-0.10, 0.10), np.random.uniform(-0.10, 0.10), np.random.uniform(0.5, 1.0)],
            [np.random.uniform(-0.10, 0.10), np.random.uniform(-0.10, 0.10), np.random.uniform(1.0, 1.5)],
            [0,0,2]
        ])

        if self.main_spindle is None:
            self.main_spindle = create_cubic_bezier_curve(pts, material=material)
            for _ in range(self.num_side_branches.value):
                self.side_branches.append(create_cylinder(height=0.10, r=0.005, material=material))

        else:
            if material is not None:
                for branch in [self.main_spindle] + self.side_branches:
                    branch.material_slots[0].material = material

            tf = self.tf_buffer.lookup_transform(self.base_frame.value, self.camera_frame.value, rclpy.time.Time())
            tl = tf.transform.translation
            q = tf.transform.rotation

            cam_pose[:3,3] = [tl.x, tl.y, tl.z]
            cam_pose[:3,:3] = Quaternion([q.w, q.x, q.y, q.z]).to_matrix()

        in_front_pos = (cam_pose @ np.array([0, 0, self.spindle_dist.value, 1]))[:3]
        base_loc = np.array([in_front_pos[0], in_front_pos[1], 0])
        pts = pts + base_loc

        self.main_spindle.data.splines[0].bezier_points[0].co = pts[0]
        self.main_spindle.data.splines[0].bezier_points[0].handle_right = pts[1]
        self.main_spindle.data.splines[0].bezier_points[1].handle_left = pts[2]
        self.main_spindle.data.splines[0].bezier_points[1].co = pts[3]

        self.main_spindle_eval = np.array(interpolate_bezier(*pts, 50))

        sb_low, sb_high = self.side_branch_range.value
        candidate_sb_points = self.main_spindle_eval[(self.main_spindle_eval[:,2] > sb_low) & (self.main_spindle_eval[:,2] < sb_high)]

        for branch in self.side_branches:

            branch.location = candidate_sb_points[np.random.choice(len(candidate_sb_points))]
            branch.rotation_euler = [0, np.random.uniform(0.25*np.pi, 0.75*np.pi), np.random.uniform(0, 2*np.pi)]

        current_pos = cam_pose[:3,3]
        self.light.location = current_pos + np.array([0,0,1]) # + np.random.uniform(-1,1,3) * np.array([0.5, 0.5, 2])

        if len(args) == 2:
            resp = args[1]
            resp.success = True
            return resp

    def load_random_image(self):
        textures_path = self.textures_location.value
        files = os.listdir(textures_path)
        if not files:
            print('No texture images located at {}!'.format(textures_path))
            return None

        file_path = os.path.join(textures_path, random.choice(files))
        img = bpy.data.images.load(file_path)
        return img


# Blender stuff
# ---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# ---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd, scene):

    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = np.array([
        (s_u, skew, u_0),
        (0, s_v, v_0),
        (0, 0, 1)
    ])
    return K


def create_light(location=(0,0,1)):
    bpy.context.scene.cursor.location.xyz = (0, 0, 0)
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=location)
    light = bpy.context.scene.collection.objects[-1]
    return light

def create_cylinder(location=(0, 0, 0), height=1.0, r=0.05, material=None):
    bpy.context.scene.cursor.location.xyz = (0, 0, 0)
    bpy.ops.mesh.primitive_cylinder_add(enter_editmode=False, align='WORLD')
    cyl = bpy.context.scene.collection.objects[-1]

    bpy.context.scene.cursor.location.xyz = (0, 0, -1)
    bpy.context.view_layer.objects.active = cyl
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
    cyl.scale = (r, r, height / 2)
    cyl.location = location

    bpy.context.view_layer.objects.active = cyl
    cyl.select_set(True)
    bpy.ops.object.transform_apply(scale=True)

    if material is not None:
        cyl.data.materials.append(material)

    return cyl

def create_cubic_bezier_curve(ctrl_pts, radius=0.01, material=None, name='Curve'):

    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 10
    curve_data.resolution_u = 20
    curve_data.use_fill_caps = True

    spline = curve_data.splines.new(type='BEZIER')
    spline.bezier_points.add(1)
    pts = spline.bezier_points
    pts[0].co = ctrl_pts[0]
    pts[0].handle_right = ctrl_pts[1]
    pts[1].handle_left = ctrl_pts[2]
    pts[1].co = ctrl_pts[3]

    curve_obj = bpy.data.objects.new(name + '_obj', curve_data)
    bpy.context.collection.objects.link(curve_obj)
    curve_obj.select_set(True)

    if material is not None:
        curve_obj.data.materials.append(material)

    return curve_obj


def create_tiled_material(img, repeat_factor=1):
    mat = bpy.data.materials.new('Test Material')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    img_node = nodes.new(type="ShaderNodeTexImage")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    coord_node = nodes.new(type="ShaderNodeTexCoord")
    bsdf_node = nodes['Principled BSDF']

    img_node.image = img
    mapping_node.inputs['Scale'].default_value = Vector([repeat_factor] * 3)

    mat.node_tree.links.new(coord_node.outputs['Object'], mapping_node.inputs['Vector'])
    mat.node_tree.links.new(mapping_node.outputs['Vector'], img_node.inputs['Vector'])
    mat.node_tree.links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    mat.node_tree.nodes['Mapping'].inputs['Rotation'].default_value = Vector(
        np.random.uniform(np.radians(-90), np.radians(90), 3))

    bsdf_node.inputs['Specular'].default_value = 0.0

    return mat


def main(args=None):
    try:
        rclpy.shutdown()
    except RuntimeError:
        pass

    try:
        rclpy.init(args=args)
        executor = MultiThreadedExecutor()
        server = ImageServer()
        rclpy.spin(server, executor=executor)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
