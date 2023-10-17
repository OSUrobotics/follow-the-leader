#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterEvent
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from image_geometry import PinholeCameraModel
from threading import Event, Lock
from scipy.spatial.transform import Rotation
import numpy as np


class PinholeCameraModelNP(PinholeCameraModel):
    """
    Modifications to the PinholeCameraModel class to make them operate with Numpy.
    """

    def project3dToPixel(self, pts):
        pts = np.array(pts)
        pts_homog = np.ones((*pts.shape[:-1], pts.shape[-1] + 1))
        pts_homog[..., :3] = pts

        x, y, w = np.array(self.P) @ pts_homog.T
        return np.array([x / w, y / w]).T


    def getDeltaU(self, deltaX, Z):
        fx = self.P[0, 0]
        return fx * deltaX / Z


def wait_for_future_synced(future):
    event = Event()

    def done_callback(_):
        nonlocal event
        event.set()

    future.add_done_callback(done_callback)
    event.wait()
    resp = future.result()
    return resp


def call_service_synced(client, request):
    future = client.call_async(request)
    return wait_for_future_synced(future)


def process_list_as_dict(msg_list, name_field, val_field):
    return {getattr(msg, name_field): getattr(msg, val_field) for msg in msg_list}


class ParameterServerNode(Node):
    def __init__(self, name, *args, desired_params=None, **kwargs):
        super().__init__(name, *args, **kwargs)

        self._params = {}
        if desired_params is not None:
            for param, val in desired_params.items():
                self.declare_parameter(param, val)
                self._params[param] = val

        self._param_sub = self.create_subscription(ParameterEvent, "/parameter_events", self._param_callback, 1)
        return


    def _param_callback(self, msg: ParameterEvent):
        if msg.node.lstrip("/") == self.get_name():
            for change in msg.changed_parameters:
                name = change.name
                if name not in self._params:
                    continue
                val_msg = change.value
                val_type = val_msg.type
                field_map = {
                    1: "bool_value",
                    2: "integer_value",
                    3: "double_value",
                    4: "string_value",
                    5: "byte_array_value",
                    6: "bool_array_value",
                    7: "integer_array_value",
                    8: "double_array_value",
                    9: "string_array_value",
                }
                self._params[name] = getattr(val_msg, field_map[val_type])
        return


    def get_param(self, name):
        return self._params[name]


class TFNode(Node):
    def __init__(self, name, *args, cam_info_topic=None, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._params = {}
        self.camera = PinholeCameraModelNP()
        if cam_info_topic is not None:
            self._cam_info_sub = self.create_subscription(CameraInfo, cam_info_topic, self._handle_cam_info, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        return


    def declare_parameter_dict(self, **kwargs):
        for key, val in kwargs.items():
            self._params[key] = self.declare_parameter(key, val)
        return

    def get_param_val(self, key):
        return self._params[key].value

    def _handle_cam_info(self, msg: CameraInfo):
        self.camera.fromCameraInfo(msg)
        return

    def lookup_transform(self, target_frame, source_frame, time=None, sync=True, as_matrix=False):
        if time is None:
            time = rclpy.time.Time()
        if sync:
            future = self.tf_buffer.wait_for_transform_async(target_frame, source_frame, time)
            wait_for_future_synced(future)

        tf = self.tf_buffer.lookup_transform(target_frame, source_frame, time)
        if not as_matrix:
            return tf

        tl = tf.transform.translation
        q = tf.transform.rotation
        mat = np.identity(4)
        mat[:3, 3] = [tl.x, tl.y, tl.z]
        mat[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return mat

    @staticmethod
    def mul_homog(mat, pt):
        pt = np.array(pt)
        pt_homog = np.ones((*pt.shape[:-1], pt.shape[-1] + 1))
        pt_homog[..., : pt.shape[-1]] = pt
        return (mat @ pt_homog.T).T[..., : pt.shape[-1]]

    def load_dummy_camera(self):
        # Based on the Realsense D405 profile
        sample_cam_info = CameraInfo(
            height=480,
            width=848,
            distortion_model="plumb_bob",
            binning_x=0,
            binning_y=0,
            d=[
                -0.05469128489494324,
                0.05773274227976799,
                7.857435412006453e-05,
                0.0003967129159718752,
                -0.018736450001597404,
            ],
            k=[437.00222778, 0.0, 418.9420166, 0.0, 439.22055054, 240.41038513, 0.0, 0.0, 1.0],
            r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            p=[437.00222778, 0.0, 418.9420166, 0.0, 0.0, 439.22055054, 240.41038513, 0.0, 0.0, 0.0, 1.0, 0.0],
            roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
        )
        sample_cam_info.header.frame_id = "camera_color_optical_frame"
        self.camera.fromCameraInfo(sample_cam_info)
        return


class SharedData:
    def __init__(self):
        self.data = {}
        self.mutex = Lock()

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __enter__(self):
        self.mutex.__enter__()

    def __exit__(self, *args, **kwargs):
        self.mutex.__exit__(*args, **kwargs)

    def __bool__(self):
        return bool(self.data)

    def delete(self, key):
        del self.data[key]

    def clear(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def pop(self, key, default=None):
        return self.data.pop(key, default)

    def items(self):
        return self.data.items()
