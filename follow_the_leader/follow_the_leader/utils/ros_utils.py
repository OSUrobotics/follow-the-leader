#!/usr/bin/env python3
import yaml
import os
import functools
from datetime import datetime
from threading import Event, Lock

import numpy as np
import rclpy
import rclpy.time
from follow_the_leader.utils.image_utils import PinholeCameraModelNP
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import ParameterEvent
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster


def log_entry_exit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = rclpy.logging.get_logger(func.__module__)
        start = datetime.now()
        logger.info(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(
            f"Exiting {func.__name__} took {(datetime.now() - start).total_seconds()}s"
        )
        return result

    return wrapper


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

        if desired_params is not None:
            for param, val in desired_params.items():
                self.declare_parameter(param, val)

        self._param_sub = self.create_subscription(
            ParameterEvent, "/parameter_events", self._param_callback, 1
        )
        return

    def _param_callback(self, msg: ParameterEvent):
        if msg.node.lstrip("/") == self.get_name():
            for change in msg.changed_parameters:
                name = change.name
                if name not in self._parameters:
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
                self._parameters[name] = getattr(val_msg, field_map[val_type])
        return

    def get_param(self, name):
        return self._parameters[name]


class TFNode(Node):
    def __init__(self, name, *args, cam_info_topic=None, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.camera = PinholeCameraModelNP()
        if cam_info_topic is not None:
            self.get_logger().info(f"Waiting for camera info on {cam_info_topic}")
            state, msg = wait_for_message(CameraInfo, self, cam_info_topic, time_to_wait=-1)
            self.get_logger().info(f"Received camera info on {cam_info_topic}")
            if state:
                self.camera.fromCameraInfo(msg)
            else:
                raise Exception("Failed to get camera info!")
            self._cam_info_sub = self.create_subscription(
                CameraInfo, cam_info_topic, self._handle_cam_info, 1
            )
        self.tf_buffer = Buffer(cache_time=rclpy.time.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self)
        return

    def declare_parameter_dict(self, **kwargs):
        for key, val in kwargs.items():
            self.declare_parameter(key, val)
        return

    def get_param_val(self, key):
        try:
            return self.get_parameter(key).value
        except Exception as ex:
            self.get_logger().error(ex)
            return None

    def _handle_cam_info(self, msg: CameraInfo):
        self.camera.fromCameraInfo(msg)
        self.total_pixels = self.camera.width * self.camera.height
        return

    async def get_future(future):
        await future
        return future.result()

    def lookup_transform(
        self,
        target_frame,
        source_frame,
        time=None,
        sync=True,
        as_matrix=False,
        timeout=rclpy.time.Duration(seconds=0.5),
    ):
        """Convenience function to lookup a transform

        :param target_frame: target
        :param source_frame: source
        :param time: time to use, defaults behaviour to use most recent transform
        :param sync: whether to use blocking sync, defaults to True. 
        :param as_matrix: returns in homogenous matrix form, defaults to False
        :param timeout: how long to block, defaults to rclpy.time.Duration(seconds=0.5)
        :return: tf or matrix, None if failed
        """
        if time is None or not isinstance(time, rclpy.time.Time):
            time = rclpy.time.Time()
        start = self.get_clock().now()
        tf = None
        log_str = f"{self.get_name()}: TF lookup {source_frame} -> {target_frame}"
        try:
            if sync:
                tf = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, time, timeout=timeout
                )
            else:
                tf = self.tf_buffer.lookup_transform(target_frame, source_frame, time)
            if tf is None:
                raise TransformException("Likely timeout!")
        except TransformException as ex:
            self.get_logger().fatal(f"{log_str}: Received TF Exception: {ex}")
            return
        except Exception as ex:
            self.get_logger().fatal(f"{log_str}: Received Exception: {ex }")
            return
        wait = self.get_clock().now() - start
        if wait > rclpy.time.Duration(seconds=0.1):
            self.get_logger().warn(
                f"{log_str} took {wait.nanoseconds / 1e9} seconds"
            )
        if not as_matrix:
            return tf

        tl = tf.transform.translation
        q = tf.transform.rotation
        mat = np.identity(4)
        mat[:3, 3] = [tl.x, tl.y, tl.z]
        mat[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return mat

    def dump_params(self, dirname):
        name = self.get_name()
        yaml_output = {name: {'ros__parameters': {}}}
        for key, val in self.get_parameters_by_prefix("").items():
            yaml_output[name]['ros__parameters'][key] = val.value

        with open(os.path.join(dirname, f'params_{name}.yaml'), 'w') as outfile:
            yaml.dump(yaml_output, outfile,  default_flow_style=False)
        return

    @staticmethod
    def mul_homog(mat, pt):
        """Multiply a homogenous matrix by points in Nxdim form and get Nxdim output"""
        pt = np.array(pt)
        # add dimension to last axis
        pt_homog = np.ones((*pt.shape[:-1], pt.shape[-1] + 1))
        pt_homog[..., : pt.shape[-1]] = pt
        return (mat @ pt_homog.T).T[..., : pt.shape[-1]]

    @staticmethod 
    def define_dummy_camera():
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
        k=[
            437.00222778,
            0.0,
            418.9420166,
            0.0,
            439.22055054,
            240.41038513,
            0.0,
            0.0,
            1.0,
        ],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[
            437.00222778,
            0.0,
            418.9420166,
            0.0,
            0.0,
            439.22055054,
            240.41038513,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        roi=RegionOfInterest(
            x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
        ),
        )
        return sample_cam_info

    def load_dummy_camera(self):
        # Based on the Realsense D405 profile
        sample_cam_info = TFNode.define_dummy_camera()
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


if __name__ == "__main__":
    camera = PinholeCameraModelNP()
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
        k=[
            437.00222778,
            0.0,
            418.9420166,
            0.0,
            439.22055054,
            240.41038513,
            0.0,
            0.0,
            1.0,
        ],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[
            437.00222778,
            0.0,
            418.9420166,
            0.0,
            0.0,
            439.22055054,
            240.41038513,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        roi=RegionOfInterest(
            x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
        ),
    )
    camera.fromCameraInfo(sample_cam_info)

    pxs = np.array([[100, 200], [300, 400]])
    rays = camera.projectPixelTo3dRay(pxs)
    print(rays)
