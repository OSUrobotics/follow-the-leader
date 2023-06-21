import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from image_geometry import PinholeCameraModel
from threading import Event, Lock
from scipy.spatial.transform import Rotation
import numpy as np


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


class TFNode(Node):
    def __init__(self, name, *args, cam_info_topic=None,  **kwargs):
        super().__init__(name, *args, **kwargs)
        self.camera = PinholeCameraModel()
        if cam_info_topic is not None:
            self._cam_info_sub = self.create_subscription(CameraInfo, cam_info_topic, self._handle_cam_info, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _handle_cam_info(self, msg: CameraInfo):
        self.camera.fromCameraInfo(msg)

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
        mat[:3,3] = [tl.x, tl.y, tl.z]
        mat[:3,:3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return mat

    @staticmethod
    def mul_homog(mat, pt):
        pt = np.array(pt)
        pt_homog = np.ones((*pt.shape[:-1], pt.shape[-1] + 1))
        pt_homog[..., :pt.shape[-1]] = pt
        return (mat @ pt_homog.T).T[..., :pt.shape[-1]]

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
