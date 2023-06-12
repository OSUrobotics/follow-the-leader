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
from threading import Event, Lock
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from follow_the_leader.utils.ros_utils import TFNode, SharedData
bridge = CvBridge()

# TODO: Refactor to use TFNode in utils


def wait_for_future_synced(future):
    event = Event()
    def done_callback(_):
        nonlocal event
        event.set()
    future.add_done_callback(done_callback)
    event.wait()
    resp = future.result()
    return resp

class RotatingQueue:
    def __init__(self, size=8):
        self.queue = [None] * size
        self.idx = 0

    def empty(self):
        self.queue = [None] * len(self.queue)
        self.idx = 0

    @property
    def is_full(self):
        return not None in self.queue

    def append(self, item):
        self.queue[self.idx] = item
        self.idx = (self.idx + 1) % len(self.queue)

    def as_list(self):
        if not self.is_full:
            return self.queue[:self.idx]
        else:
            return self.queue[self.idx:] + self.queue[:self.idx]


class PointTracker(Node):
    def __init__(self):
        super().__init__('point_tracker_node')

        # State variables
        self.current_request = SharedData()
        self.image_queue = RotatingQueue(size=8)
        self.tracker = PipsTracker(
            model_dir=os.path.join(os.path.expanduser('~'), 'repos', 'pips', 'pips', 'reference_model'))
        self.last_pos = None

        # Config
        # TODO: Retrieve configuration params properly
        # self.movement_threshold = 0
        self.movement_threshold = 0.025 / 8
        self.base_frame = 'base_link'
        self.tool_frame = 'tool0'
        # self.do_3d_point_estimation = False
        self.do_3d_point_estimation = True

        # ROS Utils
        self.custom_callback = MutuallyExclusiveCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera = PinholeCameraModel()
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.set_camera_info, 1)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_rect_raw', self.handle_image_callback, 1, callback_group=self.custom_callback)
        self.tracking_request_sub = self.create_subscription(TrackedPointRequest, '/point_tracking_request', self.handle_tracking_request, 10)
        self.tracked_3d_pub = self.create_publisher(Tracked3DPointResponse, '/point_tracking_response', 1)
        self.pc_pub = self.create_publisher(PointCloud2, '/point_tracking_response_pc', 1)

    def set_camera_info(self, msg):
        self.camera.fromCameraInfo(msg)
        # self.cam_info_sub.destroy()

    def handle_tracking_request(self, msg: TrackedPointRequest):
        with self.current_request:
            groups = msg.groups
            if msg.action == TrackedPointRequest.ACTION_REMOVE:
                for group in groups:
                    self.current_request.pop(group.name, None)
                return

            self.current_request.clear()
            for group in groups:
                print('New request {}'.format(group.name))
                self.current_request[group.name] = np.array([[pt.x, pt.y] for pt in group.points])

            self.image_queue.empty()
            if msg.image.data:
                self.process_image_info(msg.image, add_to_queue=True)


    def flatten_current_request(self):
        all_pts = []
        all_names = []
        for name, points in self.current_request.items():
            all_pts.append(points)
            all_names.extend([name] * len(points))
        return np.concatenate(all_pts, axis=0), all_names

    def process_image_info(self, img_msg: Image, add_to_queue=False):
        stamp = img_msg.header.stamp
        pose = None
        if self.do_3d_point_estimation:
            pose = self.get_camera_frame_pose(time=stamp, position_only=False)
        info = {
            'stamp': stamp,
            'frame_id': img_msg.header.frame_id,
            'image': bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8'),
            'pose': pose,
        }
        if add_to_queue:
            self.image_queue.append(info)
        return info

    def handle_image_callback(self, msg):
        if self.camera.tf_frame is None or not self.current_request:
            return

        current_pos = self.get_camera_frame_pose(position_only=True)
        if self.movement_threshold and (not (self.last_pos is None or np.linalg.norm(current_pos - self.last_pos) > self.movement_threshold)):
            return
        print('Movement threshold passed')

        self.last_pos = current_pos
        self.process_image_info(img_msg=msg, add_to_queue=True)
        self.update_tracker(last_image=msg)

    def update_tracker(self, last_image=None):
        if not self.image_queue.is_full:
            return

        print('Updating tracker')
        with self.current_request:
            all_info = self.image_queue.as_list()
            images = [info['image'] for info in all_info]
            targets, groups = self.flatten_current_request()
            trajs = self.tracker.track_points(targets, images)
            trajs = np.transpose(trajs, (1,0,2))

            pts_3d = None
            if self.do_3d_point_estimation:
                pose_t_w = np.linalg.inv(all_info[-1]['pose'])
                camera_frame_tf_matrices = [(pose_t_w @ info['pose']) for info in all_info]
                triangulator = PointTriangulator(self.camera)
                pts_3d = triangulator.compute_3d_points(camera_frame_tf_matrices, trajs)

                reprojs = triangulator.get_reprojs(pts_3d, camera_frame_tf_matrices, trajs)
                error = np.linalg.norm(trajs - reprojs, axis=2)
                avg_error = error.mean(axis=1)
                max_error = error.max(axis=1)
                print('Average pix error:\n')
                print(', '.join('{:.3f}'.format(x) for x in avg_error))
                print('Max pix error:\n')
                print(', '.join('{:.3f}'.format(x) for x in max_error))

            trajs = np.transpose(trajs, (1,0,2))

            frame_id = all_info[-1]['frame_id']
            stamp = all_info[-1]['stamp']

            response = Tracked3DPointResponse(header=Header(frame_id=frame_id, stamp=stamp))
            if pts_3d is not None:
                pc = create_cloud_xyz32(Header(frame_id=frame_id, stamp=stamp), points=pts_3d)
                self.pc_pub.publish(pc)
                for group, pts_and_errs in self.unflatten_tracked_points(zip(pts_3d, max_error), groups).items():
                    points, errors = zip(*pts_and_errs)
                    response.groups.append(Tracked3DPointGroup(name=group,
                                                               points=[Point(x=x, y=y, z=z) for x, y, z in points],
                                                               errors=errors))

            for group, points_2d in self.unflatten_tracked_points(trajs[-1].astype(np.float), groups).items():
                response.groups_2d.append(TrackedPointGroup(name=group, points=[Point2D(x=x, y=y) for x,y in points_2d]))
            if last_image is not None:
                response.image = last_image

            self.tracked_3d_pub.publish(response)
            self.update_request_from_trajectory(trajs, groups)


    def unflatten_tracked_points(self, points, groups):

        rez = defaultdict(list)
        for point, group in zip(points, groups):
            rez[group].append(point)

        return rez

    def update_request_from_trajectory(self, trajs, groups):

        w = self.camera.width
        h = self.camera.height
        final_locs = trajs[-1]
        is_outside = (final_locs[:,0] < 0) | (final_locs[:,0] >= w) | (final_locs[:,1] < 0) | (final_locs[:,1] >= h)
        idx_to_stay = np.where(~is_outside)[0]
        new_req = {}

        update_locs = trajs[1]
        for idx in idx_to_stay:
            group = groups[idx]
            if group not in new_req:
                new_req[group] = []
            new_req[group].append(update_locs[idx])

        self.current_request.clear()
        for group, points in new_req.items():
            self.current_request[group] = np.array(points)

    def reset(self, *_, **__):
        self.current_request.clear()
        self.image_queue.empty()

    def get_camera_frame_pose(self, time=None, position_only=False):

        time = time or rclpy.time.Time()
        future = self.tf_buffer.wait_for_transform_async(self.base_frame, self.camera.tf_frame, time)
        wait_for_future_synced(future)

        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera.tf_frame, time)
        except TransformException as ex:
            self.get_logger().warn('Received TF Exception: {}'.format(ex))
            return

        tl = tf.transform.translation
        if position_only:
            return np.array([tl.x, tl.y, tl.z])
        q = tf.transform.rotation
        mat = np.identity(4)
        mat[:3,3] = [tl.x, tl.y, tl.z]
        mat[:3,:3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        return mat


class PointTriangulator:
    def __init__(self, camera):
        self.camera = camera

    @property
    def k(self):
        return self.camera.K

    def run_triangulation(self, pose_matrices, point_traj):
        """
        pose_matrices: List of N 4x4 matrices
        trajs: N x 2 array of point trajectories in typical image XY format
        """

        D = np.zeros((len(pose_matrices) * 2, 4))
        for i, (pose_mat, point) in enumerate(zip(pose_matrices, point_traj)):
            proj_mat = self.k @ pose_mat[:3]
            D[2*i] = proj_mat[2] * point[0] - proj_mat[0]
            D[2*i+1] = proj_mat[2] * point[1] - proj_mat[1]

        _, _, v = np.linalg.svd(D, full_matrices=True)
        pts_3d = v[-1,:3] / v[-1,3]

        # TODO: There is some sort of bug (?) that is returning the values negative
        # Figure out if this is correct or if there is some sort of logical error
        return -pts_3d

    def compute_3d_points(self, pose_matrices, point_trajs):
        """
            pose_matrices: List of N 4x4 matrices
            trajs: T x N x 2 array of point trajectories in typical image XY format
        """

        return np.array([self.run_triangulation(pose_matrices, traj) for traj in point_trajs])

    def get_reprojs(self, points_3d, pose_matrices, point_trajs):
        """
        points_3d: K x 3 matrix of 3D points
        pose_matrices: N x 2 array of point trajectories in typical XY format
        point_traj: K x N x 2 array
        """

        rez = np.zeros(point_trajs.shape)

        for j, pose_mat in enumerate(pose_matrices):
            pose_t_base = np.linalg.inv(pose_mat)
            for i, (pt_3d, traj) in enumerate(zip(points_3d, point_trajs)):
                pt_3d_h = np.ones(4)
                pt_3d_h[:3] = pt_3d
                pt_3d_t = (pose_t_base @ pt_3d_h)[:3]

                reproj = self.camera.project3dToPixel(pt_3d_t)
                rez[i,j] = reproj

        return rez

def main(args=None):
    try:
        rclpy.init(args=args)
        executor = MultiThreadedExecutor()
        node = PointTracker()
        rclpy.spin(node, executor=executor)
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
