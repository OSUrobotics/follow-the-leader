import sys
import numpy as np
import cv_bridge
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QPainter, QPen
from PyQt5.QtCore import QTimer, Qt
import os
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from follow_the_leader_msgs.msg import TrackedPointRequest, TrackedPointGroup, Point2D, Tracked3DPointResponse

class SharedData:
    def __init__(self):
        self.data = {}
        self.mutex = threading.Lock()

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __enter__(self):
        self.mutex.__enter__()

    def __exit__(self, *args, **kwargs):
        self.mutex.__exit__(*args, **kwargs)

    def delete(self, key):
        del self.data[key]

    def clear(self):
        self.data = {}


class ROS2ProcessorNode(Node):
    def __init__(self, shared_data):
        super().__init__('point_tracking_gui_node')
        self.shared_data = shared_data
        self.image_subscriber = self.create_subscription(Image, '/camera/color/image_rect_raw', self.process_image, 1)
        self.bridge = cv_bridge.CvBridge()
        self.timer = self.create_timer(0.01, self.check_data)

        self.request_pub = self.create_publisher(TrackedPointRequest, '/point_tracking_request', 1)
        self.response_sub = self.create_subscription(Tracked3DPointResponse, '/point_tracking_response', self.handle_response, 1)

    def process_image(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.shared_data:
            self.shared_data['image'] = image

    def handle_response(self, msg: Tracked3DPointResponse):
        data = {}
        data['image'] = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='rgb8')
        data['groups'] = {group.name: np.array([[pt.x, pt.y, pt.z] for pt in group.points]) for group in msg.groups}
        data['groups_2d'] = {group.name: np.array([[pt.x, pt.y] for pt in group.points]) for group in msg.groups_2d}

        with self.shared_data:
            self.shared_data['response'] = data

    def check_data(self):

        with self.shared_data:
            try:
                points = self.shared_data['request']
            except KeyError:
                return

            req = TrackedPointRequest()
            if not points:
                req.action = TrackedPointRequest.ACTION_REMOVE

            group = TrackedPointGroup(name='main', points=[Point2D(x=float(p[0]), y=float(p[1])) for p in points])
            req.groups.append(group)
            self.request_pub.publish(req)
            self.shared_data.delete('request')


class ROS2NodeWrapper(threading.Thread):
    def __init__(self, node_class, *args, **kwargs):
        super().__init__()
        self.node_class = node_class
        self.args = args
        self.kwargs = kwargs
    def run(self):
        rclpy.init()
        node = self.node_class(*self.args, **self.kwargs)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()


class MainWindow(QMainWindow):
    def __init__(self, shared_data):
        super().__init__()

        self.setup()

        self.is_tracking = False
        self.shared_data = shared_data
        self.queued_points = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timer_callback)
        self.timer.start(50)

    def setup(self):
        # Set window title
        self.setWindowTitle("Point Tracking GUI")

        # Create main widget and layout
        self.central_widget = QWidget(self)
        self.layout = QHBoxLayout(self.central_widget)

        # Create left column widget and layout
        self.left_widget = QWidget(self.central_widget)
        self.left_layout = QVBoxLayout(self.left_widget)

        # Create QLabel for image display
        self.image_label = QLabel(self)
        self.image_label.mousePressEvent = self.handle_mouse_click

        # Add QLabel to the left layout
        self.left_layout.addWidget(self.image_label)

        # Create right column widget and layout
        self.right_widget = QWidget(self.central_widget)
        self.right_layout = QVBoxLayout(self.right_widget)

        # Create buttons
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset)
        self.clear_button = QPushButton("Clear all points", self)
        self.clear_button.clicked.connect(self.clear_points)
        self.tracking_button = QPushButton("Track points", self)
        self.tracking_button.clicked.connect(self.track_points)

        # Add buttons to the right layout
        self.right_layout.addWidget(self.reset_button)
        self.right_layout.addWidget(self.clear_button)
        self.right_layout.addWidget(self.tracking_button)

        # Add left and right columns to the main layout
        self.layout.addWidget(self.left_widget)
        self.layout.addWidget(self.right_widget)

        # Set the central widget
        self.setCentralWidget(self.central_widget)

    def timer_callback(self):
        if self.is_tracking:
            with self.shared_data:
                try:
                    data = self.shared_data['response']
                except KeyError:
                    return
                self.shared_data.delete('response')

            image = data['image']
            points_2d = data['groups_2d'].get('main', [])

        else:
            points_2d = []
            with self.shared_data:
                try:
                    image = self.shared_data['image']
                except KeyError:
                    return

        pixmap = QPixmap.fromImage(self.convert_numpy_to_qimage(image))
        painter = QPainter(pixmap)
        pen = QPen(Qt.blue)
        pen.setWidth(8)
        painter.setPen(pen)

        for point in points_2d:
            painter.drawPoint(*point.astype(int))

        pen.setWidth(4)
        pen.setColor(Qt.green)
        for point in self.queued_points:
            painter.drawPoint(*point)

        painter.end()
        self.image_label.setPixmap(pixmap)

    def handle_mouse_click(self, event: QMouseEvent):
        pos = event.pos()
        x = pos.x()
        y = pos.y()
        self.queued_points.append((x,y))
        print('Clicked pixel at ({}, {})'.format(x,y))

    def reset(self):
        self.queued_points = []
        self.is_tracking = False
        with self.shared_data:
            self.shared_data.clear()

    def clear_points(self):
        self.queued_points = []

    def track_points(self):
        self.is_tracking = bool(self.queued_points)
        with self.shared_data:
            self.shared_data['request'] = self.queued_points
        self.clear_points()

    @staticmethod
    def convert_numpy_to_qimage(img):
        h, w, c = img.shape
        bytes_per_line = c * w
        qimage = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qimage

def main(args=None):

    shared_data = SharedData()
    node_thread = ROS2NodeWrapper(ROS2ProcessorNode, shared_data)
    node_thread.start()

    app = QApplication(sys.argv)
    main_window = MainWindow(shared_data)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()