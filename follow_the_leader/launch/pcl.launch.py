import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

import launch_ros.actions
import launch_ros.descriptions

def generate_launch_description():
    default_rviz = os.path.join(get_package_share_directory('depth_image_proc'),
                                'launch', 'rviz/point_cloud_xyzrgb.rviz')
    return LaunchDescription([
        # launch_ros.actions.Node(
        #     package='realsense_ros2_camera', executable='realsense_ros2_camera',
        #     output='screen'),

        # launch plugin through rclcpp_components container
        launch_ros.actions.ComposableNodeContainer(
            name='container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                launch_ros.descriptions.ComposableNode(
                    package='depth_image_proc',
                    plugin='depth_image_proc::RegisterNode',
                    name='register_node',
                    remappings=[('depth/image_rect', '/camera/depth/image_rect_raw'),
                                ('depth/camera_info', '/camera/depth/camera_info'),
                                ('rgb/camera_info', '/camera/color/camera_info'),
                                ('depth_registered/image_rect',
                                 '/camera/depth_registered/image_rect'),
                                ('depth_registered/camera_info',
                                 '/camera/depth_registered/camera_info')]
                ),
                launch_ros.descriptions.ComposableNode(
                        package='depth_image_proc',
                        plugin='depth_image_proc::PointCloudXyzrgbNode',
                        name='point_cloud_xyzrgb_node',
                        remappings=[('rgb/camera_info', '/camera/color/camera_info'),
                                    ('rgb/image_rect_color', '/camera/color/image_raw'),
                                    ('depth_registered/image_rect',
                                    '/camera/depth_registered/image_rect'),
                                    ('points', '/camera/depth_registered/points')]
                ),
                launch_ros.descriptions.ComposableNode(
                        package='pcl_ros',
                        plugin='pcl_ros::CropBox',
                        name='cropbox_node',
                        parameters=[{'min_x': -10., 'max_x': 10., 'min_y': -10., 'max_y:':-10., 'min_z': 0., 'max_z': 0.7, 'keep_organized': True}],
                        remappings=[('input', '/camera/depth_registered/points'),
                                    ('output', '/camera/filtered'),]
                ),
        
            ],
            output='screen',
        ),

        # rviz
        launch_ros.actions.Node(
            package='rviz2', executable='rviz2', output='screen',
            arguments=['--display-config', default_rviz]), # Default QOS settings in rviz seem to be Reliable instead of BestEffort published by nodes
    ])