import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition

from launch_ros.actions import Node, ComposableNodeContainer, SetUseSimTime
import launch_ros.descriptions


def generate_launch_description():
    default_rviz = os.path.join(
        get_package_share_directory("depth_image_proc"),
        "launch",
        "rviz/point_cloud_xyzrgb.rviz",
    )
    rviz_flag = LaunchConfiguration("rviz_flag")
    use_sim_time = LaunchConfiguration("use_sim_time")
    rviz_arg = DeclareLaunchArgument("rviz_flag", default_value="false")
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="false")

    depth_register_node = ComposableNodeContainer(
        name="container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            # launch_ros.descriptions.ComposableNode(
            #     package='realsense2_camera',
            #     namespace='',
            #     plugin='realsense2_camera::' + rs_node_class,
            #     name="camera",
            #     parameters=[set_configurable_parameters(realsense_node_params)],
            #     extra_arguments=[{'use_intra_process_comms': LaunchConfiguration("intra_process_comms")}])
            
            launch_ros.descriptions.ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::RegisterNode",
                name="register_node",
                remappings=[
                    ("depth/image_rect", "/camera/depth/image_rect_raw"),
                    ("depth/camera_info", "/camera/depth/camera_info"),
                    ("rgb/camera_info", "/camera/color/camera_info"),
                    (
                        "depth_registered/image_rect",
                        "/camera/depth_registered/image_rect",
                    ),
                    (
                        "depth_registered/camera_info",
                        "/camera/depth_registered/camera_info",
                    ),
                ],
                parameters=[
                    {"use_rgb_timestamp": True},
                ],
            ),

            launch_ros.descriptions.ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::PointCloudXyzrgbNode",
                name="point_cloud_xyzrgb_node",
                remappings=[
                    ("rgb/camera_info", "/camera/color/camera_info"),
                    ("rgb/image_rect_color", "/camera/color/image_raw"),
                    (
                        "depth_registered/image_rect",
                        "/camera/depth_registered/image_rect",
                    ),
                    ("points", "/camera/depth_registered/points"),
                ],
            ),
            # launch_ros.descriptions.ComposableNode(
            #         package='pcl_ros',
            #         plugin='pcl_ros::CropBox',
            #         name='cropbox_node',
            #         parameters=[{'min_x': -10., 'max_x': 10., 'min_y': -10., 'max_y:':-10., 'min_z': 0., 'max_z': 0.7, 'keep_organized': True}],
            #         remappings=[('input', '/camera/depth_registered/points'),
            #                     ('output', '/camera/filtered'),]
            # ),
        ],
        output="screen",
    )

    # rviz
    rviz_node = Node(
        condition=IfCondition(rviz_flag),
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["--display-config", default_rviz],
    )

    return LaunchDescription(
        [
            rviz_arg,
            use_sim_time_arg,
            GroupAction(
                actions=[
                    SetUseSimTime(value=True, condition=IfCondition(use_sim_time)),
                    depth_register_node,
                    rviz_node,
                ]
            ),
        ]
    )
