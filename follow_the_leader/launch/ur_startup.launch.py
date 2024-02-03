#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetLaunchConfiguration, OpaqueFunction, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    headless_mode = LaunchConfiguration("headless_mode", default="true")
    warehouse_sqlite_path = LaunchConfiguration("warehouse_sqlite_path")

    initial_joint_controller = LaunchConfiguration(
        "initial_joint_controller", default="scaled_joint_trajectory_controller"
    )
    set_joint_controller = SetLaunchConfiguration(
        "initial_joint_controller",
        value="joint_trajectory_controller",
        condition=LaunchConfigurationEquals("use_fake_hardware", "true"),
    )

    ur_type_arg = DeclareLaunchArgument(
        "ur_type", default_value="ur3", description="Robot description name (consistent with ur_control.launch.py)"
    )
    robot_ip_arg = DeclareLaunchArgument("robot_ip", default_value="169.254.174.50", description="Robot IP")

    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware", default_value="true", description="If true, uses the fake controllers"
    )

    ur_base_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ur_robot_driver"), "launch/ur_control.launch.py")
        ),
        launch_arguments=[
            ("robot_ip", robot_ip),
            ("ur_type", ur_type),
            ("use_fake_hardware", use_fake_hardware),
            ("headless_mode", headless_mode),
            ("initial_joint_controller", initial_joint_controller),
            ("launch_rviz", "false"),
        ],
    )

    ur_moveit_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ur_moveit_config"), "launch/ur_moveit.launch.py")
        ),
        launch_arguments=[
            ("ur_type", ur_type),
            ("use_fake_hardware", use_fake_hardware),
            ("launch_rviz", "false"),
            ("warehouse_sqlite_path", warehouse_sqlite_path),
        ],
    )

    ftl_moveit_server_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ftl_move_group_server"), "launch/ftl_server.launch.py")
        ),
        launch_arguments=[
            ("ur_type", ur_type),
            ("use_fake_hardware", use_fake_hardware),
            ("launch_rviz", "true"),
            ("warehouse_sqlite_path", warehouse_sqlite_path),
            ("use_sim_time", "false"),
        ],
    )

    delay_for_ftl = TimerAction(
        period=5.0,
        actions=[
            ftl_moveit_server_launch
        ],
    )

    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": warehouse_sqlite_path,
    }

    warehouse_server_node = Node(
        package="moveit_ros_warehouse",
        executable="moveit_warehouse_services",
        output="screen",
        parameters=[
            warehouse_ros_config,
        ]
    )

    tf_node_mount = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments="0 -0.05 0.007 0 0 0 1 tool0 camera_mount_center".split(),
        condition=UnlessCondition(use_fake_hardware),
    )

    tf_node_mount_to_cam = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments="-0.009 0 0.0193 0.5 -0.5 0.5 0.5 camera_mount_center camera_link".split(),  # Z is camera thickness (23mm) minus glass (3.7mm)
        condition=UnlessCondition(use_fake_hardware),
    )

    tf_node_b = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments="0.0 0 0.0 0.5 -0.5 0.5 0.5 tool0 camera_link".split(),
        condition=IfCondition(use_fake_hardware),
    )

    tf_node_c = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments="0 0 0 0.5 -0.5 0.5 -0.5 camera_link camera_color_optical_frame".split(),
        condition=IfCondition(use_fake_hardware),
    )

    return LaunchDescription(
        [
            ur_type_arg,
            robot_ip_arg,
            use_fake_hardware_arg,
            set_joint_controller,
            ur_base_launch,
            ur_moveit_launch,
            warehouse_server_node,
            tf_node_mount,
            tf_node_mount_to_cam,
            tf_node_b,
            tf_node_c,
            delay_for_ftl,
        ]
    )
