#!/usr/bin/env python3
import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetLaunchConfiguration, EmitEvent
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode
from ament_index_python.packages import get_package_share_directory
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
import os


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch/rs_launch.py")
        ),
        launch_arguments=[
            ("enable_depth", "false"),
        ],
    )

    controller_node = Node(
        package="follow_the_leader",
        executable="controller",
        output="screen",
    )

    image_processor_node = Node(
        package="follow_the_leader",
        executable="image_processor",
        output="screen",
    )

    tf_node_a = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments="0.0 0 0.0 0 0 0 1 tool0 ee".split(),
    )

    tf_node_b = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments="-0.106048 0.00338 -0.11277 -0.00034 -0.58759 -0.00117 0.80916 ee camera_link".split(),
    )

    return LaunchDescription([realsense_launch, image_processor_node, controller_node, tf_node_a, tf_node_b])
