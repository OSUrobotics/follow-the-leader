import os
import yaml

import rclpy.logging
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction,
                            TimerAction)
from launch.conditions import (IfCondition, LaunchConfigurationEquals,
                               UnlessCondition)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import (AndSubstitution, LaunchConfiguration,
                                  NotSubstitution, PathJoinSubstitution)
from launch_ros.actions import Node


def launch_setup(context: LaunchContext, *args, **kwargs):
    camera_yaml_path = LaunchConfiguration("camera_yaml_path")
    rs_camera_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace="",
        parameters=[{
            "initial_reset": True,
            "align_depth.enable": True,
            "linear_accel_cov": 1.0,
            "unite_imu_method": 2,  # linear_interpolation
            "enable_rgbd": True,
            "enable_sync": True,
            "enable_color": True,
            "enable_depth": True,
            "enable_accel": True,
            "enable_gyro": True,
            "initial_reset": True,
            "publish_tf": True,

            "rgb_camera.color_profile":"640x480x30",
            "depth_module.infra_profile":"640x480x30", 
            }]
        )
    
    configuration = None
    with open(camera_yaml_path.perform(context), 'r') as f:
        configuration = yaml.safe_load(f)
    tf_node_mount_to_cam = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=configuration['calib_publisher']['ros__parameters']['calibration'].split(' '),
    )
    
    ros_param_set = ExecuteProcess(
        cmd=[
            "ros2",
            "param",
            "set",
            "/camera/camera",
            "rgb_camera.enable_auto_exposure True"
        ],
        shell=True,  # need to use args with options
        output="screen",
        log_cmd=True,
    )

    delay_rs = TimerAction(
        period=4.0,
        actions=[
            ros_param_set
        ],
    )
    return [
        rs_camera_node,
        delay_rs,
        tf_node_mount_to_cam,
    ]

def generate_launch_description():
    camera_yaml_arg = DeclareLaunchArgument("camera_yaml_path")
    declared_args = [
        camera_yaml_arg
    ]

    ld = LaunchDescription(declared_args + [OpaqueFunction(function=launch_setup)])
    return ld