
import os
from datetime import datetime

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
from launch_ros.parameter_descriptions import ParameterFile
from ur_moveit_config.launch_common import load_yaml

logger = rclpy.logging.get_logger("nbv.launch")


def launch_setup(context: LaunchContext, *args, **kwargs):
    package_dir = get_package_share_directory("follow_the_leader")
    params_path = os.path.join(package_dir, "config")

    # modes to run
    logging = LaunchConfiguration("logging")
    debug_mode = LaunchConfiguration("debug_mode")

    # configuration for setup
    ur_type = LaunchConfiguration("ur_type")
    camera_type = LaunchConfiguration("camera_type", default="d435")
    log_folder = LaunchConfiguration("log_folder")
    camera_yaml_path = PathJoinSubstitution([params_path, "camera_" + camera_type.perform(context) + ".yaml"])
    core_yaml_path = PathJoinSubstitution([params_path, "ftl_" + ur_type.perform(context) + ".yaml"])
    
    # create logging folder per instance
    now = datetime.now()
    date_time = now.strftime("%d%b%Y_%H:%M:%S")
    log_path = PathJoinSubstitution([log_folder, f"ftl_{date_time}"])

    realsense_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(package_dir, "camera_realsense.launch.py"
            )
        ),
        launch_arguments=[("camera_yaml_path", camera_yaml_path.perform(context))],
        condition=LaunchConfigurationEquals("camera_type", "d435"),
    )
    
    joy_node = Node(
        package="joy",
        executable="joy_node",
    )

    io_node = Node(
        package="follow_the_leader", executable="io_manager", output="screen"
    )

    bagfile_path = PathJoinSubstitution([log_path, "bag"])
    bag_recorder = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "--all",
            "--compression-mode file",  # other option is by `message`
            "--compression-format zstd",
            "--exclude \".*(compressed|theora).*\"",
            "--output",
            bagfile_path,
        ],
        shell=True,  # need to use args with options
        output="screen",
        log_cmd=True,
        condition=IfCondition(LaunchConfiguration("logging")),
    )

    core_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("follow_the_leader"),
                "core_ftl_3d.launch.py",
            )
        ),
        launch_arguments=[
            ("core_params_file", core_yaml_path),
            ("camera_params_file", camera_yaml_path),
            ("logging", logging),
            ("log_path", log_path),
            ("debug_mode", debug_mode)
        ],
    )

    _to_run = [
        realsense_launch,
        joy_node,
        io_node,
        core_launch,
        bag_recorder,
    ]
    return _to_run

def generate_launch_description():
    logging_arg = DeclareLaunchArgument(
        "logging",
        default_value="true",
        description="If true, bagfile and pickles saved",
    )
    debug_mode_arg = DeclareLaunchArgument(
        "debug_mode",
        default_value="false",
        description="If true, debug mode is enabled"
    )
    log_folder_arg = DeclareLaunchArgument(
        "log_folder",
        default_value=os.path.join(os.path.expanduser("~"), "bagfiles"),
        description="Existing folder where logs are saved",
    )
    ur_type_arg = DeclareLaunchArgument(
        "ur_type",
        default_value="ur5e",
        description="Robot description name (consistent with ur_control.launch.py)",
    )
    camera_type_arg = DeclareLaunchArgument(
        name="camera_type",
        default_value="d435",  # TODO: get this value from the orig launch file? Or declare it in the other file
        description="Path to the YAML file containing camera parameters",
    )

    declared_args = [
        logging_arg,
        debug_mode_arg,
        log_folder_arg,
        ur_type_arg,
        camera_type_arg,
    ]

    ld = LaunchDescription(declared_args + [OpaqueFunction(function=launch_setup)])

    return ld
