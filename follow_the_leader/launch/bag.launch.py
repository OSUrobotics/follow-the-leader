#!/usr/bin/env python3
import launch
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    SetLaunchConfiguration,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
    LogInfo,
    GroupAction,
)
from launch.event_handlers import (
    OnExecutionComplete,
    OnProcessExit,
    OnProcessIO,
    OnProcessStart,
    OnShutdown,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
    TextSubstitution,
    PathJoinSubstitution,
)
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, SetUseSimTime
import os
from datetime import datetime


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    camera_type = LaunchConfiguration("camera_type")
    logging = LaunchConfiguration("logging")
    log_folder = LaunchConfiguration("log_folder")

    package_dir = get_package_share_directory("follow_the_leader")
    params_path = os.path.join(package_dir, "config")

    # Load the YAML config files
    core_yaml_path = PythonExpression(
        ["'{}/ftl_{}.yaml'.format(r'", params_path, "', '", ur_type, "')"]
    )
    camera_yaml_path = PythonExpression(
        ["'{}/camera_{}.yaml'.format(r'", params_path, "', '", camera_type, "')"]
    )

    # ==============
    # Core
    # ==============
    ur_type_arg = DeclareLaunchArgument(
        "ur_type",
        default_value="ur5e",
        description="Robot description name (consistent with ur_control.launch.py)",
    )
    camera_params_arg = DeclareLaunchArgument(
        name="camera_type",
        default_value=camera_yaml_path,  # TODO: get this value from the orig launch file? Or declare it in the other file
        description="Path to the YAML file containing camera parameters",
    )
    logging_arg = DeclareLaunchArgument(
        "logging",
        default_value="true",
        description="If true, bagfile and pickles saved",
    )
    log_folder_arg = DeclareLaunchArgument(
        "log_folder",
        default_value=os.path.join(os.path.expanduser("~"), "bagfiles"),
        description="Existing folder where logs are saved",
    )
    # create logging folder per instance
    now = datetime.now()
    date_time = now.strftime("%d%b%Y_%H:%M:%S")
    log_path = PathJoinSubstitution([log_folder, f"ftl_{date_time}"])
    # os.makedirs(log_path.perform)

    state_manager_node = Node(
        package="follow_the_leader",
        executable="state_manager",
        output="screen",
        parameters=[core_yaml_path, {"use_sim_time": True}],
    )

    point_tracker_node = Node(
        package="follow_the_leader",
        executable="point_tracker",
        output="screen",
        parameters=[core_yaml_path, camera_yaml_path, {"use_sim_time": True}],
    )

    modeling_node = Node(
        name="curve_3d_model_node",
        package="follow_the_leader",
        executable="model",
        output="screen",
        parameters=[
            core_yaml_path,
            camera_yaml_path,
            {"logging": logging},
            {"log_path": log_path},
            {"use_sim_time": True},
        ],
    )

    # # TODO: only begin play after all nodes setup
    # # ==============
    # # ROS2 BAG PLAY
    # # ==============

    # ros_bag_execute = ExecuteProcess(
    #     cmd=[
    #         "ros2",
    #         "bag",
    #         "play",
    #         LaunchConfiguration("log_folder"),
    #     ],
    #     shell=True,  # need to use args with options
    #     output="screen",
    #     log_cmd=True,
    # )

    return LaunchDescription(
        [
            # Launch args
            ur_type_arg,
            logging_arg,
            log_folder_arg,
            camera_params_arg,
            GroupAction(
                actions=[
                    SetUseSimTime(value=True),
                    # Nodes, launch descriptions
                    state_manager_node,
                    point_tracker_node,
                    modeling_node,
                ]
            ),
        ]
    )
