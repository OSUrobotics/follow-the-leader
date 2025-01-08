#!/usr/bin/env python3
import launch
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    SetLaunchConfiguration,
    EmitEvent,
    ExecuteProcess,
    GroupAction,
    OpaqueFunction,
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
    PathJoinSubstitution,
)
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, SetUseSimTime
from launch.actions import SetEnvironmentVariable
import os
from datetime import datetime
from launch_ros.parameter_descriptions import ParameterFile


import rclpy.logging

logger = rclpy.logging.get_logger("bag.launch")


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type")
    camera_type = LaunchConfiguration("camera_type")
    logging = LaunchConfiguration("logging")
    log_folder = LaunchConfiguration("log_folder")
    bagfile_path = LaunchConfiguration("bagfile_path")

    # Get the path to the config files
    package_dir = get_package_share_directory("follow_the_leader")
    param_file_path_subst = PathJoinSubstitution([package_dir, "config"])

    # Load the YAML config files
    param_file_path_str = param_file_path_subst.perform(context=context)
    ur_type_str = LaunchConfiguration("ur_type").perform(context=context)
    camera_type_str = LaunchConfiguration("camera_type").perform(context=context)
    # logger.info(param_file_path_subst)
    logger.info(param_file_path_str)
    core_yaml = ParameterFile(
        os.path.join(param_file_path_str, f"ftl_{ur_type_str}.yaml"), allow_substs=True
    )
    camera_yaml = ParameterFile(
        os.path.join(param_file_path_str, f"camera_{camera_type_str}.yaml"),
        allow_substs=True,
    )

    # create logging folder per instance
    now = datetime.now()
    date_time = now.strftime("%d%b%Y_%H:%M:%S")
    log_path = PathJoinSubstitution([log_folder, f"ftl_{date_time}"])

    state_manager_node = Node(
        package="follow_the_leader",
        executable="state_manager",
        output="screen",
        parameters=[core_yaml, {"log_path": log_path}],
    )

    point_tracker_node = Node(
        package="follow_the_leader",
        executable="point_tracker",
        output="screen",
        parameters=[core_yaml, camera_yaml, {"log_path": log_path}],
    )

    modeling_node = Node(
        name="curve_3d_model_node",
        package="follow_the_leader",
        executable="model",
        output="screen",
        parameters=[
            core_yaml,
            camera_yaml,
            {"logging": logging},
            {"log_path": log_path},
        ],
    )

    controller_node = Node(
        package="follow_the_leader",
        executable="controller_3d",
        # output='screen',            ("use_sim_time", "true"),
        parameters=[
            core_yaml,
            ("use_sim_time", "true"),
            {"log_path": log_path},
        ],
    )

    servoing_node = Node(
        package="follow_the_leader",
        executable="visual_servoing",
        output="screen",
        parameters=[
            camera_yaml,
            ("use_sim_time", "true"),
        ],
    )

    realsense_depth_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("follow_the_leader"), "pcl.launch.py"
            )
        ),
        launch_arguments=[
            ("rviz_flag", "false"),
            ("use_sim_time", "true"),
        ],
    )

    ur_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("follow_the_leader"), "ur_startup.launch.py"
            )
        ),
        launch_arguments=[
            ("ur_type", "ur5e"),
            ("use_fake_hardware", "true"),
            ("use_sim_time", "true"),
        ],
    )

    ros_bag_execute = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "--clock 1.0",
            "--rate 0.1",
            # "--start-offset 10", # skipping start may result in miissing static tf
            "--delay 2.0",
            "--topics /state_announcement /image_mask /camera/depth/camera_info /camera/depth/metadata /camera/depth/image_rect_raw /camera/color/image_raw /camera/color/camera_info /robot_description /tf /tf_static /image_mask_pair /image_mask /joint_states",
            "--read-ahead-queue-size 100",
            "--disable-keyboard-controls",
            bagfile_path,
        ],
        shell=True,  # need to use args with options
        output="screen",
        log_cmd=True,
    )

    nodes_to_launch = [
        realsense_depth_launch,
        GroupAction(
            actions=[
                SetUseSimTime(value=True),
                state_manager_node,
                point_tracker_node,
                modeling_node,
                ros_bag_execute,
                controller_node,
                # servoing_node,
                # ur_launch,
            ]
        ),
    ]
    return nodes_to_launch


def generate_launch_description():
    ur_type_arg = DeclareLaunchArgument(
        "ur_type",
        default_value="ur5e",
        description="Robot description name (consistent with ur_control.launch.py)",
    )
    camera_params_arg = DeclareLaunchArgument(
        name="camera_type",
        default_value="d435",  # TODO: get this value from the orig launch file? Or declare it in the other file
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
    bagfile_path_arg = DeclareLaunchArgument(
        "bagfile_path",
        default_value=os.path.join(
            os.path.expanduser("~"),
            "bagfiles",
            "20240202_prosser_trials/bagfiles/tree4/ftl_02Feb2024_14:49:57/bag/bag_0.db3",
        ),
        description="Path to the bagfile",
    )

    declared_args = [
        ur_type_arg,
        camera_params_arg,
        logging_arg,
        log_folder_arg,
        bagfile_path_arg,
    ]

    ld = LaunchDescription(
        [SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")]
        + declared_args
        + [OpaqueFunction(function=launch_setup)]
    )

    return ld
