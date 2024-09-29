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
from launch.actions import SetEnvironmentVariable
import os
from datetime import datetime


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    camera_type = LaunchConfiguration("camera_type")
    logging = LaunchConfiguration("logging")
    log_folder = LaunchConfiguration("log_folder")

    package_dir = get_package_share_directory("follow_the_leader")
    params_path = os.path.join(package_dir, "config")
    bagfile_path = LaunchConfiguration("bagfile_path")

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
    # create logging folder per instance
    now = datetime.now()
    date_time = now.strftime("%d%b%Y_%H:%M:%S")
    log_path = PathJoinSubstitution([log_folder, f"ftl_{date_time}"])

    state_manager_node = Node(
        package="follow_the_leader",
        executable="state_manager",
        output="screen",
        parameters=[core_yaml_path, {"log_path": log_path}],
    )

    point_tracker_node = Node(
        package="follow_the_leader",
        executable="point_tracker",
        output="screen",
        parameters=[core_yaml_path, camera_yaml_path, {"log_path": log_path}],
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

    # ==============
    # ROS2 BAG PLAY
    # ==============

    ros_bag_execute = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "--clock 1.0",
            "--rate 2.0",
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

    return LaunchDescription(
        [
            # Launch args
            SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
            # SetEnvironmentVariable("ROS_LOG_DIR", log_path),
            ur_type_arg,
            camera_params_arg,
            logging_arg,
            log_folder_arg,
            bagfile_path_arg,
            realsense_depth_launch,
            GroupAction(
                actions=[
                    SetUseSimTime(value=True),
                    state_manager_node,
                    point_tracker_node,
                    modeling_node,
                    ros_bag_execute
                ]
            ),
        ]
    )
