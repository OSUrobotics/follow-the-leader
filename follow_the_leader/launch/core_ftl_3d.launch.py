import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetLaunchConfiguration, EmitEvent
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    core_params_file = LaunchConfiguration("core_params_file")
    camera_params_file = LaunchConfiguration("camera_params_file")
    # Load the YAML files
    package_dir = get_package_share_directory("follow_the_leader")
    params_path = os.path.join(package_dir, "config", "ftl_ur5e.yaml")
    params_arg = DeclareLaunchArgument(
        "core_params_file", default_value=params_path, description="Path to the YAML file containing node parameters"
    )
    # camera_params_arg = DeclareLaunchArgument(
    #     name="camera_params_file",
    #     default_value=camera_params_path, # TODO: get this value from the orig launch file? Or declare it in the other file
    #     description="Path to the YAML file containing camera parameters"
    # )

    state_manager_node = Node(
        package="follow_the_leader",
        executable="state_manager",
        output="screen",
        parameters=[core_params_file],
    )

    image_processor_node = Node(
        package="follow_the_leader",
        executable="image_processor",
        output="screen",
        parameters=[core_params_file, camera_params_file],
    )

    point_tracker_node = Node(
        package="follow_the_leader",
        executable="point_tracker",
        output="screen",
        parameters=[core_params_file, camera_params_file],
    )

    modeling_node = Node(
        package="follow_the_leader",
        executable="model",
        output="screen",
        parameters=[core_params_file, camera_params_file],
    )

    controller_node = Node(
        package="follow_the_leader",
        executable="controller_3d",
        # output='screen',
        parameters=[core_params_file],
    )

    servoing_node = Node(
        package="follow_the_leader",
        executable="visual_servoing",
        output="screen",
        parameters=[core_params_file],
    )

    return LaunchDescription(
        [
            params_arg,
            camera_params_arg,
            state_manager_node,
            image_processor_node,
            point_tracker_node,
            modeling_node,
            controller_node,
            servoing_node,
        ]
    )
