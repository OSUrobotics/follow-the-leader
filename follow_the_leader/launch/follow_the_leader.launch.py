#!/usr/bin/env python3
import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetLaunchConfiguration, EmitEvent
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PythonExpression, AndSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type')
    robot_ip = LaunchConfiguration('robot_ip')
    load_core = LaunchConfiguration('load_core')
    use_sim = LaunchConfiguration('use_sim')
    launch_blender = LaunchConfiguration('launch_blender')

    # Load the YAML file
    package_dir = get_package_share_directory('follow_the_leader')
    params_path = os.path.join(package_dir, 'config')
    yaml_file_path = PythonExpression(["'{}/ftl_{}.yaml'.format(r'", params_path, "', '", ur_type, "')"])

    ur_type_arg = DeclareLaunchArgument(
        'ur_type', default_value='ur3', description='Robot description name (consistent with ur_control.launch.py)'
    )
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value='169.254.174.50', description='Robot IP'
    )

    load_core_arg = DeclareLaunchArgument(
        'load_core', default_value='true', description='If true, loads the core modules for 3D FTL',
    )

    use_sim_arg = DeclareLaunchArgument(
        'use_sim', default_value='false', description='If true, uses the fake controllers'
    )

    ur_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('follow_the_leader'), 'ur_startup.launch.py')
        ),
        launch_arguments=[
            ('robot_ip', robot_ip),
            ('ur_type', ur_type),
            ('use_fake_hardware', use_sim),
        ]
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
    )

    io_node = Node(
        package='follow_the_leader',
        executable='io_manager',
        output='screen'
    )

    core_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('follow_the_leader'), 'core_ftl_3d.launch.py')
        ),
        launch_arguments=[
            ('params_file', yaml_file_path)
        ],
        condition=IfCondition(load_core)
    )

    # ==============
    # Non-simulation
    # ==============

    realsense_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch/rs_launch.py')
        ),
        launch_arguments=[
            ('enable_depth', 'false'),
            ('pointcloud.enable', 'false'),
            ('rgb_camera.profile', '424x240x30'),
            ('depth_module.profile', '424x240x30'),
        ],
        condition=UnlessCondition(use_sim),
    )

    # ==============
    # Simulation
    # ==============

    launch_blender_arg = DeclareLaunchArgument(
        'launch_blender', default_value='false', description='Launches Blender sim environment if enabled.',
    )

    blender_node = Node(
        package='follow_the_leader',
        executable='blender',
        parameters=[yaml_file_path],
        condition=IfCondition(use_sim) and IfCondition(launch_blender),
    )

    return LaunchDescription([
        ur_type_arg, robot_ip_arg, use_sim_arg, load_core_arg, launch_blender_arg,
        ur_launch, joy_node, io_node, realsense_launch, core_launch, blender_node,
    ])
