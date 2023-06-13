import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetLaunchConfiguration, EmitEvent
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    ur_type = LaunchConfiguration('ur_type')
    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    load_core = LaunchConfiguration('load_core')

    # Load the YAML file
    package_dir = get_package_share_directory('follow_the_leader')
    params_path = os.path.join(package_dir, 'config', 'ftl_ur3.yaml')
    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=params_path,
        description='Path to the YAML file containing node parameters'
    )



    ur_type_arg = DeclareLaunchArgument(
        'ur_type', default_value='ur3', description='Robot description name (consistent with ur_control.launch.py)'
    )
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value='169.254.57.122', description='Robot IP'
    )

    use_fake_hardware_arg = DeclareLaunchArgument(
        'use_fake_hardware', default_value='true', description='If true, uses the fake controllers'
    )

    load_core_arg = DeclareLaunchArgument(
        'load_core', default_value='true', description='If true, loads the core modules for 3D FTL',
    )


    ur_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('follow_the_leader'), 'ur_startup.launch.py')
        ),
        launch_arguments=[
            ('robot_ip', robot_ip),
            ('ur_type', ur_type),
            ('use_fake_hardware', use_fake_hardware),
        ]
    )

    realsense_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch/rs_launch.py')
        ),
        launch_arguments=[
            ('enable_depth', 'false'),
            ('pointcloud.enable', 'false'),
        ]
    )

    core_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('follow_the_leader'), 'launch/core_ftl_3d.launch')
        ),
        launch_arguments=[
            ('params_file', params_file)
        ],
        condition=IfCondition(load_core)
    )

    return LaunchDescription([
        params_arg, ur_type_arg, robot_ip_arg, use_fake_hardware_arg, load_core_arg,
        ur_launch, realsense_launch, core_launch
    ])
