from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetLaunchConfiguration, OpaqueFunction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    ur_type = LaunchConfiguration('ur_type')
    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    headless_mode = LaunchConfiguration('headless_mode', default='true')

    initial_joint_controller = LaunchConfiguration('initial_joint_controller', default='scaled_joint_trajectory_controller')
    set_joint_controller = SetLaunchConfiguration('initial_joint_controller', value='joint_trajectory_controller', condition=LaunchConfigurationEquals('use_fake_hardware', 'true'))

    ur_type_arg = DeclareLaunchArgument(
        'ur_type', default_value='ur3', description='Robot description name (consistent with ur_control.launch.py)'
    )
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value='169.254.57.122', description='Robot IP'
    )

    use_fake_hardware_arg = DeclareLaunchArgument(
        'use_fake_hardware', default_value='true', description='If true, uses the fake controllers'
    )

    ur_base_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ur_robot_driver'), 'launch/ur_control.launch.py')
        ),
        launch_arguments=[
            ('robot_ip', robot_ip),
            ('ur_type', ur_type),
            ('use_fake_hardware', use_fake_hardware),
            ('headless_mode', headless_mode),
            ('initial_joint_controller', initial_joint_controller),
            ('launch_rviz', 'false'),
        ]
    )

    ur_moveit_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ur_moveit_config'), 'launch/ur_moveit.launch.py')
        ),
        launch_arguments=[
            ('ur_type', ur_type),
            ('use_fake_hardware', use_fake_hardware),
            ('launch_rviz', 'false'),
        ]
    )

    tf_node_a = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments='0.0 0 0.0 0.7071068 0 0.7071068 0 tool0 camera_link'.split(),
        condition=UnlessCondition(use_fake_hardware)
    )
    tf_node_b = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments='0.1 0 0.1 0.5 -0.5 0.5 0.5 tool0 camera_link'.split(),
        condition=IfCondition(use_fake_hardware)
    )

    return LaunchDescription([
        ur_type_arg, robot_ip_arg, use_fake_hardware_arg, set_joint_controller,
        ur_base_launch, ur_moveit_launch, tf_node_a, tf_node_b,
    ])