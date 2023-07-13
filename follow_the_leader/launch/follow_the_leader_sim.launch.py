import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetLaunchConfiguration, EmitEvent
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type')
    robot_ip = LaunchConfiguration('robot_ip')
    load_core = LaunchConfiguration('load_core')
    launch_blender = LaunchConfiguration('launch_blender')

    # Load the YAML file
    package_dir = get_package_share_directory('follow_the_leader')
    params_path = os.path.join(package_dir, 'config')
    yaml_file_path = PythonExpression(["'{}/ftl_{}.yaml'.format(r'", params_path, "', '", ur_type, "')"])

    ur_type_arg = DeclareLaunchArgument(
        'ur_type', default_value='ur3', description='Robot description name (consistent with ur_control.launch.py)'
    )
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value='169.254.57.122', description='Robot IP'
    )

    load_core_arg = DeclareLaunchArgument(
        'load_core', default_value='true', description='If true, loads the core modules for 3D FTL',
    )

    launch_blender_arg = DeclareLaunchArgument(
        'launch_blender', default_value='true', description='Launches Blender sim environment if enabled.',
    )

    ur_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('follow_the_leader'), 'ur_startup.launch.py')
        ),
        launch_arguments=[
            ('robot_ip', robot_ip),
            ('ur_type', ur_type),
            ('use_fake_hardware', 'true'),
        ]
    )

    blender_node = Node(
        package='follow_the_leader',
        executable='blender',
        output='screen',
        parameters=[yaml_file_path],
        condition=IfCondition(launch_blender),
    )

    core_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('follow_the_leader'), 'core_ftl_3d.launch.py')
        ),
        launch_arguments={'params_file': yaml_file_path}.items(),
        condition=IfCondition(load_core)
    )

    return LaunchDescription([
        ur_type_arg, robot_ip_arg, load_core_arg, launch_blender_arg,
        ur_launch, blender_node, core_launch
    ])
