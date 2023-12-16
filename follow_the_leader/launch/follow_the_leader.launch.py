#!/usr/bin/env python3
import launch
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    SetLaunchConfiguration,
    EmitEvent,
    ExecuteProcess,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PythonExpression, AndSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    load_core = LaunchConfiguration("load_core")
    use_sim = LaunchConfiguration("use_sim")
    launch_blender = LaunchConfiguration("launch_blender")
    camera_type = LaunchConfiguration("camera_type")
    linear_slider = LaunchConfiguration("linear_slider")
    external_camera = LaunchConfiguration("external_camera")

    package_dir = get_package_share_directory("follow_the_leader")
    params_path = os.path.join(package_dir, "config")

    # ==============
    # Non-simulation
    # ==============

    # Load the YAML config files
    core_yaml_path = PythonExpression(
        ["'{}/ftl_{}.yaml'.format(r'", params_path, "', '", ur_type, "')"]
    )
    camera_yaml_path = PythonExpression(
        ["'{}/camera_{}.yaml'.format(r'", params_path, "', '", camera_type, "')"]
    )
    linear_slider_yaml_path = PythonExpression(
        ["'{}/linear_slider.yaml'.format(f'", params_path, "')"]
    )
    usb_cam_yaml_path = PythonExpression(
        ["'{}/logitech_c920.yaml'.format(f'", params_path, "')"]
    )

    camera_params_arg = DeclareLaunchArgument(
        name="camera_type",
        default_value=camera_yaml_path,  # TODO: get this value from the orig launch file? Or declare it in the other file
        description="Path to the YAML file containing camera parameters",
    )

    realsense_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"), "launch/rs_launch.py"
            )
        ),
        launch_arguments=[
            ("enable_depth", "true"),
            ("pointcloud.enable", "false"),
            ("rgb_camera.profile", "424x240x30"),
            # ("rgb_camera.profile", "640x480x30"),
            ("depth_module.profile", "424x240x30"),
        ],
        condition=UnlessCondition(
            use_sim
        ),  # TODO: add unless condition for other camera makers
    )

    # ==============
    # Simulation
    # ==============

    launch_blender_arg = DeclareLaunchArgument(
        "launch_blender",
        default_value="false",
        description="Launches Blender sim environment if enabled.",
    )

    blender_node = Node(
        package="follow_the_leader",
        executable="blender",
        parameters=[core_yaml_path, camera_yaml_path],
        condition=IfCondition(use_sim) and IfCondition(launch_blender),
    )

    # ==============
    # Core
    # ==============
    ur_type_arg = DeclareLaunchArgument(
        "ur_type",
        default_value="ur3",
        description="Robot description name (consistent with ur_control.launch.py)",
    )
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip", default_value="169.254.177.232", description="Robot IP"
    )
    load_core_arg = DeclareLaunchArgument(
        "load_core",
        default_value="true",
        description="If true, loads the core modules for 3D FTL",
    )
    use_sim_arg = DeclareLaunchArgument(
        "use_sim",
        default_value="false",
        description="If true, uses the fake controllers",
    )

    linear_slider_arg = DeclareLaunchArgument(
        "linear_slider",
        default_value="false",
        description="If true, uses the linear slider base to add an addtional degree of freedom.",
    )

    linear_slider_node = Node(
        package="follow_the_leader",
        executable="linear_slider",
        output="screen",
        parameters=[linear_slider_yaml_path],
        condition=IfCondition(linear_slider),
    )

    ur_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("follow_the_leader"), "ur_startup.launch.py"
            )
        ),
        launch_arguments=[
            ("robot_ip", robot_ip),
            ("ur_type", ur_type),
            ("use_fake_hardware", use_sim),
        ],
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
    )

    io_node = Node(
        package="follow_the_leader", executable="io_manager", output="screen"
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
        ],
        condition=IfCondition(load_core),
    )

    # ==============
    # ROS2 BAG
    # ==============

    from datetime import datetime

    now = datetime.now()
    date_time = now.strftime("%d%b%Y_%H:%M:%S")
    bagfile = os.path.join(os.path.expanduser("~"), "bagfiles", f"ftl_{date_time}")

    ros_bag_execute = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "--all",
            "--compression-mode file",  # other option is by `message`
            "--compression-format zstd",
            f'--output "{bagfile}"',
        ],
        shell=True,  # need to use args with options
        output="screen",
        log_cmd=True,
    )

    # #===============
    # # EXTERNAL CAMERA
    # #================
    # external_camera_arg = DeclareLaunchArgument("external_camera", default_value="true", description="If true, uses an external camera")
    # external_camera_node = Node(
    #     package="usb_cam",
    #     executable="usb_cam_node_exe",
    #     namespace="external_camera",
    #     output="screen",
    #     parameters=[usb_cam_yaml_path],
    #     condition=IfCondition(external_camera)
    # )

    return LaunchDescription(
        [
            # Launch args
            ur_type_arg,
            robot_ip_arg,
            use_sim_arg,
            load_core_arg,
            launch_blender_arg,
            camera_params_arg,
            linear_slider_arg,
            # external_camera_arg,

            # Nodes, launch descriptions
            ur_launch,
            joy_node,
            io_node,
            realsense_launch,
            core_launch,
            blender_node,
            # external_camera_node,
            # linear_slider_node
            ros_bag_execute,
        ]
    )
