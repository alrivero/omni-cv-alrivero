"""
=====================================================================
 * MIT License
 *
 * Copyright (c) 2025 Omni Instrument Inc.
 * ...
 * =====================================================================
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


_HOME = os.path.expanduser("~")

_CONFIG_MAP = {
    "default": os.path.join(_HOME, "ros2_ws", "src", "tsdf_saver", "config", "custom.yaml"),
    "waft":    os.path.join(_HOME, "ros2_ws", "src", "tsdf_saver", "config", "waft.yaml"),
    "quality": os.path.join(_HOME, "ros2_ws", "src", "tsdf_saver", "config", "quality.yaml"),
}


def launch_setup(context, *_):

    depth_method   = LaunchConfiguration("depth_method").perform(context)
    config_preset  = LaunchConfiguration("config_preset").perform(context)
    bag_rate       = LaunchConfiguration("bag_rate").perform(context)
    bag_path       = LaunchConfiguration("bag").perform(context)
    save_threshold = LaunchConfiguration("save_time_threshold").perform(context)
    qos_yaml       = os.path.join(_HOME, "dataset", "qos_override.yaml")

    config_path = _CONFIG_MAP.get(config_preset, _CONFIG_MAP["default"])

    # ==============================================================
    #  Startup banner
    # ==============================================================
    B, R = "\033[1m", "\033[0m"
    print(f"\n{B}{'=' * 54}{R}")
    print(f"{B}  Depth Method  : {depth_method.upper()}{R}")
    print(f"{B}  Config Preset : {config_preset.upper()}  →  {os.path.basename(config_path)}{R}")
    print(f"{B}  Bag Rate      : {bag_rate}x{R}")
    print(f"{B}{'=' * 54}{R}\n")

    # ==============================================================
    #  Stereo Depth Nodes
    # ==============================================================

    _depth_nodes = {
        "sgbm": Node(
            package="stereo_depth",
            executable="stereo_depth_node",
            name="stereo_depth_node",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
        "waft": Node(
            package="waft_stereo",
            executable="waft_stereo_node",
            name="waft_stereo_node",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "config_file": "configs/SynLarge/DAv2L-5.yaml",
                "hf_filename": "SynLarge/DAv2L-5.pth",
            }],
        ),
        "unidepth": Node(
            package="unidepth_ros",
            executable="unidepth_node",
            name="unidepth_node",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
    }
    depth_node = _depth_nodes[depth_method]

    depth_selector_node = Node(
        package="waft_stereo",
        executable="depth_selector_node",
        name="depth_selector",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "method": depth_method,
        }],
    )

    # ==============================================================
    #  PointCloud Node (XYZ)
    # ==============================================================

    pointcloud_component = ComposableNode(
        package="depth_image_proc",
        plugin="depth_image_proc::PointCloudXyzNode",
        name="depth_image_pointcloud",
        parameters=[
            {"use_sim_time": True},
            {"queue_size": 20},
        ],
        remappings=[
            ("/zed/zedxm/depth/camera_info", "/zed/zedxm/depth/depth_registered/camera_info"),
            ("image_rect", "/zed/zedxm/depth/depth_registered"),
            ("points", "/stereo/points"),
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # ==============================================================
    #  TSDF Saver Component
    # ==============================================================

    tsdf_saver_component = ComposableNode(
        package="tsdf_saver",
        plugin="tsdf_saver::ExactTimeSaver",
        name="tsdf_saver",
        parameters=[
            {"use_sim_time": True},
            {"save_time_threshold": float(save_threshold)},
        ],
        remappings=[("cloud_in", "/stereo/points")],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # ==============================================================
    #  DB-TSDF Node
    # ==============================================================

    db_tsdf_node = Node(
        package="db_tsdf",
        executable="db_tsdf_node",
        name="db_tsdf_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            config_path,
        ],
        remappings=[("cloud", "/tsdf/local_cloud")],
    )

    # ==============================================================
    #  Container
    # ==============================================================

    container = ComposableNodeContainer(
        name="stereo_tsdf_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        arguments=["--use_multi_threaded_executor", "--ros-args", "--log-level", "info"],
        output="screen",
        parameters=[{"use_sim_time": True}],
        composable_node_descriptions=[
            pointcloud_component,
            tsdf_saver_component,
        ],
    )

    # ==============================================================
    #  Bag Playback
    # ==============================================================

    # Exclude the bag's pre-recorded ZED depth: it would publish directly to
    # /zed/zedxm/depth/depth_registered and compete with our computed depth,
    # corrupting whichever pipeline method we've chosen.
    bag_proc = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            bag_path,
            "--clock",
            "--rate", bag_rate,
            "--qos-profile-overrides-path", qos_yaml,
            "--topics",
            "/tf",
            "/tf_static",
            "/zed/zedxm/left/color/rect/image",
            "/zed/zedxm/left/color/rect/image/camera_info",
            "/zed/zedxm/left/gray/rect/image",
            "/zed/zedxm/left/gray/rect/image/camera_info",
            "/zed/zedxm/right/color/rect/image",
            "/zed/zedxm/right/color/rect/image/camera_info",
            "/zed/zedxm/right/gray/rect/image",
            "/zed/zedxm/right/gray/rect/image/camera_info",
            "/zed/zedxm/odom",
        ],
        output="screen",
    )

    # ==============================================================
    #  Save mesh then shutdown once bag finishes
    # ==============================================================

    save_proc = ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/save_grid_mesh', 'std_srvs/srv/Trigger', '{}'],
        output='screen',
    )

    # After the bag finishes, wait for the depth queue to drain before saving.
    # WAFT is slow (~600 ms/frame × 566 frames) so needs up to 400 s to catch up.
    drain_time = 400.0 if depth_method == "waft" else 60.0

    save_and_shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=bag_proc,
            on_exit=[
                TimerAction(
                    period=drain_time,
                    actions=[save_proc],
                ),
                TimerAction(
                    period=drain_time + 30.0,
                    actions=[Shutdown(reason="Mesh saved; shutting down.")],
                ),
            ],
        )
    )

    return [
        depth_node,
        depth_selector_node,
        bag_proc,
        container,
        db_tsdf_node,
        save_and_shutdown,
    ]


def generate_launch_description():

    default_bag_path = os.path.join(_HOME, "dataset", "VIO_stripped")

    return LaunchDescription([

        DeclareLaunchArgument(
            "bag",
            default_value=default_bag_path,
            description="Path to rosbag directory",
        ),
        DeclareLaunchArgument(
            "save_time_threshold",
            default_value="9999999999.0",
            description="When pointcloud time exceeds this value, TSDF saver triggers (default: never)",
        ),
        DeclareLaunchArgument(
            "depth_method",
            default_value="sgbm",
            description='Depth source: "sgbm" or "waft"',
        ),
        DeclareLaunchArgument(
            "config_preset",
            default_value="default",
            description='TSDF config preset: "default" (custom.yaml), "waft" (waft.yaml), or "quality" (quality.yaml)',
        ),
        DeclareLaunchArgument(
            "bag_rate",
            default_value="1.0",
            description="Bag playback rate multiplier (e.g. 0.2 for WAFT, 1.0 for SGBM)",
        ),

        OpaqueFunction(function=launch_setup),
    ])
