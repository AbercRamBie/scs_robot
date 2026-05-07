from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share_dir = get_package_share_directory('semantic_comm_runtime')
    runtime_config = os.path.join(share_dir, 'config', 'runtime.yaml')

    # ── Launch arguments ──────────────────────────────────────────────────────
    serial_port_arg = DeclareLaunchArgument(
        'robot_serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port the Arduino is connected to'
    )
    serial_baud_arg = DeclareLaunchArgument(
        'robot_serial_baud',
        default_value='9600',
        description='Baud rate – must match Serial.begin() in Arduino sketch'
    )
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device index'
    )
    show_windows_arg = DeclareLaunchArgument(
        'show_debug_windows',
        default_value='false',
        description='Set true to open OpenCV tracking windows (requires display)'
    )
    snr_arg = DeclareLaunchArgument(
        'snr_db',
        default_value='5.0',
        description='Simulated channel SNR in dB'
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    vision = Node(
        package='semantic_comm_runtime',
        executable='vision_node',
        name='vision_node',
        output='screen',
        parameters=[{
            'camera_id': LaunchConfiguration('camera_id'),
            'show_debug_windows': LaunchConfiguration('show_debug_windows'),
        }]
    )

    encoder = Node(
        package='semantic_comm_runtime',
        executable='encoder_node',
        name='encoder_node',
        output='screen',
        parameters=[runtime_config]
    )

    channel = Node(
        package='semantic_comm_runtime',
        executable='channel_node',
        name='channel_node',
        output='screen',
        parameters=[
            runtime_config,
            {'snr_db': LaunchConfiguration('snr_db')},
        ]
    )

    decoder = Node(
        package='semantic_comm_runtime',
        executable='decoder_node',
        name='decoder_node',
        output='screen',
        parameters=[runtime_config]
    )

    nav = Node(
        package='semantic_comm_runtime',
        executable='nav_node',
        name='nav_node',
        output='screen',
    )

    robot_driver = Node(
        package='semantic_comm_runtime',
        executable='robot_driver_node',
        name='robot_driver_node',
        output='screen',
        parameters=[{
            'robot_serial_port': LaunchConfiguration('robot_serial_port'),
            'robot_serial_baud': LaunchConfiguration('robot_serial_baud'),
        }]
    )

    return LaunchDescription([
        serial_port_arg,
        serial_baud_arg,
        camera_id_arg,
        show_windows_arg,
        snr_arg,
        vision,
        encoder,
        channel,
        decoder,
        nav,
        robot_driver,
    ])
