from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share_dir = get_package_share_directory('semantic_comm_runtime')
    runtime_config = os.path.join(share_dir, 'config', 'runtime.yaml')

    robot_serial_port_arg = DeclareLaunchArgument(
        'robot_serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port the Arduino is connected to',
    )
    robot_serial_baud_arg = DeclareLaunchArgument(
        'robot_serial_baud',
        default_value='9600',
        description='Baud rate for the Arduino serial link',
    )
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device index for vision_node',
    )
    show_debug_windows_arg = DeclareLaunchArgument(
        'show_debug_windows',
        default_value='false',
        description='Open OpenCV debug windows for vision_node',
    )
    enable_vision_arg = DeclareLaunchArgument(
        'enable_vision',
        default_value='true',
        description='Start vision_node',
    )
    enable_nav_arg = DeclareLaunchArgument(
        'enable_nav',
        default_value='true',
        description='Start nav_node',
    )
    enable_robot_driver_arg = DeclareLaunchArgument(
        'enable_robot_driver',
        default_value='true',
        description='Start robot_driver_node',
    )
    enable_semantic_stack_arg = DeclareLaunchArgument(
        'enable_semantic_stack',
        default_value='false',
        description='Start encoder/channel/decoder stack; this currently expects /scan input',
    )
    snr_db_arg = DeclareLaunchArgument(
        'snr_db',
        default_value='5.0',
        description='Channel SNR when semantic stack is enabled',
    )

    vision = Node(
        package='semantic_comm_runtime',
        executable='vision_node',
        name='vision_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_vision')),
        parameters=[
            {
                'camera_id': LaunchConfiguration('camera_id'),
                'show_debug_windows': LaunchConfiguration('show_debug_windows'),
            }
        ],
    )

    nav = Node(
        package='semantic_comm_runtime',
        executable='nav_node',
        name='nav_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_nav')),
    )

    robot_driver = Node(
        package='semantic_comm_runtime',
        executable='robot_driver_node',
        name='robot_driver_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_robot_driver')),
        parameters=[
            {
                'robot_serial_port': LaunchConfiguration('robot_serial_port'),
                'robot_serial_baud': LaunchConfiguration('robot_serial_baud'),
            }
        ],
    )

    encoder = Node(
        package='semantic_comm_runtime',
        executable='encoder_node',
        name='encoder_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_semantic_stack')),
        parameters=[runtime_config],
    )

    channel = Node(
        package='semantic_comm_runtime',
        executable='channel_node',
        name='channel_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_semantic_stack')),
        parameters=[runtime_config, {'snr_db': LaunchConfiguration('snr_db')}],
    )

    decoder = Node(
        package='semantic_comm_runtime',
        executable='decoder_node',
        name='decoder_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_semantic_stack')),
        parameters=[runtime_config],
    )

    return LaunchDescription([
        robot_serial_port_arg,
        robot_serial_baud_arg,
        camera_id_arg,
        show_debug_windows_arg,
        enable_vision_arg,
        enable_nav_arg,
        enable_robot_driver_arg,
        enable_semantic_stack_arg,
        snr_db_arg,
        vision,
        nav,
        robot_driver,
        encoder,
        channel,
        decoder,
    ])
