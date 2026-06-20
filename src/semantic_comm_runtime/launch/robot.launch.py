from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import TimerAction
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
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='',
        description='Optional camera device path such as /dev/video0',
    )
    show_recording_preview_arg = DeclareLaunchArgument(
        'show_recording_preview',
        default_value='false',
        description='Show OpenCV preview while recording startup scan',
    )
    show_processing_preview_arg = DeclareLaunchArgument(
        'show_processing_preview',
        default_value='false',
        description='Show OpenCV preview during YOLO post-processing',
    )
    process_after_recording_arg = DeclareLaunchArgument(
        'process_after_recording',
        default_value='true',
        description='Run YOLO processing after startup scan recording',
    )
    enable_startup_scan_arg = DeclareLaunchArgument(
        'enable_startup_scan',
        default_value='true',
        description='Run startup_scan_node to trigger one 360 scan at startup',
    )
    startup_delay_sec_arg = DeclareLaunchArgument(
        'startup_delay_sec',
        default_value='2.0',
        description='Delay before startup scan motion begins',
    )
    startup_spin_duration_sec_arg = DeclareLaunchArgument(
        'startup_spin_duration_sec',
        default_value='20.0',
        description='Startup scan spin duration in seconds',
    )
    startup_angular_speed_z_arg = DeclareLaunchArgument(
        'startup_angular_speed_z',
        default_value='0.314',
        description='Angular velocity for startup scan spin (rad/s)',
    )
    autonomy_start_delay_sec_arg = DeclareLaunchArgument(
        'autonomy_start_delay_sec',
        default_value='0.0',
        description='Delay (seconds) before starting nav + semantic stack after launch',
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
        default_value='true',
        description='Start encoder/channel/decoder stack (expects /scan input)',
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
                'camera_device': LaunchConfiguration('camera_device'),
                'process_after_recording': LaunchConfiguration('process_after_recording'),
                'show_recording_preview': LaunchConfiguration('show_recording_preview'),
                'show_processing_preview': LaunchConfiguration('show_processing_preview'),
            }
        ],
    )

    startup_scan = Node(
        package='semantic_comm_runtime',
        executable='startup_scan_node',
        name='startup_scan_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_startup_scan')),
        parameters=[
            {
                'start_delay_sec': LaunchConfiguration('startup_delay_sec'),
                'spin_duration_sec': LaunchConfiguration('startup_spin_duration_sec'),
                'angular_speed_z': LaunchConfiguration('startup_angular_speed_z'),
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

    delayed_autonomy = TimerAction(
        period=LaunchConfiguration('autonomy_start_delay_sec'),
        actions=[nav, encoder, channel, decoder],
    )

    return LaunchDescription([
        robot_serial_port_arg,
        robot_serial_baud_arg,
        camera_id_arg,
        camera_device_arg,
        show_recording_preview_arg,
        show_processing_preview_arg,
        process_after_recording_arg,
        enable_startup_scan_arg,
        startup_delay_sec_arg,
        startup_spin_duration_sec_arg,
        startup_angular_speed_z_arg,
        autonomy_start_delay_sec_arg,
        enable_vision_arg,
        enable_nav_arg,
        enable_robot_driver_arg,
        enable_semantic_stack_arg,
        snr_db_arg,
        vision,
        startup_scan,
        robot_driver,
        delayed_autonomy,
    ])
