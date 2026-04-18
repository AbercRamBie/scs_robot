from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share_dir = get_package_share_directory('semantic_comm_runtime')
    joystick_config = os.path.join(share_dir, 'config', 'joyStick.yaml')

    robot_serial_port_arg = DeclareLaunchArgument(
        'robot_serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port for robot communication'
    )

    robot_serial_baud_arg = DeclareLaunchArgument(
        'robot_serial_baud',
        default_value='115200',
        description='Serial baud rate for robot'
    )

    robot_radius_arg = DeclareLaunchArgument(
        'robot_radius',
        default_value='0.15',
        description='Robot radius in meters (from center to wheel contact point)'
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{'dev': '/dev/input/js0', 'autorepeat_rate': 20.0}]
    )

    joy_control = Node(
        package='semantic_comm_runtime',
        executable='joy_control_node',
        name='joy_control_node',
        output='screen',
        parameters=[joystick_config]
    )

    robot_driver = Node(
        package='semantic_comm_runtime',
        executable='robot_driver_node',
        name='robot_driver_node',
        output='screen',
        parameters=[
            {'robot_serial_port': LaunchConfiguration('robot_serial_port')},
            {'robot_serial_baud': LaunchConfiguration('robot_serial_baud')},
            {'robot_radius': LaunchConfiguration('robot_radius')},
        ]
    )

    return LaunchDescription([
        robot_serial_port_arg,
        robot_serial_baud_arg,
        robot_radius_arg,
        joy_node,
        joy_control,
        robot_driver,
    ])
