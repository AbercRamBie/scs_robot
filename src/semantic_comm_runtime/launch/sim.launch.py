from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from ament_index_python.packages import get_package_share_directory
import os

# Ensure the system gz (not ROS vendor gz) is used so the 'sim' subcommand is available.
# ROS Jazzy ships gz_tools_vendor which only registers: help, log, msg, param, service, topic.
_GZ_ENV = {
    'GZ_CONFIG_PATH': '/usr/share/gz',
    'PATH': '/usr/bin:' + os.environ.get('PATH', ''),
}

def generate_launch_description():
    share_dir = get_package_share_directory('semantic_comm_runtime')
    runtime_config = os.path.join(share_dir, 'config', 'runtime.yaml')
    joystick_config = os.path.join(share_dir, 'config', 'joyStick.yaml')

    robot_sdf = os.path.join(
        share_dir, 'assets', 'robot', 'semantic_robot.sdf'
    )
    world = os.path.join(
        share_dir, 'assets', 'world', 'world_Basic.sdf'
    )

    snr_arg = DeclareLaunchArgument(
        'snr',
        default_value='5.0',
        description='Channel SNR in dB'
    )

    encoder_checkpoint_arg = DeclareLaunchArgument(
        'encoder_checkpoint',
        default_value='/home/subash/DiskD/RoboticsWorks/scs_robot/artifacts/checkpoints/encoder_snr10.pth',
        description='Path to the encoder checkpoint file'
    )

    decoder_checkpoint_arg = DeclareLaunchArgument(
        'decoder_checkpoint',
        default_value='/home/subash/DiskD/RoboticsWorks/scs_robot/artifacts/checkpoints/decoder_snr10.pth',
        description='Path to the decoder checkpoint file'
    )

    joy_control_config_arg = DeclareLaunchArgument(
        'joy_control_config',
        default_value=joystick_config,
        description='Path to the joystick configuration file'
    )

    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='joystick',
        description='Control mode: joystick or semantic'
    )

    gazebo = ExecuteProcess(
        cmd=['/usr/bin/gz', 'sim', world, '-v', '4', '-r'],
        output='screen',
        additional_env=_GZ_ENV,
    )

    spawn_robot = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    '/usr/bin/gz', 'service',
                    '-s', '/world/semantic_world/create',
                    '--reqtype', 'gz.msgs.EntityFactory',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '5000',
                    '--req',
                    f'sdf_filename: "{robot_sdf}", name: "semantic_robot"'
                ],
                output='screen',
                additional_env=_GZ_ENV,
            )
        ]
    )

    encoder = Node(
        package='semantic_comm_runtime',
        executable='encoder_node',
        name='encoder_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('control_mode'), "' == 'semantic'"])),
        parameters=[
            runtime_config,
            {
                'encoder_checkpoint': LaunchConfiguration('encoder_checkpoint')
            }
        ]
    )

    channel = Node(
        package='semantic_comm_runtime',
        executable='channel_node',
        name='channel_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('control_mode'), "' == 'semantic'"])),
        parameters=[
            runtime_config,
            {'snr_db': LaunchConfiguration('snr')}
        ]
    )

    decoder = Node(
        package='semantic_comm_runtime',
        executable='decoder_node',
        name='decoder_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('control_mode'), "' == 'semantic'"])),
        parameters=[
            runtime_config,
            {
                'decoder_checkpoint': LaunchConfiguration('decoder_checkpoint')
            }
        ]
    )

    nav = Node(
        package='semantic_comm_runtime',
        executable='nav_node',
        name='nav_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('control_mode'), "' == 'semantic'"])),
        parameters=[runtime_config]
    )

    joy_driver = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('control_mode'), "' == 'joystick'"])),
        parameters=[{'dev': '/dev/input/js0', 'autorepeat_rate': 20.0}]
    )

    joystick = Node(
        package='semantic_comm_runtime',
        executable='joy_control_node',
        name='joy_control_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('control_mode'), "' == 'joystick'"])),
        parameters=[LaunchConfiguration('joy_control_config')]
    )

    bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    name='ros_gz_bridge',
    output='screen',
    arguments=[
        '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
        '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
    ],
    )

    return LaunchDescription([
        snr_arg,
        encoder_checkpoint_arg,
        decoder_checkpoint_arg,
        joy_control_config_arg,
        control_mode_arg,
        gazebo,
        spawn_robot,
        encoder,
        channel,
        decoder,
        nav,
        joy_driver,
        joystick,
        bridge
    ])