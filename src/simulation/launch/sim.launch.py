from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():

    robot_sdf = os.path.expanduser(
        '~/DiskD/RoboticsWorks/scs_robot/src/simulation/assets/robot/semantic_robot.sdf'
    )
    world = os.path.expanduser(
        '~/DiskD/RoboticsWorks/scs_robot/src/simulation/assets/world/semantic_world.sdf'
    )

    snr_arg = DeclareLaunchArgument(
        'snr',
        default_value='5.0',
        description='Channel SNR in dB'
    )

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world, '-v', '4'],
        output='screen'
    )

    spawn_robot = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'gz', 'service',
                    '-s', '/world/semantic_world/create',
                    '--reqtype', 'gz.msgs.EntityFactory',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '5000',
                    '--req',
                    f'sdf_filename: "{robot_sdf}", name: "semantic_robot"'
                ],
                output='screen'
            )
        ]
    )

    encoder = Node(
        package='simulation',
        executable='encoder_node',
        name='encoder_node',
        output='screen'
    )

    channel = Node(
        package='simulation',
        executable='channel_node',
        name='channel_node',
        output='screen',
        parameters=[{'snr_db': LaunchConfiguration('snr')}]
    )

    decoder = Node(
        package='simulation',
        executable='decoder_node',
        name='decoder_node',
        output='screen'
    )

    nav = Node(
        package='simulation',
        executable='nav_node',
        name='nav_node',
        output='screen'
    )

    return LaunchDescription([
        snr_arg,
        gazebo,
        spawn_robot,
        encoder,
        channel,
        decoder,
        nav
    ])