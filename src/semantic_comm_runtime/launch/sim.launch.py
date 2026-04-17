from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share_dir = get_package_share_directory('semantic_comm_runtime')
    runtime_config = os.path.join(share_dir, 'config', 'runtime.yaml')

    robot_sdf = os.path.join(
        share_dir, 'assets', 'robot', 'semantic_robot.sdf'
    )
    world = os.path.join(
        share_dir, 'assets', 'world', 'semantic_world.sdf'
    )

    snr_arg = DeclareLaunchArgument(
        'snr',
        default_value='',
        description='Channel SNR in dB'
    )

    encoder_checkpoint_arg = DeclareLaunchArgument(
        'encoder_checkpoint',
        default_value='',
        description='Path to the encoder checkpoint file'
    )

    decoder_checkpoint_arg = DeclareLaunchArgument(
        'decoder_checkpoint',
        default_value='',
        description='Path to the decoder checkpoint file'
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
        package='semantic_comm_runtime',
        executable='encoder_node',
        name='encoder_node',
        output='screen',
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
        parameters=[runtime_config]
    )

    return LaunchDescription([
        snr_arg,
        encoder_checkpoint_arg,
        decoder_checkpoint_arg,
        gazebo,
        spawn_robot,
        encoder,
        channel,
        decoder,
        nav
    ])