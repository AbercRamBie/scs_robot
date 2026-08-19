from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch_ros.parameter_descriptions import ParameterValue
import os

def generate_launch_description():
    share_dir = get_package_share_directory('semantic_comm_runtime')
    runtime_config = os.path.join(share_dir, 'config', 'runtime.yaml')
    joystick_config = os.path.join(share_dir, 'config', 'joyStick.yaml')
    omnibot_share_dir = get_package_share_directory('omnibot_description')
    gz_ros2_control_lib_dir = os.path.join(
        get_package_prefix('gz_ros2_control'), 'lib'
    )
    gz_system_plugin_path = os.pathsep.join(filter(None, [
        gz_ros2_control_lib_dir,
        os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
    ]))
    default_robot_model = os.path.join(
        omnibot_share_dir, 'urdf', 'omnibot.xacro'
    )
    default_world = os.path.join(
        share_dir, 'assets', 'world', 'world_citySpace.sdf'
    )

    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=default_world,
        description='Absolute path to the SDF world file'
    )

    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='city_space',
        description='Internal name from the SDF <world name="..."> element'
    )

    robot_model_arg = DeclareLaunchArgument(
        'robot_model',
        default_value=default_robot_model,
        description='Absolute path to the Omni3WD xacro/URDF model'
    )

    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='2.0')
    spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value='-18.0')
    spawn_z_arg = DeclareLaunchArgument('spawn_z', default_value='0.0')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='1.5708')

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
        cmd=['gz', 'sim', '-v', '4', '-r', LaunchConfiguration('world_file')],
        output='screen',
        additional_env={
            'GZ_SIM_SYSTEM_PLUGIN_PATH': gz_system_plugin_path,
        },
    )

    robot_description = ParameterValue(
        Command(['xacro ', LaunchConfiguration('robot_model')]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    spawn_robot = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                name='spawn_omnibot',
                arguments=[
                    '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
                    '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                    '-world', LaunchConfiguration('world_name'),
                    '-topic', 'robot_description',
                    '-name', 'omnibot',
                    '-x', LaunchConfiguration('spawn_x'),
                    '-y', LaunchConfiguration('spawn_y'),
                    '-z', LaunchConfiguration('spawn_z'),
                    '-Y', LaunchConfiguration('spawn_yaw'),
                ],
                output='screen',
            )
        ]
    )

    controllers = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'joint_state_broadcaster',
                    '--controller-manager', '/controller_manager',
                    '--controller-manager-timeout', '30',
                ],
                output='screen',
            ),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'omnibot_controller',
                    '--controller-manager', '/controller_manager',
                    '--controller-manager-timeout', '30',
                ],
                output='screen',
            ),
        ],
    )

    cmd_vel_stamper = Node(
        package='semantic_comm_runtime',
        executable='cmd_vel_stamper_node',
        name='cmd_vel_stamper_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
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
        '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
    ],
    )

    return LaunchDescription([
        world_file_arg,
        world_name_arg,
        robot_model_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        spawn_yaw_arg,
        snr_arg,
        encoder_checkpoint_arg,
        decoder_checkpoint_arg,
        joy_control_config_arg,
        control_mode_arg,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        controllers,
        cmd_vel_stamper,
        encoder,
        channel,
        decoder,
        nav,
        joy_driver,
        joystick,
        bridge
    ])
