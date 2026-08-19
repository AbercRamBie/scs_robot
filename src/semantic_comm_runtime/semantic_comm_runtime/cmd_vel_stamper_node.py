"""Adapt the project's Twist commands for ros2_control mobile-base controllers."""

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node

class CmdVelStamperNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_stamper_node')
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/omnibot_controller/cmd_vel')
        self.declare_parameter('frame_id', 'base_footprint')

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        self.publisher = self.create_publisher(TwistStamped, output_topic, 10)
        self.subscription = self.create_subscription(
            Twist, input_topic, self._on_command, 10
        )
        self.get_logger().info(
            f'Adapting {input_topic} (Twist) to {output_topic} (TwistStamped)'
        )

    def _on_command(self, command: Twist) -> None:
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.frame_id
        stamped.twist = command
        self.publisher.publish(stamped)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelStamperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
