import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist


class NavNode(Node):

    def __init__(self):
        super().__init__('nav_node')
        self.turn_count = 0
        self.max_turns  = 10
        self.sub = self.create_subscription(Int32, '/semantic/decision', self.callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Navigation node ready')

    def callback(self, msg):
        cmd      = Twist()
        decision = msg.data
        if self.turn_count >= self.max_turns:
            decision        = 0
            self.turn_count = 0
        if decision in [1, 2]:
            self.turn_count += 1
        else:
            self.turn_count = 0
        if decision == 0:
            cmd.linear.x  = 0.2
            cmd.angular.z = 0.0
        elif decision == 1:
            cmd.linear.x  = 0.05
            cmd.angular.z = 0.4
        elif decision == 2:
            cmd.linear.x  = 0.05
            cmd.angular.z = -0.4
        elif decision == 3:
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
