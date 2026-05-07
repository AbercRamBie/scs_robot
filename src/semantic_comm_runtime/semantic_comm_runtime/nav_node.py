import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist


class NavNode(Node):

    def __init__(self):
        super().__init__('nav_node')
        self.turn_count = 0
        self.max_turns  = 10
        self.declare_parameter('obstacle_dist_mm', 150)   # stop-distance threshold
        self.obstacle_dist_mm = self.get_parameter('obstacle_dist_mm').value
        # latest ultrasonic reading (mm); start large so we don't block before first reading
        self.ultrasonic_dist_mm = 9999
        self.sub = self.create_subscription(Int32, '/semantic/decision', self.callback, 10)
        self.ultrasonic_sub = self.create_subscription(
            Int32, '/ultrasonic/distance', self._ultrasonic_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Navigation node ready')

    def _ultrasonic_callback(self, msg: Int32):
        self.ultrasonic_dist_mm = msg.data

    def callback(self, msg):
        cmd      = Twist()
        decision = msg.data
        if self.turn_count >= self.max_turns:
            decision        = 0
            self.turn_count = 0
        # Ultrasonic override: if forward is commanded but obstacle is close, turn instead
        if decision == 0 and 0 < self.ultrasonic_dist_mm < self.obstacle_dist_mm:
            self.get_logger().warn(
                f'Ultrasonic obstacle at {self.ultrasonic_dist_mm} mm — overriding forward to turn left')
            decision = 1
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
