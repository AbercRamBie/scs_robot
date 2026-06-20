import time
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool

class StartupScanNode(Node):
    """
    StartupScanNode is now responsible only for:
      1. Publishing a scan-start trigger for vision_node.
      2. Commanding the robot/base motor to rotate.
      3. Publishing scan-done when the spin is finished.

    It does NOT open the camera, record video, save frames, write metadata,
    or run any vision processing.
    """

    def __init__(self):
        super().__init__('startup_scan_node')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('scan_start_topic', '/startup_scan/start')
        self.declare_parameter('scan_done_topic', '/startup_scan/done')
        self.declare_parameter('start_delay_sec', 2.0)
        self.declare_parameter('spin_duration_sec', 20.0)
        self.declare_parameter('angular_speed_z', 0.314)  # rad/s; ~360 deg in 20 sec
        self.declare_parameter('publish_rate_hz', 20.0)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.scan_start_topic = str(self.get_parameter('scan_start_topic').value)
        self.scan_done_topic = str(self.get_parameter('scan_done_topic').value)
        self.start_delay_sec = float(self.get_parameter('start_delay_sec').value)
        self.spin_duration_sec = float(self.get_parameter('spin_duration_sec').value)
        self.angular_speed_z = float(self.get_parameter('angular_speed_z').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        if self.spin_duration_sec <= 0.0:
            self.get_logger().warn('spin_duration_sec must be > 0. Falling back to 20.0.')
            self.spin_duration_sec = 20.0

        if self.publish_rate_hz <= 0.0:
            self.get_logger().warn('publish_rate_hz must be > 0. Falling back to 20.0.')
            self.publish_rate_hz = 20.0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.scan_start_pub = self.create_publisher(Bool, self.scan_start_topic, 10)
        self.scan_done_pub = self.create_publisher(Bool, self.scan_done_topic, 10)
        self.state = 'waiting'
        self.node_start_time = time.time()
        self.spin_start_time = None
        self.start_trigger_sent = False
        self.done_sent = False
        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.update)
        self.get_logger().info('Startup scan motor node ready.')
        self.get_logger().info(f'Command velocity topic: {self.cmd_vel_topic}')
        self.get_logger().info(f'Vision start topic: {self.scan_start_topic}')
        self.get_logger().info(f'Vision done topic: {self.scan_done_topic}')
        self.get_logger().info(
            f'Starting spin after {self.start_delay_sec:.2f}s, '
            f'spinning for {self.spin_duration_sec:.2f}s at '
            f'{self.angular_speed_z:.3f} rad/s.'
        )

    def publish_scan_start(self):
        if not rclpy.ok():
            return
        msg = Bool()
        msg.data = True
        self.scan_start_pub.publish(msg)
        self.start_trigger_sent = True
        self.get_logger().info('Published scan start trigger for vision_node.')

    def publish_scan_done(self):
        if not rclpy.ok():
            return
        msg = Bool()
        msg.data = True
        self.scan_done_pub.publish(msg)
        self.done_sent = True
        self.get_logger().info('Published scan done trigger for vision_node.')

    def publish_spin_command(self):
        if not rclpy.ok():
            return
        cmd = Twist()
        cmd.angular.z = self.angular_speed_z
        self.cmd_pub.publish(cmd)

    def publish_stop_command(self):
        if not rclpy.ok():
            return
        self.cmd_pub.publish(Twist())

    def update(self):
        now = time.time()

        if self.state == 'waiting':
            elapsed = now - self.node_start_time

            self.publish_stop_command()

            if elapsed >= self.start_delay_sec:
                self.publish_scan_start()
                self.spin_start_time = now
                self.state = 'spinning'
                self.get_logger().info('Motor spin started.')

            return

        if self.state == 'spinning':
            elapsed_spin = now - self.spin_start_time

            if elapsed_spin < self.spin_duration_sec:
                self.publish_spin_command()
                return

            self.publish_stop_command()
            self.publish_scan_done()
            self.state = 'done'
            self.get_logger().info('Motor spin complete. Robot stopped.')
            rclpy.shutdown()

    def destroy_node(self):
        if rclpy.ok():
            self.publish_stop_command()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = StartupScanNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.publish_stop_command()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
