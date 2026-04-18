import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class JoyControlNode(Node):

    def __init__(self):
        super().__init__('joy_control_node')

        self.declare_parameter('axis_linear', 1)
        self.declare_parameter('axis_angular', 3)
        self.declare_parameter('scale_linear', 0.4)
        self.declare_parameter('scale_angular', 1.0)
        self.declare_parameter('deadzone', 0.05)
        self.declare_parameter('require_enable_button', True)
        self.declare_parameter('enable_button_index', 4)

        self.axis_linear = int(self.get_parameter('axis_linear').value)
        self.axis_angular = int(self.get_parameter('axis_angular').value)
        self.scale_linear = float(self.get_parameter('scale_linear').value)
        self.scale_angular = float(self.get_parameter('scale_angular').value)
        self.deadzone = float(self.get_parameter('deadzone').value)
        self.require_enable_button = bool(self.get_parameter('require_enable_button').value)
        self.enable_button_index = int(self.get_parameter('enable_button_index').value)

        self.last_joy = None
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.timer = self.create_timer(0.05, self.publish_cmd)

        self.get_logger().info('Joystick control ready')

    def joy_callback(self, msg: Joy):
        self.last_joy = msg

    def _axis_value(self, axes, index):
        if 0 <= index < len(axes):
           value = float(axes[index])
           if abs(value) < self.deadzone:
              return 0.0
           return value
        return 0.0

    def _enabled(self, buttons):
        if not self.require_enable_button:
           return True
        if 0 <= self.enable_button_index < len(buttons):
            return buttons[self.enable_button_index] == 1
        return False

    def publish_cmd(self):
        cmd = Twist()

        if self.last_joy is None:
            self.pub.publish(cmd)
            return

        if not self._enabled(self.last_joy.buttons):
            self.pub.publish(cmd)
            return

        linear = self._axis_value(self.last_joy.axes, self.axis_linear)
        angular = self._axis_value(self.last_joy.axes, self.axis_angular)

        cmd.linear.x = linear * self.scale_linear
        cmd.angular.z = angular * self.scale_angular
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = JoyControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()