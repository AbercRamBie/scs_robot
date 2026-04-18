"""
Generic three-wheel omnidirectional robot driver.

Subscribes to /cmd_vel (Twist) and converts to motor commands.
Adapt the send_motor_command() method to your actual hardware interface.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math
import serial


class ThreeWheelRobotDriver(Node):
    """
    Three-wheel omnidirectional robot kinematics.
    
    Wheel layout (top view, radians):
    - Wheel 0: 90°   (front-left, π/2)
    - Wheel 1: 210°  (rear-left, 7π/6)
    - Wheel 2: 330°  (rear-right, 11π/6)
    
    Each wheel has velocity: v_i = vx * cos(θ_i) + vy * sin(θ_i) + ω * r
    where ω is angular velocity, r is distance from center (assume 1.0 normalized).
    """

    def __init__(self):
        super().__init__('robot_driver_node')
        
        # Declare parameters
        self.declare_parameter('robot_serial_port', '/dev/ttyUSB0')
        self.declare_parameter('robot_serial_baud', 115200)
        self.declare_parameter('robot_radius', 0.15)  # meters, adjust for your robot
        self.declare_parameter('wheel_radius', 0.05)  # meters, adjust for your wheels
        self.declare_parameter('max_motor_speed', 255)  # for 8-bit motor control
        
        self.serial_port = self.get_parameter('robot_serial_port').value
        self.baud = self.get_parameter('robot_serial_baud').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.max_speed = self.get_parameter('max_motor_speed').value
        
        # Initialize serial connection (adapt to your hardware)
        self.ser = None
        try:
            self.ser = serial.Serial(self.serial_port, self.baud, timeout=1.0)
            self.get_logger().info(f'Connected to robot on {self.serial_port} @ {self.baud} baud')
        except Exception as e:
            self.get_logger().warn(f'Serial connection failed: {e}. Running in simulation mode.')
        
        # Wheel angles (radians, top-down view)
        self.wheel_angles = [
            math.pi / 2,        # Wheel 0: 90°
            7 * math.pi / 6,    # Wheel 1: 210°
            11 * math.pi / 6,   # Wheel 2: 330°
        ]
        
        # Subscribe to cmd_vel
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        
        self.get_logger().info('Three-wheel robot driver ready')
    
    def cmd_vel_callback(self, msg: Twist):
        """Convert Twist command to wheel velocities."""
        vx = msg.linear.x      # Forward velocity (m/s)
        vy = msg.linear.y      # Lateral velocity (m/s, for holonomic)
        omega = msg.angular.z  # Rotational velocity (rad/s)
        
        # Calculate wheel velocities using omnidirectional kinematics
        wheel_vels = []
        for theta in self.wheel_angles:
            v_wheel = (vx * math.cos(theta) + 
                      vy * math.sin(theta) + 
                      omega * self.robot_radius)
            wheel_vels.append(v_wheel)
        
        # Normalize if any velocity exceeds max
        max_vel = max(abs(v) for v in wheel_vels) if any(wheel_vels) else 0.0
        if max_vel > 1.0:
            wheel_vels = [v / max_vel for v in wheel_vels]
        
        # Convert to motor commands (0-255)
        motor_cmds = [int((v / 1.0) * (self.max_speed / 2) + self.max_speed / 2) 
                      for v in wheel_vels]
        motor_cmds = [max(0, min(255, cmd)) for cmd in motor_cmds]
        
        self.get_logger().debug(f'Wheel vels: {wheel_vels}, Motor cmds: {motor_cmds}')
        
        # Send to robot hardware
        self.send_motor_command(motor_cmds)
    
    def send_motor_command(self, motor_speeds):
        """
        Send motor commands to the robot.
        
        TODO: Adapt this to your robot's communication protocol.
        
        Examples:
        - Serial command: "M0,127,200,150\n" (motor indices and speeds)
        - Network: send via UDP/TCP socket
        - Direct GPIO: set PWM pins directly via gpiozero/RPi.GPIO
        
        Args:
            motor_speeds: list of 3 integers (0-255) for each motor
        """
        # Example serial protocol: "M<motor0>,<motor1>,<motor2>\n"
        cmd = f"M{motor_speeds[0]},{motor_speeds[1]},{motor_speeds[2]}\n"
        
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd.encode())
                self.get_logger().debug(f'Sent: {cmd.strip()}')
            except Exception as e:
                self.get_logger().warn(f'Serial write failed: {e}')
        else:
            # Simulation mode: just log
            self.get_logger().info(f'[SIM] Motor command: {cmd.strip()}')
    
    def destroy_node(self):
        """Clean up on shutdown."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ThreeWheelRobotDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
