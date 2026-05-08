"""
Generic three-wheel omnidirectional robot driver.

Subscribes to /cmd_vel (Twist) and converts to motor commands.
Adapt the send_motor_command() method to your actual hardware interface.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import serial
import time
import re

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
        self.declare_parameter('robot_serial_baud', 9600)  # must match Serial.begin() in Arduino sketch

        self.serial_port = self.get_parameter('robot_serial_port').value
        self.baud = self.get_parameter('robot_serial_baud').value
        
        # Initialize serial connection (adapt to your hardware)
        self.ser = None
        try:
            self.ser = serial.Serial(self.serial_port, self.baud, timeout=1.0)
            # Opening the serial port often resets Arduino-class boards.
            # Give the sketch time to boot before sending motion commands.
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.get_logger().info(f'Connected to robot on {self.serial_port} @ {self.baud} baud')
        except Exception as e:
            self.get_logger().warn(f'Serial connection failed: {e}. Running in simulation mode.')
        
        # Subscribe to cmd_vel
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        # Publish ultrasonic distance (mm) read from Arduino serial stream
        self.ultrasonic_pub = self.create_publisher(Int32, '/ultrasonic/distance', 10)
        self._serial_rx_buffer = bytearray()
        self.serial_poll_timer = self.create_timer(0.1, self._poll_serial_feedback)

        self.get_logger().info('Three-wheel robot driver ready')

    def cmd_vel_callback(self, msg: Twist):
        """Map Twist to a single-char command matching the Arduino sketch."""
        vx = msg.linear.x
        omega = msg.angular.z

        if abs(vx) < 0.01 and abs(omega) < 0.01:
            char_cmd = 'S'          # stop
        elif omega > 0.05:
            char_cmd = 'L'          # turn left
        elif omega < -0.05:
            char_cmd = 'R'          # turn right
        elif vx > 0:
            char_cmd = 'F'          # forward
        else:
            char_cmd = 'B'          # backward

        self.get_logger().info(f'cmd_vel -> {char_cmd!r}  (vx={vx:.2f}, omega={omega:.2f})')
        self._send_char(char_cmd)

    def _send_char(self, char_cmd: str):
        """Send a single character command over serial to the Arduino."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(char_cmd.encode())
                self.get_logger().info(f'Sent serial command: {char_cmd!r}')
            except Exception as e:
                self.get_logger().warn(f'Serial write failed: {e}')
        else:
            self.get_logger().info(f'[SIM] Motor command: {char_cmd!r}')

    def _poll_serial_feedback(self):
        """Read serial feedback and parse ultrasonic distance from text lines when possible."""
        if not (self.ser and self.ser.is_open):
            return
        try:
            waiting = self.ser.in_waiting
            if waiting <= 0:
                return
            raw = self.ser.read(waiting)
            self._serial_rx_buffer.extend(raw)

            # Process complete newline-terminated records only.
            while b'\n' in self._serial_rx_buffer:
                line_bytes, _, remainder = self._serial_rx_buffer.partition(b'\n')
                self._serial_rx_buffer = bytearray(remainder)
                line_bytes = line_bytes.strip(b'\r ')
                if not line_bytes:
                    continue

                # First try text parsing for formats like "DIST:235" or "distance=235".
                line_text = line_bytes.decode('ascii', errors='ignore').strip()
                if line_text:
                    dist_match = re.search(r'(?:DIST|distance)\s*[:=]\s*(-?\d+)', line_text, re.IGNORECASE)
                    if dist_match:
                        dist_mm = int(dist_match.group(1))
                        msg = Int32()
                        msg.data = dist_mm
                        self.ultrasonic_pub.publish(msg)
                        self.get_logger().info(f'Ultrasonic parsed: {dist_mm} mm')
                        continue
                    self.get_logger().info(f'Arduino: {line_text}')
                    continue

                # If not text, print the binary frame in hex so protocol can be decoded.
                self.get_logger().info(f'Arduino binary frame: {line_bytes.hex(" ")}')
        except Exception as e:
            self.get_logger().warn(f'Serial read failed: {e}')
    
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
