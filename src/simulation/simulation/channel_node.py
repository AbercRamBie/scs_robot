import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np


class ChannelNode(Node):

    def __init__(self):
        super().__init__('channel_node')
        self.declare_parameter('snr_db', 5.0)
        self.snr_db = self.get_parameter('snr_db').value
        self.get_logger().info(f'Channel SNR: {self.snr_db} dB')
        self.sub = self.create_subscription(Float32MultiArray, '/semantic/compressed', self.callback, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/semantic/received', 10)
        self.create_timer(1.0, self.update_snr)

    def update_snr(self):
        new_snr = self.get_parameter('snr_db').value
        if new_snr != self.snr_db:
            self.snr_db = new_snr
            self.get_logger().info(f'SNR updated to {self.snr_db} dB')

    def callback(self, msg):
        z            = np.array(msg.data, dtype=np.float32)
        snr_linear   = 10 ** (self.snr_db / 10.0)
        signal_power = np.mean(z ** 2)
        noise_std    = np.sqrt(signal_power / snr_linear)
        z_noisy      = z + np.random.randn(*z.shape) * noise_std
        out          = Float32MultiArray()
        out.data     = z_noisy.tolist()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ChannelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
