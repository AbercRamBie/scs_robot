import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
from packages.models.encoder import SemanticEncoder
from packages.loss.vib import reparametrize
import sys
sys.path.insert(0, '/home/subash/miniconda3/envs/semcomm/lib/python3.11/site-packages')
sys.path.insert(0, '/home/subash/DiskD/RoboticsWorks/scs_robot/src/ml/ml')


class EncoderNode(Node):

    def __init__(self):
        super().__init__('encoder_node')
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = SemanticEncoder(bottleneck_dim=2).to(self.device)
        self.encoder.load_state_dict(torch.load(
            '/home/subash/DiskD/RoboticsWorks/scs_robot/artifacts/checkpoints/encoder_snr10.pth',
            map_location=self.device
        ))
        self.encoder.eval()
        self.get_logger().info(f'Encoder loaded on {self.device}')
        self.grid_size = 64
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/semantic/compressed', 10)

    def scan_to_grid(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.clip(ranges, 0, msg.range_max)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        grid   = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        cx, cy = self.grid_size // 2, self.grid_size // 2
        scale  = self.grid_size / (2 * msg.range_max)
        for r, a in zip(ranges, angles):
            if r < msg.range_max:
                x = int(cx + r * np.cos(a) * scale)
                y = int(cy + r * np.sin(a) * scale)
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[x, y] = 0.0
        return grid

    def scan_callback(self, msg):
        grid   = self.scan_to_grid(msg)
        tensor = torch.tensor(grid).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, log_var = self.encoder(tensor)
            Z           = reparametrize(mu, log_var)
        out      = Float32MultiArray()
        out.data = Z.cpu().numpy().flatten().tolist()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EncoderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
