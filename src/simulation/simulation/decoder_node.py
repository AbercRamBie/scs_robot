import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
import torch
from packages.models.decoder import SemanticDecoder
import sys
sys.path.insert(0, '/home/subash/miniconda3/envs/semcomm/lib/python3.11/site-packages')
sys.path.insert(0, '/home/subash/DiskD/RoboticsWorks/scs_robot/src/ml/ml')

class DecoderNode(Node):

    def __init__(self):
        super().__init__('decoder_node')
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = SemanticDecoder(bottleneck_dim=2, output_dim=4).to(self.device)
        self.decoder.load_state_dict(torch.load(
            '/home/subash/DiskD/RoboticsWorks/scs_robot/artifacts/checkpoints/decoder_snr10.pth',
            map_location=self.device
        ))
        self.decoder.eval()
        self.get_logger().info(f'Decoder loaded on {self.device}')
        self.classes = {0: 'FORWARD', 1: 'LEFT', 2: 'RIGHT', 3: 'STOP'}
        self.sub = self.create_subscription(Float32MultiArray, '/semantic/received', self.callback, 10)
        self.pub = self.create_publisher(Int32, '/semantic/decision', 10)

    def callback(self, msg):
        z_hat = torch.tensor(msg.data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits   = self.decoder(z_hat)
            decision = logits.argmax(dim=1).item()
        self.get_logger().info(f'Decision: {decision} ({self.classes[decision]})')
        out      = Int32()
        out.data = int(decision)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DecoderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
