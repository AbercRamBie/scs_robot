import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
import torch
from semantic_comm_core.decoder import SemanticDecoder

class DecoderNode(Node):

    def __init__(self):
        super().__init__('decoder_node')
        self.declare_parameter('decoder_checkpoint', '')
        self.declare_parameter('bottleneck_dim', 2)
        self.declare_parameter('output_dim', 4)

        checkpoint_path = self.get_parameter(
            'decoder_checkpoint'
        ).get_parameter_value().string_value
        bottleneck_dim = self.get_parameter(
            'bottleneck_dim'
        ).get_parameter_value().integer_value
        output_dim = self.get_parameter(
            'output_dim'
        ).get_parameter_value().integer_value

        if not checkpoint_path:
            raise ValueError(
                'Parameter decoder_checkpoint is required and must point to a model file.'
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = SemanticDecoder(
            bottleneck_dim=int(bottleneck_dim),
            output_dim=int(output_dim)
        ).to(self.device)
        self.decoder.load_state_dict(torch.load(
            checkpoint_path,
            map_location=self.device
        ))
        self.decoder.eval()
        self.get_logger().info(
            f'Decoder loaded from {checkpoint_path} on {self.device}'
        )
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
