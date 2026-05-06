import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # Declare parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('hsv_lower_h', 0)
        self.declare_parameter('hsv_lower_s', 0)
        self.declare_parameter('hsv_lower_v', 0)
        self.declare_parameter('hsv_upper_h', 179)
        self.declare_parameter('hsv_upper_s', 255)
        self.declare_parameter('hsv_upper_v', 255)
        self.declare_parameter('area_threshold', 500)
        
        # Get parameters
        camera_id = self.get_parameter('camera_id').value
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.bridge = CvBridge()
        
        # Subscribe to camera/image topic (if using USB camera publisher)
        # self.sub = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        
        # Publisher for detected objects
        self.pub_centroids = self.create_publisher(Float32MultiArray, '/vision/centroids', 10)
        self.pub_image = self.create_publisher(Image, '/vision/processed', 10)
        
        # Timer to continuously grab frames
        self.timer = self.create_timer(0.033, self.process_frame)  # 30 FPS
        
        self.get_logger().info('Vision Node Started')

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Get current parameters (allows live tuning)
        l_h = self.get_parameter('hsv_lower_h').value
        l_s = self.get_parameter('hsv_lower_s').value
        l_v = self.get_parameter('hsv_lower_v').value
        u_h = self.get_parameter('hsv_upper_h').value
        u_s = self.get_parameter('hsv_upper_s').value
        u_v = self.get_parameter('hsv_upper_v').value
        area_threshold = self.get_parameter('area_threshold').value
        
        # STEP A: Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # STEP B: Create mask
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        
        # STEP D: Filter noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # STEP E: Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_threshold:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append([cx, cy])
                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        
        # Publish centroids
        msg = Float32MultiArray()
        msg.data = [float(c) for centroid in centroids for c in centroid]
        self.pub_centroids.publish(msg)
        
        # Publish processed image
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()