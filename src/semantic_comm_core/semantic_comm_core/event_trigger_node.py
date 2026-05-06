import json
import math
from typing import Optional, Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

class EventTriggerNode(Node):
    def __init__(self):
        super().__init__("event_trigger_node")

        # Tunables
        self.declare_parameter("obstacle_threshold", 0.5)  # Meters
        self.declare_parameter("obstacle_hysteresis", 0.5)  # Meters
        self.declare_parameter("goal_tolerance", 0.5)  # Meters
        self.declare_parameter("goal_change_epsilon", 0.5)  # Meters
        self.declare_parameter("progress_epsilon", 0.5)  # Meters
        self.declare_parameter("blocked_timeout_sec", 0.5)  # Seconds
        self.declare_parameter("replay_delay_sec", 0.5)  # Seconds
        self.declare_parameter("event_cooldown_sec", 0.5)  # Seconds

        self.pub = self.create_publisher(String, "/semantic/events", 20)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 20)
        self.create_subscription(Odometry, "/odom", self.on_odometry, 20)
        self.create_subscription(PoseStamped, "/goal", self.on_goal, 20)

        self.goal: Optional[PoseStamped] = None
        self.last_pose = None
        self.last_progress_pose = None

        self.obstacle_active = False
        self.obstacle_since_ns: Optional[int] = None
        self.blocked_active = False

        self.last_event_ns: Dict[str, int] = {}
        self.last_progress_ns = self.now_ns()

        self.create_timer(0.1, self.on_timer)

    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds