import json
import os
import time
from collections import defaultdict
from pathlib import Path
import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ultralytics import YOLO
                    
class VisionNode(Node):
    """
    VisionNode now owns the full camera and perception pipeline:
      1. Wait for /startup_scan/start from startup_scan_node.
            2. Wait recording_start_delay_sec, then open and warm up the camera.
            3. Record the scan while the robot spins.
            4. Continue recording for recording_stop_delay_sec after /startup_scan/done.
            5. Stop when delayed done condition is met, or when max_recording_time_sec is reached.
            6. Save video, metadata, representative angle frames, and frame-angle log.
            7. Run YOLO processing on the recorded video.
    """

    def __init__(self):
        super().__init__('vision_node')

        # Trigger topics from startup_scan_node
        self.declare_parameter('scan_start_topic', '/startup_scan/start')
        self.declare_parameter('scan_done_topic', '/startup_scan/done')

        # Camera / recording parameters moved from startup_scan_node
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('camera_device', '')  # e.g. '/dev/video0'
        self.declare_parameter('scan_root', '/home/subash/manual_perceptionPipeline/manual_scans')
        self.declare_parameter('video_file', 'full_circle_video.avi')
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 20)
        self.declare_parameter('target_rotation_time_sec', 20.0)
        self.declare_parameter('max_recording_time_sec', 25.0)
        self.declare_parameter('recording_start_delay_sec', 10.0)
        self.declare_parameter('recording_stop_delay_sec', 10.0)
        self.declare_parameter('save_angle_frames', True)
        self.declare_parameter('frame_save_angle_step_deg', 30)
        self.declare_parameter('use_v4l2', True)

        # Preview / publishing
        self.declare_parameter('show_recording_preview', True)
        self.declare_parameter('show_processing_preview', False)
        self.declare_parameter('publish_recording_preview_topic', '/startup_scan/preview')
        self.declare_parameter('publish_processed_preview_topic', '/vision/processed')

        # Origin metadata
        self.declare_parameter('origin_frame_id', 'map')
        self.declare_parameter('origin_x_m', 0.0)
        self.declare_parameter('origin_y_m', 0.0)
        self.declare_parameter('origin_z_m', 0.0)
        self.declare_parameter('origin_yaw_deg', 0.0)
        self.declare_parameter(
            'origin_note',
            'Scan starts at this user-defined origin pose.'
        )
        self.declare_parameter('use_origin_from_metadata', True)
        self.declare_parameter('origin_yaw_offset_deg', 0.0)

        # YOLO / processing
        self.declare_parameter('process_after_recording', True)
        self.declare_parameter('model_name', 'yolo26n-seg.pt')
        self.declare_parameter('conf_threshold', 0.50)
        self.declare_parameter('frame_skip', 5)
        self.declare_parameter('output_fps', 6)

        self.declare_parameter(
            'allowed_classes',
            [
                'person',
                'chair',
                'couch',
                'bed',
                'tv',
                'laptop',
                'keyboard',
                'mouse',
                'book',
                'backpack',
                'bottle',
                'cup',
                'potted plant',
            ],
        )

        self.declare_parameter(
            'blocked_classes',
            [
                'refrigerator',
                'oven',
                'microwave',
                'toilet',
                'sink',
            ],
        )

        self.scan_start_topic = str(self.get_parameter('scan_start_topic').value)
        self.scan_done_topic = str(self.get_parameter('scan_done_topic').value)

        self.camera_id = int(self.get_parameter('camera_id').value)
        self.camera_device = str(self.get_parameter('camera_device').value)
        self.scan_root = Path(str(self.get_parameter('scan_root').value))
        self.video_file = str(self.get_parameter('video_file').value)
        self.frame_width = int(self.get_parameter('frame_width').value)
        self.frame_height = int(self.get_parameter('frame_height').value)
        self.fps = int(self.get_parameter('fps').value)
        self.target_rotation_time_sec = float(self.get_parameter('target_rotation_time_sec').value)
        self.max_recording_time_sec = float(self.get_parameter('max_recording_time_sec').value)
        self.recording_start_delay_sec = float(self.get_parameter('recording_start_delay_sec').value)
        self.recording_stop_delay_sec = float(self.get_parameter('recording_stop_delay_sec').value)
        self.save_angle_frames = bool(self.get_parameter('save_angle_frames').value)
        self.frame_save_angle_step_deg = int(self.get_parameter('frame_save_angle_step_deg').value)
        self.use_v4l2 = bool(self.get_parameter('use_v4l2').value)

        self.show_recording_preview = bool(self.get_parameter('show_recording_preview').value)
        self.show_processing_preview = bool(self.get_parameter('show_processing_preview').value)
        self.recording_preview_topic = str(self.get_parameter('publish_recording_preview_topic').value)
        self.processed_preview_topic = str(self.get_parameter('publish_processed_preview_topic').value)

        self.origin_frame_id = str(self.get_parameter('origin_frame_id').value)
        self.origin_x_m = float(self.get_parameter('origin_x_m').value)
        self.origin_y_m = float(self.get_parameter('origin_y_m').value)
        self.origin_z_m = float(self.get_parameter('origin_z_m').value)
        self.origin_yaw_deg = float(self.get_parameter('origin_yaw_deg').value)
        self.origin_note = str(self.get_parameter('origin_note').value)
        self.use_origin_from_metadata = bool(self.get_parameter('use_origin_from_metadata').value)
        self.origin_yaw_offset_deg = float(self.get_parameter('origin_yaw_offset_deg').value)

        self.process_after_recording = bool(self.get_parameter('process_after_recording').value)
        self.model_name = str(self.get_parameter('model_name').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.frame_skip = int(self.get_parameter('frame_skip').value)
        self.output_fps = int(self.get_parameter('output_fps').value)

        self.allowed_classes = set(self.get_parameter('allowed_classes').value)
        self.blocked_classes = set(self.get_parameter('blocked_classes').value)

        if self.fps <= 0:
            self.get_logger().warn('fps must be > 0. Falling back to 20.')
            self.fps = 20

        if self.target_rotation_time_sec <= 0.0:
            self.get_logger().warn('target_rotation_time_sec must be > 0. Falling back to 20.0.')
            self.target_rotation_time_sec = 20.0

        if self.max_recording_time_sec <= 0.0:
            self.get_logger().warn('max_recording_time_sec must be > 0. Falling back to target_rotation_time_sec + 5.')
            self.max_recording_time_sec = self.target_rotation_time_sec + 5.0

        if self.recording_start_delay_sec < 0.0:
            self.get_logger().warn('recording_start_delay_sec must be >= 0. Falling back to 10.0.')
            self.recording_start_delay_sec = 10.0

        if self.recording_stop_delay_sec < 0.0:
            self.get_logger().warn('recording_stop_delay_sec must be >= 0. Falling back to 10.0.')
            self.recording_stop_delay_sec = 10.0

        min_recording_window_sec = self.target_rotation_time_sec + self.recording_stop_delay_sec
        if self.max_recording_time_sec < min_recording_window_sec:
            self.get_logger().warn(
                'max_recording_time_sec is shorter than '
                'target_rotation_time_sec + recording_stop_delay_sec. '
                f'Bumping max_recording_time_sec to {min_recording_window_sec:.2f}s.'
            )
            self.max_recording_time_sec = min_recording_window_sec

        if self.frame_save_angle_step_deg <= 0:
            self.get_logger().warn('frame_save_angle_step_deg must be > 0. Falling back to 30.')
            self.frame_save_angle_step_deg = 30

        if self.frame_skip <= 0:
            self.get_logger().warn('frame_skip must be > 0. Falling back to 1.')
            self.frame_skip = 1

        if self.output_fps <= 0:
            self.get_logger().warn('output_fps must be > 0. Falling back to 6.')
            self.output_fps = 6

        if self.show_recording_preview and not os.environ.get('DISPLAY'):
            self.get_logger().warn('$DISPLAY is not set. Disabling recording preview window.')
            self.show_recording_preview = False

        if self.show_processing_preview and not os.environ.get('DISPLAY'):
            self.get_logger().warn('$DISPLAY is not set. Disabling processing preview window.')
            self.show_processing_preview = False

        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.use_half = bool(torch.cuda.is_available())

        self.bridge = CvBridge()

        self.recording_preview_pub = self.create_publisher(Image, self.recording_preview_topic, 10)
        self.processed_preview_pub = self.create_publisher(Image, self.processed_preview_topic, 10)
        self.done_pub = self.create_publisher(Bool, '/vision/yolo_done', 10)

        self.start_sub = self.create_subscription(
            Bool,
            self.scan_start_topic,
            self._on_scan_start,
            10,
        )

        self.done_sub = self.create_subscription(
            Bool,
            self.scan_done_topic,
            self._on_scan_done,
            10,
        )

        self.state = 'idle'
        self.scan_dir = None
        self.frames_dir = None
        self.video_path = None
        self.metadata_path = None
        self.frame_log_path = None

        self.cap = None
        self.writer = None
        self.recording_start = None
        self.scan_start_trigger_time = None
        self.scan_done_trigger_time = None
        self.frame_count = 0
        self.frame_angle_log = []
        self.next_save_angle = 0
        self.saved_angle_frames = []
        self.stopped_by_startup_node = False
        self.stopped_after_done_delay = False
        self.stopped_by_timeout = False
        self.stopped_manually = False
        self.processing_started = False

        self.recording_timer = self.create_timer(1.0 / float(self.fps), self._recording_update)

        self.get_logger().info('Vision node ready. Waiting for startup scan trigger.')
        self.get_logger().info(f'Listening for scan start on: {self.scan_start_topic}')
        self.get_logger().info(f'Listening for scan done on: {self.scan_done_topic}')

    def _on_scan_start(self, msg):
        if not msg.data:
            return

        if self.state != 'idle':
            self.get_logger().warn(f'Ignoring scan start because state={self.state}')
            return

        self.get_logger().info('Received scan start trigger.')

        if self.recording_start_delay_sec > 0.0:
            self.scan_start_trigger_time = time.time()
            self.state = 'waiting_start'
            self.get_logger().info(
                f'Recording will start after {self.recording_start_delay_sec:.1f}s delay.'
            )
        else:
            self._start_recording()

    def _on_scan_done(self, msg):
        if not msg.data:
            return

        if self.state == 'recording':
            self.get_logger().info('Received scan done trigger from startup_scan_node.')
            self.stopped_by_startup_node = True
            self.scan_done_trigger_time = time.time()

            if self.recording_stop_delay_sec <= 0.0:
                self.stopped_after_done_delay = True
                self.state = 'finalizing'
            else:
                self.get_logger().info(
                    f'Continuing recording for {self.recording_stop_delay_sec:.1f}s '
                    'after scan done trigger.'
                )

    def _open_camera(self):
        source = self.camera_device if self.camera_device else self.camera_id

        if self.use_v4l2:
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open camera source={source!r}')

        return cap

    def _warmup_camera(self):
        for _ in range(20):
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError('Camera warmup failed')
            cv2.resize(frame, (self.frame_width, self.frame_height))

    def _start_recording(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.scan_dir = self.scan_root / f'manual_video_scan_{timestamp}'
        self.scan_dir.mkdir(parents=True, exist_ok=True)

        self.frames_dir = self.scan_dir / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.scan_dir / self.video_file
        self.metadata_path = self.scan_dir / 'metadata.json'
        self.frame_log_path = self.scan_dir / 'frame_angles.json'

        self.cap = self._open_camera()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.get_logger().info('Warming up camera...')
        self._warmup_camera()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f'Failed to open video writer: {self.video_path}')

        self.recording_start = time.time()
        self.frame_count = 0
        self.frame_angle_log = []
        self.next_save_angle = 0
        self.saved_angle_frames = []
        self.stopped_by_startup_node = False
        self.stopped_after_done_delay = False
        self.stopped_by_timeout = False
        self.stopped_manually = False
        self.processing_started = False
        self.scan_done_trigger_time = None

        self.state = 'recording'

        self.get_logger().info(f'Recording started: {self.video_path}')
        self.get_logger().info(
            f'Expecting approximately one 360 degree spin in '
            f'{self.target_rotation_time_sec:.2f}s.'
        )
        self.get_logger().info(
            f'Recording will stop {self.recording_stop_delay_sec:.2f}s '
            'after scan done trigger.'
        )

    def _publish_recording_preview(self, image_bgr):
        self.recording_preview_pub.publish(
            self.bridge.cv2_to_imgmsg(image_bgr, 'bgr8')
        )

    def _recording_update(self):
        if self.state == 'idle':
            return

        if self.state == 'waiting_start':
            elapsed_since_start_trigger = time.time() - (self.scan_start_trigger_time or time.time())
            if elapsed_since_start_trigger >= self.recording_start_delay_sec:
                self._start_recording()
            return

        if self.state == 'recording':
            self._record_frame()
            return

        if self.state == 'finalizing':
            self._finalize_recording()
            self.state = 'processing' if self.process_after_recording else 'done'

            if self.process_after_recording:
                self.process_latest_scan()
            else:
                self._publish_vision_done()

            self.state = 'done'
            rclpy.shutdown()

    def _record_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('Failed to read camera frame.')
            return

        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        display = frame.copy()

        elapsed_recording = time.time() - self.recording_start
        estimated_angle = (elapsed_recording / self.target_rotation_time_sec) * 360.0
        estimated_angle = min(360.0, estimated_angle)

        self.writer.write(frame)

        if (
            self.save_angle_frames
            and estimated_angle >= self.next_save_angle
            and self.next_save_angle < 360
        ):
            frame_filename = f'frame_{int(self.next_save_angle):03d}_deg.png'
            frame_path = self.frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            self.saved_angle_frames.append(
                {
                    'yaw_deg': int(self.next_save_angle),
                    'frame_index': self.frame_count,
                    'time_sec': round(elapsed_recording, 3),
                    'image': str(Path('frames') / frame_filename),
                }
            )

            self.get_logger().info(f'Saved angle frame: {frame_filename}')
            self.next_save_angle += self.frame_save_angle_step_deg

        self.frame_angle_log.append(
            {
                'frame_index': self.frame_count,
                'time_sec': round(elapsed_recording, 3),
                'estimated_yaw_deg': round(estimated_angle, 2),
            }
        )

        self.frame_count += 1

        cv2.putText(
            display,
            f'Recording: {elapsed_recording:.1f}s',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f'Estimated angle: {estimated_angle:.1f} / 360 deg',
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"Next frame save: {self.next_save_angle if self.next_save_angle < 360 else 'done'} deg",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if self.show_recording_preview:
            cv2.imshow('VisionNode Recording Scan', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Recording stopped manually.')
                self.stopped_manually = True
                self.state = 'finalizing'

        self._publish_recording_preview(display)

        if elapsed_recording >= self.max_recording_time_sec:
            self.get_logger().warn(
                f'Max recording time reached: {self.max_recording_time_sec:.2f}s'
            )
            self.stopped_by_timeout = True
            self.state = 'finalizing'
            return

        if self.scan_done_trigger_time is not None:
            elapsed_after_done = time.time() - self.scan_done_trigger_time
            if elapsed_after_done >= self.recording_stop_delay_sec:
                self.stopped_after_done_delay = True
                self.state = 'finalizing'

    def _finalize_recording(self):
        duration = 0.0
        if self.recording_start is not None:
            duration = time.time() - self.recording_start

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.show_recording_preview:
            cv2.destroyWindow('VisionNode Recording Scan')

        metadata = {
            'scan_type': 'robot_motor_360_video_capture',
            'video_file': self.video_file,
            'frame_angle_log': 'frame_angles.json',
            'frames_dir': 'frames',
            'save_angle_frames': self.save_angle_frames,
            'frame_save_angle_step_deg': self.frame_save_angle_step_deg,
            'saved_angle_frames': self.saved_angle_frames,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'target_rotation_time_sec': self.target_rotation_time_sec,
            'max_recording_time_sec': self.max_recording_time_sec,
            'recording_start_delay_sec': self.recording_start_delay_sec,
            'recording_stop_delay_sec': self.recording_stop_delay_sec,
            'actual_recording_duration_sec': round(duration, 3),
            'angle_estimation_method': 'time_based_estimation',
            'rotation_note': 'estimated yaw assumes constant motor rotation speed',
            'stopped_by_startup_node': self.stopped_by_startup_node,
            'stopped_after_done_delay': self.stopped_after_done_delay,
            'stopped_by_timeout': self.stopped_by_timeout,
            'stopped_manually': self.stopped_manually,
            'origin_reference': {
                'frame_id': self.origin_frame_id,
                'position_m': {
                    'x': round(self.origin_x_m, 4),
                    'y': round(self.origin_y_m, 4),
                    'z': round(self.origin_z_m, 4),
                },
                'yaw_deg': round(self.origin_yaw_deg, 3),
                'note': self.origin_note,
                'source': 'user_parameter',
            },
        }

        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        with open(self.frame_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.frame_angle_log, f, indent=2)

        self.get_logger().info(f'Saved video: {self.video_path}')
        self.get_logger().info(f'Saved metadata: {self.metadata_path}')
        self.get_logger().info(f'Saved frame angle log: {self.frame_log_path}')
        self.get_logger().info(f'Saved angle frames folder: {self.frames_dir}')
        self.get_logger().info(f'Frames recorded: {self.frame_count}')
        self.get_logger().info(f'Recording duration: {duration:.2f}s')

    @staticmethod
    def yaw_to_direction(yaw_deg):
        directions = [
            'front',
            'front_right',
            'right',
            'back_right',
            'back',
            'back_left',
            'left',
            'front_left',
        ]

        index = int(((yaw_deg + 22.5) % 360) // 45)
        return directions[index]

    def get_latest_scan_dir(self):
        scan_dirs = [
            path for path in self.scan_root.glob('manual_video_scan_*')
            if path.is_dir()
        ]

        if not scan_dirs:
            raise FileNotFoundError(
                f'No manual_video_scan_* folders found in {self.scan_root}'
            )

        return max(scan_dirs, key=lambda path: path.stat().st_mtime)

    @staticmethod
    def load_metadata(scan_dir):
        metadata_path = scan_dir / 'metadata.json'

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def estimate_yaw_from_frame(frame_index, total_frames):
        if total_frames <= 0:
            return 0.0

        return (frame_index / total_frames) * 360.0

    def process_latest_scan(self):
        if self.processing_started:
            return

        self.processing_started = True

        scan_dir = self.get_latest_scan_dir()
        self.get_logger().info(f'Using latest scan folder: {scan_dir}')

        video_path = scan_dir / self.video_file
        if not video_path.exists():
            self.get_logger().error(f'Video file not found: {video_path}')
            self._publish_vision_done()
            return

        metadata = self.load_metadata(scan_dir)

        origin_reference = {}
        if isinstance(metadata, dict):
            origin_reference = metadata.get('origin_reference', {}) or {}

        metadata_origin_yaw = float(origin_reference.get('yaw_deg', 0.0))

        if self.use_origin_from_metadata:
            origin_yaw_world_deg = metadata_origin_yaw
        else:
            origin_yaw_world_deg = self.origin_yaw_offset_deg

        origin_frame = str(origin_reference.get('frame_id', self.origin_frame_id))

        self.get_logger().info(
            f'Origin heading reference: frame={origin_frame}, '
            f'yaw_offset={origin_yaw_world_deg:.2f} deg, '
            f'use_metadata={self.use_origin_from_metadata}'
        )

        self.get_logger().info('Loading YOLO model...')
        model = YOLO(self.model_name)
        self.get_logger().info(f'Using device: {self.device}')
        if self.device != 'cpu':
            self.get_logger().info(f'GPU name: {torch.cuda.get_device_name(0)}')
        else:
            self.get_logger().warn('CUDA GPU not available. YOLO will run on CPU.')

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.get_logger().error(f'Failed to open video: {video_path}')
            self._publish_vision_done()
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        self.get_logger().info(f'Video: {video_path}')
        self.get_logger().info(f'Total frames: {total_frames}')
        self.get_logger().info(f'Original video FPS: {video_fps}')
        self.get_logger().info(f'Processing every {self.frame_skip} frames')
        self.get_logger().info(f'Annotated output FPS: {self.output_fps}')

        detections_by_direction = defaultdict(lambda: defaultdict(int))
        detections_by_class = defaultdict(int)
        frame_results = []

        output_video_path = scan_dir / 'yolo_annotated_video.avi'
        output_json_path = scan_dir / 'yolo_scene_summary.json'

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = None

        frame_index = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % self.frame_skip != 0:
                frame_index += 1
                continue

            yaw_deg = self.estimate_yaw_from_frame(frame_index, total_frames)
            direction_relative = self.yaw_to_direction(yaw_deg)
            yaw_world_deg = (origin_yaw_world_deg + yaw_deg) % 360.0
            direction_world = self.yaw_to_direction(yaw_world_deg)

            results = model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )

            detected_objects = []
            annotated_frame = frame.copy()

            for result in results:
                annotated_frame = result.plot()

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    if class_name in self.blocked_classes:
                        continue

                    if class_name not in self.allowed_classes:
                        continue

                    detected_objects.append(
                        {
                            'class_name': class_name,
                            'confidence': round(confidence, 3),
                        }
                    )

                    detections_by_direction[direction_world][class_name] += 1
                    detections_by_class[class_name] += 1

            unique_objects = sorted(
                list({obj['class_name'] for obj in detected_objects})
            )

            cv2.putText(
                annotated_frame,
                f'Yaw rel: {yaw_deg:.1f} deg ({direction_relative})',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f'Yaw world: {yaw_world_deg:.1f} deg ({direction_world}) [{origin_frame}]',
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f'Frame: {frame_index} | Objects: {len(unique_objects)}',
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if writer is None:
                h, w = annotated_frame.shape[:2]

                writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    self.output_fps,
                    (w, h),
                )

                if not writer.isOpened():
                    self.get_logger().error('Failed to open output video writer')
                    cap.release()
                    self._publish_vision_done()
                    return

            writer.write(annotated_frame)

            if self.show_processing_preview:
                cv2.imshow('YOLO 360 Processing', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info('Processing stopped manually by user.')
                    break

            self.processed_preview_pub.publish(
                self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
            )

            frame_results.append(
                {
                    'frame_index': frame_index,
                    'estimated_yaw_deg_relative': round(yaw_deg, 2),
                    'estimated_yaw_deg_world': round(yaw_world_deg, 2),
                    'direction_relative': direction_relative,
                    'direction_world': direction_world,
                    'origin_frame_id': origin_frame,
                    'objects': unique_objects,
                    'detections': detected_objects,
                }
            )

            self.get_logger().info(
                f'Frame {frame_index:05d} | '
                f'Yaw rel {yaw_deg:6.1f} / world {yaw_world_deg:6.1f} | '
                f'{direction_world:12s} | '
                f'{unique_objects}'
            )

            processed_count += 1
            frame_index += 1

        cap.release()

        if writer is not None:
            writer.release()

        if self.show_processing_preview:
            cv2.destroyWindow('YOLO 360 Processing')

        summary = {
            'scan_type': 'yolo_360_video_semantic_summary',
            'source_video': self.video_file,
            'model': self.model_name,
            'confidence_threshold': self.conf_threshold,
            'frame_skip': self.frame_skip,
            'output_fps': self.output_fps,
            'total_video_frames': total_frames,
            'processed_frames': processed_count,
            'angle_method': 'frame_index_based_estimation',
            'origin_reference': {
                'frame_id': origin_frame,
                'yaw_offset_deg': round(origin_yaw_world_deg, 3),
                'used_metadata': self.use_origin_from_metadata,
            },
            'note': (
                'Relative yaw is estimated assuming the video covers one full '
                '360 degree rotation from start to end. World yaw adds origin yaw offset.'
            ),
            'object_frequency_global': dict(
                sorted(
                    detections_by_class.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
            'direction_summary': {},
            'frames': frame_results,
        }

        for direction, object_counts in detections_by_direction.items():
            summary['direction_summary'][direction] = dict(
                sorted(
                    object_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info('YOLO processing complete.')
        self.get_logger().info(f'Saved summary: {output_json_path}')
        self.get_logger().info(f'Saved annotated video: {output_video_path}')

        self.get_logger().info('Global object frequency:')
        for obj, count in summary['object_frequency_global'].items():
            self.get_logger().info(f'  {obj}: {count}')

        self.get_logger().info('Direction summary:')
        for direction, objects in summary['direction_summary'].items():
            self.get_logger().info(f'  {direction}: {objects}')

        self._publish_vision_done()

    def _publish_vision_done(self):
        done_msg = Bool()
        done_msg.data = True
        self.done_pub.publish(done_msg)

    def destroy_node(self):
        if self.writer is not None:
            self.writer.release()

        if self.cap is not None:
            self.cap.release()

        if self.show_recording_preview or self.show_processing_preview:
            cv2.destroyAllWindows()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = VisionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
