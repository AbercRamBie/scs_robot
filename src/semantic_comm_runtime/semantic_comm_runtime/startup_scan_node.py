import os
import json
import time
from pathlib import Path
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

class StartupScanNode(Node):
    def __init__(self):
        super().__init__('startup_scan_node')

        self.declare_parameter('camera_id', 0)
        self.declare_parameter('camera_device', '')  # e.g. '/dev/video0'
        self.declare_parameter('output_root', 'manual_scans')
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 20)
        self.declare_parameter('start_delay_sec', 5.0)
        self.declare_parameter('target_rotation_time_sec', 20.0)
        self.declare_parameter('save_angle_frames', True)
        self.declare_parameter('frame_save_angle_step_deg', 30)
        self.declare_parameter('use_v4l2', True)
        self.declare_parameter('show_preview', True)
        self.declare_parameter('publish_preview_topic', '/startup_scan/preview')
        self.declare_parameter('origin_frame_id', 'map')
        self.declare_parameter('origin_x_m', 0.0)
        self.declare_parameter('origin_y_m', 0.0)
        self.declare_parameter('origin_z_m', 0.0)
        self.declare_parameter('origin_yaw_deg', 0.0)
        self.declare_parameter(
            'origin_note',
            'Manual scan starts at this user-defined origin pose.'
        )

        self.camera_id = int(self.get_parameter('camera_id').value)
        self.camera_device = str(self.get_parameter('camera_device').value)
        self.output_root = Path(str(self.get_parameter('output_root').value))
        self.frame_width = int(self.get_parameter('frame_width').value)
        self.frame_height = int(self.get_parameter('frame_height').value)
        self.fps = int(self.get_parameter('fps').value)
        self.start_delay_sec = float(self.get_parameter('start_delay_sec').value)
        self.target_rotation_time_sec = float(
            self.get_parameter('target_rotation_time_sec').value
        )
        self.save_angle_frames = bool(self.get_parameter('save_angle_frames').value)
        self.frame_save_angle_step_deg = int(
            self.get_parameter('frame_save_angle_step_deg').value
        )
        self.use_v4l2 = bool(self.get_parameter('use_v4l2').value)
        self.show_preview = bool(self.get_parameter('show_preview').value)
        self.preview_topic = str(self.get_parameter('publish_preview_topic').value)
        self.origin_frame_id = str(self.get_parameter('origin_frame_id').value)
        self.origin_x_m = float(self.get_parameter('origin_x_m').value)
        self.origin_y_m = float(self.get_parameter('origin_y_m').value)
        self.origin_z_m = float(self.get_parameter('origin_z_m').value)
        self.origin_yaw_deg = float(self.get_parameter('origin_yaw_deg').value)
        self.origin_note = str(self.get_parameter('origin_note').value)

        if self.frame_save_angle_step_deg <= 0:
            self.get_logger().warn(
                'frame_save_angle_step_deg must be > 0. Falling back to 30.'
            )
            self.frame_save_angle_step_deg = 30

        if self.target_rotation_time_sec <= 0.0:
            self.get_logger().warn(
                'target_rotation_time_sec must be > 0. Falling back to 20.0.'
            )
            self.target_rotation_time_sec = 20.0

        if self.show_preview and not os.environ.get('DISPLAY'):
            self.get_logger().warn(
                '$DISPLAY is not set, disabling OpenCV preview window.'
            )
            self.show_preview = False

        self.bridge = CvBridge()
        self.preview_pub = self.create_publisher(Image, self.preview_topic, 10)
        self.done_pub = self.create_publisher(Bool, '/startup_scan/done', 10)

        self.cap = self._open_camera()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.get_logger().info('Warming up camera...')
        self._warmup_camera()

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.scan_dir = self.output_root / f'manual_video_scan_{timestamp}'
        self.scan_dir.mkdir(parents=True, exist_ok=True)

        self.frames_dir = self.scan_dir / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.scan_dir / 'full_circle_video.avi'
        self.metadata_path = self.scan_dir / 'metadata.json'
        self.frame_log_path = self.scan_dir / 'frame_angles.json'

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )

        if not self.writer.isOpened():
            raise RuntimeError('Failed to open video writer')

        self.get_logger().info('Manual 360 video capture ready.')
        self.get_logger().info(
            f'Recording will start after {self.start_delay_sec:.1f} seconds.'
        )
        self.get_logger().info(
            'Try to complete one full 360 degree rotation in '
            f'{self.target_rotation_time_sec:.1f} seconds.'
        )
        self.get_logger().info(
            f'Saving representative frames every {self.frame_save_angle_step_deg} degrees.'
        )
        self.get_logger().info('Press q in preview window anytime to cancel.')
        self.get_logger().info(
            'Origin reference: '
            f'frame={self.origin_frame_id}, '
            f'position=({self.origin_x_m:.3f}, {self.origin_y_m:.3f}, {self.origin_z_m:.3f}) m, '
            f'yaw={self.origin_yaw_deg:.1f} deg'
        )

        self.state = 'countdown'
        self.delay_start = time.time()
        self.recording_start = None
        self.frame_count = 0
        self.frame_angle_log = []
        self.next_save_angle = 0
        self.saved_angle_frames = []
        self.stopped_manually = False

        self.timer = self.create_timer(1.0 / float(max(1, self.fps)), self.update)

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

    def _publish_preview(self, image_bgr):
        if rclpy.ok():
            self.preview_pub.publish(self.bridge.cv2_to_imgmsg(image_bgr, 'bgr8'))

    def _finalize_scan(self):
        duration = 0.0
        if self.recording_start is not None:
            duration = time.time() - self.recording_start

        metadata = {
            'scan_type': 'manual_360_video_capture',
            'video_file': 'full_circle_video.avi',
            'frame_angle_log': 'frame_angles.json',
            'frames_dir': 'frames',
            'save_angle_frames': self.save_angle_frames,
            'frame_save_angle_step_deg': self.frame_save_angle_step_deg,
            'saved_angle_frames': self.saved_angle_frames,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'start_delay_sec': self.start_delay_sec,
            'target_rotation_time_sec': self.target_rotation_time_sec,
            'actual_recording_duration_sec': round(duration, 3),
            'angle_estimation_method': 'time_based_estimation',
            'rotation_note': 'estimated yaw assumes constant manual rotation speed',
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
        self.get_logger().info(f'Angle frames saved: {len(self.saved_angle_frames)}')
        self.get_logger().info(f'Frames recorded: {self.frame_count}')
        self.get_logger().info(f'Recording duration: {duration:.2f}s')

        done_msg = Bool()
        done_msg.data = True
        self.done_pub.publish(done_msg)

    def _handle_key(self, key):
        if key == ord('q'):
            if self.state == 'countdown':
                self.get_logger().info('Cancelled before recording.')
                self.stopped_manually = True
                self.state = 'done'
                rclpy.shutdown()
            elif self.state == 'recording':
                self.get_logger().info('Stopped manually.')
                self.stopped_manually = True
                self.state = 'finalizing'

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame')
            return

        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        display = frame.copy()

        if self.state == 'countdown':
            elapsed_delay = time.time() - self.delay_start
            remaining = max(0.0, self.start_delay_sec - elapsed_delay)

            cv2.putText(
                display,
                f'Recording starts in: {remaining:.1f}s',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                'Get ready. Keep camera steady.',
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                (
                    f'Origin {self.origin_frame_id}: '
                    f'({self.origin_x_m:.2f}, {self.origin_y_m:.2f}, {self.origin_z_m:.2f}) m '
                    f'yaw {self.origin_yaw_deg:.1f} deg'
                ),
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            if elapsed_delay >= self.start_delay_sec:
                self.state = 'recording'
                self.recording_start = time.time()
                self.get_logger().info('Recording started. Rotate now.')

        elif self.state == 'recording':
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
            cv2.putText(
                display,
                'Rotate smoothly. Press q to stop early.',
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        if self.show_preview:
            cv2.imshow('Manual 360 Video Capture', display)
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)

        self._publish_preview(display)

        if self.state == 'finalizing':
            self._finalize_scan()
            self.state = 'done'
            rclpy.shutdown()

    def destroy_node(self):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.release()

        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        if self.show_preview:
            cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = StartupScanNode()
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
