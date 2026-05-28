import json
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
    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter(
            'scan_root',
            '/home/subash/manual_perceptionPipeline/manual_scans'
        )
        self.declare_parameter('video_file', 'full_circle_video.avi')
        self.declare_parameter('model_name', 'yolo26n-seg.pt')
        self.declare_parameter('conf_threshold', 0.50)
        self.declare_parameter('frame_skip', 5)
        self.declare_parameter('output_fps', 6)
        self.declare_parameter('show_preview', False)
        self.declare_parameter('publish_preview_topic', '/vision/processed')
        self.declare_parameter('use_origin_from_metadata', True)
        self.declare_parameter('origin_yaw_offset_deg', 0.0)
        self.declare_parameter('origin_frame_id', 'map')

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

        self.scan_root = Path(str(self.get_parameter('scan_root').value))
        self.video_file = str(self.get_parameter('video_file').value)
        self.model_name = str(self.get_parameter('model_name').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.frame_skip = int(self.get_parameter('frame_skip').value)
        self.output_fps = int(self.get_parameter('output_fps').value)
        self.show_preview = bool(self.get_parameter('show_preview').value)
        self.preview_topic = str(self.get_parameter('publish_preview_topic').value)
        self.use_origin_from_metadata = bool(
            self.get_parameter('use_origin_from_metadata').value
        )
        self.origin_yaw_offset_deg = float(
            self.get_parameter('origin_yaw_offset_deg').value
        )
        self.origin_frame_id = str(self.get_parameter('origin_frame_id').value)

        self.allowed_classes = set(self.get_parameter('allowed_classes').value)
        self.blocked_classes = set(self.get_parameter('blocked_classes').value)

        if self.frame_skip <= 0:
            self.get_logger().warn('frame_skip must be > 0. Falling back to 1.')
            self.frame_skip = 1

        if self.output_fps <= 0:
            self.get_logger().warn('output_fps must be > 0. Falling back to 6.')
            self.output_fps = 6

        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.use_half = bool(torch.cuda.is_available())

        self.bridge = CvBridge()
        self.preview_pub = self.create_publisher(Image, self.preview_topic, 10)
        self.done_pub = self.create_publisher(Bool, '/vision/yolo_done', 10)

        # Run once after node starts; this node is batch-style, not continuous.
        self.started = False
        self.timer = self.create_timer(0.1, self._run_once)

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

    def _run_once(self):
        if self.started:
            return

        self.started = True

        try:
            self.process_latest_scan()
        except Exception as exc:
            self.get_logger().error(f'Vision processing failed: {exc}')
        finally:
            done_msg = Bool()
            done_msg.data = True
            self.done_pub.publish(done_msg)
            rclpy.shutdown()

    def process_latest_scan(self):
        scan_dir = self.get_latest_scan_dir()
        self.get_logger().info(f'Using latest scan folder: {scan_dir}')

        video_path = scan_dir / self.video_file
        if not video_path.exists():
            self.get_logger().error(f'Video file not found: {video_path}')
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
            self.get_logger().warn(
                'CUDA GPU not available. YOLO will run on CPU.'
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.get_logger().error(f'Failed to open video: {video_path}')
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
                    return

            writer.write(annotated_frame)

            if self.show_preview:
                cv2.imshow('YOLO 360 Processing', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info('Stopped manually by user.')
                    break

            self.preview_pub.publish(
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

        if self.show_preview:
            cv2.destroyAllWindows()

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

    def destroy_node(self):
        if self.show_preview:
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
