import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Track:
    id: int
    x: float
    y: float
    w: float
    h: float
    misses: int = 0
    hits: int = 1

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('camera_id', 0)
        self.declare_parameter('camera_device', '')   # e.g. '/dev/video0'
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('show_debug_windows', False)

        camera_id = self.get_parameter('camera_id').value
        camera_device = self.get_parameter('camera_device').value
        frame_width = self.get_parameter('frame_width').value
        frame_height = self.get_parameter('frame_height').value
        fps = self.get_parameter('fps').value
        self.show_debug_windows = self.get_parameter('show_debug_windows').value

        # On headless systems (e.g. SSH into Orin Nano) there is no display.
        # Disable windows automatically so the node doesn't crash.
        if self.show_debug_windows and not os.environ.get('DISPLAY'):
            self.get_logger().warn(
                '$DISPLAY is not set — debug windows disabled. '
                'Connect a monitor or run with X forwarding (ssh -X) to enable them.'
            )
            self.show_debug_windows = False

        cv2.setUseOptimized(True)

        # On Jetson platforms the default GStreamer backend often fails for USB
        # webcams.  Explicitly request V4L2 and fall back to auto-detect.
        source = camera_device if camera_device else camera_id
        self.cap = self._open_camera(source, frame_width, frame_height, fps)

        self.bridge = CvBridge()

        self.pub_centroids = self.create_publisher(Float32MultiArray, '/vision/centroids', 10)
        self.pub_image = self.create_publisher(Image, '/vision/processed', 10)

        self.tick_freq = cv2.getTickFrequency()
        self.prev_tick = cv2.getTickCount()
        self.fps_smooth = 0.0

        self.tracks = []
        self.next_track_id = 1
        self.max_misses = 8

        self.alpha_pos = 0.27
        self.alpha_size = 0.22
        self.match_iou_min = 0.10
        self.match_dist_max = 95.0
        self.merge_iou = 0.14
        self.merge_center_gap = 58.0

        self.is_fullscreen = False
        if self.show_debug_windows:
            cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Object Detection', 960, 540)
            cv2.namedWindow('Obstacle Mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Obstacle Mask', 960, 540)

        self.timer = self.create_timer(1.0 / float(max(1, fps)), self.process_frame)
        self.get_logger().info('Vision Node Started')

    def _open_camera(self, source, width, height, fps):
        """Try V4L2 backend first (required on Jetson), then fall back to auto."""
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        cap = None
        for backend in backends:
            attempt = cv2.VideoCapture(source, backend)
            if attempt.isOpened():
                attempt.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                attempt.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                attempt.set(cv2.CAP_PROP_FPS, fps)
                # Verify we can actually read a frame
                ok, _ = attempt.read()
                if ok:
                    self.get_logger().info(
                        f'Camera opened: source={source!r} backend={backend}'
                    )
                    cap = attempt
                    break
                attempt.release()

        if cap is None or not cap.isOpened():
            msg = (
                f'Failed to open camera source={source!r}. '
                'Check that the device is connected and not in use by another process. '
                'Try setting the camera_device parameter to /dev/video0 (or video1, etc.).'
            )
            self.get_logger().fatal(msg)
            raise RuntimeError(msg)

        return cap

    @staticmethod
    def iou_xywh(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        union = aw * ah + bw * bh - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def center_distance(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        acx = ax + 0.5 * aw
        acy = ay + 0.5 * ah
        bcx = bx + 0.5 * bw
        bcy = by + 0.5 * bh
        return abs(acx - bcx) + abs(acy - bcy)

    def merge_close_boxes(self, boxes, iou_threshold=0.2, center_gap_threshold=34.0):
        if not boxes:
            return []

        merged = [tuple(map(int, b)) for b in boxes]
        changed = True
        while changed:
            changed = False
            out = []
            used = [False] * len(merged)

            for i in range(len(merged)):
                if used[i]:
                    continue
                x1, y1, w1, h1 = merged[i]
                used[i] = True

                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    x2, y2, w2, h2 = merged[j]

                    should_merge = (
                        self.iou_xywh((x1, y1, w1, h1), (x2, y2, w2, h2)) >= iou_threshold
                        or self.center_distance((x1, y1, w1, h1), (x2, y2, w2, h2)) <= center_gap_threshold
                    )

                    if should_merge:
                        nx1 = min(x1, x2)
                        ny1 = min(y1, y2)
                        nx2 = max(x1 + w1, x2 + w2)
                        ny2 = max(y1 + h1, y2 + h2)
                        x1, y1, w1, h1 = nx1, ny1, nx2 - nx1, ny2 - ny1
                        used[j] = True
                        changed = True

                out.append((x1, y1, w1, h1))

            merged = out

        return merged

    def suppress_nested_boxes(self, boxes, overlap_keep_threshold=0.85):
        if not boxes:
            return []

        keep = [True] * len(boxes)
        areas = [w * h for (_, _, w, h) in boxes]

        for i, bi in enumerate(boxes):
            if not keep[i]:
                continue
            xi, yi, wi, hi = bi
            ai = max(1, areas[i])
            for j, bj in enumerate(boxes):
                if i == j or not keep[j]:
                    continue
                xj, yj, wj, hj = bj
                aj = max(1, areas[j])

                ix1 = max(xi, xj)
                iy1 = max(yi, yj)
                ix2 = min(xi + wi, xj + wj)
                iy2 = min(yi + hi, yj + hj)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0:
                    continue

                overlap_i = inter / ai
                overlap_j = inter / aj

                if overlap_i >= overlap_keep_threshold and ai < aj:
                    keep[i] = False
                    break
                if overlap_j >= overlap_keep_threshold and aj < ai:
                    keep[j] = False

        return [b for k, b in zip(keep, boxes) if k]

    @staticmethod
    def preprocess_for_obstacles(frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        obstacle_mask = cv2.dilate(edges, kernel, iterations=2)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return gray, obstacle_mask

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_width = self.get_parameter('frame_width').value
        frame_height = self.get_parameter('frame_height').value
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
        vis = frame.copy()

        _, obstacle_mask = self.preprocess_for_obstacles(frame)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
        vertical_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, vertical_kernel)

        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 11))
        vertical_mask = cv2.morphologyEx(vertical_mask, cv2.MORPH_CLOSE, connect_kernel, iterations=1)

        contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = max(1400, int(frame_area * 0.006))
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh <= int(1.15 * bw):
                continue
            raw_boxes.append((x, y, bw, bh))

        detections = self.merge_close_boxes(
            raw_boxes,
            iou_threshold=self.merge_iou,
            center_gap_threshold=self.merge_center_gap,
        )
        detections = self.suppress_nested_boxes(detections)

        candidates = []
        for ti, tr in enumerate(self.tracks):
            tbox = (tr.x, tr.y, tr.w, tr.h)
            for di, det in enumerate(detections):
                dbox = tuple(float(v) for v in det)
                ov = self.iou_xywh(tbox, dbox)
                dist = self.center_distance(tbox, dbox)
                if ov > self.match_iou_min or dist < self.match_dist_max:
                    score = ov - 0.0015 * dist
                    candidates.append((score, ti, di))
        candidates.sort(reverse=True, key=lambda x: x[0])

        matched_tracks = set()
        matched_dets = set()
        for _, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            tr = self.tracks[ti]
            dx, dy, dw, dh = detections[di]

            tr.x = tr.x + self.alpha_pos * (dx - tr.x)
            tr.y = tr.y + self.alpha_pos * (dy - tr.y)
            tr.w = tr.w + self.alpha_size * (dw - tr.w)
            tr.h = tr.h + self.alpha_size * (dh - tr.h)

            if abs(dx - tr.x) < 1.0:
                tr.x = round(tr.x)
            if abs(dy - tr.y) < 1.0:
                tr.y = round(tr.y)

            tr.misses = 0
            tr.hits += 1

            matched_tracks.add(ti)
            matched_dets.add(di)

        for ti, tr in enumerate(self.tracks):
            if ti not in matched_tracks:
                tr.misses += 1

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            x, y, w_box, h_box = det
            self.tracks.append(Track(id=self.next_track_id, x=float(x), y=float(y), w=float(w_box), h=float(h_box)))
            self.next_track_id += 1

        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]

        h_frame, w_frame = vis.shape[:2]
        centroids = []
        visible_count = 0
        for tr in self.tracks:
            if tr.hits < 2:
                continue

            sx = int(max(0, min(w_frame - 2, tr.x)))
            sy = int(max(0, min(h_frame - 2, tr.y)))
            sw = int(max(2, min(w_frame - sx - 1, tr.w)))
            sh = int(max(2, min(h_frame - sy - 1, tr.h)))

            if tr.misses > 0:
                color = (0, 190, 255)
                label = f'Object #{tr.id} (hold)'
            else:
                color = (0, 255, 0)
                label = f'Object #{tr.id}'

            cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), color, 2)
            cv2.putText(vis, label, (sx, max(15, sy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cx = sx + sw // 2
            cy = sy + sh // 2
            centroids.append([cx, cy])
            visible_count += 1

        now_tick = cv2.getTickCount()
        dt = (now_tick - self.prev_tick) / self.tick_freq
        self.prev_tick = now_tick
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        self.fps_smooth = 0.9 * self.fps_smooth + 0.1 * instant_fps if self.fps_smooth > 0 else instant_fps

        h, w = vis.shape[:2]
        cv2.putText(vis, f'Objects: {visible_count}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f'FPS: {self.fps_smooth:.1f}', (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        hud_y = h - 80
        cv2.putText(
            vis,
            f'a/z pos:{self.alpha_pos:.2f}  s/x size:{self.alpha_size:.2f}  d/c miss:{self.max_misses}',
            (12, hud_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            vis,
            f'g/b mergeIOU:{self.merge_iou:.2f}  h/n mergeGap:{self.merge_center_gap:.0f}  j/m matchDist:{self.match_dist_max:.0f}',
            (12, hud_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            vis,
            'r reset tracks  f fullscreen  q quit',
            (12, hud_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )

        if self.show_debug_windows:
            cv2.imshow('Object Detection', vis)
            cv2.imshow('Obstacle Mask', vertical_mask)
            key = cv2.waitKey(1) & 0xFF
            self.handle_keyboard(key)

        msg = Float32MultiArray()
        msg.data = [float(c) for centroid in centroids for c in centroid]
        self.pub_centroids.publish(msg)

        self.pub_image.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))

    def handle_keyboard(self, key):
        if key == ord('a'):
            self.alpha_pos = min(0.90, self.alpha_pos + 0.02)
        if key == ord('z'):
            self.alpha_pos = max(0.05, self.alpha_pos - 0.02)
        if key == ord('s'):
            self.alpha_size = min(0.90, self.alpha_size + 0.02)
        if key == ord('x'):
            self.alpha_size = max(0.05, self.alpha_size - 0.02)
        if key == ord('d'):
            self.max_misses = min(30, self.max_misses + 1)
        if key == ord('c'):
            self.max_misses = max(0, self.max_misses - 1)
        if key == ord('g'):
            self.merge_iou = min(0.90, self.merge_iou + 0.02)
        if key == ord('b'):
            self.merge_iou = max(0.01, self.merge_iou - 0.02)
        if key == ord('h'):
            self.merge_center_gap = min(120.0, self.merge_center_gap + 2.0)
        if key == ord('n'):
            self.merge_center_gap = max(4.0, self.merge_center_gap - 2.0)
        if key == ord('j'):
            self.match_dist_max = min(220.0, self.match_dist_max + 4.0)
        if key == ord('m'):
            self.match_dist_max = max(20.0, self.match_dist_max - 4.0)
        if key == ord('r'):
            self.tracks.clear()
        if key == ord('f'):
            self.is_fullscreen = not self.is_fullscreen
            mode = cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN, mode)
            if not self.is_fullscreen:
                cv2.resizeWindow('Object Detection', 960, 540)
        if key == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        if self.show_debug_windows:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()