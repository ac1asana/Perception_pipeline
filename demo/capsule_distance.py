import cv2
import numpy as np
import pyrealsense2 as rs


# Update this to the actual capsule color.
# Example default below is green, because the broader earlier version was matching skin/finger tones.
TARGET_HSV_RANGE = (
    np.array([35, 80, 60], dtype=np.uint8),
    np.array([90, 255, 255], dtype=np.uint8),
)


class CapsuleDistanceTracker:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.current_depth_frame = None
        self.current_color_frame = None

    def start(self):
        self.pipeline.start(self.config)

    def stop(self):
        self.pipeline.stop()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        self.current_color_frame = color_frame
        self.current_depth_frame = depth_frame
        return color_frame, depth_frame

    def detect_capsule_mask(self, color_img):
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        lower, upper = TARGET_HSV_RANGE
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if cv2.countNonZero(mask) == 0:
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < 200:
            return None

        x, y, w, h = cv2.boundingRect(best)
        if w <= 0 or h <= 0:
            return None

        # Keep a reasonable capsule-like contour by requiring not too tiny and not too huge
        if min(w, h) < 10:
            return None

        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [best], -1, 255, thickness=-1)
        return mask, (x, y, w, h)

    def distance_from_mask(self, depth_frame, mask):
        if mask is None:
            return None
        depth_data = np.asanyarray(depth_frame.get_data())
        valid = depth_data[mask > 0]
        valid = valid[valid > 0]
        if valid.size == 0:
            return None
        avg_mm = float(np.mean(valid))
        return avg_mm / 1000.0, avg_mm

    def run(self):
        self.start()
        cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and self.current_depth_frame is not None:
                dist_m = self.current_depth_frame.get_distance(x, y)
                if dist_m > 0:
                    print(f"Clicked pixel ({x}, {y}) -> {dist_m:.3f} m ({dist_m * 1000:.1f} mm)")
                else:
                    print(f"Clicked pixel ({x}, {y}) -> no valid depth")

        cv2.setMouseCallback("Color", click_event)

        try:
            while True:
                color_frame, depth_frame = self.get_frames()
                if color_frame is None or depth_frame is None:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())

                result = self.detect_capsule_mask(color_img)
                mask = None
                bbox = None
                if result is not None:
                    mask, bbox = result

                if mask is not None:
                    dist_info = self.distance_from_mask(depth_frame, mask)
                    if dist_info is not None:
                        avg_m, avg_mm = dist_info
                        print(f"Detected capsule avg distance: {avg_m:.3f} m ({avg_mm:.1f} mm)")

                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_data, alpha=0.05), cv2.COLORMAP_JET
                )

                if mask is not None:
                    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            color_img,
                            "capsule",
                            (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                else:
                    mask_vis = np.zeros((480, 640, 3), dtype=np.uint8)

                cv2.imshow("Color", color_img)
                cv2.imshow("Mask", mask_vis)
                cv2.imshow("Depth", depth_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = CapsuleDistanceTracker()
    tracker.run()
