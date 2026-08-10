import os

import cv2
import numpy as np
import pyrealsense2 as rs


def start_realsense_pipeline():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found. Connect the camera and try again.")

    pipeline = rs.pipeline()
    config = rs.config()

    try:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        print("Trying color and depth streams at 640x480 @ 30 fps")
        profile = pipeline.start(config)
    except RuntimeError as exc:
        print(f"Stream request failed: {exc}")
        print("Falling back to the camera's default stream settings")
        profile = pipeline.start()

    return pipeline, profile


def segment_capsule(color_image, lower, upper):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 400]
    if not valid_contours:
        return None

    filled_mask = np.zeros_like(mask)
    for cnt in valid_contours:
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    filled_mask = cv2.dilate(filled_mask, kernel, iterations=1)
    return filled_mask


def nothing(_):
    pass


pipeline, profile = start_realsense_pipeline()
align = rs.align(rs.stream.color)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth scale:", depth_scale)

cv2.namedWindow("Capsule View")
cv2.namedWindow("Mask View")
cv2.namedWindow("Depth View")
cv2.namedWindow("HSV Settings")

cv2.createTrackbar("H min", "HSV Settings", 15, 179, nothing)
cv2.createTrackbar("H max", "HSV Settings", 80, 179, nothing)
cv2.createTrackbar("S min", "HSV Settings", 60, 255, nothing)
cv2.createTrackbar("S max", "HSV Settings", 255, 255, nothing)
cv2.createTrackbar("V min", "HSV Settings", 60, 255, nothing)
cv2.createTrackbar("V max", "HSV Settings", 255, 255, nothing)

print("Warming up camera...")
for _ in range(20):
    pipeline.wait_for_frames()

print("Camera ready. Adjust the HSV sliders until the capsule is isolated.")
print("Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_mm = depth_raw.astype(np.float32) * depth_scale * 1000.0

        h_min = cv2.getTrackbarPos("H min", "HSV Settings")
        h_max = cv2.getTrackbarPos("H max", "HSV Settings")
        s_min = cv2.getTrackbarPos("S min", "HSV Settings")
        s_max = cv2.getTrackbarPos("S max", "HSV Settings")
        v_min = cv2.getTrackbarPos("V min", "HSV Settings")
        v_max = cv2.getTrackbarPos("V max", "HSV Settings")

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        mask = segment_capsule(color, lower, upper)
        if mask is None:
            isolated_color = color
            mask_view = np.zeros(color.shape[:2], dtype=np.uint8)
            distance_mm = float("nan")
        else:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (color.shape[1], color.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            isolated_color = cv2.bitwise_and(color, color, mask=mask)
            mask_view = mask

            masked_depth = depth_mm[mask > 0]
            valid_depth = masked_depth[masked_depth > 0]
            if valid_depth.size > 0:
                distance_mm = float(np.median(valid_depth))
            else:
                distance_mm = float("nan")

        depth_display = cv2.applyColorMap(
            cv2.convertScaleAbs(np.nan_to_num(depth_mm, nan=0.0), alpha=0.05),
            cv2.COLORMAP_JET,
        )
        if mask_view.size > 0:
            depth_mask_3ch = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)
            depth_display = cv2.bitwise_and(depth_display, depth_mask_3ch)

        cv2.putText(
            isolated_color,
            f"Depth: {distance_mm:.1f} mm" if np.isfinite(distance_mm) else "Depth: n/a",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Capsule View", isolated_color)
        cv2.imshow("Mask View", cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR))
        cv2.imshow("Depth View", depth_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print("Done.")
