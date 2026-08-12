import cv2
import numpy as np
import pyrealsense2 as rs

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

current_depth_frame = None
i = 1


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and current_depth_frame is not None:
        dist_m = current_depth_frame.get_distance(x, y)
        if dist_m > 0:
            print(f"Pixel ({x}, {y}) -> {dist_m:.3f} m ({dist_m * 1000:.1f} mm)")
        else:
            print(f"Pixel ({x}, {y}) -> no valid depth")


cv2.namedWindow("Color")
cv2.setMouseCallback("Color", click_event)

try:
    while True:
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        current_depth_frame = depth_frame

        # display color
        cv2.imshow("Color", color_img)

        # display depth as colorized map
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.05),
            cv2.COLORMAP_JET
        )
        cv2.imshow("Depth", depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(f"color_{i}.png", color_img)
            np.save(f"depth_{i}.npy", depth_img)
            print(f"Saved color_{i}.png and depth_{i}.npy")
            i += 1
        elif key == ord("q") or key == 27:
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()