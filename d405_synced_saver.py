#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ROS 2 Message Synchronization
import message_filters

class D405SyncedSaverNode(Node):
    def __init__(self):
        super().__init__('d405_synced_saver')
        self.bridge = CvBridge()
        
        # Output Directories
        self.output_dir = "captured_d405_data"
        self.rgb_dir = os.path.join(self.output_dir, "rgb")
        self.depth_dir = os.path.join(self.output_dir, "depth_raw")
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        self.saved_count = 0

        # Topic Subscribers (Adjust prefix if camera_name parameter differs)
        rgb_sub = message_filters.Subscriber(self, Image, '/d405/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/d405/depth/image_rect_raw')

        # Synchronize topics within a 10ms window
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 
            queue_size=10, 
            slop=0.01
        )
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info(
            f"Synced Saver Initialized.\n"
            f"Focus on the preview window and press SPACE to capture, or 'q' to quit."
        )

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            # Convert ROS RGB Image to OpenCV BGR
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            
            # Convert ROS Depth Image to 16-bit Unsigned Raw Depth (millimeters)
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Create visual depth display (Normalized 8-bit for UI preview only)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(cv_depth, alpha=0.03), 
                cv2.COLORMAP_JET
            )

            # Stack images side-by-side for live preview
            preview = np.hstack((cv_rgb, depth_colormap))
            
            # Calculate sharpness score on RGB
            gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Draw overlay info on preview
            status_text = f"Count: {self.saved_count} | Sharpness: {sharpness:.1f} | Press SPACE to Save"
            cv2.putText(
                preview, status_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            cv2.imshow("D405 Live Feed (RGB | Depth Preview)", preview)
            key = cv2.waitKey(1) & 0xFF

            # --- KEYPRESS LOGIC ---
            if key == ord(' '):  # SPACEBAR to Save
                timestamp = int(time.time() * 1000)
                
                rgb_filename = os.path.join(self.rgb_dir, f"frame_{self.saved_count:04d}_{timestamp}.jpg")
                depth_filename = os.path.join(self.depth_dir, f"frame_{self.saved_count:04d}_{timestamp}.png")

                # Save RGB as JPEG
                cv2.imwrite(rgb_filename, cv_rgb)
                
                # Save Raw 16-bit Depth as PNG (Preserves precise mm depth values)
                cv2.imwrite(depth_filename, cv_depth)

                self.saved_count += 1
                self.get_logger().info(
                    f"[{self.saved_count}] Saved synced pair:\n"
                    f"  RGB:   {rgb_filename}\n"
                    f"  Depth: {depth_filename}"
                )

            elif key == ord('q'):  # 'q' to Quit
                self.get_logger().info("Exit key pressed. Shutting down...")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error processing synced frame: {e}")

def main():
    rclpy.init()
    node = D405SyncedSaverNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()