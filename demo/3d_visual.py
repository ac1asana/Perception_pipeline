import numpy as np
import open3d as o3d

# Load depth array
depth_mm = np.load("demo_captures/sample_02_depth.npy").astype(np.float32)

# Convert depth image to Open3D image object
depth_o3d = o3d.geometry.Image(depth_mm)

# Use D405 default intrinsic parameters for 640x480 resolution
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=640,
    height=480,
    fx=384.0,  # Approximate focal length
    fy=384.0,
    cx=320.0,  # Principal point center X
    cy=240.0,  # Principal point center Y
)

# Generate 3D point cloud
pcd = o3d.geometry.PointCloud.create_from_depth_image(
    depth_o3d, intrinsics, depth_scale=1000.0  # mm to meters
)

# Visualize interactive 3D window
o3d.visualization.draw_geometries([pcd])