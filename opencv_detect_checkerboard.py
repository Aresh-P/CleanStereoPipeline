# Usage:
# uv run opencv_detect_checkerboard.py --square-size 0.154 --width 11 --height 5 --input '../pipeline/camera_datasets/ov580/checkerboards/*.jpg' --output ov_calibration.npz

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2
import argparse
import os
from utils import UnrectStereoPair, StereoPairSource, CalibData, save_calibration, RectMaps
from stereo_sources import StereoImageSource, OpenCVLiveSource
from calib_models import OpenCVCalibModel, OpenCVCalibParams

# Get screen resolution for fullscreen
import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

@dataclass
class CheckerboardConfig:
    """Configuration for checkerboard calibration."""
    square_size: float  # Side length of a square in meters
    width: int  # Number of internal corners horizontally
    height: int  # Number of internal corners vertically
    stereo_pair: UnrectStereoPair  # Unrectified stereo image pair


def detect_checkerboard(config: CheckerboardConfig) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Detect checkerboard corners in a stereo pair.
    
    Returns:
        Optional tuple of (object_points, image_points_left, image_points_right)
        Returns None if detection fails on either image.
    """
    # Define the checkerboard pattern size
    pattern_size = (config.width, config.height)
    
    # Prepare object points based on checkerboard dimensions
    object_points = np.zeros((config.width * config.height, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:config.width, 0:config.height].reshape(2, -1).T
    object_points *= config.square_size
    
    # Convert images to grayscale if needed
    left_gray = cv2.cvtColor(config.stereo_pair.left, cv2.COLOR_BGR2GRAY) if len(config.stereo_pair.left.shape) == 3 else config.stereo_pair.left
    right_gray = cv2.cvtColor(config.stereo_pair.right, cv2.COLOR_BGR2GRAY) if len(config.stereo_pair.right.shape) == 3 else config.stereo_pair.right
    
    # Find corners in both images
    ret_left, corners_left = cv2.findChessboardCorners(left_gray, pattern_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(right_gray, pattern_size, None)
    
    # If both detections succeed
    if ret_left and ret_right:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
        
        # Prepare images for display
        left_display = config.stereo_pair.left.copy()
        right_display = config.stereo_pair.right.copy()
        
        # Draw corners
        cv2.drawChessboardCorners(left_display, pattern_size, corners_left, ret_left)
        cv2.drawChessboardCorners(right_display, pattern_size, corners_right, ret_right)
        
        # Combine images side by side
        combined = np.hstack([left_display, right_display])
        
        # Display for user review
        cv2.namedWindow('Detected Checkerboard Corners', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('Detected Checkerboard Corners', combined)

        cv2.resizeWindow('Detected Checkerboard Corners', screen_width, screen_height)
        cv2.moveWindow('Detected Checkerboard Corners', 0, 0)
        
        print("Press 'q' to accept, 'ESC' to reject")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return object_points, corners_left.reshape(-1, 2), corners_right.reshape(-1, 2)
            elif key == 27:  # ESC key
                cv2.destroyAllWindows()
                return None
            # Ignore all other keys and continue waiting
    
    return None


class LiveCalibrationSource(StereoPairSource):
    """
    Live source that shows preview with optional checkerboard overlay,
    waits for user to capture snapshots for calibration.
    """
    def __init__(self, device_id: int, checker_width: int, checker_height: int):
        self.live_source = OpenCVLiveSource(device_id)
        self.checker_size = (checker_width, checker_height)
        
    def get_pair(self) -> Optional[UnrectStereoPair]:
        print("\nLive preview: SPACE to capture when checkerboard found, ENTER to finish")
        while True:
            # Try to get a valid frame up to 20 times
            stereo_pair = None
            for _ in range(20):
                stereo_pair = self.live_source.get_pair()
                if stereo_pair is not None:
                    break
            
            if stereo_pair is None:
                print("Too many dropped frames, stopping...")
                return None
                
            left = stereo_pair.left
            right = stereo_pair.right
            
            # Try to detect corners for overlay
            # Create display by combining left and right images
            display = np.hstack([left.copy(), right.copy()])
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if len(left.shape) == 3 else left
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if len(right.shape) == 3 else right
            
            ret_l, corners_l = cv2.findChessboardCorners(left_gray, self.checker_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(right_gray, self.checker_size, None)
            
            # Draw corners if found
            half_width = left.shape[1]
            if ret_l:
                cv2.drawChessboardCorners(display[:, :half_width], self.checker_size, corners_l, ret_l)
            if ret_r:
                cv2.drawChessboardCorners(display[:, half_width:], self.checker_size, corners_r, ret_r)
                
            # Show status
            if ret_l and ret_r:
                status = "Checkerboard: FOUND (press SPACE to capture)"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                status = f"Checkerboard: Left {'OK' if ret_l else 'NO'}, Right {'OK' if ret_r else 'NO'}"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow('Live Calibration', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and ret_l and ret_r:  # Space to capture
                cv2.destroyWindow('Live Calibration')
                return UnrectStereoPair(left, right)
            elif key == 13:  # Enter key to finish
                cv2.destroyWindow('Live Calibration')
                return None


def collect_calibration_data(
    square_size: float,
    width: int,
    height: int,
    source: StereoPairSource
) -> CalibData:
    """
    Collect calibration data from a stereo source by detecting checkerboards.
    
    Args:
        square_size: Side length of a checkerboard square in meters
        width: Number of internal corners horizontally
        height: Number of internal corners vertically
        source: Source of stereo image pairs
    
    Returns:
        CalibData object containing all detected checkerboard points
    """
    object_points_list: List[np.ndarray] = []
    image_points_left_list: List[np.ndarray] = []
    image_points_right_list: List[np.ndarray] = []
    image_size = None
    
    while True:
        # Get next stereo pair
        stereo_pair = source.get_pair()
        if stereo_pair is None:
            break
        
        # Store image size from first image
        if image_size is None:
            image_size = (stereo_pair.left.shape[1], stereo_pair.left.shape[0])
        
        # Create config for this pair
        config = CheckerboardConfig(
            square_size=square_size,
            width=width,
            height=height,
            stereo_pair=stereo_pair
        )
        
        # Detect checkerboard
        result = detect_checkerboard(config)
        if result is not None:
            obj_pts, img_pts_left, img_pts_right = result
            object_points_list.append(obj_pts)
            image_points_left_list.append(img_pts_left)
            image_points_right_list.append(img_pts_right)
            print(f"Collected calibration pair {len(object_points_list)}")
    
    if not object_points_list:
        raise ValueError("No calibration pairs were collected")
    
    print(f"Total calibration pairs collected: {len(object_points_list)}")
    
    return CalibData(
        object_points=object_points_list,
        image_points_left=image_points_left_list,
        image_points_right=image_points_right_list,
        image_size=image_size
    )


def main():
    """
    Command-line interface for checkerboard calibration.
    """
    parser = argparse.ArgumentParser(description='Calibrate stereo camera using checkerboard pattern')
    parser.add_argument('--square-size', type=float, required=True,
                        help='Size of checkerboard square in meters')
    parser.add_argument('--width', type=int, required=True,
                        help='Number of internal corners horizontally')
    parser.add_argument('--height', type=int, required=True,
                        help='Number of internal corners vertically')
    parser.add_argument('--output', type=str, default='calibration.npz',
                        help='Output calibration file (default: calibration.npz)')
    
    # Create mutually exclusive group for source options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--input', type=str,
                              help='Glob pattern for stereo images (e.g., "*.png")')
    source_group.add_argument('--device', type=int,
                              help='Camera device ID for live capture')
    
    args = parser.parse_args()
    
    # Create stereo source
    if args.input:
        source = StereoImageSource(args.input)
    else:  # args.device
        source = LiveCalibrationSource(args.device, args.width, args.height)
    
    # Collect calibration data
    print(f"Collecting calibration data...")
    print(f"Checkerboard: {args.width}x{args.height} corners, {args.square_size}m squares")
    print("Press 'q' to accept detected corners, ESC to skip")
    
    calib_data = collect_calibration_data(
        square_size=args.square_size,
        width=args.width,
        height=args.height,
        source=source
    )
    
    # Perform calibration using the same approach as stereo_calibrate.py
    print("\nPerforming stereo calibration...")
    
    # Initialize camera matrices and distortion coefficients
    K1 = np.eye(3, dtype=np.float64)
    D1 = np.zeros((5, 1), dtype=np.float64)
    K2 = np.eye(3, dtype=np.float64)
    D2 = np.zeros((5, 1), dtype=np.float64)
    
    # Set flags exactly as in stereo_calibrate.py
    flags = 0
    flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_SAME_PRINCIPAL_POINT  # Commented out as it might not exist
    
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    
    # Stereo calibration
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        calib_data.object_points, 
        calib_data.image_points_left, 
        calib_data.image_points_right,
        K1, D1, K2, D2, 
        calib_data.image_size,
        criteria=criteria,
        flags=flags
    )
    
    print(f"\nStereo calibration RMS reprojection error: {ret:.4f} pixels")
    print(f"Left camera matrix:\n{K1}")
    print(f"Right camera matrix:\n{K2}")
    print(f"Left distortion coefficients: {D1.ravel()}")
    print(f"Right distortion coefficients: {D2.ravel()}")
    print(f"Rotation between cameras:\n{R}")
    print(f"Translation between cameras: {T.ravel()}")
    baseline_mm = np.linalg.norm(T) * 1000
    print(f"Baseline: {baseline_mm:.2f} mm")
    
    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, calib_data.image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    # Create rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, calib_data.image_size, cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, calib_data.image_size, cv2.CV_32FC1
    )
    
    rect_maps = RectMaps(map1_left, map2_left, map1_right, map2_right)
    
    # Save calibration
    save_calibration(args.output, rect_maps, Q)
    print(f"\nCalibration saved to {args.output}")


if __name__ == '__main__':
    main()
