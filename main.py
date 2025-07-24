# Usage:
# uv run main.py --device 2 ov_calibration.npz

import argparse
import cv2
import numpy as np
import sys
import os
import time

from utils import load_calibration, rectify, disp_to_depth
from stereo_sources import OpenCVLiveSource, StereoImageSource
from stereo_matchers import OpenCVSGBMMatcher, OpenCVSGBMParams

# Get screen resolution for fullscreen
import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

def main():
    parser = argparse.ArgumentParser(description='Display depth map from stereo camera or images')
    parser.add_argument('calibration_file', help='Path to calibration file (.npz)')
    
    # Create mutually exclusive group for source options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--device', type=int, help='Camera device ID for live capture')
    source_group.add_argument('--images', type=str, help='Glob pattern for stereo image files')
    
    args = parser.parse_args()
    
    # Load calibration data
    try:
        rect_maps, Q = load_calibration(args.calibration_file)
        print(f"Loaded calibration from {args.calibration_file}")
    except Exception as e:
        print(f"Error loading calibration: {e}")
        sys.exit(1)
    
    # Initialize source based on arguments
    try:
        if args.device is not None:
            source = OpenCVLiveSource(args.device)
            print(f"Initialized camera device {args.device}")
            is_live = True
        else:
            source = StereoImageSource(args.images)
            print(f"Initialized image source with pattern: {args.images}")
            is_live = False
    except Exception as e:
        print(f"Error initializing source: {e}")
        sys.exit(1)
    
    # Initialize stereo matcher with default parameters
    sgbm_params = OpenCVSGBMParams()
    matcher = OpenCVSGBMMatcher(sgbm_params)
    print("Initialized SGBM matcher with WLS filter")
    
    # Create windows
    cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Disparity Map', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
    
    print("\nControls:")
    print("  ESC or 'q' - Quit")
    print("  's' - Save current depth map")
    if not is_live:
        print("  Left/Right arrows - Navigate between images")
        print("  Home - First image")
        print("  End - Last image")
    
    frame_count = 0
    
    while True:
        try:
            # Capture stereo pair
            unrect_pair = source.get_pair()
            if unrect_pair is None:
                print("Didn't get a pair, exiting")
                exit()
            
            # Debug: Check if images are black
            if is_live and frame_count == 0:
                left_mean = np.mean(unrect_pair.left)
                right_mean = np.mean(unrect_pair.right)
                print(f"Debug: Left image mean brightness: {left_mean:.2f}")
                print(f"Debug: Right image mean brightness: {right_mean:.2f}")
                print(f"Debug: Left image shape: {unrect_pair.left.shape}")
                print(f"Debug: Right image shape: {unrect_pair.right.shape}")
            
            # Rectify images
            rect_pair = rectify(unrect_pair, rect_maps)
            
            # Compute disparity
            disparity = matcher.match(rect_pair)
            
            # Convert disparity to depth
            depth_map = disp_to_depth(disparity, Q)
            
            # Normalize disparity for visualization
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            
            # Normalize depth for visualization (clip to reasonable range)
            depth_vis = np.clip(depth_map, 0, 5000)  # Clip to 5 meters
            depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # Debug: Save rectified images on first frame
            if is_live and frame_count == 0:
                cv2.imwrite('debug_rect_left.jpg', rect_pair.left)
                cv2.imwrite('debug_rect_right.jpg', rect_pair.right)
                print("Debug: Saved rectified images as debug_rect_left.jpg and debug_rect_right.jpg")
            
            # Display images
            cv2.imshow('Left Image', rect_pair.left)
            cv2.imshow('Right Image', rect_pair.right)
            cv2.imshow('Disparity Map', disp_vis)
            cv2.imshow('Depth Map', depth_vis)
            
            cv2.resizeWindow('Depth Map', screen_width, screen_height)
            cv2.moveWindow('Depth Map', 0, 0)
            
            # Show current filename for image source
            if not is_live:
                filename = os.path.basename(source.get_current_filename())
                print(f"\rProcessing: {filename} ({source.current_index + 1}/{len(source.image_paths)})", end='', flush=True)
            
            # Handle keyboard input
            wait_time = 1 if is_live else 0  # Wait indefinitely for image source
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
            elif key == ord('s'):  # Save depth map
                filename = f'depth_map_{frame_count:04d}.npz'
                np.savez_compressed(filename, depth=depth_map, disparity=disparity)
                print(f"\nSaved depth map to {filename}")
            elif not is_live:
                # Navigation controls for image source
                update_needed = False
                if key == 83:  # Right arrow
                    update_needed = source.next_image()
                elif key == 81:  # Left arrow
                    update_needed = source.previous_image()
                elif key == 80:  # Home key
                    source.current_index = 0
                    update_needed = True
                elif key == 87:  # End key
                    source.current_index = len(source.image_paths) - 1
                    update_needed = True
                
                if not update_needed and key != 255:  # No update needed and not a special key
                    continue
            
            frame_count += 1
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    if not is_live:
        print()  # New line after progress indicator
    print("\nApplication terminated")


if __name__ == '__main__':
    main()
