from utils import StereoPairSource, UnrectStereoPair
import cv2
import numpy as np
import glob
from typing import List, Optional

class OpenCVLiveSource(StereoPairSource):
    def __init__(self, device_id):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {device_id}")
        
        # Try to set highest resolution (4416x1242)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)
        
        # Print actual resolution achieved
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")
        
        if actual_width != 4416 or actual_height != 1242:
            print(f"Warning: Could not set camera to 4416x1242, using {actual_width}x{actual_height}")
        
        # Capture and discard frames for autoexposure
        print("Warming up camera (capturing frames for autoexposure)...")
        for i in range(20):
            ret, frame = self.cap.read()
            if ret:
                print(f"Captured warmup frame {i+1}/20", end='\r')
        print("\nCamera warmup complete")
    
    def get_pair(self) -> Optional[UnrectStereoPair]:
        ret, frame = self.cap.read()
        if not ret:
            return None
                
        # Split side-by-side stereo image
        height, width = frame.shape[:2]
        half_width = width // 2
        
        left_image = frame[:, :half_width]
        right_image = frame[:, half_width:]
                
        return UnrectStereoPair(left=left_image, right=right_image)
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


class StereoImageSource(StereoPairSource):
    def __init__(self, image_pattern: str):
        self.image_paths: List[str] = sorted(glob.glob(image_pattern))
        if not self.image_paths:
            raise RuntimeError(f"No images found matching pattern: {image_pattern}")
        self.current_index = 0
        print(f"Found {len(self.image_paths)} images")
    
    def get_pair(self) -> Optional[UnrectStereoPair]:
        if self.current_index >= len(self.image_paths):
            return None
        
        image_path = self.image_paths[self.current_index]
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        
        # Split side-by-side stereo image
        height, width = frame.shape[:2]
        half_width = width // 2
        
        left_image = frame[:, :half_width]
        right_image = frame[:, half_width:]

        self.current_index += 1
        return UnrectStereoPair(left=left_image, right=right_image)
    
    def next_image(self):
        """Move to the next image"""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            return True
        return False
    
    def previous_image(self):
        """Move to the previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
    
    def get_current_filename(self) -> str:
        """Get the filename of the current image"""
        if self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return ""
    
