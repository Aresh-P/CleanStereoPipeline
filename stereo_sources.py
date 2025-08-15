from utils import StereoPairSource, UnrectStereoPair
import cv2
import numpy as np
import glob
from typing import List, Optional
import threading
import time
class FrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def update(self, frame):
        with self.lock:
            self.frame = frame

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

class OpenCVLiveSource(StereoPairSource):
    def __init__(self, device_id):

        # Backends:
        #   python -c "import cv2; print([(b, cv2.videoio_registry.getBackendName(b)) for b in cv2.videoio_registry.getBackends()])"
        #   [(1900, 'FFMPEG'), (1800, 'GSTREAMER'), (2300, 'INTEL_MFX'), (1400, 'MSMF'), (1400, 'MSMF'), 
        #    (700, 'DSHOW'), (2000, 'CV_IMAGES'), (2200, 'CV_MJPEG'), (2500, 'UEYE'), (2600, 'OBSENSOR')]
        #   1900 (FFMPEG) doesn't open
        #   1800 (GSTREAMER) doesn't open
        #   2300 (INTEL_MFX) doesn't open
        #   1400 (MSMF) (default) eventually freezes
        #   700 (DSHOW) eventually goes black
        #   2000 (CV_IMAGES) can't open by index
        #   2200 (CV_MJPEG) can't open by index
        #   2500 (UEYE) doesn't open
        #   2600 (OBSENSOR) doesn't open

        self.cap = cv2.VideoCapture(device_id, 700)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")

        if actual_width != 4416 or actual_height != 1242:
            print(f"Warning: Could not set camera to 4416x1242, using {actual_width}x{actual_height}")

        print("Warming up camera (capturing frames for autoexposure)...")
        for i in range(20):
            ret, frame = self.cap.read()
            if ret:
                print(f"Captured warmup frame {i+1}/20", end='\r')
        print("\nCamera warmup complete")

        self.frame_buffer = FrameBuffer()
        self.running = True
        self.thread = threading.Thread(target=self._capture_thread, daemon=True)
        self.thread.start()

    def _capture_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.update(frame)
            time.sleep(0.01)  # Slight delay to avoid busy-waiting

    def get_pair(self) -> Optional[UnrectStereoPair]:
        frame = self.frame_buffer.get()
        if frame is None:
            return None

        height, width = frame.shape[:2]
        half_width = width // 2

        left_image = frame[:, :half_width]
        right_image = frame[:, half_width:]

        return UnrectStereoPair(left=left_image, right=right_image)

    def __del__(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)
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
    
