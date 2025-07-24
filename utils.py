from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import cv2

Matrix4x4 = NDArray[np.float64]

@dataclass
class CalibData:
    """
    Arrays of calibration pattern points in 3D space,
    their corresponding image points in the left and right cameras,
    and the image size
    Meant to be passed to the calibrate method of CalibModel,
    which in the derived OpenCVCalibModel class passes its members to cv2.stereoCalibrate
    """
    object_points: List[np.ndarray]  # 3D points for each calibration image
    image_points_left: List[np.ndarray]  # 2D points in left camera
    image_points_right: List[np.ndarray]  # 2D points in right camera
    image_size: Tuple[int, int]  # (width, height)

@dataclass
class RectMaps:
    """
    Rectification maps for the left and right stereo images
    Returned by the calibrate method of CalibModel,
    which in the derived OpenCVCalibModel class uses cv2.initUndistortRectifyMap
    """
    map1_left: np.ndarray
    map2_left: np.ndarray
    map1_right: np.ndarray
    map2_right: np.ndarray

class CalibModel(ABC):
    @abstractmethod
    def calibrate(self, data: CalibData) -> tuple[RectMaps, Matrix4x4]:
        """
        Returns rectification maps (as expected by cv2.remap)
        and Q matrix (as expected by cv2.reprojectImageTo3D)
        """
        pass

@dataclass
class UnrectStereoPair:
    """
    Unrectified stereo image pair
    """
    left: np.ndarray
    right: np.ndarray

class StereoPairSource(ABC):
    @abstractmethod
    def get_pair(self) -> Optional[UnrectStereoPair]:
        pass

@dataclass
class RectStereoPair:
    """
    Rectified stereo image pair
    Same data as UnrectStereoPair, different type to avoid mistakes
    """
    left: np.ndarray
    right: np.ndarray

def rectify(unrect_pair: UnrectStereoPair, maps: RectMaps) -> RectStereoPair:
    """
    Wrapper around cv2.remap
    """
    rectified_left = cv2.remap(unrect_pair.left, maps.map1_left, maps.map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(unrect_pair.right, maps.map1_right, maps.map2_right, cv2.INTER_LINEAR)
    return RectStereoPair(rectified_left, rectified_right)

# Type alias for disparity map
DispMap = NDArray[np.float32]  # Disparity map for the left camera in pixels (rectified coordinates)

class StereoMatcher(ABC):
    @abstractmethod
    def match(self, rect_pair: RectStereoPair) -> DispMap:
        pass


# Type alias for depth map
DepthMap = NDArray[np.float32]  # Depth map for the left camera in the distance units used for the initial object points

def disp_to_depth(disp: DispMap, Q: Matrix4x4) -> DepthMap:
    '''
    Wrapper around cv2.reprojectImageTo3D
    '''
    # Reproject to 3D
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    
    # Extract just the Z (depth) channel
    depth_map = points_3d[:, :, 2].astype(np.float32)
    
    # Clean up invalid depths
    depth_map[disp <= 0] = 0  # Invalid disparity
    depth_map[np.isinf(depth_map)] = 0  # Infinite depth
    depth_map[np.isnan(depth_map)] = 0  # NaN depth
    
    return depth_map

def save_calibration(filename: str, rect_maps: RectMaps, Q: Matrix4x4) -> None:
    """Save rectification maps and Q matrix to a numpy file."""
    np.savez_compressed(filename,
                      map1_left=rect_maps.map1_left,
                      map2_left=rect_maps.map2_left,
                      map1_right=rect_maps.map1_right,
                      map2_right=rect_maps.map2_right,
                      Q=Q)

def load_calibration(filename: str) -> Tuple[RectMaps, Matrix4x4]:
    """Load rectification maps and Q matrix from a numpy file."""
    data = np.load(filename)
    rect_maps = RectMaps(
        map1_left=data['map1_left'],
        map2_left=data['map2_left'],
        map1_right=data['map1_right'],
        map2_right=data['map2_right']
    )
    Q = data['Q']
    return rect_maps, Q
