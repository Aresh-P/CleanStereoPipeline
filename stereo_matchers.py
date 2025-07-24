from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import cv2

from utils import StereoMatcher, RectStereoPair, DispMap

# TODO: verify completeness and default values of OpenCV parameters
@dataclass
class OpenCVSGBMParams:
    """
    Parameters for OpenCV's SGBM matching and (optionally) WLS filtering
    """
    min_disparity: int = 0
    num_disparities: int = 16 * 6  # Must be divisible by 16
    block_size: int = 15  # Odd number, typically 3-21
    P1: int = 8 * 3 * 15 * 15  # Penalty for disparity change of 1
    P2: int = 32 * 3 * 15 * 15  # Penalty for disparity change > 1
    disp12_max_diff: int = 1  # Maximum allowed difference in left-right check
    pre_filter_cap: int = 63
    uniqueness_ratio: int = 10  # Margin in percentage by which best cost should win
    # WLS filter parameters
    use_wls: bool = True  # Whether to apply WLS filtering
    wls_lambda: float = 8000.0  # Regularization parameter for smoothness
    wls_sigma: float = 1.5  # Sensitivity to edges based on disparity gradient

class OpenCVSGBMMatcher(StereoMatcher):
    def __init__(self, match_params: OpenCVSGBMParams):
        """
        Initialize OpenCV's SGBM and WLS filter as members
        """
        self.use_wls = match_params.use_wls
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=match_params.min_disparity,
            numDisparities=match_params.num_disparities,
            blockSize=match_params.block_size,
            P1=match_params.P1,
            P2=match_params.P2,
            disp12MaxDiff=match_params.disp12_max_diff,
            preFilterCap=match_params.pre_filter_cap,
            uniquenessRatio=match_params.uniqueness_ratio,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Always create right matcher and WLS filter
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
        self.wls_filter.setLambda(match_params.wls_lambda)
        self.wls_filter.setSigmaColor(match_params.wls_sigma)
    
    def update(self, new_match_params: OpenCVSGBMParams):
        """
        Set new parameters for SGBM and WLS filters if possible,
        or re-initialize them if not (e.g. SGBM mode changed)
        """
        self.use_wls = new_match_params.use_wls
        self.stereo.setMinDisparity(new_match_params.min_disparity)
        self.stereo.setNumDisparities(new_match_params.num_disparities)
        self.stereo.setBlockSize(new_match_params.block_size)
        self.stereo.setP1(new_match_params.P1)
        self.stereo.setP2(new_match_params.P2)
        self.stereo.setDisp12MaxDiff(new_match_params.disp12_max_diff)
        self.stereo.setPreFilterCap(new_match_params.pre_filter_cap)
        self.stereo.setUniquenessRatio(new_match_params.uniqueness_ratio)
        
        # Always update WLS filter parameters
        self.wls_filter.setLambda(new_match_params.wls_lambda)
        self.wls_filter.setSigmaColor(new_match_params.wls_sigma)
    
    def match(self, rect_pair: RectStereoPair) -> DispMap:
        """
        Wrapper around StereoSGBM's compute() and DisparityWLSFilter's filter()
        """
        if self.use_wls:
            # Compute disparity for both left-right and right-left
            left_disp = self.stereo.compute(rect_pair.left, rect_pair.right)
            right_disp = self.right_matcher.compute(rect_pair.right, rect_pair.left)
            
            # Apply WLS filter
            filtered_disp = self.wls_filter.filter(left_disp, rect_pair.left, None, right_disp)
            return filtered_disp.astype(np.float32) / 16.0
        else:
            # SGBM can work with color images directly
            disparity = self.stereo.compute(rect_pair.left, rect_pair.right).astype(np.float32) / 16.0
            return disparity


class DownsampleStereoMatcher(StereoMatcher):
    def __init__(self, inner_matcher: StereoMatcher, downsampling_factor: int):
        """
        Initialize with an inner StereoMatcher and a downsampling factor.
        
        Args:
            inner_matcher: The StereoMatcher to use on downsampled images
            downsampling_factor: Factor by which to downsample the images (must be >= 1)
        """
        if downsampling_factor < 1:
            raise ValueError("Downsampling factor must be >= 1")
        self.inner_matcher = inner_matcher
        self.downsampling_factor = downsampling_factor
    
    def match(self, rect_pair: RectStereoPair) -> DispMap:
        """
        Downsample input images, run inner matcher, then upsample result.
        
        Args:
            rect_pair: The rectified stereo pair to process
            
        Returns:
            Disparity map at original resolution with values scaled appropriately
        """
        if self.downsampling_factor == 1:
            # No downsampling needed
            return self.inner_matcher.match(rect_pair)
        
        # Downsample the images
        downsampled_left = cv2.resize(
            rect_pair.left, 
            None, 
            fx=1.0/self.downsampling_factor, 
            fy=1.0/self.downsampling_factor,
            interpolation=cv2.INTER_AREA
        )
        downsampled_right = cv2.resize(
            rect_pair.right,
            None,
            fx=1.0/self.downsampling_factor,
            fy=1.0/self.downsampling_factor,
            interpolation=cv2.INTER_AREA
        )
        
        # Create downsampled stereo pair
        downsampled_pair = RectStereoPair(downsampled_left, downsampled_right)
        
        # Run inner matcher on downsampled images
        downsampled_disp = self.inner_matcher.match(downsampled_pair)
        
        # Upsample the disparity map back to original resolution
        original_height, original_width = rect_pair.left.shape[:2]
        upsampled_disp = cv2.resize(
            downsampled_disp,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Scale disparity values by the downsampling factor
        # (disparity in original image = disparity in downsampled image * downsampling_factor)
        scaled_disp = upsampled_disp * self.downsampling_factor
        
        return scaled_disp.astype(np.float32)
