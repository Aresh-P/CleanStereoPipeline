import numpy as np
from typing import List, Tuple

from utils import (
    CalibData,
    UnrectStereoPair, rectify,
    disp_to_depth
)
from stereo_matchers import OpenCVSGBMParams, OpenCVSGBMMatcher
from calib_models import OpenCVCalibParams, OpenCVCalibModel

def stereo_to_depth(
        calib_data: CalibData, # Calibration data containing object and image points
        unrect_pair: UnrectStereoPair, # Unrectified stereo image pair
        calib_params: OpenCVCalibParams, # Calibration parameters for OpenCV model
        match_params: OpenCVSGBMParams # Stereo matching parameters for OpenCV SGBM
) -> np.ndarray:
    """
    Complete stereo pipeline from calibration points to depth map.
    Shows the data flow using the utils module components.
    """
    
    # Set up calibration model
    calib_model = OpenCVCalibModel(calib_params)
    
    # Perform calibration to get rectification maps and Q matrix
    rect_maps, Q = calib_model.calibrate(calib_data)

    # Apply rectification
    rect_pair = rectify(unrect_pair, rect_maps)
    
    # Set up stereo matcher
    matcher = OpenCVSGBMMatcher(match_params)
    
    # Compute disparity map
    disparity = matcher.get_disparity(rect_pair)
    
    # Convert disparity to depth
    depth_map = disp_to_depth(disparity, Q)
    
    return depth_map
