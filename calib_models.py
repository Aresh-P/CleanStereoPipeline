from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np
import cv2

from utils import CalibData, RectMaps, Matrix4x4, CalibModel


@dataclass
class OpenCVCalibParams:
    """
    Parameters for OpenCV stereo calibration
    This class attempts to provide a type-safe way to specify the parameters needed to turn object point data into rectification maps
    This class only allows a subset of possible OpenCV calibrations
    In particular, it does NOT support:
    - Initial guesses for intrinsic matrices: makes the interactions between flags too complicated
      - Cameras are calibrated individually with no initial guess
    - Different distortion models for the left and right cameras: while possible in theory, this is kind of cursed
    - Using higher-degree polynomial terms in the numerator/denominator of the rational model WITHOUT lower-degree terms
      - In other words, K3 -> K2 -> K1, K6 -> K5 -> K4
    """
    
    class SolverMethod(Enum):
        """Solver method for camera calibration optimization"""
        SVD = 0 # Default: Singular Value Decomposition (most precise, slowest)
        LU = 1 # LU decomposition (faster, less precise)
        QR = 2 # QR decomposition (faster, less precise)
    
    class FocalLengthConstraint(Enum):
        """Constraint on focal length optimization during stereo calibration"""
        NONE = 0 # No constraints - all 4 values (fx_left, fy_left, fx_right, fy_right) vary independently
        FIX_ASPECT_RATIO = 1 # Each camera keeps its fy/fx ratio from mono calibration
        FIX_FOCAL_LENGTH = 2 # No refinement of focal lengths from mono calibration
        SAME_FOCAL_LENGTH = 3 # Enforce fx_left = fx_right and fy_left = fy_right
    
    class RadialDistortionNumerator(Enum):
        """Number of radial distortion terms in the numerator (k1, k2, k3)"""
        NONE = 0 # No radial distortion
        K1_ONLY = 1 # k1 * r^2
        K1_K2 = 2 # k1 * r^2 + k2 * r^4
        K1_K2_K3 = 3 # k1 * r^2 + k2 * r^4 + k3 * r^6
    
    class RadialDistortionDenominator(Enum):
        """Number of radial distortion terms in the denominator (k4, k5, k6) for rational model"""
        NONE = 0 # No denominator (standard polynomial model)
        K4_ONLY = 1 # 1 + k4 * r^2
        K4_K5 = 2 # 1 + k4 * r^2 + k5 * r^4
        K4_K5_K6 = 3 # 1 + k4 * r^2 + k5 * r^4 + k6 * r^6
    
    # Mono calibration flags
    zero_tangent_dist: bool = False # Set tangential distortion coefficients to zero
    thin_prism_model: bool = False # Use thin prism model with s1, s2, s3, s4
    tilted_model: bool = False # Use tilted sensor model with tauX, tauY
    fix_principal_point: bool = False # Fix principal point during optimization
    
    # Stereo calibration flags
    fix_intrinsic: bool = False # Don't refine intrinsics during stereo calibration
    same_principal_point: bool = False # Enforce cx_left = cx_right and cy_left = cy_right
    
    # Rectification flags
    zero_disparity: bool = True # Enforce parallel epipolar lines (used in stereoRectify)
    
    # Optional parameters that automatically set corresponding flags
    initial_stereo_rt: Optional[Tuple[np.ndarray, np.ndarray]] = None # Sets USE_EXTRINSIC_GUESS
    
    # Calibration constraints
    focal_length_constraint: FocalLengthConstraint = FocalLengthConstraint.NONE # Constraint on focal length optimization
    
    # Radial distortion model complexity
    radial_distortion_numerator: RadialDistortionNumerator = RadialDistortionNumerator.K1_K2_K3 # Number of numerator terms (k1, k2, k3)
    radial_distortion_denominator: RadialDistortionDenominator = RadialDistortionDenominator.NONE # Number of denominator terms (k4, k5, k6)
    
    # Solver selection
    solver_method: SolverMethod = SolverMethod.SVD # Optimization solver method
    
    # Rectification parameters
    alpha: float = 0 # Free scaling parameter for rectification (0-1)
    new_image_size: Optional[Tuple[int, int]] = None # Output size for rectified images
    crop_to_roi: bool = False # Whether to crop output to valid region
    
    
    def get_mono_flags(self) -> int:
        """Get flags for monocular calibration (calibrateCamera)"""
        flags = 0
        
        # Boolean flags
        if self.zero_tangent_dist:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        if self.thin_prism_model:
            flags |= cv2.CALIB_THIN_PRISM_MODEL
        if self.tilted_model:
            flags |= cv2.CALIB_TILTED_MODEL
        if self.fix_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            
        # Radial distortion model flags
        # Set flags to fix unused k coefficients based on the chosen model complexity
        if self.radial_distortion_numerator == self.RadialDistortionNumerator.NONE:
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        elif self.radial_distortion_numerator == self.RadialDistortionNumerator.K1_ONLY:
            flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        elif self.radial_distortion_numerator == self.RadialDistortionNumerator.K1_K2:
            flags |= cv2.CALIB_FIX_K3
        # For K1_K2_K3, don't fix any numerator coefficients
        
        # Handle denominator (rational model)
        if self.radial_distortion_denominator != self.RadialDistortionDenominator.NONE:
            flags |= cv2.CALIB_RATIONAL_MODEL
            if self.radial_distortion_denominator == self.RadialDistortionDenominator.K4_ONLY:
                flags |= cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
            elif self.radial_distortion_denominator == self.RadialDistortionDenominator.K4_K5:
                flags |= cv2.CALIB_FIX_K6
            # For K4_K5_K6, don't fix any denominator coefficients
            
        # Solver method flags
        if self.solver_method == self.SolverMethod.LU:
            flags |= cv2.CALIB_USE_LU
        elif self.solver_method == self.SolverMethod.QR:
            flags |= cv2.CALIB_USE_QR
        # SVD is the default (no flag needed)
            
        return flags
    
    def get_stereo_flags(self) -> int:
        """Get flags for stereo calibration (stereoCalibrate)"""
        # Start with base mono flags
        flags = self.get_mono_flags()
        
        # Only set USE_INTRINSIC_GUESS if we did individual calibrations
        # (not when using SAME_FOCAL_LENGTH with identity matrices)
        if self.focal_length_constraint != self.FocalLengthConstraint.SAME_FOCAL_LENGTH:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        
        # Focal length constraint flags
        if self.focal_length_constraint == self.FocalLengthConstraint.FIX_ASPECT_RATIO:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        elif self.focal_length_constraint == self.FocalLengthConstraint.FIX_FOCAL_LENGTH:
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        elif self.focal_length_constraint == self.FocalLengthConstraint.SAME_FOCAL_LENGTH:
            flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        
        # Add stereo-specific boolean flags
        if self.fix_intrinsic:
            flags |= cv2.CALIB_FIX_INTRINSIC
        # Note: CALIB_SAME_PRINCIPAL_POINT may not exist in all OpenCV versions
        # if self.same_principal_point and hasattr(cv2, 'CALIB_SAME_PRINCIPAL_POINT'):
        #     flags |= cv2.CALIB_SAME_PRINCIPAL_POINT
            
        # Flags set by optional parameters
        if self.initial_stereo_rt is not None:
            flags |= cv2.CALIB_USE_EXTRINSIC_GUESS
            
        return flags


class OpenCVCalibModel(CalibModel):
    def __init__(self, calib_params: OpenCVCalibParams):
        self.calib_params = calib_params
    
    def calibrate(self, data: CalibData) -> tuple[RectMaps, Matrix4x4]:
        """
        Uses cv2.calibrateCamera, cv2.stereoCalibrate, cv2.stereoRectify, and cv2.initUndistortRectifyMap
        """
        print("OPENCV CALIBRATION IN THIS PIPELINE IS PROBABLY BUGGED")
        # Get calibration flags
        mono_flags = self.calib_params.get_mono_flags()
        
        # Check if we should skip individual calibration
        if self.calib_params.focal_length_constraint == self.calib_params.FocalLengthConstraint.SAME_FOCAL_LENGTH:
            # Initialize camera matrices and distortion coefficients
            K1 = np.eye(3, dtype=np.float64)
            K2 = np.eye(3, dtype=np.float64)
            
            # Determine distortion coefficient size based on model
            if self.calib_params.radial_distortion_denominator != self.calib_params.RadialDistortionDenominator.NONE:
                # Rational model uses more coefficients
                dist_size = 8
            elif self.calib_params.tilted_model:
                # Tilted model uses 14 coefficients
                dist_size = 14
            elif self.calib_params.thin_prism_model:
                # Thin prism model uses 12 coefficients
                dist_size = 12
            else:
                # Standard model uses 5 coefficients
                dist_size = 5
                
            dist1 = np.zeros((dist_size, 1), dtype=np.float64)
            dist2 = np.zeros((dist_size, 1), dtype=np.float64)
        else:
            # Individual camera calibration (always starting from scratch, no initial guess)
            ret_l, K1, dist1, rvecs_l, tvecs_l = cv2.calibrateCamera(
                data.object_points, data.image_points_left, data.image_size, 
                None, None, flags=mono_flags
            )
            ret_r, K2, dist2, rvecs_r, tvecs_r = cv2.calibrateCamera(
                data.object_points, data.image_points_right, data.image_size, 
                None, None, flags=mono_flags
            )
        
        # Get stereo calibration flags
        stereo_flags = self.calib_params.get_stereo_flags()
        
        # Prepare initial stereo transformation if provided
        if self.calib_params.initial_stereo_rt is not None:
            R_init, T_init = self.calib_params.initial_stereo_rt
        else:
            R_init, T_init = None, None
        
        # Stereo calibration
        ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
            data.object_points, data.image_points_left, data.image_points_right,
            K1, dist1, K2, dist2, data.image_size,
            R=R_init, T=T_init,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            flags=stereo_flags
        )
        
        # Print calibration results
        print(f"\nStereo calibration RMS reprojection error: {ret:.4f} pixels")
        print(f"Left camera matrix:\n{K1}")
        print(f"Right camera matrix:\n{K2}")
        print(f"Left distortion coefficients: {dist1.ravel()}")
        print(f"Right distortion coefficients: {dist2.ravel()}")
        print(f"Rotation between cameras:\n{R}")
        print(f"Translation between cameras: {T.ravel()}")
        baseline_mm = np.linalg.norm(T) * 1000
        print(f"Baseline: {baseline_mm:.2f} mm")
        
        # Compute rectification transforms
        rect_flags = cv2.CALIB_ZERO_DISPARITY if self.calib_params.zero_disparity else 0
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, dist1, K2, dist2, data.image_size, R, T,
            newImageSize=self.calib_params.new_image_size or data.image_size,
            alpha=self.calib_params.alpha,
            flags=rect_flags
        )
        
        # Handle optional cropping to ROI
        output_size = self.calib_params.new_image_size or data.image_size
        
        if self.calib_params.crop_to_roi:
            # Find intersection of both ROIs
            x = max(roi1[0], roi2[0])
            y = max(roi1[1], roi2[1])
            w = min(roi1[0] + roi1[2], roi2[0] + roi2[2]) - x
            h = min(roi1[1] + roi1[3], roi2[1] + roi2[3]) - y
            
            # Adjust projection matrices for cropped coordinates
            T_crop = np.array([
                [1, 0, -x],
                [0, 1, -y],
                [0, 0,  1]
            ], dtype=np.float64)
            
            P1_cropped = T_crop @ P1
            P2_cropped = T_crop @ P2
            
            # Adjust Q matrix for cropped coordinates
            translation_matrix = np.array([
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            Q = Q @ translation_matrix
            
            # Create maps for the cropped size
            map1_left, map2_left = cv2.initUndistortRectifyMap(
                K1, dist1, R1, P1_cropped, (w, h), cv2.CV_32FC1
            )
            map1_right, map2_right = cv2.initUndistortRectifyMap(
                K2, dist2, R2, P2_cropped, (w, h), cv2.CV_32FC1
            )
        else:
            # Create maps for the full output size
            map1_left, map2_left = cv2.initUndistortRectifyMap(
                K1, dist1, R1, P1, output_size, cv2.CV_32FC1
            )
            map1_right, map2_right = cv2.initUndistortRectifyMap(
                K2, dist2, R2, P2, output_size, cv2.CV_32FC1
            )
        
        return RectMaps(map1_left, map2_left, map1_right, map2_right), Q
