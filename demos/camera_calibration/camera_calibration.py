"""

"""
from pathlib import Path
import os
import time
import numpy as np

import cv2

from utils.base_video_demo import BaseVideoDemo
from utils.helper import load_images_from_folder
from utils.image_extractor import ImageExtractor
from utils.logger import logger
from utils.thread_pool import get_thread_pool as get_thread_pool

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

def detect_corners(image, chessboard_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    return ret, corners

def calibrate_camera(objpoints, imgpoints, image_size, settings):
    if len(objpoints) == 0:
        logger.error("No 3d points for camera calibration.")
        return

    # Perform camera calibration
    ret, settings.matrix, settings.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, settings.matrix, settings.dist)

    # Compute reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], settings.matrix, settings.dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    if settings.print_results:
        print("\n--- Camera Calibration Results ---")
        print(f"Camera Matrix:\n{settings.matrix}")
        print(f"Distortion Coefficients:\n{settings.dist.ravel()}")
        print(f"Mean Reprojection Error: {mean_error:.4f}")

    success = mean_error < settings.projection_error_threshold
    if success:
        logger.info("Calibration successful within the error threshold!")
    else:
        logger.warning("Reprojection error (%.4f) exceeds threshold (%.4f)!", mean_error, settings.projection_error_threshold)

    return success

class CameraCalibrationDemo(BaseVideoDemo):
    """."""

    def __init__(self):
        super().__init__()

        if self.settings.demo.extracted_input_folder != "":
            # Load images in another thread
            input_folder = self.get_asset_path(self.settings.demo.extracted_input_folder)
            images_future = get_thread_pool().submit_task(load_images_from_folder, input_folder)
            logger.debug("Waiting for images...")
            try:
                images = images_future.result(timeout=10)
                self.calibrate_from_images( images )
            except Exception as e:
                logger.error("Exception during loading images task: %s", e)
            
        self._image_extractor = ImageExtractor(self)
        self.initialize_calibration_parameters()

    def initialize_calibration_parameters(self):
        """."""
        chessboard_size = self.settings.demo.calibration.chessboard_size
        self.points_3d = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.points_3d[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.points_3d *= self.settings.demo.calibration.square_size
        
        if self.settings.demo.calibration.matrix is not None:
            self.settings.demo.calibration.matrix = np.array(self.settings.demo.calibration.matrix)
        if self.settings.demo.calibration.dist is not None:
            self.settings.demo.calibration.dist = np.array(self.settings.demo.calibration.dist)

    def process_frame(self, frame):
        """."""
        self._image_extractor.process_frame(frame)
        self._visual_debugger.add_debugging_frame("Last Extracted", self._image_extractor.get_last_extracted_image())
        chessboard_size = self.settings.demo.calibration.chessboard_size
        ret, corners = detect_corners(frame.image, chessboard_size)
        if ret:
            cv2.drawChessboardCorners(frame.image, chessboard_size, corners, ret)
        return frame

    def calibrate_from_images(self, images):
        """."""
        def calibration_task():
            logger.debug("Starting calibrationing...")
            obj_points = []  # 3D points in real-world space
            img_points = []  # 2D points in image plane
            for i, image in enumerate(images):
                chessboard_size = self.settings.demo.calibration.chessboard_size
                ret, corners = detect_corners(image, chessboard_size)
                if ret:
                    obj_points.append(self.points_3d)
                    img_points.append(corners)
                else:
                    logger.warning("Could not detect corners in image %i, ignoring.", i)
            image_size = (image.shape[1], image.shape[0])
            scs = calibrate_camera(obj_points, img_points, image_size, self.settings.demo.calibration)
            logger.debug("Finished calibration %s success.", "with" if scs else "without")
            return scs

        if not images:
            logger.error("No images for calibration.")
            return

        # Calibrate in another thread
        calibration_future = get_thread_pool().submit_task(calibration_task)

        logger.debug("Waiting for calibration...")
        try:
            calibration_result = calibration_future.result(timeout=10)
        except Exception as e:
            logger.error("Exception during calibration task: %s", e)

    def run_calibration(self):
        images = self._image_extractor.get_extracted_images()
        self.calibrate_from_images(images)

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.
        """
        super(CameraCalibrationDemo, self).register_keys()

        key_bindings = [
            # General keys
            (ord('e'), "Extract frame(screenshot) ", lambda : self._image_extractor.interactive_extract(), None),
            (ord('r'), "Toggle recording extraction ", lambda : self._image_extractor.toggle_recording(), None),
            (ord('p'), "Finalize calibration ", lambda : self.run_calibration(), None),
            (ord('x'), "Delete last extraction ", lambda : self._image_extractor.delete_last_extracted_image(), None)
        ]
    
        # Register all key bindings
        for key, description, callback, callback_arg, *args in key_bindings:
            self._key_manager.register_key(key, description, callback, callback_arg, *args, name_space=self.get_window_name())

    def check_captured_frame(self, frame):
        """Hook for operations to run when there are not enough captured video frames."""
        frame_status = super(CameraCalibrationDemo, self).check_captured_frame(frame)
        if frame_status == BaseVideoDemo.FrameStatus.ABORT:
            self.run_calibration()
        return frame_status

    def get_window_name(self):
        """Return the name of the demo window."""
        return "Camera Calibration Demo"


if __name__ == "__main__":
    demo = CameraCalibrationDemo()
    demo.run()
