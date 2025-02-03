# Camera Calibration Demo

## Overview
The **Camera Calibration Demo** is a real-time video processing application designed to calibrate cameras using chessboard patterns.
It extracts calibration frames, detects chessboard corners, and computes the camera matrix and distortion coefficients to improve
image accuracy.

## Features
- **Real-Time Camera Calibration:** Automatically detects chessboard patterns and computes calibration parameters.
- **Image Extraction:** Allows extracting frames for calibration via keyboard controls.
- **Threaded Processing:** Utilizes multi-threading to load images and run calibration asynchronously.
- **Interactive Debugging:** Displays extracted frames and detected corners for visual verification.
- **Configurable Parameters:** Supports user-defined chessboard sizes, square sizes, and calibration thresholds.

## How It Works
1. **Frame Extraction:** The system captures frames from a video stream or loads them from a folder.
2. **Corner Detection:** Chessboard corners are identified in each frame.
3. **Camera Calibration:** The system computes intrinsic parameters, including the camera matrix and distortion coefficients.
4. **Reprojection Error Analysis:** Evaluates calibration accuracy using reprojection error.

## Usage Scenarios
- **Robotics & Computer Vision:** Enhances the accuracy of depth perception and object recognition.
- **Augmented Reality (AR):** Improves tracking accuracy for AR applications.
- **3D Reconstruction:** Essential for stereo vision and 3D scanning applications.
- **Photogrammetry & Surveying:** Enhances precision in camera-based measurement systems.

