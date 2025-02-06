# OpenCVDemos

This repository contains a collection of Python-based OpenCV demos.


## Brief Overview of the Demos

### Background Removal
A real-time video processing application that replaces the background in video streams.
Using advanced segmentation techniques, this demo identifies people in each frame and seamlessly replaces the existing background
with a custom image.

### Camera Calibration
A real-time video processing application designed to calibrate cameras using chessboard patterns.
It extracts calibration frames, detects chessboard corners, and computes the camera matrix and distortion coefficients to improve
image accuracy.

### Cartoonize
Application of various stylization effects to video frames in real-time, leveraging OpenCV's
advanced image processing capabilities. Users can switch between different stylization modes interactively
during the demo.

### Face Detection
A real-time identification of faces in video streams and overlaying rectangles around the detected faces.

### Hand Detection
A real-time identification of hands and finger joints in video streams and its visualisation as well as gesture estimation.

### Image Stitching
Demonstrates real-time image stitching using video frame extraction. It captures frames from a video stream,
processes them, and stitches them together for panaroma or environment map creation.

### Video Recorder
A basic lightweight video recorder from a live video stream.


## Running the Demos

To run these demos successfully, follow these steps:

- **Create the Conda Environment**: Use the `requirements_conda` file to create a corresponding Conda environment.

  ```bash
  conda env create -f ./requirements_conda.yaml --prefix ./env
  conda activate ./env
  ```

- **Run the Demo in Module Mode**: After setting up the environment, run the demo from the root directory in module mode. For example:


  ```bash
  python -m demos.face_detection.face_detection
  ```
Ensure you are in the root directory before executing the Python command to run each demo.
