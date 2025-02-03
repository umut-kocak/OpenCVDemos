# Image Stitching Demo

This project demonstrates real-time image stitching using video frame extraction. It captures frames from a video stream,
processes them, and stitches them together using OpenCV's stitching algorithms.

## Features
- Extract frames from a video feed.
- Supports both OpenCV’s built-in and detailed stitching methods.
- Performs stitching in a separate thread for optimized performance.
- Saves and displays the stitched output.

## Usage
The demo initializes an image extractor to capture frames and a stitcher to combine them into a panoramic image.
The stitched result is saved and displayed in a separate window and stored in the output folder.
