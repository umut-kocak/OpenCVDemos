# Face Detection Demo

This project provides a real-time **Face Detection Demo** built upon the `BaseVideoDemo` framework.
The application uses the `FaceDetector` utility to identify faces in video streams and overlay rectangles around
the detected faces. This demo includes a user-friendly interface for visualizing processed video frames and logs
runtime events for debugging and analysis.

## How It Works

1. **Face Detection**: 
   - Each video frame is passed to the `FaceDetector` utility, which detects faces.
   - Bounding boxes are drawn around detected faces.
   
2. **Real-Time Visualization**: 
   - The processed frames are displayed in a window titled "Face Detection Demo".

3. **Logging**: 
   - A logger is initialized at runtime to record events to both the console and a log file.
