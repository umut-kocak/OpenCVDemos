# Cartoonize Demo

This project provides a real-time **Cartoonize Demo** built upon the `BaseVideoDemo` framework.  
The application applies various stylization effects to video frames in real-time, leveraging OpenCV's
advanced image processing capabilities. Users can switch between different stylization modes interactively
during the demo. This demo includes a user-friendly interface for visualizing processed video frames and
logs runtime events for debugging and analysis.

## How It Works

1. **Stylization Effects**:
   - The application processes video frames and applies one of the following effects based on the selected mode:
     - **Edge-Preserving Filter**: Smoothens the image while preserving edges.
     - **Detail Enhancement**: Enhances fine details in the frame.
     - **Pencil Sketch (Grayscale)**: Converts the frame into a grayscale pencil sketch.
     - **Pencil Sketch (Color)**: Creates a colorful pencil sketch version of the frame.
     - **Stylization**: Applies artistic stylization for a painting-like effect.

2. **Real-Time Visualization**:
   - Processed frames are displayed in a window titled "Cartoonize Demo".

3. **Keyboard Interaction**:
   - Users can press the `m` key to cycle through the available stylization modes.

4. **Logging**:
   - A logger is initialized at runtime to record events to both the console and a log file for debugging and analysis.
