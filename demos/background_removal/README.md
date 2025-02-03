# Real-Time Background Removal Demo

## Overview
The **Background Removal Demo** is a real-time video processing application that replaces the background in video streams.
Using advanced segmentation techniques, this demo identifies people in each frame and seamlessly replaces the existing background
with a custom image. The system leverages the **BackgroundRemover** and **Segmentation** utilities to achieve high-quality
background substitution.

## Features
- **Real-Time Processing:** Efficiently processes video streams to detect and replace backgrounds in real-time.
- **Person Segmentation:** Utilizes the `Segmentation` utility to identify people in the frame.
- **Custom Backgrounds:** Supports loading custom background images for replacement.
- **Logger Integration:** Tracks runtime events and errors through a structured logging system.
- **User Interface:** Displays processed video output in a dedicated visualization window.

## How It Works
1. **Frame Processing:** Each video frame is passed through the `Segmentation` model to detect individuals.
2. **Background Replacement:** The detected foreground is retained while the background is swapped with a user-defined image.
3. **Real-Time Display:** The transformed video is displayed in a window, allowing users to see the effect instantly.

## Use Cases
- **Virtual Backgrounds for Video Calls** – Enhance video meetings with custom backgrounds.
- **Live Streaming & Broadcasting** – Create engaging content with real-time background replacement.
- **Security & Surveillance** – Improve visibility and context in monitoring applications.
- **Augmented Reality Applications** – Implement interactive background effects in AR systems.
