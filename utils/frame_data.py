"""
This module defines the `FrameData` class, which encapsulates data for a video frame,
including the frame image and its capture timestamp.

Usage:
    Create an instance of `FrameData` with an image and a corresponding capture time.
"""

from dataclasses import dataclass

@dataclass
class FrameData:
    """
    A simple class that encapsulates data for a video frame.
    """
    def __init__(self, _image, _time):
        self.image = _image
        self.capture_time = _time
