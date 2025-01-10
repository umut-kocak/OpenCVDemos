"""
This module implements a `FaceDetectionDemo` class that extends the `BaseVideoDemo` framework to
perform real-time face detection in video streams. It uses the `FaceDetector` utility to identify
faces in each frame and overlays rectangles around detected faces. The demo initializes a logger
for tracking runtime events and provides a user interface window for visualizing the processed
video output.
"""
import logging
import os
import time

import cv2

from utils.logger_initializer import initialize_logger

# Initialize the global logger before importing other modules
logger_name = os.path.splitext(os.path.basename(__file__))[0]
logger_file_name = logger_name + time.strftime("%Y%m%d-%H%M%S") + ".log"

initialize_logger(
    logger_name,
    logger_file_name,
    _log_to_console=True,
    _log_to_file=logger_file_name,
    _console_level=logging.DEBUG,
    _file_level=logging.DEBUG
)


from utils.base_module import BaseVideoDemo # pylint: disable=C0413
from utils.face_detector import FaceDetector # pylint: disable=C0413


class FaceDetectionDemo(BaseVideoDemo):
    """Demo for detecting faces in video frames."""

    def __init__(self):
        super().__init__()
        self._face_detector = FaceDetector()

    def process_frame(self, frame):
        """Detect faces and draw rectangles around them."""

        faces = self._face_detector.detect_faces(frame.image)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def get_window_name(self):
        """Return the name of the demo window."""
        return "Face Detection Demo"


if __name__ == "__main__":
    demo = FaceDetectionDemo()
    demo.run()
