"""
This module implements a `ObjectDetectionDemo` class that extends the `BaseVideoDemo` framework to
perform real-time face detection in video streams.
"""
import cv2

from utils.base_module import BaseVideoDemo
from utils.object_detector import ObjectDetectorAndTracker

class ObjectDetectionDemo(BaseVideoDemo):
    """Demo for detecting objects in video frames."""

    def __init__(self):
        super().__init__()

        #self._strategies = {
        #    0: OCVDnnFaceDetection,
        #    1: HaarCascadeFaceDetection,
        #}
        #self._current_strategy = 0
        #self._face_detector = FaceDetector(OCVDnnFaceDetection())
        self._detector_tracker = ObjectDetectorAndTracker(
            model_type="yolo",
            yolo_weights="yolov5s.pt",  # Replace with your YOLOv5 weights
            confidence_threshold=0.5
        )

    def process_frame(self, frame):
        """Detect objects and draw rectangles around them."""
        frame.image = detector_tracker.process_frame(frame.image)
        return frame

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.

        - Press 'm' to cycle through the available stylization modes.
        """
        super(ObjectDetectionDemo, self).register_keys()
        return


    def get_window_name(self):
        """Return the name of the demo window."""
        return "Face Detection Demo"


if __name__ == "__main__":
    demo = ObjectDetectionDemo()
    demo.run()
