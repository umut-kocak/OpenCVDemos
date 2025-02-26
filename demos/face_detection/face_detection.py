"""
This module implements a `FaceDetectionDemo` class that extends the `BaseVideoDemo` framework to
perform real-time face detection in video streams. It uses the `FaceDetector` utility to identify
faces in each frame and overlays rectangles around detected faces. The demo initializes a logger
for tracking runtime events and provides a user interface window for visualizing the processed
video output.
"""
import cv2

from utils.base_video_demo import BaseVideoDemo
from utils.class_factory import ClassFactory
from utils.face_detector import FaceDetector, HaarCascadeFaceDetection, OCVDnnFaceDetection
from utils.logger import logger

class FaceDetectionDemo(BaseVideoDemo):
    """Demo for detecting faces in video frames."""

    def __init__(self):
        super().__init__()
        strategies = {
            0: ( OCVDnnFaceDetection,
                {
                "model_path": self.get_asset_path("models/res10_300x300_ssd_iter_140000.caffemodel"),
                "config_path": self.get_asset_path("models/deploy.proto.txt") }
            ),
            1: ( HaarCascadeFaceDetection,
                {
                "model_path": self.get_asset_path("./models/haarcascade_frontalface_default.xml")
                }
            )
        }
        self._strategy_factory = ClassFactory( strategies )
        self._current_strategy = 0
        self._number_of_strategies = len(strategies)
        self._face_detector = FaceDetector(self._strategy_factory.create_class(self._current_strategy))

    def process_frame(self, frame):
        """Detect faces and draw rectangles around them."""

        faces = self._face_detector.detect_faces(frame.image)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.
        """
        super().register_keys()

        def adjust_detection_strategy(delta):
            self._current_strategy = (self._current_strategy + delta) % self._number_of_strategies
            logger.info("Changing detection to strategy %s ", self._strategy_factory.get_class(self._current_strategy))
            self._face_detector.set_strategy(self._strategy_factory.create_class(self._current_strategy))

        key_bindings = [
            # General keys
            (ord('m'), "Change the detection strategy ", lambda delta: adjust_detection_strategy(delta), 1)
        ]

        # Register all key bindings
        for key, description, callback, callback_arg, *args in key_bindings:
            self._key_manager.register_key(key, description, callback, callback_arg, *args,
                name_space=self.get_window_name())

    def get_window_name(self):
        """Return the name of the demo window."""
        return "Face Detection Demo"

if __name__ == "__main__":
    demo = FaceDetectionDemo()
    demo.run()
