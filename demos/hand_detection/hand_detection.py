"""
Hand Detection Demo

This module provides a demonstration of real-time hand detection using multiple
hand detection strategies. It extends the BaseVideoDemo class to process video
frames and detect hands, display gestures, and allow switching between
detection strategies using keyboard input.

"""

from utils.base_video_demo import BaseVideoDemo
from utils.class_factory import ClassFactory
from utils.hand_detector import HandDetector, MediaPipeHandDetection
from utils.logger import logger
from utils.text_manager import TextProperties

class HandDetectionDemo(BaseVideoDemo):
    """Demo for detecting hands in video frames."""

    def __init__(self):
        """Initialize the hand detection demo with different detection strategies."""
        super().__init__()
        strategies = {
            0: (MediaPipeHandDetection, {})
        }
        self._strategy_factory = ClassFactory(strategies)
        self._current_strategy = 0
        self._number_of_strategies = len(strategies)
        self._hand_detector = HandDetector(self._strategy_factory.create_class(self._current_strategy))

    def process_frame(self, frame):
        """Process a video frame to detect hands and display detected gestures."""

        # Detect and draw hands
        self._hand_detector.strategy.detect_hands(frame.image, draw_hands=True)

        # Detect gestures
        gestures = self._hand_detector.strategy.recognize_hand_gestures()
        gestures = [key + " : " + value for key, value in gestures.items()]

        # Show gestures
        height, width, _ = frame.image.shape
        x, y = self.settings.demo.gestures_text_position
        x, y = (int(x * width), int(y * height))
        self._text_manager.draw_text(frame.image, gestures, pos=(x, y),
          properties=TextProperties(color=TextProperties.RED))

        return frame

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.
        """
        super().register_keys()

        def adjust_detection_strategy(delta):
            """Adjust the detection strategy based on user input."""
            self._current_strategy = (self._current_strategy + delta) % self._number_of_strategies
            logger.info("Changing detection to strategy %s ", self._strategy_factory.get_class(self._current_strategy))
            self._hand_detector.set_strategy(self._strategy_factory.create_class(self._current_strategy))

        key_bindings = [
            # General keys
            (ord('m'), "Change the detection strategy", lambda delta: adjust_detection_strategy(delta), 1)
        ]

        # Register all key bindings
        for key, description, callback, callback_arg, *args in key_bindings:
            self._key_manager.register_key(key, description, callback, callback_arg, *args,
                name_space=self.get_window_name())

    def get_window_name(self):
        """Return the name of the demo window."""
        return "Hand Detection Demo"

if __name__ == "__main__":
    demo = HandDetectionDemo()
    demo.run()
