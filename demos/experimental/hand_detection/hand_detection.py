"""
"""
import cv2

from utils.base_module import BaseVideoDemo
from utils.hand_detector import HandDetector, MediaPipeHandDetection, ContourBasedHandDetection
from utils.logger import logger


class HandDetectionDemo(BaseVideoDemo):
    """Demo for detecting faces in video frames."""

    def __init__(self):
        super().__init__()

        self._strategies = {
            0: MediaPipeHandDetection,
            1: ContourBasedHandDetection,
        }
        self._current_strategy = 0
        self._detector = HandDetector(self._create_strategy(self._current_strategy))

    def process_frame(self, frame):
        """Detect faces and draw rectangles around them."""

        frame.image = self._detector.detect_hands(frame.image)
        return frame

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.

        - Press 'm' to cycle through the available stylization modes.
        """
        super(HandDetectionDemo, self).register_keys()

        def adjust_detection_strategy(demo, delta):
            demo._current_strategy = (demo._current_strategy + delta) % len(demo._strategies)
            logger.info("Changing detection to strategy %s ", demo._strategies.get(demo._current_strategy))
            demo._detector.set_strategy(demo._create_strategy(demo._current_strategy))

        key_bindings = [
            # General keys
            (ord('m'), "Change the detection strategy ", lambda m: adjust_detection_strategy(m, +1), self)
        ]

        # Register all key bindings
        for key, description, callback, callback_arg, *args in key_bindings:
            self._key_manager.register_key(key, description, callback, callback_arg, *args, name_space=self.get_window_name())

    def get_window_name(self):
        """Return the name of the demo window."""
        return "Hand Detection Demo"

    def _create_strategy(self, strategy_key, **kwargs):
        """
        Retrieve and instantiate a face detection strategy based on the given key.
    
        Args:
            strategy_key (int): The key corresponding to the desired strategy.
            kwargs: Additional arguments required by specific strategies.
    
        Returns:
            HandDetectionStrategy: An instantiated strategy object.
    
        Raises:
            ValueError: If the strategy_key is not valid.
        """
        strategy_class = self._strategies.get(strategy_key)
        if strategy_class is None:
            raise ValueError(f"Invalid strategy key: {strategy_key}. Available keys: {list(self._strategies.keys())}")

        return strategy_class(**kwargs)


if __name__ == "__main__":
    demo = HandDetectionDemo()
    demo.run()
