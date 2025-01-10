"""
This module implements the StylingDemo class, a video processing demo that applies various 
stylization effects to video frames in real-time. The demo supports multiple modes, allowing 
the user to cycle through different effects such as edge-preserving filtering, detail 
enhancement, pencil sketch, and stylization.

Usage:
Press 'm' during the demo to change the stylization mode.
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

from utils.base_module import BaseVideoDemo  # pylint: disable=C0413
from utils.face_detector import FaceDetector  # pylint: disable=C0413


class StylingDemo(BaseVideoDemo):
    """
    Demo for applying different stylization filters to video frames.

    Modes:
    - 0: Edge-preserving filter
    - 1: Detail enhancement
    - 2: Pencil sketch (grayscale)
    - 3: Pencil sketch (color)
    - 4: Stylization
    """

    def __init__(self):
        """
        Initialize the StylingDemo with default mode and total number of modes.
        """
        super().__init__()
        self._mode = 0  # Current stylization mode
        self._nr_of_modes = 5  # Total number of stylization modes

    def process_frame(self, frame):
        """
        Process the given frame and apply the selected stylization effect.

        Args:
            frame: A video frame containing an `image` attribute to be stylized.

        Returns:
            The modified frame with the stylization effect applied.
        """
        mode = self._mode % self._nr_of_modes
        match mode:
            case 0:
                frame.image = cv2.edgePreservingFilter(frame.image, flags=1, sigma_s=60, sigma_r=0.4)
            case 1:
                frame.image = cv2.detailEnhance(frame.image, sigma_s=10, sigma_r=0.15)
            case 2:
                frame.image, _ = cv2.pencilSketch(frame.image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            case 3:
                _, frame.image = cv2.pencilSketch(frame.image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            case 4:
                frame.image = cv2.stylization(frame.image, sigma_s=60, sigma_r=0.07)
            case _:  # Fallback case
                frame.image = cv2.edgePreservingFilter(frame.image, flags=1, sigma_s=60, sigma_r=0.4)

        return frame

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.

        - Press 'm' to cycle through the available stylization modes.
        """
        super(StylingDemo, self).register_keys()

        self._key_manager.register_key(
            ord('m'), 
            "Change the styling mode",
            lambda module: setattr(module, '_mode', getattr(module, '_mode') + 1),
            self
        )

    def get_window_name(self):
        """
        Return the name of the demo window.

        Returns:
            str: The name of the window.
        """
        return "Styling Demo"


if __name__ == "__main__":
    demo = StylingDemo()
    demo.run()
