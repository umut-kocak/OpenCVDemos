"""
This module implements a `BackgroundRemovalDemo` class that extends the `BaseVideoDemo` framework to
perform real-time background reoval in video streams. It uses the `BackgroundRemover` and 'Segmentation'
utilities to identify faces in each frame and replaces the background with a provided background image.
The demo initializes a logger for tracking runtime events and provides a user interface window for visualizing
the processed video output.
"""
import logging
import os
import time

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

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
from utils.background_remover import BackgroundRemover # pylint: disable=C0413
from utils.background_remover import Segmentation, SegmentationClass # pylint: disable=C0413

class BackgroundRemovalDemo(BaseVideoDemo):
    """Demo for replacing the background in video frames."""

    def __init__(self):
        super().__init__()
        segmentation = Segmentation()
        self._remover = BackgroundRemover(segmentation)
        
        # Load a custom background image
        _background_path = self.get_demo_folder() / "assets" / self.settings.demo.background_file
        self._background = cv2.imread(_background_path)
        if self._background is None:
            logging.error("No background image found: %s", _background_path)

    def process_frame(self, frame):
        """Replaces the background of the frame."""
        frame.image = self._remover.replace_background(frame.image, self._background, SegmentationClass.PERSON)
        return frame

    def get_window_name(self):
        """Return the name of the demo window."""
        return "Real-Time Background Removal"

if __name__ == "__main__":
    demo = BackgroundRemovalDemo()
    demo.run()
