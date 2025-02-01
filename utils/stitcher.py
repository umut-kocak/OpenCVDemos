"""
"""
import cv2
import numpy as np
from abc import ABC, abstractmethod

from utils.logger import logger

class Stitcher(ABC):
    """
    """

    @abstractmethod
    def stitch_images(self, images, mode=cv2.Stitcher_PANORAMA):
        """
        """
        pass


class StitcherOpenCVBuiltIn(Stitcher):
    """
    """
    def __init__(self):
        """
        """
        self._stitcher = cv2.Stitcher_create()

    def stitch_images(self, images, mode=cv2.Stitcher_PANORAMA):
        self._stitcher.setPanoConfidenceThresh(0.5)
        status, stitched = self._stitcher.stitch(images)
    
        if status != cv2.Stitcher_OK:
            logger.error(f"Stitching went wrong: {status}")
            return None
        return stitched

class StitcherOpenCVDetailed(Stitcher):
    """
    """
    def stitch_images(self, images, mode=cv2.Stitcher_PANORAMA):
        pass



