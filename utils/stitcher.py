"""
Module for stitching multiple images into a panorama.

This module defines an abstract base class `Stitcher` and its concrete implementations
using OpenCV's built-in stitching functionalities.
"""

from abc import ABC, abstractmethod

import cv2

from utils.logger import logger

class Stitcher(ABC):
    """
    Abstract base class for image stitching.

    Subclasses must implement the `stitch_images` method to provide specific stitching
    functionalities.
    """

    @abstractmethod
    def stitch_images(self, images, mode=cv2.Stitcher_PANORAMA):
        """
        Stitch multiple images into a single panorama.

        Args:
            images (list): List of images to be stitched.
            mode (int, optional): Stitching mode. Defaults to `cv2.Stitcher_PANORAMA`.

        Returns:
            The stitched image if successful, otherwise None.
        """


class StitcherOpenCVBuiltIn(Stitcher):
    """
    Image stitcher using OpenCV's built-in stitching functionality.
    """

    def __init__(self):
        """
        Initialize the OpenCV built-in stitcher.
        """
        self._stitcher = cv2.Stitcher_create()

    def stitch_images(self, images, mode=cv2.Stitcher_PANORAMA):
        """
        Perform image stitching using OpenCV's built-in stitcher.

        Args:
            images (list): List of images to be stitched.
            mode (int, optional): Stitching mode. Defaults to `cv2.Stitcher_PANORAMA`.

        Returns:
            The stitched image if successful, otherwise None.
        """
        self._stitcher.setPanoConfidenceThresh(0.5)
        status, stitched = self._stitcher.stitch(images)

        if status != cv2.Stitcher_OK:
            logger.error("Stitching went wrong: {%d}", status)
            return None
        return stitched


class StitcherOpenCVDetailed(Stitcher):
    """
    Image stitcher using a more detailed OpenCV approach (to be implemented).
    """

    def stitch_images(self, images, mode=cv2.Stitcher_PANORAMA):
        """
        Perform image stitching using a detailed approach.

        Args:
            images (list): List of images to be stitched.
            mode (int, optional): Stitching mode. Defaults to `cv2.Stitcher_PANORAMA`.

        Returns:
            The stitched image if successful, otherwise None.
        """
