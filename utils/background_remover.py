"""
This script provides a `BackgroundRemover` class that uses segmentation helper class
to segment the "person" category in an image and replace the background with a custom image 
or a default black background.
"""
import cv2
import numpy as np

from utils.segmentation import Segmentation, SegmentationClass

class BackgroundRemover:
    """
    A utility class for replacing the background of an image based on a segmentation mask.
    """

    def __init__(self, segmentation: Segmentation):
        """
        Initialize the BackgroundRemover.

        Args:
            segmentation (Segmentation): An instance of the Segmentation class to generate masks.
        """
        self._segmentation = segmentation

    def replace_background(self,image: np.ndarray,background: np.ndarray = None,
        target_class: SegmentationClass = SegmentationClass.PERSON,
    ) -> np.ndarray:
        """
        Replace the background of the given image using a segmentation mask.

        Args:
            image (np.ndarray): The input image.
            background (np.ndarray, optional): The custom background image. If None,
                a black background is used.
            target_class (SegmentationClass): The foreground class to isolate.

        Returns:
            np.ndarray: The image with the background replaced.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("The input 'image' must be a numpy array.")
        if background is not None and not isinstance(background, np.ndarray):
            raise ValueError("The 'background' must be a numpy array or None.")

        # Generate the mask using the segmentation class
        mask = self._segmentation.generate_mask(image)

        # Blend the frame with the custom or default background
        return self._blend_background(image, mask, background, target_class.value)

    @staticmethod
    def _blend_background(frame: np.ndarray, mask: np.ndarray,
        background: np.ndarray,
        target_class_id: int,
    ) -> np.ndarray:
        """
        Blend the segmented frame with a custom background.

        Args:
            frame (np.ndarray): The original image frame.
            mask (np.ndarray): The segmentation mask.
            background (np.ndarray, optional): The custom background image. If None,
                a black background is used.
            target_class_id (int): The class ID for the foreground to retain.

        Returns:
            np.ndarray: The blended image.
        """
        # Resize the mask to match the frame size
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_binary = (mask_resized == target_class_id).astype(np.uint8)

        # Use a black background if none is provided
        if background is None:
            background_resized = np.zeros_like(frame, dtype=np.uint8)
        else:
            background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

        # Blend the frame and the background
        blended_frame = np.where(mask_binary[:, :, None] == 1, frame, background_resized)
        return blended_frame
