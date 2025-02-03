"""
Module for extracting images from video streams.

This module defines the `ImageExtractor` class, which provides functionality to
extract and store images from a video stream based on different extraction modes:
interactive extraction or recording mode.
"""

from dataclasses import dataclass, field
from enum import Enum
import weakref
from pathlib import Path
import time

import cv2

from utils.logger import logger
from utils.thread_pool import get_thread_pool
from utils.helper import load_images_from_folder

class ExtractionMode(Enum):
    """Enumeration for different image extraction modes."""
    INTERACTIVE = "INTERACTIVE"
    RECORD = "RECORD"

@dataclass
class ImageExtractor:
    """
    A class responsible for extracting images from a video stream.

    Supports both interactive extraction (manual image capture) and
    recording-based extraction (automated image capture at intervals).

    Attributes:
        output_folder (Path): Path where extracted images are stored.
        extracted_images (list): List of images stored in memory.
        output_suffix_index (int): Index for naming extracted images.
        record_last_time (float): Timestamp of the last recorded extraction.
        mode (ExtractionMode): Current extraction mode (INTERACTIVE or RECORD).
        enabled (bool): Flag to enable or disable extraction.
        cached_image (object): Cached image for interactive extraction.
    """

    output_folder: Path = field(init=False)
    extracted_images: list = field(default_factory=list)
    output_suffix_index: int = 0  # TODO: Make it thread-safe
    record_last_time: float = 0
    mode: ExtractionMode = ExtractionMode.INTERACTIVE
    enabled: bool = False
    cached_image: object = None

    def __init__(self, owner):
        """
        Initialize the ImageExtractor with the given owner.

        Args:
            owner: The parent object that owns this extractor.
        """
        self.owner = weakref.ref(owner)
        self.output_folder = Path(owner.get_output_folder()) / "extracted"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def toggle_recording(self):
        """Toggle recording mode on or off."""
        self.mode = ExtractionMode.RECORD
        self.enabled = not self.enabled

    def interactive_extract(self):
        """Perform interactive image extraction."""
        self.mode = ExtractionMode.INTERACTIVE
        self.enabled = True
        if self.cached_image is not None:
            self._extract_image(self.cached_image)

    def get_extracted_images(self):
        """
        Retrieve extracted images based on settings.

        Returns:
            list or None: A list of extracted images if available, otherwise None.
        """
        self.enabled = False
        if self.owner().settings.demo.extraction.keep_in_memory:
            return self.extracted_images
        if self.owner().settings.demo.extraction.save:
            # Load images in another thread
            images_future = get_thread_pool().submit_task(load_images_from_folder, self.output_folder)
            try:
                images = images_future.result(timeout=10)
                return images
            except Exception as e:
                logger.error("Exception during loading images task: %s", e)
                return None
        return None

    def get_last_extracted_image(self):
        """
        Retrieve the last extracted image.

        Returns:
            The most recent extracted image if available, otherwise None.
        """
        if self.owner().settings.demo.extraction.keep_in_memory:
            if len(self.extracted_images) > 0:
                return self.extracted_images[-1]
        elif self.owner().settings.demo.extraction.save:
            if self.output_suffix_index > 0:
                return cv2.imread(self._get_full_file_path(self.output_suffix_index - 1))
        return None

    def delete_last_extracted_image(self):
        """Delete the last extracted image from memory or disk."""
        if self.owner().settings.demo.extraction.keep_in_memory:
            if len(self.extracted_images) > 0:
                logger.debug("Deleting the last extracted image.")
                del self.extracted_images[-1]
        elif self.owner().settings.demo.extraction.save:
            if self.output_suffix_index > 0:
                file_to_rem = self._get_full_file_path(self.output_suffix_index - 1)
                logger.debug("Deleting the last extracted image: %s", file_to_rem)
                file_to_rem.unlink()
                self.output_suffix_index -= 1

    def process_frame(self, frame):
        """
        Process a video frame and extract an image if necessary.

        Args:
            frame: The current video frame.

        Returns:
            The processed video frame.
        """
        self.cached_image = None
        if self.enabled:
            if self.mode == ExtractionMode.RECORD:
                if self._should_extract():
                    self._extract_image(frame.image.copy())
            elif self.mode == ExtractionMode.INTERACTIVE:
                self.cached_image = frame.image.copy()
        return frame

    def _should_extract(self) -> bool:
        """
        Check if it's time to extract an image in RECORD mode.

        Returns:
            bool: True if an image should be extracted, False otherwise.
        """
        ct = time.time()
        if ct - self.record_last_time > self.owner().settings.demo.record_period:
            self.record_last_time = ct
            return True
        return False

    def _extract_image(self, image):
        """
        Extract an image and save it to memory or disk.

        Args:
            image: The image to be extracted.
        """
        def extract_task():
            if self.owner().settings.demo.extraction.save:
                output_path = self._get_full_file_path(self.output_suffix_index)
                logger.debug("Saving to %s ", output_path)
                cv2.imwrite(str(output_path), image)
                self.output_suffix_index += 1
            if self.owner().settings.demo.extraction.keep_in_memory:
                self.extracted_images.append(image)

        get_thread_pool().submit_task(extract_task)

    def _get_full_file_path(self, index):
        """
        Generate the full file path for saving an extracted image.

        Args:
            index (int): The index of the image.

        Returns:
            Path: The full file path.
        """
        return self.output_folder / f'img-{index}.jpg'
