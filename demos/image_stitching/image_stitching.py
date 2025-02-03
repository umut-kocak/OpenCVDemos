"""
Image Stitching Demo

This module implements an `ImageStitchingDemo` class that extends the `BaseVideoDemo` framework.
It enables real-time image extraction and stitching using OpenCV-based algorithms. Users can extract
video frames and stitch them into a panoramic image using different stitching methods.

Key Features:
- Extract frames from video streams.
- Supports OpenCV's built-in and detailed stitching methods.
- Runs stitching in a separate thread for efficient processing.
- Saves and displays the stitched output.
"""
import cv2

from utils.base_video_demo import BaseVideoDemo
from utils.helper import load_images_from_folder
from utils.image_extractor import ImageExtractor
from utils.logger import logger
import utils.stitcher
from utils.thread_pool import get_thread_pool


class ImageStitchingDemo(BaseVideoDemo):
    """
    Demo for performing real-time image stitching using video frame extraction.
    This class allows capturing frames, processing them, and stitching them together.
    """

    def __init__(self):
        """Initialize the image stitching demo, setting up the stitcher and image extractor."""
        super().__init__()

        self._stitcher = None
        match self.settings.demo.stitcher.type:
            case "OPENCVBUILTIN":
                self._stitcher = utils.stitcher.StitcherOpenCVBuiltIn()
            case "OPENCVDETAILED":
                self._stitcher = utils.stitcher.StitcherOpenCVDetailed()
            case _:  # Fallback case
                logger.error("Invalid stitcher type %s ", self.settings.demo.stitcher.type)

        if self.settings.demo.extracted_input_folder:
            # Load images in another thread
            input_folder = self.get_asset_path(self.settings.demo.extracted_input_folder)
            images_future = get_thread_pool().submit_task(load_images_from_folder, input_folder)
            logger.debug("Waiting for images...")
            try:
                images = images_future.result(timeout=10)
                self.stitch_images(images)
            except Exception as e:
                logger.error("Exception during loading images task: %s", e)

        self._image_extractor = ImageExtractor(self)

    def process_frame(self, frame):
        """Processes the incoming video frame and updates the visual debugger."""
        self._image_extractor.process_frame(frame)
        self._visual_debugger.add_debugging_frame("Last Extracted", self._image_extractor.get_last_extracted_image())
        return frame

    def stitch_images(self, images):
        """Stitches extracted images into a panoramic image."""
        def stitch_task():
            logger.debug("Starting stitching...")
            stitched = self._stitcher.stitch_images(images, mode=cv2.Stitcher_PANORAMA)
            logger.debug("Finished stitching...")
            return stitched

        if not images:
            logger.error("No images to stitch.")
            return

        # Stitch in another thread
        stitched_future = get_thread_pool().submit_task(stitch_task)

        stitched_image = None
        logger.debug("Waiting for stitching...")
        try:
            stitched_image = stitched_future.result(timeout=10)
        except Exception as e:
            logger.error("Exception during stitching task: %s", e)

        if stitched_image is not None:
            output_path = self.get_output_folder() / "stitched_output.jpg"
            cv2.imwrite(output_path, stitched_image)
            logger.info("Stitched output saved to %s", output_path)
            cv2.imshow('Stitched Result', stitched_image)

    def run_stitching(self):
        """Triggers the stitching process from extracted images."""
        images = self._image_extractor.get_extracted_images()
        self.stitch_images(images)

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.
        """
        super().register_keys()

        key_bindings = [
            (ord('e'), "Extract frame (screenshot)", lambda: self._image_extractor.interactive_extract(), None),
            (ord('r'), "Toggle recording extraction", lambda: self._image_extractor.toggle_recording(), None),
            (ord('p'), "Finalize stitch", lambda: self.run_stitching(), None)
        ]

        for key, description, callback, callback_arg, *args in key_bindings:
            self._key_manager.register_key(key, description, callback, callback_arg, *args,
                name_space=self.get_window_name())

    def check_captured_frame(self, frame):
        """Handles operations when there are insufficient captured frames."""
        frame_status = super().check_captured_frame(frame)
        if frame_status == BaseVideoDemo.FrameStatus.ABORT:
            self.run_stitching()
        return frame_status

    def get_window_name(self):
        """Returns the name of the demo window."""
        return "Image Stitching Demo"


if __name__ == "__main__":
    demo = ImageStitchingDemo()
    demo.run()
