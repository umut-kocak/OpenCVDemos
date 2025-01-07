"""
This module provides a flexible video capture management system using strategy patterns.
It supports both single-threaded and multi-threaded approaches for frame capture, allowing for efficient
and adaptable video streaming. The `VideoStreamManager` class orchestrates the video capture process
with methods to start, stop, toggle strategies, and retrieve frames, while ensuring resource management.
"""
import threading
import time
from queue import Queue

import cv2

from utils.frame_data import FrameData
from utils.logger import logger


class VideoCaptureStrategy:
    """
    Abstract base class for video capture strategies.
    """

    def get_frame(self, supress_warnings=False):
        """
        Retrieve a video frame.

        Args:
            supress_warnings (bool): If True, suppress warnings on capture failure.

        Returns:
            FrameData: The captured frame with metadata, or None if capture failed.
        """
        raise NotImplementedError(
            "get_frame must be implemented by subclasses")

    def start(self):
        """
        Start the video capture strategy.
        """
        raise NotImplementedError("start must be implemented by subclasses")

    def stop(self):
        """
        Stop the video capture strategy.
        """
        raise NotImplementedError("stop must be implemented by subclasses")

    def get_queue_size(self):
        """
        Get the current size of the frame queue.

        Returns:
            int: The number of frames in the queue.
        """
        return 0


class SingleThreadedStrategy(VideoCaptureStrategy):
    """
    Strategy for single-threaded video capture.
    """

    def __init__(self, capture):
        """
        Initialize the single-threaded strategy.

        Args:
            capture (cv2.VideoCapture): The video capture object.
        """
        self._capture = capture

    def start(self):
        """
        Start the single-threaded capture strategy.
        """
        logger.debug("Single-threaded strategy started.")

    def get_frame(self, supress_warnings=False):
        """
        Retrieve a frame in single-threaded mode.

        Args:
            supress_warnings (bool): If True, suppress warnings on capture failure.

        Returns:
            FrameData: The captured frame with metadata, or None if capture failed.
        """
        ret, frame = self._capture.read()
        if not ret:
            if not supress_warnings:
                logger.warning("Frame capture failed in single-threaded mode.")
            return None
        return FrameData(frame, time.time())

    def stop(self):
        """
        Stop the single-threaded capture strategy.
        """
        logger.debug("Single-threaded strategy stopped.")


class MultiThreadedStrategy(VideoCaptureStrategy):
    """
    Strategy for multi-threaded video capture with queuing.
    """

    def __init__(self, capture):
        """
        Initialize the multi-threaded strategy.

        Args:
            capture (cv2.VideoCapture): The video capture object.
        """
        self._capture = capture
        self._max_queue_size = 30
        self._queue = Queue(maxsize=self._max_queue_size)
        self._started = False
        self._thread = None

    def start(self):
        """
        Start the multi-threaded capture strategy.
        """
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._capture_frames, daemon=True)
        self._thread.start()
        logger.debug("Multi-threaded strategy started.")

    def get_frame(self, supress_warnings=False):
        """
        Retrieve a frame from the queue in multi-threaded mode.

        Args:
            supress_warnings (bool): If True, suppress warnings when the queue is empty.

        Returns:
            FrameData: The captured frame with metadata, or None if no frame is available.
        """
        if not self._queue.empty():
            return self._queue.get()
        if not supress_warnings:
            logger.warning(
                "Frame queue is empty; no frame to provide in multi-threaded mode.")
        return None

    def stop(self):
        """
        Stop the multi-threaded capture strategy and clear the frame queue.
        """
        if not self._started:
            return
        self._started = False
        if self._thread:
            self._thread.join()
            self._thread = None
        # Clear the queue by reinitializing after the thread is stopped
        self._queue = Queue(maxsize=self._max_queue_size)
        logger.debug("Multi-threaded strategy stopped.")

    def get_queue_size(self):
        """
        Get the current size of the frame queue.

        Returns:
            int: The number of frames in the queue.
        """
        return self._queue.qsize()

    def _capture_frames(self):
        """
        Continuously capture frames and enqueue them for processing.
        """
        while self._started:
            ret, frame = self._capture.read()
            if not ret:
                logger.warning("Frame capture failed in multi-threaded mode.")
                break

            if not self._queue.full():
                self._queue.put(FrameData(frame, time.time()))
            else:
                logger.warning(
                    "Frame queue is full; dropping frame in multi-threaded mode.")


class VideoStreamManager:
    """
    Manages video stream capture with support for strategy patterns.
    """

    def __init__(self, source=0, width=None, height=None):
        """
        Initialize the video stream manager.

        Args:
            source (int or str): The video source (camera index or file path).
            width (int, optional): The desired frame width.
            height (int, optional): The desired frame height.
        """
        self._capture = cv2.VideoCapture(source)
        if width and height:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.debug("Video captured with dimensions: %dx%d", self.width, self.height)

        self._strategy = MultiThreadedStrategy(self._capture)
        self._started = False

    def set_strategy(self, strategy):
        """
        Set the video capture strategy.

        Args:
            strategy (VideoCaptureStrategy): The new capture strategy to use.
        """
        was_started = self._started
        if was_started:
            self.stop()
        self._strategy = strategy
        logger.debug("Video capture strategy set to %s.", self._strategy.__class__.__name__)
        if was_started:
            self.start()

    def toggle_strategy(self):
        """
        Toggle between single-threaded and multi-threaded strategies.
        """
        if isinstance(self._strategy, SingleThreadedStrategy):
            self.set_strategy(MultiThreadedStrategy(self._capture))
        else:
            self.set_strategy(SingleThreadedStrategy(self._capture))
        logger.debug("Strategy toggled.")

    def start(self, wait_for_the_first_frame=True, max_waiting_time=1):
        """
        Start the video capture using the current strategy.

        Args:
            wait_for_the_first_frame (bool): If True, wait for the first frame to be available.
            max_waiting_time (int): Maximum time to wait for the first frame, in seconds.
        """
        if self._started:
            logger.debug("VideoStreamManager is already started.")
            return
        self._started = True
        self._strategy.start()
        if wait_for_the_first_frame:
            waiting_start = time.time()
            while self._strategy.get_frame(True) is None:
                if time.time() - waiting_start > max_waiting_time:
                    logger.warning(
                        "VideoStreamManager can not get frames after %f seconds upon start.", max_waiting_time)
                    return
        logger.debug("VideoStreamManager started.")

    def get_frame(self):
        """
        Retrieve a frame using the current strategy.

        Returns:
            FrameData: The captured frame with metadata, or None if no frame is available.
        """
        if not self._started:
            logger.warning(
                "VideoStreamManager is not started. Cannot get frame.")
            return None
        return self._strategy.get_frame()

    def stop(self):
        """
        Stop the video capture and current strategy.
        """
        if not self._started:
            logger.debug("VideoStreamManager is already stopped.")
            return
        self._started = False
        self._strategy.stop()
        logger.debug("VideoStreamManager stopped.")

    def release(self):
        """
        Release the video capture resources.
        """
        self.stop()
        if self._capture.isOpened():
            self._capture.release()

    def get_queue_size(self):
        """
        Get the size of the frame queue from the current strategy.

        Returns:
            int: The number of frames in the queue.
        """
        return self._strategy.get_queue_size()

    def __del__(self):
        """
        Ensure video capture resources are released upon object deletion.
        """
        self.release()
