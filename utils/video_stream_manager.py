"""
This module provides a flexible video capture management system using strategy patterns.
It supports both single-threaded and multi-threaded approaches for frame capture, allowing for efficient
and adaptable video streaming. The `VideoStreamManager` class orchestrates the video capture process
with methods to start, stop, toggle strategies, and retrieve frames, while ensuring resource management.
"""
from enum import Enum
import threading
import time
from queue import Queue
import weakref

import cv2

from utils.frame_data import FrameData
from utils.logger import logger

class FrameFilteringMethod(Enum):
    NONE = 1
    SKIP_FRAME = 2
    ADAPT_QUEUE_SIZE = 3

    def next(self):
        members = list(type(self))
        next_index = (members.index(self) + 1) % len(members)
        return members[next_index]

    @classmethod
    def get_from_string(cls, name):
        try:
            # Access the enum member by name
            return cls.__members__[name.upper()]
        except KeyError:
            logger.warning("Invalid FrameFilteringMethod ignored: %s", name)
            return cls.__members__["NONE"]

class VideoCaptureStrategy:
    """
    Abstract base class for video capture strategies.
    """

    def __init__(self, owner):
        """
        Initialize the strategy.

        Args:
            owner (VideoStreamManager): The owner.
        """
        self._owner = weakref.ref(owner)
        self.frame_filtering = FrameFilteringMethod.NONE
        self.set_frame_filtering_method( FrameFilteringMethod.get_from_string(self._owner()._settings().video.capture.frame_filtering.method) )

        self._index_skipped_frames = 0

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

    def set_frame_filtering_method(self, method):
        """
        Sets the method for frame filtering.

        Args:
            method (FrameFilteringMethod): Method to set.

        Returns:
            None
        """

    def get_frame_filtering_skip_size(self):
        """
        Get the current skip frame for the frame filtering.

        Returns:
            int: The skipping frames.
        """
        if self.frame_filtering.value == 2:
            return self._owner()._settings().video.capture.frame_filtering.skip_frame.period
        return 0

    def _should_skip_frame(self):
        """
        Checks if the frame needs to be skipped.

        Returns:
            bool: True if the frame needs to be skipped.
        """
        should_skip = False
        if self.frame_filtering == FrameFilteringMethod.SKIP_FRAME:
            in_period_begin = False
            stngs = self._owner()._settings().video.capture.frame_filtering.skip_frame
            self._index_skipped_frames += 1
            if self._index_skipped_frames % stngs.period == 0:
                self._index_skipped_frames = 0
                in_period_begin = True
            if stngs.action == "SKIP":
                should_skip = in_period_begin
            elif stngs.action == "PICKUP":
                should_skip = not in_period_begin
        return should_skip

class SingleThreadedStrategy(VideoCaptureStrategy):
    """
    Strategy for single-threaded video capture.
    """

    def start(self):
        """
        Start the single-threaded capture strategy.
        """
        logger.debug("Single-threaded strategy started.")

    def get_frame(self, supress_warnings=True):
        """
        Retrieve a frame in single-threaded mode.

        Args:
            supress_warnings (bool): If True, suppress warnings on capture failure.

        Returns:
            FrameData: The captured frame with metadata, or None if capture failed.
        """
        ret, frame = self._owner()._capture.read()
        if not ret:
            if not supress_warnings:
                logger.warning("Frame capture failed in single-threaded mode.")
            return None
        
        if self._should_skip_frame():
            number_of_frames_to_skip = 1
            # Create a separate thread to skip frames
            thread = threading.Thread(target=self._skip_frames, args=(number_of_frames_to_skip,))
            thread.start()
            thread.join()  # Wait for the thread to complete before proceeding

        return FrameData(frame, time.time())

    def _skip_frames(self, n):
        for _ in range(n):
            self._owner()._capture.read()  # Ignore the output

    def stop(self):
        """
        Stop the single-threaded capture strategy.
        """
        logger.debug("Single-threaded strategy stopped.")

    def set_frame_filtering_method(self, method):
        """
        Sets the method for frame filtering.

        Args:
            method (FrameFilteringMethod): Method to set.

        Returns:
            None
        """
        new_method = method
        match method:
            case FrameFilteringMethod.NONE:
                new_method = FrameFilteringMethod.NONE
            case FrameFilteringMethod.SKIP_FRAME:
                if self._owner()._is_camera:
                    logger.warning("Single-threaded video capture ignores the filtering method %s when the input is from camera.", method.name)
                    new_method = FrameFilteringMethod.NONE
            case FrameFilteringMethod.ADAPT_QUEUE_SIZE:
                logger.warning("Single-threaded video capture ignores the filtering method %s.", method.name)
                new_method = FrameFilteringMethod.NONE
            case _:
                new_method = FrameFilteringMethod.NONE
        logger.debug("Single-threaded video capture filtering method changed to %s.", new_method.name)
        self.frame_filtering = new_method


class MultiThreadedStrategy(VideoCaptureStrategy):
    """
    Strategy for multi-threaded video capture with queuing.
    """

    def __init__(self, owner):
        """
        Initialize the multi-threaded strategy.

        Args:
            owner (VideoStreamManager): The owner.
        """
        super(MultiThreadedStrategy, self).__init__(owner)
        self._initial_max_queue_size = 30
        self._current_max_queue_size = self._initial_max_queue_size
        self._started = False
        self._thread = None

    def start(self):
        """
        Start the multi-threaded capture strategy.
        """
        if self._started:
            return
        self._queue = Queue(maxsize=self._initial_max_queue_size)
        self._started = True
        self._thread = threading.Thread(
            target=self._capture_frames, daemon=True)
        self._thread.start()
        logger.debug("Multi-threaded strategy started.")

    def get_frame(self, supress_warnings=True):
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
            logger.warning("Frame queue is empty; no frame to provide in multi-threaded mode.")
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
        self._queue = None
        logger.debug("Multi-threaded strategy stopped.")

    def get_queue_size(self):
        """
        Get the current size of the frame queue.

        Returns:
            int: The number of frames in the queue.
        """
        return self._queue.qsize()

    def set_frame_filtering_method(self, method):
        """
        Sets the method for frame filtering. It is not thread-safe.

        Args:
            method (FrameFilteringMethod): Method to set.

        Returns:
            None
        """
        new_method = method
        match method:
            case FrameFilteringMethod.NONE:
                logger.warning("Multi-threaded-threaded video capture with filtering method 'NONE' can cause dropped frames.")
            case FrameFilteringMethod.SKIP_FRAME:
                pass
            case FrameFilteringMethod.ADAPT_QUEUE_SIZE:
                if self._owner()._is_camera:
                    logger.warning("Multi-threaded-threaded video capture with filtering method 'ADAPT_QUEUE_SIZE' with camera can cause high latency.")
                else:
                    logger.warning("Multi-threaded-threaded video capture with filtering method 'ADAPT_QUEUE_SIZE' with video might consume high memory use, consider switching to single-threaded video processing.")
            case _:
                new_method = FrameFilteringMethod.NONE
        logger.debug("Multi-threaded video capture filtering method changed to %s.", new_method.name)
        self.frame_filtering = new_method

    def _capture_frames(self):
        """
        Continuously capture frames and enqueue them for processing.
        """
        self._index_skipped_frames = 0
        while self._started:
            ret, frame = self._owner()._capture.read()
            if not ret:
                logger.warning("Frame capture failed in multi-threaded mode.")
                break

            if self._should_skip_frame():
                continue
            if self._should_adapt_queue_size():
                self._adapt_queue_size()
            
            if not self._queue.full():
                self._queue.put(FrameData(frame, time.time()))
            else:
                logger.warning("Frame queue is full; dropping frame in multi-threaded mode.")

    def _should_adapt_queue_size(self):
        match self.frame_filtering:
            case FrameFilteringMethod.NONE:
                return False
            case FrameFilteringMethod.SKIP_FRAME:
                return False
            case FrameFilteringMethod.ADAPT_QUEUE_SIZE:
                if self._queue.full():
                    return True
            case _:
                return False
        return False

    def _adapt_queue_size(self):
        self._current_max_queue_size *= 2
        logger.warning("Doubling the size of the queue to new size %d.", self._current_max_queue_size)
        self._resize_queue(self._current_max_queue_size)

    def _resize_queue(self, max_size):
        new_queue = Queue(maxsize=max_size)
        while not self._queue.empty():
            new_queue.put(self._queue.get())
        self._queue = new_queue  # Replace the old queue with the new one


class VideoStreamManager:
    """
    Manages video stream capture with support for strategy patterns.
    """

    def __init__(self, settings, source_path="./"):
        """
        Initialize the video stream manager.

        Args:
            settings: The settings file.
        """
        self._settings = weakref.ref(settings)
        capture_source = 0 if settings.video.capture.source == 0 else source_path / settings.video.capture.source
        self._capture = cv2.VideoCapture(capture_source)
        self._is_camera = (capture_source == 0)
        if settings.video.capture.width and settings.video.capture.height:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.video.capture.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.video.capture.height)

        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._capture.get(cv2.CAP_PROP_FPS))        
        logger.debug("Video captured with dimensions: %dx%d and %d fps", self.width, self.height, self.fps)

        self._started = False
        self.capture_strategy = None
        self.set_strategy(SingleThreadedStrategy(self) if settings.video.capture.threading == "SINGLE" else MultiThreadedStrategy(self))

    def set_strategy(self, strategy):
        """
        Set the video capture strategy.

        Args:
            strategy (VideoCaptureStrategy): The new capture strategy to use.
        """
        was_started = self._started
        if was_started:
            self.stop()
        self.capture_strategy = strategy
        logger.debug("Video capture strategy set to %s.", self.capture_strategy.__class__.__name__)
        if was_started:
            self.start()

    def toggle_strategy(self):
        """
        Toggle between single-threaded and multi-threaded strategies.
        """
        if isinstance(self.capture_strategy, SingleThreadedStrategy):
            self.set_strategy(MultiThreadedStrategy(self))
        else:
            self.set_strategy(SingleThreadedStrategy(self))
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
        self.capture_strategy.start()
        if wait_for_the_first_frame:
            waiting_start = time.time()
            while self.capture_strategy.get_frame(True) is None:
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
        return self.capture_strategy.get_frame()

    def stop(self):
        """
        Stop the video capture and current strategy.
        """
        if not self._started:
            logger.debug("VideoStreamManager is already stopped.")
            return
        self._started = False
        self.capture_strategy.stop()
        logger.debug("VideoStreamManager stopped.")

    def release(self):
        """
        Release the video capture resources.
        """
        self.stop()
        self.strategy = None
        if self._capture.isOpened():
            self._capture.release()

    def get_queue_size(self):
        """
        Get the size of the frame queue from the current strategy.

        Returns:
            int: The number of frames in the queue.
        """
        return self.capture_strategy.get_queue_size()

    def __del__(self):
        """
        Ensure video capture resources are released upon object deletion.
        """
        self.release()
