import cv2
import time
import threading
from queue import Queue
from utils.frame_data import FrameData
from utils.logger import logger

class VideoCaptureStrategy:
    """
    Abstract base class for video capture strategies.
    """
    def get_frame(self, supress_warnings=False):
        raise NotImplementedError("get_frame must be implemented by subclasses")

    def start(self):
        raise NotImplementedError("start must be implemented by subclasses")

    def stop(self):
        raise NotImplementedError("stop must be implemented by subclasses")

    def getQueueSize(self):
        return 0


class SingleThreadedStrategy(VideoCaptureStrategy):
    """
    Strategy for single-threaded video capture.
    """
    def __init__(self, capture):
        self._capture = capture

    def start(self):
        logger.debug("Single-threaded strategy started.")

    def get_frame(self, supress_warnings=False):
        ret, frame = self._capture.read()
        if not ret:
            if not supress_warnings:
                logger.warning("Frame capture failed in single-threaded mode.")
            return None
        return FrameData(frame, time.time())

    def stop(self):
        logger.debug("Single-threaded strategy stopped.")


class MultiThreadedStrategy(VideoCaptureStrategy):
    """
    Strategy for multi-threaded video capture with queuing.
    """
    def __init__(self, capture):
        self._capture = capture
        self._max_queue_size = 30
        self._queue = Queue(maxsize=self._max_queue_size)
        self._started = False
        self._thread = None

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._thread.start()
        logger.debug("Multi-threaded strategy started.")

    def get_frame(self, supress_warnings=False):
        if not self._queue.empty():
            return self._queue.get()
        if not supress_warnings:
            logger.warning("Frame queue is empty; no frame to provide in multi-threaded mode.")
        return None

    def stop(self):
        if not self._started:
            return
        self._started = False
        if self._thread:
            self._thread.join()
            self._thread = None
        # Clear the queue by reinitializing after the thread is stopped
        self._queue = Queue(maxsize=self._max_queue_size)
        logger.debug("Multi-threaded strategy stopped.")

    def getQueueSize(self):
        return self._queue.qsize()

    def _capture_frames(self):
        while self._started:
            ret, frame = self._capture.read()
            if not ret:
                logger.warning("Frame capture failed in multi-threaded mode.")
                break

            if not self._queue.full():
                self._queue.put(FrameData(frame, time.time()))
            else:
                logger.warning("Frame queue is full; dropping frame in multi-threaded mode.")


class VideoStreamManager:
    """
    Manages video stream capture with support for strategy patterns.
    """
    def __init__(self, source=0, width=None, height=None):
        self._capture = cv2.VideoCapture(source)
        if width and height:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.debug(f"Video captured with dimensions: {self.width}x{self.height}")

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
        logger.debug(f"Video capture strategy set to {self._strategy.__class__.__name__}.")
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
                    logger.warning(f"VideoStreamManager can not get frames after {max_waiting_time} seconds upon start. ")
                    return
                pass
        logger.debug("VideoStreamManager started.")

    def get_frame(self):
        """
        Retrieve a frame using the current strategy.
        """
        if not self._started:
            logger.warning("VideoStreamManager is not started. Cannot get frame.")
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

    def getQueueSize(self):
        return self._strategy.getQueueSize()

    def __del__(self):
        self.release()
