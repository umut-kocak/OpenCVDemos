"""
This module provides a `VisualDebugger` class for real-time debugging of visual data by displaying
frames either in separate windows or overlaying them on a main frame. It supports toggling between
display modes, managing window states, and enabling/disabling debugging functionality. The debugger
is designed to aid in visual inspection during development by offering flexibility in frame display
and cycling through debugging views.
"""
import time
import cv2
import numpy as np

from utils.frame_data import FrameData
from utils.logger import logger


class VisualDebugger:
    """
    A utility class to assist in debugging visual frames by displaying them
    in separate or overlaid modes during runtime.
    """

    def __init__(self):
        """Initialize the visual debugger."""
        self._debugging_frames = {}
        self._opened_windows = set()
        self._closed_windows = set()
        self._enabled = False
        self._separate_mode = True
        self._override_mode_index = 0

    def toggle_mode(self):
        """
        Toggle between separate and override modes for displaying debugging frames.
        """
        was_enabled = self.is_enabled()
        if was_enabled:
            self.toggle()
        self._separate_mode = not self._separate_mode
        logger.debug("Toggling visual debugger mode to %s mode", ("separate" if self._separate_mode else "override"))
        if was_enabled:
            self.toggle()

    def add_debugging_frame(self, window_name, frame):
        """
        Add a debugging frame to be displayed.

        Args:
            window_name (str): The name of the window for the debugging frame.
            frame (ndarray): The frame to be displayed.
        """
        if not self.is_enabled():
            return
        if self._separate_mode and self._is_window_closed(window_name):
            return
        if frame is None :
            if not (window_name in self._debugging_frames):
                return
            width, height = self._debugging_frames[window_name].image.shape[:2]
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        if not isinstance(frame, FrameData):
            frame = FrameData(frame, time.time())
        self._debugging_frames[window_name] = frame

    def display_debug_frames(self, main_frame):
        """
        Display debugging frames in either separate or override mode.

        Args:
            main_frame (ndarray): The main frame to display.

        Returns:
            ndarray: The main frame or an overridden debugging frame.
        """
        if not self.is_enabled() or len(self._debugging_frames) == 0:
            return main_frame
        if self._separate_mode:
            self._show_separate()
            return main_frame
        return self._override_on_main(main_frame)

    def override_next(self):
        """
        Cycle to the next debugging frame in override mode.
        """
        if not self.is_enabled() or self._separate_mode:
            return
        self._override_mode_index += 1

    def override_previous(self):
        """
        Cycle to the previous debugging frame in override mode.
        """
        if not self.is_enabled() or self._separate_mode:
            return
        self._override_mode_index -= 1

    def toggle(self):
        """
        Enable or disable the visual debugger.
        """
        self._enabled = not self._enabled
        if not self.is_enabled():
            if self._separate_mode:
                for name in self._debugging_frames:
                    if self._is_window_closed(name):
                        continue
                    cv2.destroyWindow(name)
            self.cleanup()

    def is_enabled(self):
        """
        Check if the debugger is enabled.

        Returns:
            bool: True if enabled, False otherwise.
        """
        return self._enabled

    def cleanup(self):
        """
        Clear all debugging frames and reset state.
        """
        self._opened_windows.clear()
        self._closed_windows.clear()
        self._debugging_frames.clear()

    def _show_separate(self):
        """
        Display debugging frames in separate windows.
        """
        for name, frame in self._debugging_frames.items():
            if self._is_window_closed(name):
                continue
            cv2.imshow(name, frame.image)
            self._opened_windows.add(name)

    def _override_on_main(self, main_frame):
        """
        Override the main frame with a selected debugging frame in override mode.

        Args:
            main_frame (ndarray): The main frame to be overridden.

        Returns:
            ndarray: The overridden debugging frame.
        """
        self._override_mode_index %= len(self._debugging_frames)
        for i, frame in enumerate(self._debugging_frames.values()):
            if i == self._override_mode_index:
                return frame
        return main_frame

    def _is_window_closed(self, window_name):
        """
        Check if a window is closed and handle cleanup if necessary.

        Args:
            window_name (str): The name of the window to check.

        Returns:
            bool: True if the window is closed, False otherwise.
        """
        if window_name in self._closed_windows:
            return True
        if window_name not in self._opened_windows:
            return False

        is_closed = False
        try:
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            is_closed = prop < 0
        except cv2.error: # pylint: disable=E0712
            is_closed = True

        if is_closed:
            logger.debug("Closing the window: %s", window_name)
            self._opened_windows.remove(window_name)
            del self._debugging_frames[window_name]
            self._closed_windows.add(window_name)
            if len(self._opened_windows) == 0:
                logger.debug("No windows open, toggling off.")
                self.toggle()

        return is_closed
