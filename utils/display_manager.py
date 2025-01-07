"""
This module provides the `DisplayManager` class, which facilitates managing OpenCV windows for
visualizing video frames. It includes functionality to create windows with customizable properties,
display frames with optional resizing, and clean up by closing all windows.
"""
import cv2

from utils.logger import logger


class DisplayManager:
    """
    Manages the creation, display, and destruction of OpenCV windows for video frames.
    """

    @staticmethod
    def create_window(window_name, resizable=True, default_size=None):
        """
        Create an OpenCV window with optional resizing capabilities.

        Args:
            window_name (str): Name of the window to create.
            resizable (bool, optional): Whether the window is resizable. Defaults to True.
            default_size (tuple, optional): Default size of the window as (width, height). Defaults to None.
        """
        if resizable:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        if default_size:
            cv2.resizeWindow(window_name, *default_size)

        window_rect = cv2.getWindowImageRect(window_name)
        logger.debug("Window(%s) %dx%d is created.", window_name, window_rect[2], window_rect[3])

    @staticmethod
    def show_frame(window_name, frame, resize_to_window=False):
        """
        Display a video frame in the specified OpenCV window.

        Args:
            window_name (str): Name of the window where the frame will be displayed.
            frame (ndarray): The video frame to display.
            resize_to_window (bool, optional): Whether to resize the frame to match the window size. Defaults to False.
        """
        if resize_to_window:
            window_rect = cv2.getWindowImageRect(window_name)
            width, height = window_rect[2], window_rect[3]

            frame = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_LINEAR)

        cv2.imshow(window_name, frame)

    @staticmethod
    def destroy_all_windows():
        """
        Close all OpenCV windows.
        """
        cv2.destroyAllWindows()
