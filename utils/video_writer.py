""" A lightweight video writer. """
import cv2

from utils.logger import logger


class VideoWriter:
    """ A lightweight video writer. """

    def __init__(self, filename, fourcc='MJPG', fps=30, frame_size=(640, 480)):
        """
        Initializes the VideoWriter.
        
        :param filename: Output video file name.
        :param fourcc: FourCC codec code (default: 'MJPG').
        :param fps: Frames per second (default: 30).
        :param frame_size: Frame size (default: (640, 480)).
        """
        # Create the VideoWriter object
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.out = cv2.VideoWriter(self.filename, self.fourcc, self.fps, self.frame_size)

        if not self.out.isOpened():
            logger.error("Failed to open VideoWriter with filename: %s", self.filename)

    def write(self, frame):
        """
        Write a single frame to the video.
        
        :param frame: A single frame (numpy array) to write.
        """
        if self.out.isOpened():
            self.out.write(frame)
        else:
            logger.error("VideoWriter is not opened. Can not write the frame.")

    def release(self):
        """
        Release the VideoWriter and close the output file.
        """
        if self.out.isOpened():
            self.out.release()
            logger.info("Video saved to %s.", self.filename)
            logger.debug("VideoWriter is already released")
        else:
            logger.debug("VideoWriter is already released.")
