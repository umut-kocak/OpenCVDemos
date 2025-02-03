"""
Module for real-time video recording.

This module implements a `VideoRecorder` class that extends the `BaseVideoDemo` framework.
It enables video recording from a live video stream, providing functionality to start and stop recording using
keyboard shortcuts. The recorded video is saved to an output file specified in the settings.
"""

from utils.base_video_demo import BaseVideoDemo
from utils.video_writer import VideoWriter
from utils.logger import logger

class VideoRecorder(BaseVideoDemo):
    """Demo for recording video from a live video stream."""

    def __init__(self):
        """Initialize the video recorder with output file settings."""
        super().__init__()

        stngs = self.settings.demo.output
        self._video_writer = VideoWriter(
            self.get_output_folder() / stngs.file_name,
            fourcc=stngs.fourcc,
            fps=self._video_manager.fps,
            frame_size=(self._video_manager.width, self._video_manager.height)
        )
        self._recording = False

    def process_frame(self, frame):
        """Process each frame and write to the video file if recording is enabled.

        Args:
            frame: The current video frame to process.

        Returns:
            The processed video frame.
        """
        if self._recording:
            self._video_writer.write(frame.image)
        return frame

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.
        """
        super().register_keys()

        def toggle_recording():
            """Toggle the recording state and log the action."""
            logger.info("%s recording.", ("Stopping" if self._recording else "Starting"))
            setattr(self, '_recording', not getattr(self, '_recording'))

        key_bindings = [
            (ord('r'), "Toggle recording", lambda: toggle_recording(), None),
        ]

        for key, description, callback, callback_arg, *args in key_bindings:
            self._key_manager.register_key(
                key, description, callback, callback_arg, *args, name_space=self.get_window_name()
            )

    def cleanup(self):
        """Perform cleanup tasks when the demo is stopped, such as releasing the video writer."""
        super().cleanup()
        self._video_writer.release()

    def get_window_name(self):
        """Return the name of the demo window."""
        return "Video Recorder"


if __name__ == "__main__":
    demo = VideoRecorder()
    demo.run()
