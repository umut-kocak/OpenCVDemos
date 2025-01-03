from utils.video_stream_manager import VideoStreamManager
from utils.display_manager import DisplayManager
from utils.key_manager import KeyManager
from utils.logger import logger
from utils.stats_manager import StatsManager
from utils.text_manager import TextManager
from utils.text_manager import TextProperties
from utils.visual_debugger import VisualDebugger
from utils.frame_timer import FrameTimer
from utils.settings import Settings

from abc import ABC, abstractmethod
import time
import cv2

SEC_TO_MSEC = 1000

def overridable(method):
    """Decorator to indicate that a method can be overridden by subclasses."""
    method._overridable = True
    return method

class BaseVideoDemo(ABC):
    """
    Base class for video demos that handle video streaming, displaying, and key management.
    Subclasses should implement specific frame processing logic and window naming.
    """

    def __init__(self, source=0, settings_file=None):
        """Initialize the video demo with video source and settings."""

        # Settings
        self._settings = Settings(["DefaultSettings.json", settings_file])

        # Handlers
        self._video_manager = VideoStreamManager(source, self._settings.inputWidth, self._settings.inputHeight)
        self._video_manager.start()

        DisplayManager.create_window(self.get_window_name(), 
                                     resizable=self._settings.resizable, 
                                     default_size=(self._settings.windowWidth, self._settings.windowHeight))

        self._key_manager = KeyManager()
        self._stats_manager = StatsManager(calculation_frequency=30)
        self._text_manager = TextManager()
        self._text_manager.register_properties("stats", TextProperties(color=TextProperties.GREEN))

        self._visual_debugger = VisualDebugger()
        self._frame_timer = FrameTimer(calculation_frequency=30)

        self.register_keys()

        self._quit_loop = False
        self._number_of_null_frames = 0
        self._paused = False

    @abstractmethod
    def process_frame(self, frame):
        """Process the frame and return the processed frame."""
        pass

    @abstractmethod
    def get_window_name(self):
        """Return the name of the window."""
        pass

    @overridable
    def pre_frame_loop(self):
        """Hook for operations to run before the frame loop starts."""
        pass

    @overridable
    def frame_loop_begin(self):
        """Capture and return a frame. Can be overridden by subclasses."""
        frame = self._video_manager.get_frame()
        if frame is None:
            logger.warning("Cannot get any frame from video manager.")
        return frame

    @overridable
    def frame_loop_end(self, frame):
        """Handle post-frame operations such as displaying and stats updates."""
        self._stats_manager.update_stat("FrWaitT(ms)", self._settings.frameWaitTime)
        self._stats_manager.update_stat("VidQueueSize", self._video_manager.getQueueSize())
        
        if self._settings.showHelp:
            x, y = (int(0.2 * self._video_manager.width), int(0.2 * self._video_manager.height))
            self._text_manager.draw_text(frame.image, self._key_manager.get_help_text(), pos=(x, y))

        if self._settings.showStats:
            x, y = (int(0.02 * self._video_manager.width), int(0.05 * self._video_manager.height))
            self._text_manager.draw_text(
                frame.image,
                self._frame_timer.get_formatted_stats(self._settings.detailedStats) + self._stats_manager.get_formatted_stats(self._settings.detailedStats),
                pos=(x, y),
                properties="stats",
            )

        DisplayManager.show_frame(self.get_window_name(), frame.image, self._settings.fit_frame_to_window_size)

        key = cv2.waitKey(self._settings.frameWaitTime) & 0xFF
        # cv2.waitKey is the actual call causing anything to be drawn, so measure latency after this.
        self._stats_manager.update_stat("FrLatency(ms)", (time.time() - frame.captureTime)*SEC_TO_MSEC)

        if not self._key_manager.check_events(key):
            self._quit_loop = True

    @overridable
    def pause_loop(self):
        """."""
        key = cv2.waitKey(self._settings.frameWaitTime) & 0xFF
        if not self._key_manager.check_events(key):
            self._quit_loop = True
            return False
        return True

    @overridable
    def cleanup(self):
        """Perform cleanup tasks when the demo is stopped."""
        logger.debug("BaseModule::cleanup")
        self._video_manager.release()
        DisplayManager.destroy_all_windows()
        self._stats_manager.cleanup()
        self._visual_debugger.cleanup()

    @overridable
    def togglePause(self):
        """Toggle pause."""
        logger.debug(("Un" if self._paused else "") + "Pausing.")
        if self._paused:
            self._video_manager.start()
        else:
            self._video_manager.stop()
        self._paused = not self._paused

    def run(self):
        """Run the main loop for video processing."""
        self.pre_frame_loop()

        while True:
            if self._paused:
                # Run the paused loop: Could be some processing etc.
                if not self.pause_loop():
                    # Quit the main loop
                    break
                continue

            # Beginning of the frame
            self._frame_timer.start_frame()
            frame = self.frame_loop_begin()
            if frame is None:
                # Check if the number of consequent None frames is within the tolerated limits
                self._number_of_null_frames += 1
                self._frame_timer.undo_start_frame()
                if self._number_of_null_frames > self._settings.missing_frames_to_tolerate:
                    # Too many None frames, Quit the main loop
                    break
                continue
            
            # Processing the frame
            self._frame_timer.begin_label("PocessT(ms)")
            frame = self.process_frame(frame)
            self._frame_timer.end_label("PocessT(ms)")

            # End of the frame
            self.frame_loop_end(frame)
            self._frame_timer.end_frame()
            self._number_of_null_frames = 0
            if self._should_quit_frame_loop():
                break

        logger.debug("End of the main loop.")
        self.cleanup()

    @overridable
    def register_keys(self):
        """Register keys and their respective handlers."""

        self._key_manager.register_key(ord('q'), "Quit the application", self.cleanup, None, True)
        self._key_manager.register_key(ord(' '), "Pause", self.togglePause, None )

        self._key_manager.register_key(ord('h'), "Print help text", self._key_manager.print_help, None)
        self._key_manager.register_key(ord('H'), "Show help text", 
            lambda settings: setattr(settings, 'showHelp', not getattr(settings, 'showHelp')),
            self._settings)

        self._key_manager.register_key(ord('s'), "Toggle stats", 
            lambda settings: (setattr(settings, 'showStats', not getattr(settings, 'showStats')), setattr(settings, 'detailedStats', False)),
            self._settings)

        self._key_manager.register_key(ord('S'), "Toggle stats(detailed)", 
            lambda settings: (setattr(settings, 'showStats', not getattr(settings, 'showStats')), setattr(settings, 'detailedStats', True)),
            self._settings)

        self._key_manager.register_key(ord('d'), "Enable/Disable VisualDebugger.", 
            lambda debugger: debugger.toggle(),
            self._visual_debugger)
        self._key_manager.register_key(ord('D'), "Toggle VisualDebugger mode: Override or Separate.", 
            lambda debugger: debugger.toggle_mode(),
            self._visual_debugger)
        self._key_manager.register_key(ord('+'), "Override next VisualDebugger image.",
            lambda debugger: debugger.isEnabled() and debugger.override_next(),
            self._visual_debugger)
        self._key_manager.register_key(ord('-'), "Override previous VisualDebugger image.", 
            lambda debugger: debugger.isEnabled() and debugger.override_previous(),
            self._visual_debugger)

        self._key_manager.register_key(ord('V'), "Toggle VideoCapture mode: Single or multithreaded.", 
            lambda vid_manager: vid_manager.toggle_strategy(),
            self._video_manager)
        self._key_manager.register_key(ord('u'), "Increase waiting time at the end of the frame.",
            lambda settings: setattr(settings, 'frameWaitTime', getattr(settings, 'frameWaitTime') + 1),
            self._settings)
        self._key_manager.register_key(ord('j'), "Decrease waiting time at the end of the frame.",
            lambda settings: setattr(settings, 'frameWaitTime', max(1, getattr(settings, 'frameWaitTime') - 1)),
            self._settings)

        self._key_manager.register_key(ord('C'), "Save the current settings.", 
            lambda settings: settings.save_to_json("CurrentSettings.json"),
            self._settings)

    def _should_quit_frame_loop(self):
        """Determine if the frame loop should quit."""
        return self._quit_loop

