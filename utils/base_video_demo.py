"""
Defines `BaseVideoDemo`, an abstract base class for video processing applications.

Key Features:
- Manages video streaming, display, and key input handling.
- Requires subclasses to implement `process_frame` for custom frame processing
  and `get_window_name` for window naming.
- Provides overridable hooks (`pre_frame_loop`, `frame_loop_begin`, `frame_loop_end`)
  for customizing behavior.

Usage:
Subclass `BaseVideoDemo` and override required methods to create custom video demos.
"""
from abc import ABC, abstractmethod
from enum import Enum
import os
import logging
from pathlib import Path
import sys
import time

import cv2

from utils.base_demo import BaseDemo
from utils.display_manager import DisplayManager
from utils.frame_timer import FrameTimer
from utils.helper import resize_image
from utils.logger import logger
from utils.video_stream_manager import VideoStreamManager
from utils.visual_debugger import VisualDebugger

SEC_TO_MSEC = 1000


class BaseVideoDemo(BaseDemo):
    """
    Base class for video demos that handle video streaming, displaying, and key management.
    Subclasses should implement specific frame processing logic and window naming.
    """
    class FrameStatus(Enum):
        SUCCESS = 0
        IGNORE = 1
        ABORT = 2

    def __init__(self):
        """Initialize the video demo with video source and settings."""

        super().__init__()

        # Handlers
        source = 0
        if self.settings.video.capture.source != 0:
            source_file = self.get_asset_path(self.settings.video.capture.source)
            if source_file:
                source = source_file
            else: 
                logger.error("Could not find the video source %s. Using camera instead.", self.settings.video.capture.source)
        self._video_manager = VideoStreamManager(self.settings, capture_source=source)
        self._video_manager.start()

        DisplayManager.create_window(self.get_window_name(),
                                     resizable=self.settings.display.resizable,
                                     default_size=(self.settings.display.width, self.settings.display.height))

        self._frame_timer = FrameTimer(calculation_frequency=self.settings.frame.timer.calculation_frequency)
        self._quit_loop = False
        self._number_of_null_frames = 0
        self._paused = False
        
        self._visual_debugger = VisualDebugger()

        self.register_keys()

    @abstractmethod
    def process_frame(self, frame):
        """Process the frame and return the processed frame."""

    @abstractmethod
    def get_window_name(self):
        """Return the name of the window."""
        return "Base Video Demo"

    def pre_frame_loop(self):
        """Hook for operations to run before the frame loop starts."""

    def check_captured_frame(self, frame):
        """Hook for operations to run when there are not enough captured video frames."""
        if frame is None:
            # Check if the number of consequent None frames is within the
            # tolerated limits
            self._number_of_null_frames += 1
            self._frame_timer.undo_start_frame()
            nr_tolerate = self.settings.frame.missing_to_tolerate
            if nr_tolerate > 0 and self._number_of_null_frames > nr_tolerate:
                return self.FrameStatus.ABORT
            return self.FrameStatus.IGNORE
        self._number_of_null_frames = 0
        return self.FrameStatus.SUCCESS

    def frame_loop_begin(self):
        """Capture and return a frame. Can be overridden by subclasses."""
        frame = self._video_manager.get_frame()
        #if frame is None: logger.warning("Cannot get any frame from video manager.")
        return frame

    def frame_loop_end(self, frame):
        """Handle post-frame operations such as displaying and stats updates."""
        self._stats_manager.update_stat(
            "FrWaitT(ms)", self.settings.frame.wait_time)
        self._stats_manager.update_stat(
            "VidQueueSize", self._video_manager.get_queue_size())
        self._stats_manager.update_stat(
            "VidSkipFrame", self._video_manager.capture_strategy.get_frame_filtering_skip_size())

        if self.settings.stats.show_help:
            # Base help text
            x, y = self.settings.stats.default_help_text_position
            x, y = (int( x * self._video_manager.width),
                    int( y * self._video_manager.height))
            self._text_manager.draw_text(
                frame.image, self._key_manager.get_help_text(), pos=(x, y))
            
            # Module-specific help text
            x, y = self.settings.stats.demo_help_text_position
            x, y = (int( x * self._video_manager.width),
                    int( y * self._video_manager.height))
            self._text_manager.draw_text(
                frame.image, self._key_manager.get_help_text(self.get_window_name()), pos=(x, y))

        if self.settings.stats.show_stats:
            x, y = self.settings.stats.text_position
            x, y = (int( x * self._video_manager.width),
                    int( y * self._video_manager.height))
            self._text_manager.draw_text(
                frame.image,
                self._frame_timer.get_formatted_stats(self.settings.stats.detailed) +
                    self._stats_manager.get_formatted_stats(self.settings.stats.detailed),
                pos=(x, y),
                properties="stats",
            )

        DisplayManager.show_frame(self.get_window_name(
        ), frame.image, self.settings.display.resize_output_to_window)

        key = cv2.waitKey(self.settings.frame.wait_time) & 0xFF
        # cv2.waitKey is the actual call causing anything to be drawn, so
        # measure latency after this.
        self._stats_manager.update_stat(
            "FrLatency(ms)", (time.time() - frame.capture_time) * SEC_TO_MSEC)

        if not self._key_manager.check_events(key):
            self._quit_loop = True

    def pause_loop(self):
        """."""
        key = cv2.waitKey(self.settings.frame.wait_time) & 0xFF
        if not self._key_manager.check_events(key):
            self._quit_loop = True
            return False
        return True

    def cleanup(self):
        """Perform cleanup tasks when the demo is stopped."""
        logger.debug("BaseVideoDemo::cleanup")
        super(BaseVideoDemo, self).cleanup()
        self._video_manager.release()
        DisplayManager.destroy_all_windows()

    def toggle_pause(self, explicit_set=False, explicit_value=True):
        """Toggle pause."""
        if explicit_set:
            self._paused = explicit_value
        else: 
            self._paused = not self._paused
        logger.debug("%sPausing.",("" if self._paused else "Un"))
        if self._paused:
            self._video_manager.stop()
        else:
            self._video_manager.start()

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
            frame_status = self.check_captured_frame(frame)
            if frame_status:
                if frame_status == self.FrameStatus.ABORT:
                    logger.warning("Too many None frames, Quitting the main loop.")
                    break
                elif frame_status == self.FrameStatus.IGNORE :
                    continue

            # Processing the frame
            self._frame_timer.begin_label("PocessT(ms)")
            if self.settings.frame.processing.resize_capture_before_process:
                frame.image = resize_image(frame.image, self.settings.frame.processing.width, self.settings.frame.processing.height)
            frame = self.process_frame(frame)
            if self.settings.frame.processing.resize_capture_before_process:
                frame.image = resize_image(frame.image, self._video_manager.width, self._video_manager.height)
            self._frame_timer.end_label("PocessT(ms)")
            
            frame = self._visual_debugger.display_debug_frames(frame)

            # End of the frame
            self.frame_loop_end(frame)
            self._frame_timer.end_frame()
            if self._should_quit_frame_loop():
                break

        logger.debug("End of the main loop.")
        self.cleanup()

    def register_keys(self):
        """Register keys and their respective handlers."""
        
        def toggle_show_help(settings):
            setattr(settings, 'show_help', not getattr(settings, 'show_help'))
        
        def toggle_stats(settings, detailed=False):
            setattr(settings, 'show_stats', not getattr(settings, 'show_stats'))
            setattr(settings, 'detailed', detailed)
        
        def adjust_wait_time(settings, delta):
            new_time = max(1, getattr(settings, 'wait_time') + delta)
            setattr(settings, 'wait_time', new_time)
    
        def adjust_skip_period(settings, delta):
            new_skip = max(2, getattr(settings, 'period') + delta)
            setattr(settings, 'period', new_skip)
        
        key_bindings = [
            # General keys
            (ord('q'), "Quit the application", self.cleanup, None, True),
            (ord(' '), "Pause", self.toggle_pause, None),
            (ord('h'), "Print help text", self._key_manager.print_help, None),
            (ord('H'), "Show help text", toggle_show_help, self.settings.stats),
            
            # Stats toggling
            (ord('s'), "Toggle stats", lambda s: toggle_stats(s, detailed=False), self.settings.stats),
            (ord('S'), "Toggle stats (detailed)", lambda s: toggle_stats(s, detailed=True), self.settings.stats),
            
            # Visual Debugger
            (ord('d'), "Enable/Disable VisualDebugger", lambda dbg: dbg.toggle(), self._visual_debugger),
            (ord('D'), "Toggle VisualDebugger mode: Override or Separate.", lambda dbg: dbg.toggle_mode(), self._visual_debugger),
            (ord('+'), "Override next VisualDebugger image.", lambda dbg: dbg.isEnabled() and dbg.override_next(), self._visual_debugger),
            (ord('-'), "Override previous VisualDebugger image.", lambda dbg: dbg.isEnabled() and dbg.override_previous(), self._visual_debugger),
            
            # Video Manager
            (ord('V'), "Toggle VideoCapture mode: Single or multithreaded.", lambda vm: vm.toggle_strategy(), self._video_manager),
            (ord('f'), "Change the frame filter method of video capture strategy.", 
            lambda vm: vm.capture_strategy.set_frame_filtering_method(vm.capture_strategy.frame_filtering.next()), self._video_manager),
            
            # Frame settings
            (ord('u'), "Increase waiting time at the end of the frame.", lambda s: adjust_wait_time(s, +1), self.settings.frame),
            (ord('j'), "Decrease waiting time at the end of the frame.", lambda s: adjust_wait_time(s, -1), self.settings.frame),
            
            # Saving and settings
            (ord('C'), "Save the current settings.", lambda s: s.save_to_json(self.get_output_folder()/"CurrentSettings.json"), self.settings),
            
            # Video capture skip number
            (ord('i'), "Increase video_capture_frame_filter_skip_number.", lambda s: adjust_skip_period(s, +1), self.settings.video.capture.frame_filtering.skip_frame),
            (ord('k'), "Decrease video_capture_frame_filter_skip_number.", lambda s: adjust_skip_period(s, -1), self.settings.video.capture.frame_filtering.skip_frame),
        ]
    
        # Register all key bindings
        for key, description, callback, callback_arg, *args in key_bindings:
            self._key_manager.register_key(key, description, callback, callback_arg, *args)

    def _should_quit_frame_loop(self):
        """Determine if the frame loop should quit."""
        return self._quit_loop
