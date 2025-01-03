import cv2
import numpy as np
from collections import defaultdict

from utils.logger import logger

SEC_TO_MSEC = 1000

class FrameTimer:
    """
    A utility class for measuring frame times and calculating statistics such as FPS and labeled intervals.
    """

    def __init__(self, calculation_frequency=30):
        """
        Initialize the FrameTimer.

        Args:
            calculation_frequency (int): The number of frames after which statistics are calculated. Defaults to 30.
        """
        self._calculation_frequency = calculation_frequency
        self._frame_count = 0
        self._ticks_per_unit = cv2.getTickFrequency()
        self._last_frame_start_tick = cv2.getTickCount()

        self._active_labels = {}
        self._ignored_labels = set()
        self._label_times = defaultdict(list)
        self._label_times['FPS'] = list()
        self._label_times['FrT(ms)'] = list()
        self._label_stats = {}

        self._last_frame_start = None
        self._last_frame_end = None

    def start_frame(self):
        """
        Call at the beginning of each frame to mark the start time.

        Raises:
            RuntimeError: If called consecutively without an `end_frame`.
        """
        if self._last_frame_start is not None and self._last_frame_end is None:
            raise RuntimeError("start_frame called consecutively without an end_frame.")
        self._last_frame_start = cv2.getTickCount()
        self._active_labels.clear()

    def undo_start_frame(self):
        """
        Undo the start of a frame. Useful if the frame processing is aborted.

        Raises:
            RuntimeError: If called without a preceding `start_frame`.
        """
        if self._last_frame_start is None:
            raise RuntimeError("undo_start_frame called without a preceding start_frame.")
        self._last_frame_start = None

    def end_frame(self):
        """
        Call at the end of each frame to update frame count and calculate stats when needed.

        Raises:
            RuntimeError: If called without a preceding `start_frame`.
        """
        if self._last_frame_start is None:
            raise RuntimeError("end_frame called without a preceding start_frame.")

        self._last_frame_end = cv2.getTickCount()
        frame_time = (self._last_frame_end - self._last_frame_start) / self._ticks_per_unit
        self._label_times['FPS'].append(1.0 / frame_time if frame_time > 0 else 0.0)
        self._label_times['FrT(ms)'].append(frame_time)
        self._last_frame_start = None
        self._last_frame_end = None

        self._frame_count += 1

        # Calculate FPS and other stats every `calculation_frequency` frames
        if self._frame_count % self._calculation_frequency == 0:
            self._frame_count = 0
            self._label_stats = {}
            for label, times in self._label_times.items():
                if not times:
                    continue
                arr = np.array(times)
                if label != "FPS":
                    arr *= SEC_TO_MSEC
                self._label_stats[label] = {
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'avg': np.mean(arr),
                    'stddev': np.std(arr)
                }

            # Clear the label times for the next interval
            self._label_times.clear()
            self._label_times['FPS'] = list()
            self._label_times['FrT(ms)'] = list()
            

    def begin_label(self, label):
        """
        Mark the start of a labeled interval.

        Args:
            label (str): The label for the interval.

        Logs a warning if the label is already active.
        """
        current_tick = cv2.getTickCount()

        if label in self._active_labels:
            if label not in self._ignored_labels:
                logger.warning(f"Warning: 'begin_label' for '{label}' called without a matching 'end_label'. Ignoring further warnings for this label.")
                self._ignored_labels.add(label)
            return

        self._active_labels[label] = current_tick

    def end_label(self, label):
        """
        Mark the end of a labeled interval and record its elapsed time.

        Args:
            label (str): The label for the interval.

        Logs a warning if the label is not active or if the end is called before the start.
        """
        current_tick = cv2.getTickCount()

        if label not in self._active_labels:
            if label not in self._ignored_labels:
                logger.warning(f"Warning: 'end_label' for '{label}' called without a matching 'begin_label'. Ignoring further warnings for this label.")
                self._ignored_labels.add(label)
            return

        start_tick = self._active_labels.pop(label)
        if start_tick > current_tick:
            if label not in self._ignored_labels:
                logger.warning(f"Warning: 'end_label' for '{label}' called before 'begin_label'. Ignoring further warnings for this label.")
                self._ignored_labels.add(label)
            return

        elapsed_time = (current_tick - start_tick) / self._ticks_per_unit
        self._label_times[label].append(elapsed_time)

    def get_fps(self):
        """
        Get the average FPS from the most recent statistics.

        Returns:
            float: The average FPS, or 0.0 if no stats are available.
        """
        return self._label_stats.get('FPS', {}).get('avg', 0.0)

    def get_stats(self):
        """
        Get the most recently calculated statistics for all labels.

        Returns:
            dict: A dictionary of statistics for each label.
        """
        return self._label_stats

    def get_formatted_stats(self, detailed=False):
        """
        Get the most recent statistics formatted as strings.
        
        Args:
            detailed (bool): Whether to include detailed stats (min, max, stddev) or just avg.
        
        Returns:
            list: A list of strings formatted based on the detailed flag:
                - If detailed: "label: avg: X min: Y max: Z stdv: W"
                - If not detailed: "label: avg: X"
        """
        formatted_stats = []
        
        for label, stats in self._label_stats.items():
            if detailed:
                formatted_stats.append(
                    f"{label}: avg: {stats['avg']:.2f} "
                    f"min: {stats['min']:.2f} "
                    f"max: {stats['max']:.2f} "
                    f"stdv: {stats['stddev']:.2f}"
                )
            else:
                formatted_stats.append(
                    f"{label}:{stats['avg']:.2f}"
                )
        
        return formatted_stats
