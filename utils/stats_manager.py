"""
This script defines a `StatsManager` class to manage and track statistical data over time.
It allows updating statistics, calculating aggregated metrics (such as min, max, average, and
standard deviation) at specified intervals, and retrieving formatted results. The class is designed
for efficient tracking and periodic calculation of statistics in real-time or near-real-time
applications.
"""
from collections import defaultdict

import numpy as np

class StatsManager:
    """
    A manager for tracking and updating statistics.
    """

    def __init__(self, calculation_frequency=30):
        """
        Initialize the StatsManager with an empty stats dictionary.
        """
        self._calculation_frequency = calculation_frequency
        self._frame_count = 0
        self._stats = defaultdict(list)
        self._labeled_results = {}

    def update_stat(self, name, value):
        """
        Update or add a stat with the given name and value.

        Args:
            name (str): The name of the stat.
            value (float): The value to set for the stat.
        """
        self._stats[name].append(value)

    def get_formatted_stats(self, detailed=False):
        """
        Generate and return formatted stat strings.

        Returns:
            list: A list of formatted strings, one for each stat.
        """
        self._frame_count += 1

        # Calculate FPS and other stats every `calculation_frequency` frames
        if self._frame_count % self._calculation_frequency == 0:
            self._frame_count = 0
            self._labeled_results = {}
            for label, values in self._stats.items():
                if not values:
                    continue
                arr = np.array(values)
                self._labeled_results[label] = {
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'avg': np.mean(arr),
                    'stddev': np.std(arr)
                }

            # Clear the label times for the next interval
            self._stats.clear()

        formatted_stats = []

        for label, stats in self._labeled_results.items():
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

    def cleanup(self):
        """
        Clear all stats.
        """
        self._stats.clear()
        self._labeled_results.clear()
