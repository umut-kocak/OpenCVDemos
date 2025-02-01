"""
Usage:
Subclass `BaseVideoDemo` and override required methods to create custom video demos.
"""
from abc import ABC, abstractmethod
import os
import logging
from pathlib import Path
import sys
import time

import cv2

# Initialize the global logger before importing other modules
def get_entry_point_filename():
    filename_with_ext = os.path.basename(sys.argv[0])
    filename = os.path.splitext(filename_with_ext)[0]
    return filename

logger_name = get_entry_point_filename()
logger_time = time.strftime("%Y%m%d-%H%M%S")
logger_file_name = logger_name + '-' + logger_time + ".log"
from utils.logger_initializer import initialize_global_logger
initialize_global_logger(
    logger_name,
    logger_file_name,
    _log_to_console=True,
    _log_to_file=logger_file_name,
    _console_level=logging.DEBUG,
    _file_level=logging.DEBUG
)

# Import modules after the global logger is initialized
from utils.key_manager import KeyManager  # pylint: disable=C0413
from utils.logger import logger  # pylint: disable=C0413
from utils.settings import Settings  # pylint: disable=C0413
from utils.stats_manager import StatsManager  # pylint: disable=C0413
from utils.text_manager import TextManager, TextProperties  # pylint: disable=C0413
from utils.visual_debugger import VisualDebugger  # pylint: disable=C0413

SEC_TO_MSEC = 1000


class BaseDemo(ABC):
    """
    """

    def __init__(self):
        """"""

        # Settings
        self.settings = Settings(self.get_all_setting_files())
        if self.settings.print_ocv_info:
            print(cv2.getBuildInformation())

        self._key_manager = KeyManager()
        self._stats_manager = StatsManager(calculation_frequency=self.settings.stats.calculation_frequency)
        self._text_manager = TextManager()
        self._text_manager.register_properties(
            "stats", TextProperties(color=TextProperties.GREEN))
        self._visual_debugger = VisualDebugger()

    @abstractmethod
    def get_window_name(self):
        """Return the name of the window."""

    @abstractmethod
    def run(self):
        """Run the main loop for video processing."""

    def cleanup(self):
        """Perform cleanup tasks when the demo is stopped."""
        logger.debug("BaseDemo::cleanup")
        self._stats_manager.cleanup()
        self._visual_debugger.cleanup()

    def get_demo_folder(self):
        """ Gets the folder(Path) where the demo resides."""
        module_name = self.__class__.__module__
        module_file = sys.modules[module_name].__file__
        return Path(os.path.dirname(os.path.abspath(module_file)))

    def get_asset_path(self, relative_path):
        """ Gets the full asset path given the relative_path."""
        full_path = self.get_demo_folder() / "assets" / relative_path
        if full_path.exists():
            return full_path
        full_path = self.get_demo_folder() / relative_path
        if full_path.exists():
            return full_path
        full_path = Path("./assets") / relative_path
        if full_path.exists():
            return full_path
        full_path = Path(relative_path)
        if full_path.exists():
            return full_path
        return None

    def get_output_folder(self):
        """ Gets the folder(Path) where the demo should save outputs."""
        global logger_time
        _output_path = self.get_demo_folder() / ("output-" + logger_time )
        Path(_output_path).mkdir(parents=True, exist_ok=True)
        return _output_path

    def get_all_setting_files(self):
        """ Gets all the settings files."""
        all_settings = ["DefaultSettings.json"]
        full_path = self.get_demo_folder() / "Settings.json"
        if full_path.exists():
            all_settings.append(full_path)
        return all_settings

