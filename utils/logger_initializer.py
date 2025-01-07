"""
This script sets up a logging infrastructure for the application. It ensures a dedicated directory
for log files is created and provides a function to initialize loggers with configurable options
such as logging levels, output destinations (console and/or file), and logger names.
"""
import logging
import os
from pathlib import Path

from utils.logger import initialize_logger as init_logger

# Initialize the global logger before importing other modules
LOGGING_PATH = './logs'
Path(LOGGING_PATH).mkdir(parents=True, exist_ok=True)


def initialize_logger(_logger_name, _logger_file_name, _log_to_console=True,
                      _log_to_file=None, _console_level=logging.INFO, _file_level=logging.DEBUG):
    """ Initializes the global logger.
    """
    init_logger(
        name=_logger_name,
        log_to_console=_log_to_console,
        log_to_file=None if _log_to_file is None else os.path.join(
            LOGGING_PATH, _log_to_file),
        console_level=_console_level,
        file_level=_file_level
    )
