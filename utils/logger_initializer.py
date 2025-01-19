"""
This script sets up a logging infrastructure for the application. It ensures a dedicated directory
for log files is created and provides a function to initialize loggers with configurable options
such as logging levels, output destinations (console and/or file), and logger names.
"""
import logging
import os
from pathlib import Path
import utils.logger


def initialize_global_logger(_logger_name, _logger_file_name, _log_to_console=True,
                      _log_to_file=None, _console_level=logging.INFO, _file_level=logging.DEBUG,
                      _logger_type="SEPARATE_THREAD"):
    """ Initializes the global logger.
    """
    LOGGING_PATH = './logs/' + _logger_name
    Path(LOGGING_PATH).mkdir(parents=True, exist_ok=True)

    if _logger_type == "MAIN_THREAD":
        utils.logger.initialize_global_logger_main(
            name=_logger_name,
            log_to_console=_log_to_console,
            log_to_file=None if _log_to_file is None else os.path.join(
                LOGGING_PATH, _log_to_file),
            console_level=_console_level,
            file_level=_file_level
        )
    elif _logger_type == "SEPARATE_THREAD":
        utils.logger.initialize_global_logger_threaded(
            name=_logger_name,
            log_to_console=_log_to_console,
            log_to_file=None if _log_to_file is None else os.path.join(
                LOGGING_PATH, _log_to_file),
            console_level=_console_level,
            file_level=_file_level
        )
    elif _logger_type == "NONE":
        utils.logger.initialize_global_logger_none(
            name=_logger_name,
            log_to_console=_log_to_console,
            log_to_file=None if _log_to_file is None else os.path.join(
                LOGGING_PATH, _log_to_file),
            console_level=_console_level,
            file_level=_file_level
        )
    else:
        print("Invalid logger type ", _logger_type)