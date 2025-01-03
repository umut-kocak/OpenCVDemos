import logging
import os
from pathlib import Path
import time

# Initialize the global logger before importing other modules
logging_path = './logs'
Path(logging_path).mkdir(parents=True, exist_ok=True)

def initialize_logger(_logger_name, _logger_file_name, _log_to_console=True, _log_to_file=None, _console_level=logging.INFO, _file_level=logging.DEBUG):
    from utils.logger import initialize_logger as init_logger
    init_logger(
        name=_logger_name,
        log_to_console=_log_to_console,
        log_to_file=None if _log_to_file is None else os.path.join(logging_path, _log_to_file),
        console_level=_console_level,
        file_level=_file_level
    )
