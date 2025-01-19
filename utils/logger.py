"""
This module provides functionality for initializing and managing loggers using Python's logging framework.
It includes a global logger for general application use and a method for creating separate, customized loggers
with options for console and file logging, configurable levels, and formatted output.
"""
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

class ThreadedLogger:
    """
        A class to handle logging operations asynchronously using a thread pool.
    
        This class uses a `ThreadPoolExecutor` to perform logging operations in 
        a separate thread, allowing for non-blocking and concurrent logging.
    
        Args:
            max_workers (int): The maximum number of threads in the thread pool. 
                Defaults to 2.
    """
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = None

    def initialize_logger(self, name="ApplicationLogger", log_to_console=True,
                          log_to_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
        """
        Initialize the logger with asynchronous threading for log calls.

        Args:
            name (str): The name of the logger.
            log_to_console (bool): Whether to log to the console.
            log_to_file (Optional[str]): Path to the log file. If None, file logging is disabled.
            console_level (int): Logging level for the console handler.
            file_level (int): Logging level for the file handler.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Define formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            file_handler = RotatingFileHandler(
                log_to_file, maxBytes=5 * 1024 * 1024, backupCount=3)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.logger.info("Threaded logger '%s' initialized.", name)

    def debug(self, msg, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(logging.ERROR, msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """
        Submit logging tasks to the thread pool.

        Args:
            level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
            msg (str): Log message.
            *args: Positional arguments for the logger.
            **kwargs: Keyword arguments for the logger.
        """
        if self.logger:
            self.executor.submit(self.logger.log, level, msg, *args, **kwargs)

    def shutdown(self):
        """
        Shutdown the thread pool and wait for all logging tasks to complete.
        """
        self.executor.shutdown(wait=True)
        self.logger = None

    def __del__(self):
        """
        """
        self.shutdown()

class NoneLogger:
    """
        An empty class to disable logging without breaking the existing code. 
    """

    def debug(self, msg, *args, **kwargs):
        """
        """

    def info(self, msg, *args, **kwargs):
        """
        """

    def warning(self, msg, *args, **kwargs):
        """
        """

    def error(self, msg, *args, **kwargs):
        """
        """

# Define a default global logger with basic configuration
logger = None # pylint: disable=C0103

def initialize_global_logger_main(name="ApplicationLogger", log_to_console=True,
                      log_to_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Initialize the global logger instance with specific settings.

    Args:
        name (str): The name of the logger.
        log_to_console (bool): Whether to log to the console.
        log_to_file (Optional[str]): Path to the log file. If None, file logging is disabled.
        console_level (int): Logging level for the console handler.
        file_level (int): Logging level for the file handler.
    """
    global logger # pylint: disable=global-statement

    # Reconfigure the global logger
    logger = logging.getLogger(name)
    # Set the logger's level to the lowest level you want to capture.
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define the formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler()
        # Set level for console handler
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if specified
    if log_to_file:
        file_handler = RotatingFileHandler(
            log_to_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(file_level)  # Set level for file handler
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logger '%s' initialized.", name)

def initialize_global_logger_threaded(name="ApplicationLogger", log_to_console=True,
                      log_to_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Initialize the global logger instance with specific settings.

    Args:
        name (str): The name of the logger.
        log_to_console (bool): Whether to log to the console.
        log_to_file (Optional[str]): Path to the log file. If None, file logging is disabled.
        console_level (int): Logging level for the console handler.
        file_level (int): Logging level for the file handler.
    """
    global logger # pylint: disable=global-statement

    # Reconfigure the global logger
    logger = ThreadedLogger(max_workers=2)
    logger.initialize_logger(name, log_to_console, log_to_file, console_level, file_level)
    logger.debug("Logger '%s' initialized.", name)

def initialize_global_logger_none(name="ApplicationLogger", log_to_console=True,
                      log_to_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Initialize the global logger instance with specific settings.

    Args:
        name (str): The name of the logger.
        log_to_console (bool): Whether to log to the console.
        log_to_file (Optional[str]): Path to the log file. If None, file logging is disabled.
        console_level (int): Logging level for the console handler.
        file_level (int): Logging level for the file handler.
    """
    global logger # pylint: disable=global-statement

    # Reconfigure the global logger
    logger = NoneLogger()
    logger.debug("Logger '%s' initialized.", name)


def create_separate_logger(name, log_to_console=True,
                           log_to_file=None, level=logging.INFO):
    """
    Create a separate logger for specific purposes.

    Args:
        name (str): The name of the new logger.
        log_to_console (bool): Whether to log to the console.
        log_to_file (Optional[str]): Path to the log file. If None, file logging is disabled.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: A new logger instance.
    """
    new_logger = logging.getLogger(name)
    new_logger.setLevel(level)

    # Avoid duplicate handlers
    if new_logger.hasHandlers():
        new_logger.handlers.clear()

    # Define formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        new_logger.addHandler(console_handler)

    # Add file handler if log_to_file is specified
    if log_to_file:
        file_handler = RotatingFileHandler(
            log_to_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        new_logger.addHandler(file_handler)

    return new_logger
