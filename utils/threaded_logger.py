from concurrent.futures import ThreadPoolExecutor
import logging
from logging.handlers import RotatingFileHandler

class ThreadedLogger:
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = None

    def initialize_logger(self, name="ApplicationLogger", log_to_console=True, log_to_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
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
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            file_handler = RotatingFileHandler(log_to_file, maxBytes=5 * 1024 * 1024, backupCount=3)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.logger.info(f"Threaded logger '{name}' initialized.")

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

# Usage Example
if __name__ == "__main__":
    threaded_logger = ThreadedLogger(max_workers=2)
    threaded_logger.initialize_logger(log_to_file="app.log")

    try:
        for i in range(10):
            threaded_logger.log(logging.INFO, f"Log message {i}")
    finally:
        threaded_logger.shutdown()
