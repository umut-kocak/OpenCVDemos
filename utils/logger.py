import logging
from logging.handlers import RotatingFileHandler

# Define a default global logger with basic configuration
logger = None

def initialize_logger(name="ApplicationLogger", log_to_console=True, log_to_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Initialize the global logger instance with specific settings.

    Args:
        name (str): The name of the logger.
        log_to_console (bool): Whether to log to the console.
        log_to_file (Optional[str]): Path to the log file. If None, file logging is disabled.
        console_level (int): Logging level for the console handler.
        file_level (int): Logging level for the file handler.
    """
    global logger

    # Reconfigure the global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logger's level to the lowest level you want to capture.

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
        console_handler.setLevel(console_level)  # Set level for console handler
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

    logger.info(f"Logger '{name}' initialized.")



def create_separate_logger(name, log_to_console=True, log_to_file=None, level=logging.INFO):
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
