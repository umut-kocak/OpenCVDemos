"""
This script defines a `Settings` class for managing application configuration parameters.
It supports initializing with default values, loading settings from JSON files, and saving the
current settings back to a JSON file. The configuration is organized under a section named after
the class for better modularity and reusability.
"""
import json

from utils.logger import logger


class Settings:
    """
    Initializes the Settings class with default values. Optionally loads settings
    from a JSON file if provided.

    Args:
        setting_files (array of str, optional): Path to a JSON files containing setting
                                      parameters. If not provided, default values are used.
    """

    def __init__(self, setting_files: str = None):
        self.frame_wait_time = 1
        self.show_help = False
        self.show_stats = False
        self.input_width = 640
        self.input_height = 480
        self.window_width = 640
        self.window_height = 480
        self.resizable = True
        self.fit_frame_to_window_size = False

        # Load settings from JSON file if provided
        if setting_files:
            for setting_file in setting_files:
                if setting_file:
                    self.load_from_json(setting_file)

    def load_from_json(self, setting_file: str):
        """
        Loads setting parameters from a JSON file, overwriting default values.

        Args:
            setting_file (str): Path to the JSON file containing setting parameters.

        Raises:
            FileNotFoundError: If the specified JSON file is not found.
            json.JSONDecodeError: If the JSON file is not valid.
        """
        try:
            with open(setting_file, 'r', encoding='utf-8') as f:
                setting_data = json.load(f)

            # Find the section that matches the class name and loop within it
            class_name = self.__class__.__name__
            if class_name in setting_data:
                section = setting_data[class_name]

                # Set attributes dynamically based on the settings found in the
                # section
                for key, value in section.items():
                    setattr(self, key, value)
            else:
                logger.warning("No settings found for '%s' in the file.", class_name)

        except FileNotFoundError:
            logger.warning("Setting file '%s' not found. Using default values.", setting_file)
        except json.JSONDecodeError as e:
            logger.error("Setting file '%s' is not a valid JSON. Using default values.", setting_file)
            logger.error("%s at line %d, column %d", e.msg, e.lineno, e.colno)

    def save_to_json(self, setting_file: str):
        """
        Saves the current settings to a JSON file, storing them under a section named
        the same as the class name.

        Args:
            setting_file (str): Path to the JSON file where the settings should be saved.
        """
        try:
            # Get the class name for the section
            class_name = self.__class__.__name__

            # Prepare the data to be saved under the class name section
            settings_to_save = {class_name: self.__dict__}

            with open(setting_file, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, indent=4)
            logger.info("Settings saved to '%s' successfully.", setting_file)
        except IOError as e:
            logger.error("Unable to save settings to '%s'. %s", setting_file, e)
