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
        self.frameWaitTime = 1
        self.showHelp = False
        self.showStats = False
        self.inputWidth = 640
        self.inputHeight = 480
        self.windowWidth = 640
        self.windowHeight = 480
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

                # Set attributes dynamically based on the settings found in the section
                for key, value in section.items():
                    setattr(self, key, value)
            else:
                logger.warning(f"No settings found for '{class_name}' in the file.")

        except FileNotFoundError:
            logger.warning(f"Setting file '{setting_file}' not found. Using default values.")
        except json.JSONDecodeError as e:
            logger.error(f"Setting file '{setting_file}' is not a valid JSON. Using default values.")
            logger.error(f"{e.msg} at line {e.lineno}, column {e.colno}")

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
            logger.info(f"Settings saved to '{setting_file}' successfully.")
        except IOError as e:
            logger.error(f"Unable to save settings to '{setting_file}'. {e}")
