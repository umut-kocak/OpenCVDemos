"""
This script defines a `Settings` class for managing application configuration parameters.
It supports initializing with default values, loading settings from JSON files, and saving the
current settings back to a JSON file. The configuration is organized under a section named after
the class for better modularity and reusability.
"""
import json
import numpy as np
from utils.logger import logger

class SubSection:
    """
    Represents a subsection of settings. Supports recursive merging of settings.
    """

    def __init__(self, initial_data=None):
        """
        Initialize the SubSection with optional initial data.
        """
        self._data = {}
        if initial_data:
            self.update(initial_data)

    def update(self, new_data):
        """
        Updates the SubSection with new data, merging recursively if needed.
        """
        for key, value in new_data.items():
            if isinstance(value, dict):
                # If the key already exists and is a SubSection, merge recursively
                if key in self._data and isinstance(self._data[key], SubSection):
                    self._data[key].update(value)
                else:
                    self._data[key] = SubSection(value)
            else:
                self._data[key] = value

    def to_dict(self):
        """
        Converts the SubSection to a dictionary.
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value.to_dict() if isinstance(value, SubSection) else value
        return result

    def __getattr__(self, key):
        return self._data.get(key)

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __repr__(self):
        return repr(self._data)

class Settings:
    """
    A class for managing application settings with support for subsections and recursive merging.
    """

    def __init__(self, setting_files=None):
        """
        Initializes the Settings class with default values.
        """
        if setting_files:
            for setting_file in setting_files:
                if setting_file:
                    self.load_from_json(setting_file)

    def load_from_json(self, setting_file: str):
        """
        Loads setting parameters from a JSON file, recursively merging subsections.
        """
        try:
            with open(setting_file, 'r', encoding='utf-8') as f:
                setting_data = json.load(f)

            # Find the section matching the class name
            class_name = self.__class__.__name__
            if class_name in setting_data:
                section = setting_data[class_name]
                self._merge_settings(self, section)
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
        after the class name.
        """
        try:
            class_name = self.__class__.__name__
            settings_to_save = {class_name: self._to_dict(self)}

            with open(setting_file, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, indent=4)
            logger.info("Settings saved to '%s' successfully.", setting_file)
        except IOError as e:
            logger.error("Unable to save settings to '%s'. %s", setting_file, e)

    def _merge_settings(self, parent, new_data):
        """
        Recursively merges settings into the parent object.
        """
        for key, value in new_data.items():
            if isinstance(value, dict):
                key = key.lower()  # Convert section names to lowercase
                # Handle subsections
                if hasattr(parent, key):
                    existing_value = getattr(parent, key)
                    if isinstance(existing_value, SubSection):
                        existing_value.update(value)
                    else:
                        setattr(parent, key, SubSection(value))
                else:
                    setattr(parent, key, SubSection(value))
            else:
                setattr(parent, key, value)

    def _to_dict(self, obj):
        """
        Converts a Settings object to a dictionary, including SubSections.
        """
        result = {}
        for key, value in vars(obj).items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, SubSection):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
