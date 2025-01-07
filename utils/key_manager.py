"""
This module provides the `KeyManager` class for handling keyboard events in an application.
It allows registering keys with associated callbacks and descriptions, managing quit keys,
and providing a help text summary for the registered keys and their functions.
"""


from utils.logger import logger


class KeyManager:
    """
    Manages key bindings, callbacks, and descriptions for keyboard event handling.
    """

    def __init__(self):
        """
        Initialize the KeyManager.
        """
        self._key_callbacks = {}  # Maps keys to (callback, param) pairs.
        self._key_descriptions = {}  # Maps keys to descriptions.
        self._quit_keys = []  # List of keys that signal quitting.

    def register_key(self, key, description, callback,
                     param=None, is_quit_key=False):
        """
        Register a key with a callback and description.

        Args:
            key (int): The key code (ASCII) to register.
            description (str): A description of the key's function.
            callback (callable): The function to execute when the key is pressed.
            param (optional): An optional parameter to pass to the callback.
            is_quit_key (bool): If True, pressing this key will signal quitting.

        Raises:
            ValueError: If the key or callback is invalid.
        """
        if not callable(callback):
            raise ValueError("The callback must be a callable function or method.")

        if key in self._key_descriptions:
            logger.warning("%s is already registered.", chr(key))
            return

        self._key_descriptions[key] = description
        self._key_callbacks[key] = (callback, param)

        if is_quit_key:
            self._quit_keys.append(key)

    def check_events(self, key):
        """
        Check for key events and execute associated callbacks.

        Args:
            key (int): The key code (ASCII) to check.

        Returns:
            bool: False if the key is a quit key, True otherwise.
        """
        if key in self._key_callbacks:
            callback, param = self._key_callbacks[key]
            if param is not None:
                callback(param)
            else:
                callback()

        return key not in self._quit_keys

    def get_help_text(self):
        """
        Get a list of key descriptions for display or logging.

        Returns:
            list: A list of strings in the format "Key: Description".
        """
        return [f"{chr(key)}: {desc}" for key,
                desc in self._key_descriptions.items()]

    def print_help(self):
        """
        Print the help text to the console.
        """
        print("\n".join(self.get_help_text()))

    # Private Methods
    def _is_key_registered(self, key):
        """
        Check if a key is already registered.

        Args:
            key (int): The key code (ASCII) to check.

        Returns:
            bool: True if the key is registered, False otherwise.
        """
        return key in self._key_descriptions
