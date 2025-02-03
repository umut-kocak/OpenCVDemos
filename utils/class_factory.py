"""
A factory class to create and retrieve classes based on a provided catalog of strategies.
"""
from utils.logger import logger

class ClassFactory():
    """
    A factory class to create and retrieve classes based on a provided catalog of strategies.

    The class catalog should be a dictionary where keys are strategy identifiers, and values
    are tuples containing the class reference and a dictionary of default arguments for the class.

    Example:
        class_catalog = {
            "strategy_a": (StrategyAClass, {"arg1": 10, "arg2": 20}),
            "strategy_b": (StrategyBClass, {"arg1": 5})
        }
        factory = ClassFactory(class_catalog)

        strategy_a_instance = factory.create_class("strategy_a", arg1=15)
        strategy_b_class = factory.get_class("strategy_b")
    """

    def __init__(self, class_catalog):
        """
        Initialize the factory with a catalog of classes and their default arguments.

        Args:
            class_catalog (dict): A dictionary mapping strategy keys to tuples of 
                                  (class_reference, default_arguments).
        """
        super().__init__()
        self._catalog = class_catalog

    def get_class(self, strategy_key):
        """
        Retrieve a class reference from the catalog based on the provided strategy key.

        Args:
            strategy_key (str): The key identifying the strategy.

        Returns:
            type: The class reference associated with the strategy key.

        Raises:
            ValueError: If the strategy key does not exist in the catalog.
        """
        entry = self._catalog.get(strategy_key)
        if entry is None:
            raise ValueError(
                f"Invalid strategy key: {strategy_key}. Available keys: {list(self._catalog.keys())}"
            )
        strategy_class, _ = entry
        return strategy_class

    def create_class(self, strategy_key, **kwargs):
        """
        Create an instance of a class based on the strategy key and provided arguments.

        Args:
            strategy_key (str): The key identifying the strategy.
            **kwargs: Arguments to override the default arguments for the class instantiation.

        Returns:
            object: An instance of the class associated with the strategy key.

        Raises:
            ValueError: If the strategy key does not exist in the catalog.
        """
        entry = self._catalog.get(strategy_key)
        if entry is None:
            raise ValueError(
                f"Invalid strategy key: {strategy_key}. Available keys: {list(self._catalog.keys())}"
            )
        strategy_class, default_arguments = entry

        if kwargs is not None and len(kwargs) > 0:
            logger.debug("Creating class '%s' with given parameters: %s", strategy_key, kwargs)
            return strategy_class(**kwargs)

        # Fallback to default parameters
        logger.debug("Creating class '%s' with default parameters: %s", strategy_key, default_arguments)
        return strategy_class(**default_arguments)
