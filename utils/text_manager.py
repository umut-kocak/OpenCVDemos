"""
This module provides classes for managing text properties and rendering text on images using OpenCV.
`TextProperties` defines styling options such as color, font, and spacing, while `TextManager` facilitates
drawing text with customizable or predefined properties, supporting multi-line text rendering and
named property configurations.
"""
from dataclasses import dataclass

import cv2

@dataclass
class TextProperties:
    """
    Represents the styling properties for drawing text on an image.
    Includes options for color, font, thickness, spacing, and more.
    """
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)

    def __init__(self, y_spacing=20, color=WHITE, thickness=1, font_scale=0.5,
                 font_face=cv2.FONT_HERSHEY_SIMPLEX, line_type=cv2.LINE_AA):
        """
        Initialize text properties.

        Args:
            y_spacing (int): Vertical spacing between lines of text.
            color (tuple): Color of the text (B, G, R).
            thickness (int): Thickness of the text.
            font_scale (float): Scale factor for the font size.
            font_face (int): Font face type from cv2 (e.g., cv2.FONT_HERSHEY_SIMPLEX).
            line_type (int): Line type for drawing text (e.g., cv2.LINE_AA).
        """
        self.y_spacing = y_spacing
        self.color = color
        self.thickness = thickness
        self.font_scale = font_scale
        self.font_face = font_face
        self.line_type = line_type


class TextManager:
    """
    Manages text drawing on images, allowing customization through
    default or named text properties.
    """

    def __init__(self, default_properties=None):
        """
        Initialize the TextManager.

        Args:
            default_properties (TextProperties, optional): Default text properties.
            If None, a default instance of TextProperties is used.
        """
        self._default_properties = default_properties if default_properties else TextProperties()
        self._registered_properties = {}

    def set_default_properties(self, properties):
        """
        Set new default text properties.

        Args:
            properties (TextProperties): The new default text properties.

        Raises:
            ValueError: If the provided properties are not an instance of TextProperties.
        """
        if not isinstance(properties, TextProperties):
            raise ValueError(
                "default_properties must be an instance of TextProperties")
        self._default_properties = properties

    def register_properties(self, name, properties):
        """
        Register a named set of text properties for later use.

        Args:
            name (str): The name to associate with the text properties.
            properties (TextProperties): The text properties to register.

        Raises:
            ValueError: If the provided properties are not an instance of TextProperties.
        """
        if not isinstance(properties, TextProperties):
            raise ValueError(
                "properties must be an instance of TextProperties")
        self._registered_properties[name] = properties

    def draw_text(self, image, text, pos=(10, 20), properties=None):
        """
        Draw text on an image using the specified or default text properties.

        Args:
            image (ndarray): The target image to draw the text on.
            text (str, list, tuple): The text to draw. Supports multi-line text.
            pos (tuple): (x, y) coordinates for the starting position of the text.
            properties (str or TextProperties, optional): The text properties to use.
                - If a string, uses the registered properties with the given name.
                - If an instance of TextProperties, uses the given properties.
                - If None, uses the default properties.

        Raises:
            ValueError: If the properties argument is invalid.

        Supports rendering multiple lines if the text is a list or tuple. Each line
        is spaced vertically by the `y_spacing` value in the selected TextProperties.
        """
        props = self._select_properties(properties)
        x, y = pos

        if isinstance(text, (list, tuple)):
            for line in text:
                cv2.putText(
                    img=image,
                    text=line,
                    org=(x, y),
                    fontFace=props.font_face,
                    fontScale=props.font_scale,
                    color=props.color,
                    thickness=props.thickness,
                    lineType=props.line_type
                )
                y += props.y_spacing
        else:
            cv2.putText(
                img=image,
                text=text,
                org=(x, y),
                fontFace=props.font_face,
                fontScale=props.font_scale,
                color=props.color,
                thickness=props.thickness,
                lineType=props.line_type
            )

    def _select_properties(self, properties):
        """
        Select the appropriate text properties based on the input.

        Args:
            properties (str or TextProperties or None): The text properties to select.

        Returns:
            TextProperties: The selected text properties.

        Raises:
            ValueError: If the input is not valid.
        """
        if isinstance(properties, str):
            return self._registered_properties.get(
                properties, self._default_properties)
        if isinstance(properties, TextProperties):
            return properties
        if properties is None:
            return self._default_properties
        raise ValueError("Invalid properties argument. Must be None, a string, or TextProperties.")
