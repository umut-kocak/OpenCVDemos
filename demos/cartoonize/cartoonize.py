"""
This module implements the CartoonizeDemo class, a video processing demo that applies various 
stylization effects to video frames in real-time. The demo supports multiple modes, allowing 
the user to cycle through different effects such as edge-preserving filtering, detail 
enhancement, pencil sketch, and stylization.

Usage:
Press 'm' during the demo to change the stylization mode.
"""
import cv2

from utils.base_module import BaseVideoDemo
from utils.face_detector import FaceDetector


class CartoonizeDemo(BaseVideoDemo):
    """
    Demo for applying different stylization filters to video frames.

    Modes:
    - 0: Edge-preserving filter
    - 1: Detail enhancement
    - 2: Pencil sketch (grayscale)
    - 3: Pencil sketch (color)
    - 4: Stylization
    """

    def __init__(self):
        """
        Initialize the CartoonizeDemo with default mode and total number of modes.
        """
        super().__init__()
        self._processing_mode = 0  # Current stylization mode
        self._nr_of_modes = 5  # Total number of stylization modes

    def process_frame(self, frame):
        """
        Process the given frame and apply the selected stylization effect.

        Args:
            frame: A video frame containing an `image` attribute to be stylized.

        Returns:
            The modified frame with the stylization effect applied.
        """
        mode = self._processing_mode % self._nr_of_modes
        current_sigma_s = getattr(self.settings.demo, 'sigma_s'+str(mode))
        current_sigma_r = getattr(self.settings.demo, 'sigma_r'+str(mode))
        match mode:
            case 0:
                frame.image = cv2.edgePreservingFilter(frame.image, flags=1, sigma_s=current_sigma_s, sigma_r=current_sigma_r)
            case 1:
                frame.image = cv2.detailEnhance(frame.image, sigma_s=current_sigma_s, sigma_r=current_sigma_r)
            case 2:
                frame.image, _ = cv2.pencilSketch(frame.image, sigma_s=current_sigma_s, sigma_r=current_sigma_r, shade_factor=0.05)
            case 3:
                _, frame.image = cv2.pencilSketch(frame.image, sigma_s=current_sigma_s, sigma_r=current_sigma_r, shade_factor=0.05)
            case 4:
                frame.image = cv2.stylization(frame.image, sigma_s=current_sigma_s, sigma_r=current_sigma_r)
            case _:  # Fallback case
                frame.image = cv2.edgePreservingFilter(frame.image, flags=1, sigma_s=60, sigma_r=0.4)

        return frame

    def register_keys(self):
        """
        Register keyboard keys and their corresponding handlers for the demo.

        - Press 'm' to cycle through the available stylization modes.
        """
        super(CartoonizeDemo, self).register_keys()

        def adjust_processing_mode(demo, delta):
            new_mode = max(1, getattr(demo, '_processing_mode') + delta)
            setattr(demo, '_processing_mode', new_mode % self._nr_of_modes)
    
        def adjust_sigma_s(settings, delta):
            key = 'sigma_s'+str(self._processing_mode % self._nr_of_modes)
            new_sigma = max(1, getattr(self.settings.demo, key) + delta)
            setattr(settings, key, new_sigma)

        def adjust_sigma_r(settings, delta):
            key = 'sigma_r'+str(self._processing_mode % self._nr_of_modes)
            new_sigma = max(0.01, getattr(self.settings.demo, key) + delta)
            setattr(settings, key, new_sigma)
        
        key_bindings = [
            # General keys
            (ord('m'), "Change the processing mode", lambda m: adjust_processing_mode(m, +1), self),

            # Sigma settings
            (ord('o'), "Increase current sigma_s.", lambda s: adjust_sigma_s(s, +1), self.settings.demo),
            (ord('l'), "Decrease current sigma_s.", lambda s: adjust_sigma_s(s, -1), self.settings.demo),
            (ord('p'), "Increase current sigma_r.", lambda s: adjust_sigma_r(s, +0.1), self.settings.demo),
            (ord(';'), "Decrease current sigma_r.", lambda s: adjust_sigma_r(s, -0.1), self.settings.demo),

        ]
    
        # Register all key bindings
        for key, description, callback, callback_arg, *args in key_bindings:
            self._key_manager.register_key(key, description, callback, callback_arg, *args, name_space=self.get_window_name())

    def get_window_name(self):
        """
        Return the name of the demo window.

        Returns:
            str: The name of the window.
        """
        return "Cartoonize Demo"


if __name__ == "__main__":
    demo = CartoonizeDemo()
    demo.run()
