"""
This module provides the `HandDetector` class for detecting hands in images using different strategies
It initializes the required model and processes image frames to identify hand gestures.

Usage:
    Instantiate the `HandDetector` class with a strategy and use it to detect hands in image frames,
    returning results with some joint information.
"""
from abc import ABC, abstractmethod
import concurrent.futures

import cv2
import mediapipe as mp
import numpy as np

from utils.logger import logger
from utils.thread_pool import get_thread_pool

class HandDetectionStrategy(ABC):
    """
    Abstract base class for hand detection strategies.
    """

    @abstractmethod
    def detect_hands(self, image: np.ndarray, **kwargs):
        """
        Detect hands in the given image.

        Parameters:
            image (numpy.ndarray): Input image.
        """


class MediaPipeHandDetection(HandDetectionStrategy):
    """
    Hand detection strategy using Google's MediaPipe library.
    """

    def __init__(self):
        """
        Initialize MediaPipe components for hand detection and drawing utilities.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.detection_results = None
        self.detection_future = None

    def detect_hands(self, image: np.ndarray, **kwargs):
        """
        Detects hand gestures and draws annotations using MediaPipe.
        
        Parameters:
            image (numpy.ndarray): Input BGR image.
        """
        self._update_detection(image)
        if not self.has_results():
            return

        for hand_landmarks in self.detection_results.multi_hand_landmarks:
            # Convert normalized landmark coordinates to pixel values
            height, width, _ = image.shape
            landmarks = [ # todo : Not used is it needed.
                mp.solutions.drawing_utils._normalized_to_pixel_coordinates( # pylint: disable=W0212
                    landmark.x, landmark.y, width, height
                )
                for landmark in hand_landmarks.landmark
            ]

            if kwargs.get('draw_hands', True):
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

    def has_results(self):
        """
        Check if the detection has valid results.

        Returns:
            bool: True if results are available, False otherwise.
        """
        if self.detection_results is None or self.detection_results.multi_hand_landmarks is None:
            return False
        return True

    def _update_detection(self, image: np.ndarray):
        """
        Run hand detection asynchronously using a thread pool.

        Parameters:
            image (numpy.ndarray): Input image.
        """
        def detection_task():
            with self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
                return hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return None

        if self.detection_future is None:
        # If not enabled yet trigger detection
            self.detection_future = get_thread_pool().submit_task(detection_task)

        try:
            # Check if detection is finished
            res = self.detection_future.result(timeout=0.0)
            if res is None:
                logger.warning("Hand detection returned None")
            else:
                # Update the results
                self.detection_results = res
            # Trigger the next detection
            self.detection_future = get_thread_pool().submit_task(detection_task)
        except concurrent.futures.TimeoutError :
            # Detection still ongoing
            pass
        except concurrent.futures.CancelledError as e:
            logger.error("Hand detection is cancelled: %s", e)
        except Exception as e:
            logger.error("Exception during detecting hands task: %s", e)

    def recognize_hand_gestures(self):
        """
        Recognizes gestures based on detected hand landmarks.

        Returns:
            dict: Dictionary mapping hand labels to recognized gestures.
        """
        if not self.has_results():
            return {}
        gestures = {}
        for i, hand_classification in enumerate(self.detection_results.multi_handedness):
            if hand_classification.classification:
                classification = hand_classification.classification[0]
                hand_label = classification.label  # 'Left' or 'Right'
                gestures[hand_label] = self._recognize_hand_gesture(
                    self.detection_results.multi_hand_landmarks[i].landmark)
            else:
                logger.warning("Warning: Hand detected but classification is empty!")
        return gestures

    def _recognize_hand_gesture(self, landmarks):
        """
        Recognizes basic hand gestures based on the position of hand landmarks.
    
        Parameters:
            landmarks (list): List of normalized hand landmark positions.
    
        Returns:
            str: Recognized gesture name.
        """
        if landmarks and None not in landmarks:
            # Assign landmark indices for convenience
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            thumb_mcp = landmarks[2]
            wrist = landmarks[0]

            # Calculate distances or relative positions
            thumb_open = thumb_tip.x > thumb_mcp.x if thumb_mcp.x < wrist.x else thumb_tip.x < thumb_mcp.x
            index_open = index_tip.y < landmarks[6].y  # Index finger is raised
            middle_open = middle_tip.y < landmarks[10].y
            ring_open = ring_tip.y < landmarks[14].y
            pinky_open = pinky_tip.y < landmarks[18].y

            # Determine gesture based on which fingers are open
            if all([thumb_open, index_open, middle_open, ring_open, pinky_open]):
                return "Open Hand"
            if all([not thumb_open, index_open, middle_open, ring_open, pinky_open]):
                return "Four"
            if all([not thumb_open, index_open, not middle_open, not ring_open, not pinky_open]):
                return "One"
            if all([thumb_open, not index_open, not middle_open, not ring_open, not pinky_open]):
                return "Thumbs Up"
            if not any([thumb_open, index_open, middle_open, ring_open, pinky_open]):
                return "Fist"

        return "Unknown Gesture"

class HandDetector:
    """
    A hand detection class that uses a pluggable detection strategy.
    """

    def __init__(self, strategy: HandDetectionStrategy):
        """
        Initialize the HandDetector with a specific detection strategy.

        Args:
            strategy (HandDetectionStrategy): The hand detection strategy to use.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: HandDetectionStrategy):
        """
        Set a new detection strategy.

        Args:
            strategy (HandDetectionStrategy): The new detection strategy to use.
        """
        self.strategy = strategy
