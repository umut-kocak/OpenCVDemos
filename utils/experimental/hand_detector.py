"""
"""
import numpy as np
import math
import cv2
from abc import ABC, abstractmethod

import os
import sys
import logging
from absl import logging as absl_logging

# Suppress TensorFlow and MediaPipe Logs via Environment Variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow log level (suppress INFO, WARNING, and ERROR)
os.environ['MEDIAPIPE_DISABLE_LOGGING'] = '1'  # Disable MediaPipe-specific logging

# Suppress Python Warnings Globally
#import warnings
#warnings.filterwarnings("ignore")

# Suppress absl logs (MediaPipe uses absl)
#logging.getLogger('absl').setLevel(logging.FATAL)
#absl_logging.set_verbosity(absl_logging.FATAL)

# Redirect stderr and stdout globally (including low-level C/C++ logs)
def suppress_output():
    sys.stderr.flush()
    #sys.stdout.flush()
    stderr_fd = sys.stderr.fileno()
    #stdout_fd = sys.stdout.fileno()

    # Open /dev/null and redirect file descriptors
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, stderr_fd)  # Redirect stderr to /dev/null
    #os.dup2(devnull, stdout_fd)  # Redirect stdout to /dev/null

def restore_output():
    sys.stderr.flush()
    #sys.stdout.flush()
    sys.stderr = sys.__stderr__
    #sys.stdout = sys.__stdout__

# Suppress Output During MediaPipe Import
#suppress_output()
import mediapipe as mp
#restore_output()


class HandDetectionStrategy(ABC):
    """
    Abstract base class for hand detection strategies.
    """

    @abstractmethod
    def detect_hands(self, frame: np.ndarray, **kwargs):
        """
        """
        pass

class MediaPipeHandDetection(HandDetectionStrategy):
    """
    """

    def detect_hands(self, image: np.ndarray, **kwargs):
        """
        Detects hand gestures and draws annotations using MediaPipe.
        
        Parameters:
            image (numpy.ndarray): Input BGR image.
        
        Returns:
            numpy.ndarray: Annotated image with hand gestures and landmarks.
        """
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
            # Convert the BGR image to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            # Process the image to detect hands
            results = hands.process(rgb_image)
        
            # Annotate the image if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
        
                    # Convert normalized landmark coordinates to pixel values
                    height, width, _ = image.shape
                    landmarks = [
                        mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                            landmark.x, landmark.y, width, height
                        )
                        for landmark in hand_landmarks.landmark
                    ]
        
                    if landmarks and None not in landmarks:
                        # Recognize the gesture
                        gesture = self._recognize_hand_gesture(hand_landmarks.landmark)
                        cv2.putText(
                            image,
                            f"Gesture: {gesture}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
        return image

    def _recognize_hand_gesture(self, landmarks):
        """
        Recognizes basic hand gestures based on the position of hand landmarks.
    
        Parameters:
            landmarks (list): List of normalized hand landmark positions.
    
        Returns:
            str: Recognized gesture name.
        """
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
        elif all([not thumb_open, index_open, middle_open, ring_open, pinky_open]):
            return "Four"
        elif all([not thumb_open, index_open, not middle_open, not ring_open, not pinky_open]):
            return "One"
        elif all([thumb_open, not index_open, not middle_open, not ring_open, not pinky_open]):
            return "Thumbs Up"
        elif not any([thumb_open, index_open, middle_open, ring_open, pinky_open]):
            return "Fist"
    
        return "Unknown Gesture"        


class ContourBasedHandDetection(HandDetectionStrategy):
    """
    """

    def detect_hands(self, image: np.ndarray, **kwargs):
        """
        Detects hand gestures, poses, and finger angles in an image.
        
        Parameters:
            image (numpy.ndarray): Input image.
            
        Returns:
            Annotated image with contours, convex hull, and recognized gestures.
        """
        # Convert the image to HSV color space and apply skin color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return image  # Return the original image if no hand is detected
    
        # Find the largest contour (assuming it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 2000:  # Ignore small contours
            return image
    
        # Draw the contour
        cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    
        # Convex hull
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        hull_points = cv2.convexHull(largest_contour)
        cv2.drawContours(image, [hull_points], -1, (255, 0, 0), 2)
    
        # Find convexity defects
        defects = cv2.convexityDefects(largest_contour, hull)
        if defects is not None:
            count_fingers = 0
            for i in range(defects.shape[0]):
                start_idx, end_idx, far_idx, depth = defects[i, 0]
                start = tuple(largest_contour[start_idx][0])
                end = tuple(largest_contour[end_idx][0])
                far = tuple(largest_contour[far_idx][0])
    
                # Calculate the angle between the fingers
                a = math.dist(start, end)
                b = math.dist(start, far)
                c = math.dist(end, far)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    
                # If the angle is less than 90 degrees, it is considered a finger
                if angle <= math.pi / 2:
                    count_fingers += 1
                    cv2.circle(image, far, 8, (0, 0, 255), -1)
    
                # Draw lines for the defect points
                cv2.line(image, start, end, (0, 255, 255), 2)
                cv2.circle(image, start, 8, (255, 255, 0), -1)
                cv2.circle(image, end, 8, (255, 255, 0), -1)
    
            # Gesture recognition
            gesture = ""
            if count_fingers == 0:
                gesture = "Fist"
            elif count_fingers == 1:
                gesture = "One"
            elif count_fingers == 2:
                gesture = "Two"
            elif count_fingers == 3:
                gesture = "Three"
            elif count_fingers == 4:
                gesture = "Four"
            elif count_fingers == 5:
                gesture = "Five"
    
            # Display the gesture
            cv2.putText(image, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        return image


class HandDetector:
    """
    A hand detection class that uses a pluggable detection strategy.
    """

    def __init__(self, strategy: HandDetectionStrategy):
        """
        Initialize the HandDetector with a specific detection strategy.

        Args:
            strategy (HnadDetectionStrategy): The hand detection strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: HandDetectionStrategy):
        """
        Set a new detection strategy.

        Args:
            strategy (HandDetectionStrategy): The new detection strategy to use.
        """
        self._strategy = strategy

    def detect_hands(self, frame: np.ndarray, **kwargs):
        """
        Detect hands using the current strategy.

        Args:
            frame (np.ndarray): The input image frame.
            **kwargs: Additional parameters specific to the detection strategy.

        """
        return self._strategy.detect_hands(frame, **kwargs)

