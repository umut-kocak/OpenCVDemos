"""
This module provides the `FaceDetector` class for detecting faces in images using different strategies
It initializes the required model and processes image frames to identify face regions.

Usage:
    Instantiate the `FaceDetector` class with a strategy and use it to detect faces in image frames,
    returning their bounding rectangles.
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np

class FaceDetectionStrategy(ABC):
    """
    Abstract base class for face detection strategies.
    """

    @abstractmethod
    def detect_faces(self, frame: np.ndarray, **kwargs):
        """
        Detect faces in the given frame.

        Args:
            frame (np.ndarray): The input image frame.
            **kwargs: Additional parameters specific to the detection strategy.

        Returns:
            list: A list of rectangles where faces were detected, each represented as (x, y, w, h).
        """


class HaarCascadeFaceDetection(FaceDetectionStrategy):
    """
    Face detection using OpenCV's Haar cascades.
    """

    def __init__(self, model_path="./models/face_detection/haarcascade_frontalface_default.xml"):
        """
        Initialize Haar cascade face detection.

        Args:
            model_path (str): Path to the Haar cascade model file.
        """
        self._face_cascade = cv2.CascadeClassifier(model_path)
        if self._face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade model from path: {model_path}")

    def detect_faces(self, frame: np.ndarray, **kwargs):
        """
        Detect faces using Haar cascades.

        Args:
            frame (np.ndarray): The input image frame.
            scale_factor (float): Image scale factor.
            min_neighbors (int): Minimum neighbors to retain a detection.
            min_size (tuple): Minimum face size.

        Returns:
            list: A list of rectangles where faces were detected.
        """
        scale_factor = kwargs.get('scale_factor', 1.1)
        min_neighbors = kwargs.get('min_neighbors', 5)
        min_size = kwargs.get('min_size', (30, 30))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        return faces


class OCVDnnFaceDetection(FaceDetectionStrategy):
    """
    Face detection using OpenCV's deep learning-based face detector.
    """

    def __init__(self, model_path="./models/face_detection/res10_300x300_ssd_iter_140000.caffemodel",
                 config_path="./models/face_detection/deploy.proto.txt"):
        """
        Initialize DNN face detection.

        Args:
            model_path (str): Path to the pre-trained model weights.
            config_path (str): Path to the model configuration file.
        """
        self._net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    def detect_faces(self, frame: np.ndarray, **kwargs):
        """
        Detect faces using a DNN-based face detector.

        Args:
            frame (np.ndarray): The input image frame.
            confidence_threshold (float): Minimum confidence for face detection.

        Returns:
            list: A list of rectangles where faces were detected.
        """
        confidence_threshold = kwargs.get('confidence_threshold', 0.6)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self._net.setInput(blob)
        detections = self._net.forward()
        faces = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                faces.append((start_x, start_y, end_x - start_x, end_y - start_y))

        return faces

class FaceDetector:
    """
    A face detection class that uses a pluggable detection strategy.
    """

    def __init__(self, strategy: FaceDetectionStrategy):
        """
        Initialize the FaceDetector with a specific detection strategy.

        Args:
            strategy (FaceDetectionStrategy): The face detection strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FaceDetectionStrategy):
        """
        Set a new detection strategy.

        Args:
            strategy (FaceDetectionStrategy): The new detection strategy to use.
        """
        self._strategy = strategy

    def detect_faces(self, frame: np.ndarray, **kwargs):
        """
        Detect faces using the current strategy.

        Args:
            frame (np.ndarray): The input image frame.
            **kwargs: Additional parameters specific to the detection strategy.

        Returns:
            list: A list of rectangles where faces were detected.
        """
        return self._strategy.detect_faces(frame, **kwargs)
