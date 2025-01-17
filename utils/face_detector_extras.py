import cv2
import numpy as np
from abc import ABC, abstractmethod

import dlib
from scipy.spatial import distance
from abc import ABC, abstractmethod
import numpy as np
import cv2
from torchvision import transforms


class FaceNetDetection(FaceDetectionStrategy):
    """
    Face detection using a pre-trained FaceNet model.
    """

    def __init__(self, model_path="./models/facenet_model.pt"):
        """
        Initialize the FaceNet-based face detection.

        Args:
            model_path (str): Path to the pre-trained FaceNet model file.
        """
        import torch
        from facenet_pytorch import InceptionResnetV1

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = InceptionResnetV1(pretrained='vggface2').eval().to(self._device)
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_faces(self, frame: np.ndarray, **kwargs):
        """
        Detect faces using FaceNet embeddings.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            list: A list of bounding boxes representing detected faces.
        """
        raise NotImplementedError("FaceNet detection requires a separate face detector for bounding boxes. Use DNN or Haar first.")


class DlibFaceDetection(FaceDetectionStrategy):
    """
    Face detection using Dlib's CNN-based face detector.
    """

    def __init__(self):
        """
        Initialize Dlib-based face detection.
        """
        self._detector = dlib.get_frontal_face_detector()

    def detect_faces(self, frame: np.ndarray, upsample=1):
        """
        Detect faces using Dlib's face detector.

        Args:
            frame (np.ndarray): The input image frame.
            upsample (int): Number of times to upsample the image before detecting.

        Returns:
            list: A list of bounding boxes representing detected faces.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._detector(gray, upsample)
        faces = [(d.left(), d.top(), d.width(), d.height()) for d in detections]
        return faces


class FaceRecognition(FaceDetectionStrategy):
    """
    A strategy for recognizing faces using embeddings and comparison.
    """

    def __init__(self, known_faces_embeddings, known_faces_labels):
        """
        Initialize face recognition.

        Args:
            known_faces_embeddings (list): Pre-computed face embeddings for known individuals.
            known_faces_labels (list): Corresponding labels for the known faces.
        """
        self.known_faces_embeddings = known_faces_embeddings
        self.known_faces_labels = known_faces_labels

    def detect_faces(self, frame: np.ndarray, embedding_model, **kwargs):
        """
        Recognize faces using embeddings.

        Args:
            frame (np.ndarray): The input image frame.
            embedding_model: The embedding model for feature extraction.

        Returns:
            list: A list of recognized face labels or "Unknown".
        """
        face_embeddings = embedding_model.extract_embeddings(frame)
        recognized_faces = []
        for embedding in face_embeddings:
            distances = [distance.euclidean(embedding, known_embedding) for known_embedding in self.known_faces_embeddings]
            min_distance = min(distances)
            if min_distance < 0.6:  # Threshold for face recognition
                recognized_faces.append(self.known_faces_labels[distances.index(min_distance)])
            else:
                recognized_faces.append("Unknown")
        return recognized_faces
