import cv2


class FaceDetector:
    """
    A simple face detection class using OpenCV's Haar cascades.
    """

    def __init__(self):
        """
        Initialize the FaceDetector and load the Haar cascade model for frontal face detection.
        """
        self._face_cascade = self._initialize_cascade()

    def detect_faces(self, frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in a given image frame.

        Args:
            frame (numpy.ndarray): The input image frame in which to detect faces.
            scale_factor (float): Parameter specifying how much the image size is reduced at each image scale.
            min_neighbors (int): Parameter specifying how many neighbors each rectangle should have to retain it.
            min_size (tuple): Minimum possible object size (width, height).

        Returns:
            list: A list of rectangles where faces were detected. Each rectangle is represented as (x, y, w, h).
        """
        # Convert the frame to grayscale, as the face detection algorithm works on grayscale images.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image.
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        return faces

    def _initialize_cascade(self):
        """
        Load the Haar cascade model for face detection.

        Returns:
            cv2.CascadeClassifier: The loaded Haar cascade classifier.

        Raises:
            RuntimeError: If the Haar cascade model cannot be loaded.
        """
        #model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        model_path = "./models/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(model_path)

        if face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade model from path: {model_path}")

        return face_cascade
