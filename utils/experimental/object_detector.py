import cv2
import numpy as np
from yolov5 import YOLOv5  # For YOLOv5 integration, requires YOLOv5 Python library
from sort import Sort  # SORT tracking algorithm

class ObjectDetectorAndTracker:
    """
    Object Detection and Tracking using YOLO or SSD, integrated with SORT for multi-object tracking.
    """

    def __init__(self, model_type="yolo", yolo_weights="yolov5s.pt", ssd_prototxt=None, ssd_model=None, confidence_threshold=0.5):
        """
        Initialize the object detector and tracker.
        
        Args:
            model_type (str): Type of model to use ("yolo" or "ssd").
            yolo_weights (str): Path to YOLOv5 weights file (required for YOLO).
            ssd_prototxt (str): Path to SSD prototxt file (required for SSD).
            ssd_model (str): Path to SSD model file (required for SSD).
            confidence_threshold (float): Minimum confidence for detections.
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.tracker = Sort()  # SORT tracker instance

        if model_type == "yolo":
            self.detector = YOLOv5(yolo_weights)  # YOLOv5 detector instance
        elif model_type == "ssd":
            if not ssd_prototxt or not ssd_model:
                raise ValueError("SSD requires both prototxt and model paths.")
            self.net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_model)
        else:
            raise ValueError("Invalid model type. Choose 'yolo' or 'ssd'.")

    def process_frame(self, frame):
        """
        Detect and track objects in a video frame.

        Args:
            frame (numpy.ndarray): Input video frame.

        Returns:
            numpy.ndarray: Frame with detections and tracking information drawn.
        """
        detections = self.detect_objects(frame)
        tracked_objects = self.track_objects(detections)

        # Draw bounding boxes and labels
        for obj_id, x1, y1, x2, y2, category in tracked_objects:
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {obj_id} {category}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def detect_objects(self, frame):
        """
        Detect objects in a frame using the selected model.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            list: List of detections in the format [x1, y1, x2, y2, confidence, class_id].
        """
        detections = []
        if self.model_type == "yolo":
            results = self.detector.predict(frame)
            for result in results:
                x1, y1, x2, y2, conf, class_id = result[:6]
                if conf > self.confidence_threshold:
                    detections.append([int(x1), int(y1), int(x2), int(y2), conf, class_id])

        elif self.model_type == "ssd":
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            outputs = self.net.forward()

            for i in range(outputs.shape[2]):
                confidence = outputs[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    class_id = int(outputs[0, 0, i, 1])
                    box = outputs[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    detections.append([x1, y1, x2, y2, confidence, class_id])

        return detections

    def track_objects(self, detections):
        """
        Track objects across frames using SORT.

        Args:
            detections (list): List of detections in the format [x1, y1, x2, y2, confidence, class_id].

        Returns:
            list: List of tracked objects in the format [id, x1, y1, x2, y2, category].
        """
        if len(detections) == 0:
            return []

        # Convert detections to SORT format [x1, y1, x2, y2, score]
        sort_detections = np.array([[det[0], det[1], det[2], det[3], det[4]] for det in detections])
        tracked_objects = self.tracker.update(sort_detections)

        # Map tracking IDs back to categories
        results = []
        for track in tracked_objects:
            track_id, x1, y1, x2, y2 = map(int, track[:5])
            # Map the ID to the category (defaulting to "Unknown" if missing)
            category = next((det[5] for det in detections if det[0] == x1 and det[1] == y1), "Unknown")
            results.append((track_id, x1, y1, x2, y2, category))

        return results


# Example usage
if __name__ == "__main__":
    # Initialize detector with YOLO or SSD
    detector_tracker = ObjectDetectorAndTracker(
        model_type="yolo",
        yolo_weights="yolov5s.pt",  # Replace with your YOLOv5 weights
        confidence_threshold=0.5
    )

    # Capture video from webcam or file
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for detection and tracking
        output_frame = detector_tracker.process_frame(frame)

        # Display the output
        cv2.imshow("Object Detection and Tracking", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
