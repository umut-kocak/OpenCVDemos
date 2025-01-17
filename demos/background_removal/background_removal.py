"""
This module implements a `FaceDetectionDemo` class that extends the `BaseVideoDemo` framework to
perform real-time face detection in video streams. It uses the `FaceDetector` utility to identify
faces in each frame and overlays rectangles around detected faces. The demo initializes a logger
for tracking runtime events and provides a user interface window for visualizing the processed
video output.
"""
import logging
import os
import time

import cv2
import numpy as np
import torch
from torchvision import models, transforms

from utils.logger_initializer import initialize_logger

# Initialize the global logger before importing other modules
logger_name = os.path.splitext(os.path.basename(__file__))[0]
logger_file_name = logger_name + time.strftime("%Y%m%d-%H%M%S") + ".log"

initialize_logger(
    logger_name,
    logger_file_name,
    _log_to_console=True,
    _log_to_file=logger_file_name,
    _console_level=logging.DEBUG,
    _file_level=logging.DEBUG
)


from utils.base_module import BaseVideoDemo # pylint: disable=C0413


# Load the pre-trained DeepLabV3 model from torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
model.eval()

# Preprocessing transform for the DeepLabV3 model
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def segment_frame(frame):
    """Segment the input frame using DeepLabV3."""
    input_tensor = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

def blend_background(frame, mask, background):
    """Blend the segmented frame with a custom background."""
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask_binary = (mask_resized == 15).astype(np.uint8)  # Class 15 corresponds to "person"

    # Ensure background matches the frame size
    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Blend the frame and the background
    blended_frame = np.where(mask_binary[:, :, None] == 1, frame, background_resized)
    return blended_frame

class BackgroundRemoval(BaseVideoDemo):
    """Demo for replacing the background in video frames."""

    def __init__(self, source=0):
        super().__init__(source)
        
        # Load a custom background image
        self.background = cv2.imread('background.jpg')  # Replace with your custom background image
        if background is None:
            print("Background image not found!")

    def process_frame(self, frame):
        """Replaces the background of the frame."""

        # Perform segmentation
        mask = segment_frame(frame)

        # Blend the frame with the background
        output_frame = blend_background(frame, mask, self.background)
        
        return output_frame


    def get_window_name(self):
        """Return the name of the demo window."""
        return "Real-Time Background Removal"

def main():
    # Load a custom background image
    background = cv2.imread('background.jpg')  # Replace with your custom background image
    if background is None:
        print("Background image not found!")
        return

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform segmentation
        mask = segment_frame(frame)

        # Blend the frame with the background
        output_frame = blend_background(frame, mask, background)

        # Display the result
        cv2.imshow('Real-Time Background Removal', output_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = FaceDetectionDemo()
    demo.run()
