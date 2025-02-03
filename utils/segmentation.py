"""
This script provides a `Segmentation` class that uses a pre-trained DeepLabV3 model 
for semantic segmentation. It supports GPU acceleration if available.
"""
from enum import Enum
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

class SegmentationClass(Enum):
    """
    Enum representing segmentation classes based on the COCO dataset.
    """
    BACKGROUND = 0
    AEROPLANE = 1
    BICYCLE = 2
    BIRD = 3
    BOAT = 4
    BOTTLE = 5
    BUS = 6
    CAR = 7
    CAT = 8
    CHAIR = 9
    COW = 10
    DINING_TABLE = 11
    DOG = 12
    HORSE = 13
    MOTORBIKE = 14
    PERSON = 15
    POTTED_PLANT = 16
    SHEEP = 17
    SOFA = 18
    TRAIN = 19
    TV_MONITOR = 20

class Segmentation:
    """
    A utility class for semantic segmentation using a pre-trained DeepLabV3 model.
    """

    def __init__(self):
        """
        Initialize the Segmentation class with a pre-trained DeepLabV3 model.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = models.segmentation.deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.DEFAULT
        ).to(self._device)
        self._model.eval()

        # Preprocessing pipeline
        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a segmentation mask for the given image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: A 2D array where each value represents the predicted class ID.
        """
        input_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._model(input_tensor)['out'][0]
        mask = output.argmax(0).byte().cpu().numpy()
        return mask
