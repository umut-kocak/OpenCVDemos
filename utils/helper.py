""" A set of general helper functions. """
import os
import cv2

def load_images_from_folder(folder_path):
    """ Loads images within a folder. """
    images = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)
        if img is not None:
            images.append(img)
    return images

def resize_image(image, width, height):
    """ Resizes a given image to the given dimensions. """
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image
