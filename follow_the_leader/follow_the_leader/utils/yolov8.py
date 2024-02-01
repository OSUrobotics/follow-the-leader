import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
import os

weights_path = os.path.join(os.path.expanduser("~"), "follow-the-leader-deps", "yolov8", "weights", "best.pt")

from enum import Enum
class TreeLabel(Enum):
    TRUNK = 0
    SIDE_BRANCH = 1

class YoloInference:
    def __init__(self, input_size=(640, 448), output_size=(640, 448),
                    model_path = weights_path):
        self.input_size = input_size
        self.output_size = output_size

        self.model = YOLO(model_path)  # load a pretrained YOLOv8 segmentation model
        print("Model weights loaded")

    def preprocess(self, image):
        """Preprocess image to yolov8 size"""
        pass

    def reset(self):
        """reset model"""
        pass

    def process(self, image):
        """predicting image
        """
        #TODO: peprocess and post process mask size
        result = self.model(source=image, retina_masks=True)[0]
        if result.masks is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        masks = result.masks.data
        cls = result.boxes.data #bbox, (N, 6)
        cls = cls[:, 5] #Class value
        trunk_indices = torch.where(cls == 0)
        # use these indices to extract the relevant masks
        trunk_masks = masks[trunk_indices]
        # scale for visualizing results
        trunk_mask = torch.any(trunk_masks, dim=0).int() * 255

        return trunk_mask.cpu().numpy()

if __name__ == "__main__":
    model_path = weights_path
    yolo = YoloInference(model_path=model_path)
    #load image using cv2
    image = cv2.imread('/home/abhinav/Downloads/0300_jpg.rf.3472cc9e93c32d8c0d507d87d5efd296.jpg')
    mask = yolo.process(image).cpu().numpy()
    plt.imshow(mask, cmap='gray')
    plt.show()