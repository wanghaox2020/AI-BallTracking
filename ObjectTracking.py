from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import cv2
import numpy as np

from KalmanFilter import KalmanFilter

# recall the function of the pretrained model and find the centriod of the function
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


def ObjectTracking(figure):
    # load the pretrained model from MC_coco 2017, we will take the sports ball model

    inputs = feature_extractor(images=figure, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([figure.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
    # defined a centerail and find the correlated score
    cur_center = []
    cur_score = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.5 and find the sport ball
        if score > 0.5:
            if label.item() == 37:  # where the sport ball item is labeled as 37 in the pretrained model
                center = np.array([[int(box[0]) + (int(box[2]) - int(box[0])) / 2],
                                   [int(box[1]) + (int(box[3]) - int(box[1])) / 2]])
                cur_center.append(center)
                cur_score.append(score)

    return cur_center, cur_score
