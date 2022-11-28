from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import cv2


def SingleObjectDetector():
    # load the pretrained model from MC_coco 2017, we will take the sports ball model
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    for i in range(51):
        image = Image.open('SingleBallRawFolder/SingleBall%d.jpg' % i)
        img = cv2.imread('SingleBallRawFolder/SingleBall%d.jpg' % i)
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # let's only keep detections with score > 0.9
            if score > 0.9:
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                if label.item() == 37: #where the sport ball item is labeled as 37 in the pretrained model
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3) #add rectangle to figure
        cv2.imwrite('SingleBallResultFolder/SingleBallResult%d.jpg' % i, img)


def MultipleObjectDetector():
    # load the pretrained model from MC_coco 2017, we will take the sports ball model
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    for i in range(41):
        image = Image.open('MultipleBallRawFolder/MultiBall%d.jpg' % i)
        img = cv2.imread('MultipleBallRawFolder/MultiBall%d.jpg' % i)
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # let's only keep detections with score > 0.9
            if score > 0.9:
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                if label.item() == 37: #where the sport ball item is labeled as 37 in the pretrained model
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
        cv2.imwrite('MultipleBallResultFolder/MultiBallResult%d.jpg' % i, img)