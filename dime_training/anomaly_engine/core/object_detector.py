import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class ObjectDetector:
    def __init__(self, device, detection_threshold=0.7):
        self.device = device
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
        self.model.eval()
        self.detection_threshold = detection_threshold
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
    def detect(self, image_tensor):
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # Process predictions
        detections = []
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            if score > self.detection_threshold:
                detections.append({
                    "box": box,
                    "class": self.class_names[label],
                    "score": score
                })
                
        return detections