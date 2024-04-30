import cv2
import torch
import pandas as pd

def detect_objects(image_path, model_path='best.pt'):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.eval()

        image = cv2.imread(image_path)
        results = model(image)
        predictions = results.pandas().xyxy[0]

        detected_labels = []
        for _, prediction in predictions.iterrows():
            label = prediction['name']
            confidence = prediction['confidence']
            xmin, ymin, xmax, ymax = prediction['xmin'], prediction['ymin'], prediction['xmax'], prediction['ymax']
            detected_labels.append(label)

        ingredients = ', '.join(detected_labels)
        return ingredients

    except Exception as e:
        print(f"An error occurred during object detection: {str(e)}")
        return None

