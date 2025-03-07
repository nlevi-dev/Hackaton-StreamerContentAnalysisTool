import os
while 'source' not in os.listdir():
    os.chdir('..')

from ultralytics import YOLO
import torch
import cv2

model = YOLO('yolo11x.pt')

image_path = 'data/1/images/Our_New_4500_Workstation_PCs_for_Editing_000001.jpg'
image = cv2.imread(image_path)

results = model(image)

print(results[0].names)
print(results[0].boxes)