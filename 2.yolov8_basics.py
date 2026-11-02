
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")


detection_output = model.predict(
    source="/Users/aryanchandra/Documents/GitHub/yolo/image.jpg",
    save=True
)


for result in detection_output:
    print(result.boxes)

    
    numpy_array = result.boxes.xyxy.cpu().numpy()
    print(numpy_array)


    result.show()
