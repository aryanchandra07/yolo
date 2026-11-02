
import random
import cv2
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent


model_path = BASE_DIR / "yolov8n.pt"
video_path = BASE_DIR / "video" / "video_sample1.mp4"


model = YOLO(str(model_path))

class_list = model.names
detection_colors = []

for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))


frame_wid = 640
frame_hyt = 480


cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Cannot open video file. Check the video path.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended. Exiting...")
        break

    
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

   
    detect_params = model.predict(
        source=frame,
        conf=0.45,
        verbose=False
    )

   
    result = detect_params[0]

    
    for box in result.boxes:

        clsID = int(box.cls.item())
        conf = float(box.conf.item())

      
        bb = box.xyxy[0].cpu().numpy()

    
        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[clsID],
            3
        )

        
        label = f"{class_list[clsID]} {conf * 100:.1f}%"

        cv2.putText(
            frame,
            label,
            (int(bb[0]), int(bb[1]) - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("YOLOv8 Object Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
