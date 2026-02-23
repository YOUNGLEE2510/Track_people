from ultralytics import YOLO

class PersonDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # tự tải weight
        self.model.fuse()

    def detect(self, frame):
        results = self.model(frame, conf=0.4, classes=[0])
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1], conf, "person")
                )

        return detections
detector = PersonDetector()
def get_detector():
    return detector