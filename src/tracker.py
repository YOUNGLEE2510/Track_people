from ultralytics import YOLO
import numpy as np
import os

class PersonTracker:
    """
    Sử dụng ByteTracker từ ultralytics để tracking người.
    ByteTracker tốt hơn DeepSort khi:
    - Bị che khuất (occlusion) - giữ ID tốt hơn khi bị che
    - Đông người - xử lý tốt hơn với nhiều đối tượng
    - Giữ ID ổn định hơn qua các frame
    """
    def __init__(self, roi=None):
        """
        Args:
            roi: Region of Interest dạng (x1, y1, x2, y2) hoặc None để detect toàn bộ frame
                 Nếu None, sẽ detect toàn bộ frame
        """
        # ByteTracker được tích hợp trong YOLO model
        # Sử dụng model nhẹ để tracking
        self.model = YOLO("yolov8n.pt")
        self.track_history = {}  # Lưu lịch sử tracking để giữ ID ổn định
        
        # ROI (Region of Interest) để giảm detect thừa
        # Chỉ detect trong vùng này, giúp tăng tốc độ và giảm false positive
        self.roi = roi  # (x1, y1, x2, y2) hoặc None
        
        # Đường dẫn đến file cấu hình ByteTracker tùy chỉnh
        # File này được tối ưu để giữ ID ổn định hơn khi bị che khuất hoặc đông người
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tracker_config_path = os.path.join(base_dir, "bytetrack_custom.yaml")
        
        # Kiểm tra xem file config có tồn tại không
        if not os.path.exists(self.tracker_config_path):
            print(f"[WARNING] ByteTracker config file not found: {self.tracker_config_path}")
            print("[INFO] Using default ByteTracker configuration")
            self.tracker_config_path = None
    
    def set_roi(self, x1, y1, x2, y2):
        """Thiết lập ROI (Region of Interest)"""
        self.roi = (int(x1), int(y1), int(x2), int(y2))
        print(f"[TRACKER] ROI set to: ({x1}, {y1}, {x2}, {y2})")
    
    def clear_roi(self):
        """Xóa ROI, detect toàn bộ frame"""
        self.roi = None
        print("[TRACKER] ROI cleared, detecting entire frame")
    
    def reset(self):
        """Reset tracker state khi video mới"""
        self.track_history.clear()
        print("[TRACKER] Tracker state reset")
        
    def update(self, detections, frame):
        """
        Update tracker với detections và frame.
        ByteTracker sẽ tự động detect và track trong một lần gọi.
        
        Args:
            detections: List of detections từ detector (không dùng với ByteTracker, giữ để tương thích)
            frame: Frame image (numpy array)
            
        Returns:
            List of tracks với format tương thích: track objects có track_id và to_ltrb()
        """
        # Áp dụng ROI nếu có để giảm detect thừa và tăng tốc độ
        roi_offset_x = 0
        roi_offset_y = 0
        roi_frame = frame
        
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            # Đảm bảo ROI trong phạm vi frame
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1, min(x2, w))
            y2 = max(y1, min(y2, h))
            
            # Crop ROI từ frame
            roi_frame = frame[y1:y2, x1:x2]
            roi_offset_x = x1
            roi_offset_y = y1
            
            # Nếu ROI rỗng hoặc quá nhỏ, return empty tracks
            if roi_frame.size == 0 or roi_frame.shape[0] < 10 or roi_frame.shape[1] < 10:
                return []
        
        # ByteTracker tự động detect và track trong một lần gọi
        # Sử dụng YOLO với tracking mode (ByteTracker)
        # persist=True để giữ ID qua các frame (quan trọng cho tracking ổn định)
        # Sử dụng ROI frame để giảm detect thừa
        # Tối ưu FPS: giảm imgsz và sử dụng half precision nếu có GPU
        results = self.model.track(
            roi_frame,
            conf=0.4,  # Confidence threshold cho detection
            classes=[0],  # Chỉ track person (class 0)
            persist=True,  # Giữ ID qua các frame - QUAN TRỌNG để tránh ID switch
            tracker="bytetrack",  # Sử dụng ByteTracker (tốt hơn DeepSort cho occlusion và đông người)
            verbose=False,  # Không in log
            imgsz=640,  # Input size (có thể giảm xuống 480 để tăng FPS nếu cần)
            half=False,  # Sử dụng FP16 nếu GPU hỗ trợ (có thể set True để tăng tốc)
            device=None,  # Tự động chọn device (CPU/GPU)
        )
        
        # Extract tracks từ results
        tracks = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # Kiểm tra xem có track_ids không
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        track_id = int(track_ids[i])
                        box = boxes[i].copy()  # [x1, y1, x2, y2] - coordinates trong ROI frame (copy để có thể chỉnh sửa)
                        conf = float(confs[i])
                        
                        # Điều chỉnh coordinates về frame gốc nếu có ROI
                        if self.roi is not None:
                            box[0] += roi_offset_x  # x1
                            box[1] += roi_offset_y  # y1
                            box[2] += roi_offset_x  # x2
                            box[3] += roi_offset_y  # y2
                        
                        # Tạo track object tương thích với code hiện tại
                        track = TrackObject(track_id, box, conf)
                        tracks.append(track)
        
        return tracks


class TrackObject:
    """
    Wrapper class để tương thích với API của DeepSort
    Giữ interface giống nhau để không cần sửa code ở nơi khác
    """
    def __init__(self, track_id, bbox, conf):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.conf = conf
        
    def to_ltrb(self):
        """Trả về bbox dạng (left, top, right, bottom)"""
        x1, y1, x2, y2 = self.bbox
        return (float(x1), float(y1), float(x2), float(y2))
    
    def is_confirmed(self):
        """ByteTracker luôn confirmed nếu có ID"""
        return True


tracker = PersonTracker()
def get_tracker():
    return tracker