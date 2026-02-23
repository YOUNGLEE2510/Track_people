import cv2
import os
import time
import math
import threading
import json
import tempfile
from shared_state import counter_state
# Detector không cần thiết vì tracker tự động detect - đã bỏ để tối ưu FPS
from src.tracker import PersonTracker
from src.counter import PeopleCounter

LINE_Y = 300
output_frame = None
output_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REALTIME_DIR = os.path.join(BASE_DIR, "realtime")
LATEST_JPG_PATH = os.path.join(REALTIME_DIR, "latest.jpg")
STATS_JSON_PATH = os.path.join(REALTIME_DIR, "stats.json")
HISTORY_JSONL_PATH = os.path.join(REALTIME_DIR, "history.jsonl")
os.makedirs(REALTIME_DIR, exist_ok=True)


def _atomic_write_bytes(path, data: bytes):
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_write_json(path, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(path, data)


def _clear_history():
    try:
        with open(HISTORY_JSONL_PATH, "w", encoding="utf-8"):
            pass
    except Exception:
        pass


def _append_history(stats):
    try:
        row = {"t": round(time.time(), 2), "in": stats.get("in", 0), "out": stats.get("out", 0), "net": stats.get("net", 0)}
        with open(HISTORY_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass

def auto_detect_line_position(video_path, num_samples=30):
    """
    Tự động phát hiện vị trí line tối ưu bằng cách phân tích chuyển động của người trong video
    
    Args:
        video_path: Đường dẫn đến video
        num_samples: Số frame để phân tích (mặc định 30 frame đầu)
    
    Returns:
        dict: Cấu hình line tối ưu {y, angle, x1, x2}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sử dụng một số frame để phân tích (không quá 30% video)
    sample_frames = min(num_samples, max(10, total_frames // 10))
    
    # Sử dụng tracker thay vì detector riêng để tối ưu (tracker tự động detect)
    tracker = PersonTracker()
    
    # Lưu trữ các vị trí trung tâm của người qua các frame
    person_positions = []
    movement_heatmap = {}  # Đếm số lần người đi qua tại mỗi vị trí Y
    
    frame_count = 0
    sample_interval = max(1, total_frames // sample_frames) if total_frames > 0 else 1
    
    print(f"[AUTO-DETECT] Analyzing {sample_frames} frames for optimal line position...")
    
    while frame_count < sample_frames * sample_interval:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ phân tích một số frame nhất định
        if frame_count % sample_interval == 0:
            try:
                # Tracker tự động detect và track
                tracks = tracker.update(None, frame)
                
                for track in tracks:
                    if track.is_confirmed():
                        try:
                            l, t, r, b = map(int, track.to_ltrb())
                            cx = int((l + r) / 2)
                            cy = int((t + b) / 2)
                            
                            person_positions.append((cx, cy))
                            
                            # Tạo heatmap - đếm số lần người xuất hiện ở mỗi vị trí Y
                            y_bucket = (cy // 10) * 10  # Làm tròn về bội số của 10
                            movement_heatmap[y_bucket] = movement_heatmap.get(y_bucket, 0) + 1
                        except Exception:
                            continue
            except Exception as e:
                print(f"[AUTO-DETECT] Error analyzing frame {frame_count}: {e}")
        
        frame_count += 1
    
    cap.release()
    
    if not movement_heatmap:
        print("[AUTO-DETECT] No movement detected, using default position")
        return {
            "y": frame_height // 2,
            "angle": 0,
            "x1": 0,
            "x2": frame_width
        }
    
    # Tìm vị trí Y có nhiều chuyển động nhất
    best_y = max(movement_heatmap.items(), key=lambda x: x[1])[0]
    
    # Phân tích hướng chuyển động để xác định góc
    if len(person_positions) > 10:
        # Tính toán vector chuyển động trung bình
        movements = []
        for i in range(1, len(person_positions)):
            dx = person_positions[i][0] - person_positions[i-1][0]
            dy = person_positions[i][1] - person_positions[i-1][1]
            if abs(dx) > 5 or abs(dy) > 5:  # Chỉ tính chuyển động đáng kể
                movements.append((dx, dy))
        
        if movements:
            avg_dx = sum(m[0] for m in movements) / len(movements)
            avg_dy = sum(m[1] for m in movements) / len(movements)
            
            # Tính góc từ vector chuyển động trung bình
            if abs(avg_dx) > 1:
                angle = math.degrees(math.atan(avg_dy / avg_dx))
                # Giới hạn góc trong khoảng hợp lý
                angle = max(-30, min(30, angle))
            else:
                angle = 0
        else:
            angle = 0
    else:
        angle = 0
    
    # Xác định điểm bắt đầu và kết thúc của line
    # Phân tích phân bố ngang của người để tìm vùng có nhiều người nhất
    x_positions = [pos[0] for pos in person_positions]
    if x_positions:
        x_min = max(0, min(x_positions) - 50)
        x_max = min(frame_width, max(x_positions) + 50)
    else:
        x_min = frame_width // 4
        x_max = 3 * frame_width // 4
    
    # Đảm bảo line có độ dài hợp lý
    if x_max - x_min < frame_width // 3:
        center_x = (x_min + x_max) // 2
        x_min = max(0, center_x - frame_width // 4)
        x_max = min(frame_width, center_x + frame_width // 4)
    
    result = {
        "y": max(50, min(best_y, frame_height - 50)),
        "angle": round(angle, 1),
        "x1": x_min,
        "x2": x_max
    }
    
    print(f"[AUTO-DETECT] Optimal line detected: Y={result['y']}, Angle={result['angle']}°, X1={result['x1']}, X2={result['x2']}")
    
    return result

def process_video(video_path, line_config=None, auto_detect=True):
    global output_frame
    # Reset tất cả state khi video mới bắt đầu
    counter_state.reset()  # Reset counter state (count_in, count_out)
    counter_state.running = True
    _clear_history()

    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with output_lock:
            output_frame = None  # Reset frame (module-level global)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Detector không cần thiết vì tracker tự động detect
        # detector = PersonDetector()  # Đã bỏ để tối ưu
        
        # Get video properties trước để có thể thiết lập ROI
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Tối ưu: resize frame nếu quá lớn để tăng FPS
        # Giữ nguyên kích thước nếu <= 1280x720, resize nếu lớn hơn
        max_width, max_height = 1280, 720
        if frame_width > max_width or frame_height > max_height:
            scale = min(max_width / frame_width, max_height / frame_height)
            frame_width = int(frame_width * scale)
            frame_height = int(frame_height * scale)
            print(f"[FPS OPTIMIZATION] Resizing frames to {frame_width}x{frame_height} for better performance")
        
        # Lấy ROI config từ line_config nếu có
        roi_config = (line_config or {}).get("roi", None)
        if roi_config:
            roi_x1 = max(0, min(int(roi_config.get("x1", 0)), frame_width))
            roi_y1 = max(0, min(int(roi_config.get("y1", 0)), frame_height))
            roi_x2 = max(roi_x1, min(int(roi_config.get("x2", frame_width)), frame_width))
            roi_y2 = max(roi_y1, min(int(roi_config.get("y2", frame_height)), frame_height))
            tracker = PersonTracker(roi=(roi_x1, roi_y1, roi_x2, roi_y2))
            print(f"[ROI] Using ROI: ({roi_x1}, {roi_y1}, {roi_x2}, {roi_y2})")
        else:
            tracker = PersonTracker()
            print("[ROI] No ROI configured, detecting entire frame")
        
        # Reset tracker state khi video mới
        tracker.reset()
        
        line_type = (line_config or {}).get("line_type", "horizontal")
        is_vertical = (line_type == "vertical")

        if is_vertical:
            # Đường dọc giữa: trái→phải = Vào, phải→trái = Ra
            if auto_detect and (not line_config or line_config.get("auto", False)):
                line_x = frame_width // 2
                print(f"[AUTO-DETECT] Vertical line at center X={line_x}")
            elif line_config and "line_x" in line_config:
                line_x = max(50, min(int(line_config["line_x"]), frame_width - 50))
            else:
                line_x = frame_width // 2
            counter = PeopleCounter(
                line_y=frame_height // 2, line_angle=0,
                line_x1=0, line_x2=frame_width,
                frame_width=frame_width, frame_height=frame_height,
                line_type="vertical", line_x=line_x
            )
        else:
            # Đường ngang/nghiêng
            if auto_detect and (not line_config or line_config.get("auto", False)):
                print("[AUTO-DETECT] Starting automatic line detection...")
                auto_config = auto_detect_line_position(video_path)
                if auto_config:
                    line_y = int(auto_config.get("y", frame_height // 2))
                    line_angle = float(auto_config.get("angle", 0))
                    line_x1 = int(auto_config.get("x1", 0))
                    line_x2 = int(auto_config.get("x2", frame_width))
                    print(f"[AUTO-DETECT] Using auto-detected line: Y={line_y}, Angle={line_angle}°")
                else:
                    line_y = frame_height // 2
                    line_angle = 0
                    line_x1 = 0
                    line_x2 = frame_width
            elif line_config:
                line_y = int(line_config.get("y", frame_height // 2))
                line_angle = float(line_config.get("angle", 0))
                line_x1 = int(line_config.get("x1", 0))
                line_x2 = int(line_config.get("x2", frame_width))
            else:
                line_y = min(LINE_Y, frame_height - 50) if frame_height > 0 else LINE_Y
                line_angle = 0
                line_x1 = 0
                line_x2 = frame_width
            line_y = max(50, min(line_y, frame_height - 50))
            line_x1 = max(0, min(line_x1, frame_width))
            line_x2 = max(0, min(line_x2, frame_width))
            counter = PeopleCounter(line_y, line_angle, line_x1, line_x2, frame_width, frame_height,
                                    line_type="horizontal")
        
        # Reset counter state khi video mới (reset tất cả tracking state)
        counter.reset()
        print("[COUNTER] Counter state reset for new video")

        # Tối ưu FPS: skip frames để tăng tốc độ xử lý
        # Process mỗi N frame để tăng FPS (ví dụ: process_frame_interval = 1 nghĩa là xử lý mọi frame)
        # Tăng lên 2 hoặc 3 để skip frames và tăng FPS (nhưng có thể giảm độ chính xác tracking)
        process_frame_interval = 1  # Có thể tăng lên 2 hoặc 3 để skip frames và tăng FPS
        
        # Calculate delay between frames to maintain video speed
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

        frame_count = 0
        last_processed_frame = None  # Lưu frame cuối cùng đã xử lý để hiển thị
        
        # Ghi frame đầu tiên ngay để stream có dữ liệu
        ret, first_frame = cap.read()
        if ret:
            if first_frame.shape[1] != frame_width or first_frame.shape[0] != frame_height:
                first_frame = cv2.resize(first_frame, (frame_width, frame_height))
            try:
                ok, buf = cv2.imencode(".jpg", first_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    _atomic_write_bytes(LATEST_JPG_PATH, buf.tobytes())
                    _atomic_write_json(STATS_JSON_PATH, counter_state.get())
            except Exception:
                pass
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Resize frame nếu cần để tăng FPS
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                frame = cv2.resize(frame, (frame_width, frame_height))
            
            # Skip frames để tăng FPS (chỉ xử lý mỗi N frame)
            should_process = (frame_count % process_frame_interval == 0)
            
            try:
                if should_process:
                    # Get current counts before processing
                    current_counts = counter_state.get()
                    
                    # Tracker tự động detect và track, không cần detector riêng
                    tracks = tracker.update(None, frame)
                    
                    # Lưu frame đã xử lý để hiển thị
                    last_processed_frame = frame.copy()
                else:
                    # Skip frame này, sử dụng frame đã xử lý trước đó
                    if last_processed_frame is not None:
                        frame = last_processed_frame.copy()
                    tracks = []

                # Chỉ vẽ khi đã xử lý frame (không vẽ khi skip)
                if should_process:
                    # Vẽ ROI nếu có để người dùng thấy vùng đang detect
                    if tracker.roi is not None:
                        roi_x1, roi_y1, roi_x2, roi_y2 = tracker.roi
                        # Vẽ rectangle cho ROI với màu xanh dương
                        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
                        # Thêm label
                        cv2.putText(frame, "ROI (Detection Area)", 
                                   (roi_x1, roi_y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                   (255, 0, 0), 2)

                    # Draw counting line
                if is_vertical:
                    # Đường dọc giữa: trái→phải = Vào, phải→trái = Ra
                    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 255, 255), 3)
                    cv2.putText(frame, "Trai -> Phai = VAO", (line_x + 15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "Phai -> Trai = RA", (line_x + 15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    if line_angle == 0:
                        cv2.line(frame, (line_x1, line_y), (line_x2, line_y), (0, 255, 255), 3)
                    else:
                        angle_rad = math.radians(line_angle)
                        center_x = (line_x1 + line_x2) // 2
                        center_y = line_y
                        length = abs(line_x2 - line_x1) // 2
                        dx = length * math.cos(angle_rad)
                        dy = length * math.sin(angle_rad)
                        pt1 = (int(center_x - dx), int(center_y - dy))
                        pt2 = (int(center_x + dx), int(center_y + dy))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 3)
                    label_x = min(line_x1, line_x2) + 10
                    label_y = line_y - 10 if line_y > 30 else line_y + 25
                    cv2.putText(frame, f"Counting Line ({line_angle}°)",
                                (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Process tracks
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        try:
                            l, t, r, b = map(int, track.to_ltrb())
                            cx = int((l + r) / 2)  # Center x
                            cy = int((t + b) / 2)  # Center y

                            counter.update(track.track_id, cx, cy, counter_state)

                            # Draw bounding box and ID
                            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID {track.track_id}",
                                        (l, t - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error processing track: {e}")
                            continue

                # Get updated counts (luôn cập nhật để hiển thị đúng)
                updated_counts = counter_state.get()
                
                # Draw statistics on frame
                stats_y = 30
                cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
                
                # IN count
                cv2.putText(frame, f"IN: {updated_counts['in']}", 
                           (20, stats_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                           (0, 255, 0), 2)
                
                # OUT count
                cv2.putText(frame, f"OUT: {updated_counts['out']}", 
                           (20, stats_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                           (0, 0, 255), 2)
                
                # NET count
                net_color = (255, 255, 0) if updated_counts['net'] >= 0 else (0, 165, 255)
                cv2.putText(frame, f"NET: {updated_counts['net']}", 
                           (20, stats_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                           net_color, 2)
                
                # Frame info và FPS
                import time as time_module
                if not hasattr(process_video, 'last_fps_time'):
                    process_video.last_fps_time = time_module.time()
                    process_video.fps_counter = 0
                
                process_video.fps_counter += 1
                current_time = time_module.time()
                if current_time - process_video.last_fps_time >= 1.0:
                    process_video.current_fps = process_video.fps_counter / (current_time - process_video.last_fps_time)
                    process_video.fps_counter = 0
                    process_video.last_fps_time = current_time
                
                fps_text = f"FPS: {process_video.current_fps:.1f}" if hasattr(process_video, 'current_fps') else "FPS: --"
                cv2.putText(frame, f"Frame: {frame_count} | {fps_text}", 
                           (20, stats_y + 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)

                # Update output frame immediately so it can be displayed
                with output_lock:
                    output_frame = frame.copy()

                # Ghi realtime artifacts - ghi mỗi frame để stream mượt hơn
                # Ghi cả khi skip frame để đảm bảo stream luôn có dữ liệu
                try:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        _atomic_write_bytes(LATEST_JPG_PATH, buf.tobytes())
                    if should_process:
                        st = counter_state.get()
                        _atomic_write_json(STATS_JSON_PATH, st)
                        # Ghi history mỗi 10 frame để đồng bộ với online mode và cập nhật biểu đồ tốt hơn
                        if frame_count % 10 == 0:
                            _append_history(st)
                except Exception as e:
                    print(f"Error writing frame: {e}")
                    pass
                
                # Tối ưu delay: chỉ delay khi cần thiết để không làm chậm xử lý
                # Delay nhỏ hơn khi skip frames để tăng FPS
                if should_process:
                    # Delay nhỏ hơn khi đã skip frames
                    time.sleep(frame_delay * 0.3)
                else:
                    # Delay rất nhỏ khi skip frame
                    time.sleep(frame_delay * 0.1)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Still show the frame even if processing fails
                with output_lock:
                    output_frame = frame.copy()
                continue

        print(f"Video processing completed. Processed {frame_count} frames.")
        
        # Ghi frame cuối cùng khi video kết thúc
        if last_processed_frame is not None:
            try:
                ok, buf = cv2.imencode(".jpg", last_processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    _atomic_write_bytes(LATEST_JPG_PATH, buf.tobytes())
                st = counter_state.get()
                _atomic_write_json(STATS_JSON_PATH, st)
                _append_history(st)
            except Exception:
                pass

    except Exception as e:
        print(f"Error in process_video: {e}")
        import traceback
        traceback.print_exc()
        # Create error frame
        error_frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
        cv2.putText(error_frame, f"Error: {str(e)}", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
        with output_lock:
            output_frame = error_frame
        try:
            ok, buf = cv2.imencode(".jpg", error_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                _atomic_write_bytes(LATEST_JPG_PATH, buf.tobytes())
            _atomic_write_json(STATS_JSON_PATH, counter_state.get())
        except Exception:
            pass
    finally:
        counter_state.running = False
        if 'cap' in locals():
            cap.release()
        try:
            st = counter_state.get()
            _atomic_write_json(STATS_JSON_PATH, st)
            _append_history(st)
        except Exception:
            pass

def get_output_frame():
    """Return current output frame (numpy array) for encoding in app, or None."""
    with output_lock:
        if output_frame is None:
            return None
        return output_frame.copy()
