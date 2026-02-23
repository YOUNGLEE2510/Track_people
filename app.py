import os
import threading
import json
import time
import tempfile
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from ai_worker import process_video

UPLOAD_FOLDER = "uploads"
REALTIME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime")
LATEST_JPG = os.path.join(REALTIME_DIR, "latest.jpg")
STATS_JSON = os.path.join(REALTIME_DIR, "stats.json")
HISTORY_JSONL = os.path.join(REALTIME_DIR, "history.jsonl")
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REALTIME_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

def _create_placeholder_frame():
    """Tạo frame placeholder khi chưa có video"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Waiting for video...", (150, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Upload a video to start tracking", (80, 260), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        return buf.tobytes()
    return None

# Tạo placeholder frame ban đầu nếu chưa có
if not os.path.exists(LATEST_JPG):
    placeholder = _create_placeholder_frame()
    if placeholder:
        with open(LATEST_JPG, "wb") as f:
            f.write(placeholder)

def _read_stats():
    """Đọc stats từ file JSON"""
    try:
        if os.path.exists(STATS_JSON):
            with open(STATS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"in": 0, "out": 0, "net": 0, "running": False}

def _read_history():
    """Đọc history từ file JSONL"""
    out = []
    try:
        if os.path.exists(HISTORY_JSONL):
            with open(HISTORY_JSONL, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-2000:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    # Sử dụng chính app.py làm realtime server cho offline mode
    # Nếu có REALTIME_BASE_URL thì dùng, không thì dùng chính server này
    realtime_base_url = os.environ.get("REALTIME_BASE_URL", "").rstrip("/")
    if not realtime_base_url:
        # Sử dụng chính server này
        realtime_base_url = request.url_root.rstrip("/")
    return render_template("index.html", realtime_base_url=realtime_base_url)

@app.route("/video_feed")
def video_feed():
    """Stream video feed từ latest.jpg"""
    def gen():
        last_mtime = 0
        last_frame_data = None
        
        while True:
            try:
                if os.path.exists(LATEST_JPG):
                    mtime = os.path.getmtime(LATEST_JPG)
                    if mtime != last_mtime:
                        last_mtime = mtime
                        with open(LATEST_JPG, "rb") as f:
                            last_frame_data = f.read()
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_frame_data + b"\r\n")
                    elif last_frame_data:
                        # Gửi lại frame cuối cùng nếu không có thay đổi để giữ stream sống
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_frame_data + b"\r\n")
                else:
                    # Nếu không có file, tạo và gửi placeholder
                    placeholder = _create_placeholder_frame()
                    if placeholder:
                        last_frame_data = placeholder
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n")
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                # Nếu có lỗi, gửi placeholder frame
                try:
                    placeholder = _create_placeholder_frame()
                    if placeholder:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n")
                except:
                    pass
                time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/result")
def api_result():
    """API trả về kết quả đếm hiện tại"""
    return jsonify(_read_stats())

@app.route("/api/history")
def api_history():
    """API trả về lịch sử đếm"""
    return jsonify(_read_history())

@app.route("/api/export/csv")
def api_export_csv():
    """API xuất CSV"""
    import csv
    from io import BytesIO, StringIO
    from flask import send_file
    history = _read_history()
    si = StringIO()
    w = csv.writer(si)
    w.writerow(["Thoi_gian", "IN", "OUT", "NET"])
    for row in history:
        t = row.get("t", "")
        if isinstance(t, (int, float)):
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
        w.writerow([t, row.get("in", 0), row.get("out", 0), row.get("net", 0)])
    buf = BytesIO(si.getvalue().encode("utf-8-sig"))
    return send_file(
        buf, mimetype="text/csv", as_attachment=True,
        download_name=f"counting_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )

@app.route("/upload", methods=["POST"])
def upload():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video = request.files.get("video")
    if video.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    if not allowed_file(video.filename):
        return jsonify({"error": "Invalid file type. Allowed types: mp4, avi, mov, mkv, flv, wmv, webm"}), 400

    try:
        # Check if auto-detect is enabled
        auto_detect = request.form.get("auto_detect", "true").lower() == "true"
        
        # Line type: "horizontal" (ngang/trên-dưới) hoặc "vertical" (dọc/trái-phải)
        line_type = request.form.get("line_type", "horizontal").lower()
        if line_type not in ("horizontal", "vertical"):
            line_type = "horizontal"

        # Get line configuration from form (only if not auto-detect)
        if not auto_detect:
            try:
                if line_type == "vertical":
                    line_x = int(request.form.get("line_x", 320))
                    line_config = {
                        "line_type": "vertical",
                        "line_x": line_x,
                        "auto": False
                    }
                else:
                    line_y = int(request.form.get("line_y", 300))
                    line_angle = float(request.form.get("line_angle", 0))
                    line_x1 = int(request.form.get("line_x1", 0))
                    line_x2 = int(request.form.get("line_x2", 640))
                    line_config = {
                        "line_type": "horizontal",
                        "y": line_y,
                        "angle": line_angle,
                        "x1": line_x1,
                        "x2": line_x2,
                        "auto": False
                    }
            except ValueError:
                return jsonify({"error": "Invalid line configuration values"}), 400
        else:
            line_config = {"auto": True, "line_type": line_type}
        
        # Sanitize filename
        filename = os.path.basename(video.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(path)
        
        # Verify file was saved and is readable
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return jsonify({"error": "Failed to save video file"}), 500

        threading.Thread(
            target=process_video,
            args=(path, line_config, auto_detect),
            daemon=True
        ).start()

        return jsonify({"message": "Processing started", "line_config": line_config})
    except ValueError as e:
        return jsonify({"error": f"Invalid line configuration: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error uploading file: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(threaded=True, debug=False, use_reloader=False)
