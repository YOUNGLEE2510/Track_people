"""
Realtime: (1) Flask server phục vụ stream + API đọc từ thư mục realtime/
          (2) Mode webcam: capture + detect + đếm, ghi latest.jpg & stats.json
Chạy server:  python realtime.py   hoặc  python realtime.py --serve
Chạy webcam: python realtime.py --cam 0 [--write-artifacts] [--show]
"""
import argparse
import os
import time
import json
import tempfile
import ultralytics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REALTIME_DIR = os.path.join(BASE_DIR, "realtime")
LATEST_JPG = os.path.join(REALTIME_DIR, "latest.jpg")
STATS_JSON = os.path.join(REALTIME_DIR, "stats.json")
HISTORY_JSONL = os.path.join(REALTIME_DIR, "history.jsonl")
os.makedirs(REALTIME_DIR, exist_ok=True)


def _read_stats():
    try:
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"in": 0, "out": 0, "net": 0, "running": False}


def _atomic_write_bytes(path, data: bytes):
    fd, tmp = tempfile.mkstemp(prefix="tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _atomic_write_json(path, obj):
    _atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def _read_history():
    """Đọc history (JSONL) thành list dict {t, in, out, net}. Giới hạn 2000 dòng gần nhất."""
    out = []
    try:
        if not os.path.exists(HISTORY_JSONL):
            return out
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


def _append_history(stats):
    try:
        row = {"t": round(time.time(), 2), "in": stats.get("in", 0), "out": stats.get("out", 0), "net": stats.get("net", 0)}
        with open(HISTORY_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _clear_history():
    try:
        with open(HISTORY_JSONL, "w", encoding="utf-8") as f:
            pass
    except Exception:
        pass


# --- Flask server (stream + API) ---
def create_app():
    from flask import Flask, jsonify, Response
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/result")
    def api_result():
        return jsonify(_read_stats())

    @app.route("/api/history")
    def api_history():
        return jsonify(_read_history())

    @app.route("/api/export/csv")
    def api_export_csv():
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

    @app.route("/video_feed")
    def video_feed():
        def gen():
            last_mtime = 0
            while True:
                try:
                    if os.path.exists(LATEST_JPG):
                        mtime = os.path.getmtime(LATEST_JPG)
                        if mtime != last_mtime:
                            last_mtime = mtime
                            with open(LATEST_JPG, "rb") as f:
                                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f.read() + b"\r\n")
                    time.sleep(0.03)
                except Exception:
                    time.sleep(0.1)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def run_server(port=5001):
    app = create_app()
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)


# --- Webcam mode ---
def run_webcam(args):
    import cv2
    from shared_state import counter_state
    from src.tracker import PersonTracker
    from src.counter import PeopleCounter

    counter_state.reset()
    counter_state.running = True
    _clear_history()
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[REALTIME] Cannot open camera {args.cam}")
        counter_state.running = False
        if args.write_artifacts:
            _atomic_write_json(STATS_JSON, counter_state.get())
        return

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
    line_x = args.line_x if args.line_x is not None else (w // 2)
    line_x = max(50, min(line_x, w - 50))

    # Tracker tự động detect, không cần detector riêng
    tracker = PersonTracker()
    counter = PeopleCounter(
        line_y=h // 2, line_angle=0, line_x1=0, line_x2=w,
        frame_width=w, frame_height=h, line_type="vertical", line_x=line_x,
    )
    print(f"[REALTIME] Camera {args.cam} {w}x{h} | Line X={line_x} (Trái→Phải=IN, Phải→Trái=OUT)")

    win = "Realtime (q=quit)"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            if args.flip:
                frame = cv2.flip(frame, 1)
            n += 1

            # Tracker tự động detect và track, không cần detector riêng
            tracks = tracker.update(None, frame)

            cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 3)
            cv2.putText(frame, "Trai -> Phai = VAO", (line_x + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Phai -> Trai = RA", (line_x + 15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                try:
                    l, t, r, b = map(int, track.to_ltrb())
                    cx, cy = (l + r) // 2, (t + b) // 2
                    counter.update(track.track_id, cx, cy, counter_state)
                    cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track.track_id}", (l, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    continue

            stats = counter_state.get()
            cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
            cv2.putText(frame, f"IN: {stats['in']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {stats['out']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"NET: {stats['net']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0) if stats["net"] >= 0 else (0, 165, 255), 2)

            # Luôn ghi frame khi write_artifacts được bật để stream mượt hơn
            if args.write_artifacts:
                try:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        _atomic_write_bytes(LATEST_JPG, buf.tobytes())
                    # Chỉ cập nhật stats và history theo write_every để giảm I/O
                    if n % max(1, args.write_every) == 0:
                        _atomic_write_json(STATS_JSON, stats)
                        # Ghi history mỗi 10 frame để đồng bộ với offline mode
                        if n % 10 == 0:
                            _append_history(stats)
                except Exception as e:
                    print(f"[REALTIME] Error writing frame: {e}")
                    pass

            if args.show:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(delay * 0.5)
    except KeyboardInterrupt:
        print("\n[REALTIME] Stopped.")
    finally:
        counter_state.running = False
        if args.write_artifacts:
            try:
                _atomic_write_json(STATS_JSON, counter_state.get())
            except Exception:
                pass
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Realtime: Flask stream server hoặc webcam đếm người.")
    p.add_argument("--serve", action="store_true", help="Chạy Flask server (video_feed + api/result), port 5001")
    p.add_argument("--cam", type=int, default=None, metavar="INDEX", help="Camera index → chạy mode webcam")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--flip", action="store_true", help="Lật ngang frame")
    p.add_argument("--line-x", type=int, default=None, help="Vị trí đường dọc (mặc định giữa)")
    p.add_argument("--show", action="store_true", help="Hiện cửa sổ OpenCV")
    p.add_argument("--write-artifacts", action="store_true", help="Ghi realtime/latest.jpg và stats.json")
    p.add_argument("--write-every", type=int, default=2)
    p.add_argument("--port", type=int, default=None, help="Port Flask (mặc định 5001)")
    args = p.parse_args()

    port = args.port or int(os.environ.get("REALTIME_PORT", "5001"))

    if args.cam is not None:
        run_webcam(args)
    else:
        run_server(port)


if __name__ == "__main__":
    main()
