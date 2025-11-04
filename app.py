from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import os

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolo11n.pt")
names = model.names

# Video path
video_path = "D:/Github/Footfall Counter/peoplecount1.mp4"
output_path = "static/peoplecount_output.mp4"

# Globals
track_paths = {}
heatmap = np.zeros((600, 1020), dtype=np.float32)
area1 = [(251, 445), (516, 575), (466, 589), (210, 447)]
area2 = [(466, 589), (210, 447), (167, 460), (386, 595)]

enter, exit_ = {}, {}
list_enter, list_exit = [], []

# Control flags
show_heatmap = False  # default real view
paused = False
restart_flag = False


def generate_frames():
    global paused, restart_flag, show_heatmap
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    while True:
        if restart_flag:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            restart_flag = False

        if paused:
            cv2.waitKey(100)
            continue

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = names[class_id]
                if 'person' in class_name:
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Entry/Exit logic
                    result = cv2.pointPolygonTest(np.array(area2, np.int32), ((x1, y2)), False)
                    if result >= 0:
                        enter[track_id] = (x1, y2)
                    if track_id in enter:
                        result1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x1, y2)), False)
                        if result1 >= 0 and track_id not in list_enter:
                            list_enter.append(track_id)

                    result2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x1, y2)), False)
                    if result2 >= 0:
                        exit_[track_id] = (x1, y2)
                    if track_id in exit_:
                        result3 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x1, y2)), False)
                        if result3 >= 0 and track_id not in list_exit:
                            list_exit.append(track_id)

                    # Trajectory path
                    if track_id not in track_paths:
                        track_paths[track_id] = []
                    track_paths[track_id].append((cx, cy))
                    for j in range(1, len(track_paths[track_id])):
                        cv2.line(frame, track_paths[track_id][j - 1], track_paths[track_id][j], (0, 255, 0), 2)

                    # Heatmap update
                    if 0 <= cy < 600 and 0 <= cx < 1020:
                        heatmap[cy, cx] += 1

        # Text overlays
        cvzone.putTextRect(frame, f'Enter: {len(list_enter)}', (50, 60), 2, 2, colorR=(0, 0, 255))
        cvzone.putTextRect(frame, f'Exit: {len(list_exit)}', (50, 100), 2, 2, colorR=(0, 0, 255))
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 255), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 255), 2)

        # Heatmap overlay toggle
        if show_heatmap:
            heat_display = cv2.applyColorMap(
                cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            frame = cv2.addWeighted(frame, 0.7, heat_display, 0.3, 0)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Control Endpoints ---
@app.route('/pause', methods=['POST'])
def pause_video():
    global paused
    paused = True
    return "paused"


@app.route('/play', methods=['POST'])
def play_video():
    global paused
    paused = False
    return "playing"


@app.route('/restart', methods=['POST'])
def restart_video():
    global restart_flag
    restart_flag = True
    return "restarted"


@app.route('/toggle_view', methods=['POST'])
def toggle_view():
    global show_heatmap
    show_heatmap = not show_heatmap
    return "heatmap" if show_heatmap else "real"


@app.route('/download')
def download():
    directory = os.path.join(app.root_path, 'static')
    return send_from_directory(directory, 'peoplecount_output.mp4', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
