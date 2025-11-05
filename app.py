# Importing necessaries Libraries
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
video_path = "peoplecount1.mp4"
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
    count = 0
    while True:

        success, frame = cap.read()
        if not success:
            print("End of video or cannot read frame any more")
            break
        
        count+=1
        # Skip alternate frames for faster processing (optional)
        if count % 2 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)
        # Draw boxes and track IDs for detected persons
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = names[class_id]
                if 'person' in class_name:  # Only track persons
                    x1, y1, x2, y2 = box

                    # Entry Logic start here
                    result = cv2.pointPolygonTest(np.array(area2, np.int32), ((x1, y2)), False)
                    if result >= 0:
                        enter[track_id]=(x1, y2)
                    if track_id in enter:
                        result1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x1, y2)), False)
                        if result1 >= 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1 - 10), 1, 1, colorR=(0, 0, 255))
                            cv2.circle(frame, (x1, y2), 4, (255, 0, 0), -1)
                            if list_enter.count(track_id)==0:
                                list_enter.append(track_id)

                    # Exit logic start Here
                    result2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x1, y2)), False)
                    if result2 >= 0:
                        exit_[track_id]=(x1, y2)
                    if track_id in exit_:
                        result3 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x1, y2)), False)
                        if result3 >= 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1 - 10), 1, 1, colorR=(0, 0, 255))
                            cv2.circle(frame, (x1, y2), 4, (255, 0, 0), -1)
                            if list_exit.count(track_id)==0:
                                list_exit.append(track_id)

                    # --- Trajectory Path Logic ---
                    # Calculate the center of the bounding box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Store trajectory path
                    if track_id not in track_paths:
                        track_paths[track_id] = []
                    track_paths[track_id].append((cx, cy))

                    # Limit the trail length to avoid memory overflow
                    if len(track_paths[track_id]) > 50:
                        track_paths[track_id].pop(0)

                    # Draw the trajectory path (green line)
                    for j in range(1, len(track_paths[track_id])):
                        cv2.line(frame, track_paths[track_id][j - 1], track_paths[track_id][j], (0, 255, 0), 2)

                    # Optional: Update heatmap with the trajectory points
                    if 0 <= cy < 600 and 0 <= cx < 1020:
                        heatmap[cy, cx] += 1
                    # --- End of Trajectory Path Logic ---


        # Text Representation
        cvzone.putTextRect(frame, f'Enter: {len(list_enter)}', (50, 60), 2, 2, colorR=(0, 0, 255))
        cvzone.putTextRect(frame, f'Exit: {len(list_exit)}', (50, 100), 2, 2, colorR=(0, 0, 255))
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 255), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 255), 2)

        # Heatmap Representation
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

# UI Parts

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# UI button control
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
