# Footfall Counter (People Counting with YOLO)

## Project overview

This repository contains a small Flask web application that performs people detection, tracking, simple entry/exit counting, and heatmap generation on a video using a YOLO model (Ultralytics). The app serves a live MJPEG video stream via an HTTP endpoint and exposes simple endpoints for playback control and toggling a density heatmap overlay.

The primary script is `app.py`. A YOLO model file (`yolo11n.pt`) and a sample video path are referenced by the app. The web UI is served from `templates/index.html` and static assets are stored in `static/`.

## Features

- Real-time object detection and multi-object tracking using an Ultralytics YOLO model.
- Entry/exit counting based on two polygonal areas.
- Trajectory visualization (drawn tracks for tracked persons).
- Heatmap overlay showing where people appear most frequently.
- Playback control endpoints: pause, play, restart.
- Toggle between real camera view and heatmap overlay.
- Download endpoint for the output video (if generated at `static/peoplecount_output.mp4`).

## Files in this repository

- `app.py` — Main Flask application. Implements video capture, detection/tracking, counting logic, heatmap accumulation, and control endpoints.
- `yolo11n.pt` — YOLO model weights used by `ultralytics.YOLO`. (Large binary file; not shown here.)
- `templates/index.html` — Front-end UI used to display the video stream and control buttons.
- `static/` — Static assets and (optionally) the generated `peoplecount_output.mp4` download file.

## How it works (high level)

1. The app opens a video file defined by `video_path` in `app.py`.
2. For each frame, it calls `model.track(frame, persist=True)` from the Ultralytics API to perform detection + tracking.
3. For each tracked box corresponding to a person, the app:
   - Computes a center point and appends it to a trajectory list for that track ID.
   - Uses two polygonal regions (`area1` and `area2`) and `cv2.pointPolygonTest` to determine if the person crosses from one area to another. It maintains simple `enter`/`exit` dictionaries and `list_enter`/`list_exit` lists to count unique entries and exits.
   - Updates a `heatmap` (a 600x1020 float array) by incrementing the pixel at the person's center.
4. The frame is annotated with track trajectories, polygon outlines, and simple on-screen text for Enter/Exit counts.
5. The frame is encoded to JPEG and streamed to the browser via the Flask endpoint `/video_feed` using multipart MJPEG.

## Important configuration variables (in `app.py`)

- `video_path` — Absolute path to the input video file. Default in the script: `D:/Github/Footfall Counter/peoplecount1.mp4`.
- `output_path` — Path to the output MP4 (downloaded via `/download` if present): `static/peoplecount_output.mp4`.
- `heatmap` — A NumPy array sized (600, 1020) matching the resized frame dimensions.
- `area1`, `area2` — Two lists of polygon vertex coordinates used for entry/exit detection. Adjust these polygons to match the camera view.
- Control flags: `show_heatmap` (toggle overlay), `paused`, and `restart_flag`.



## Installation and dependencies

This project requires Python 3.8+ and the following Python packages (examples):

- flask
- opencv-python
- numpy
- ultralytics
- cvzone

## Running the app

1. Make sure `yolo11n.pt` is present in the project root (the file path used by `YOLO("yolo11n.pt")`).
2. Adjust `video_path` in `app.py` to point to your local video file if needed.
3. Start the Flask app:
4. Open a browser to http://127.0.0.1:5000/ to see the UI. The video stream is embedded in the page via the `/video_feed` endpoint.

## UI and Controls

The front-end (`index.html`) is expected to provide buttons that call the control endpoints:

- Pause: POST `/pause`
- Play: POST `/play`
- Restart: POST `/restart`
- Toggle view (heatmap / real): POST `/toggle_view`
- Download output: GET `/download`


