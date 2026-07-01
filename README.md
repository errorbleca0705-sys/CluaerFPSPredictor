# cluaerAI FPS Predictor

Python desktop app for continuous, local screen capture and FPS situation prediction.

This app only analyzes frames captured from your own screen. It does not read game memory,
inspect packets, bypass anti-cheat, automate input, draw an enemy overlay, or control aim.

## Features

- Continuous screen capture with `mss`
- Live preview and prediction panel
- Heuristic visual analysis for damage flashes, whiteout, smoke-like screens, motion, and visible detections
- Optional deep-learning detector hook for YOLO or ONNX models
- Manual context fields for health, ammo, allies, enemies, map, position, and utility
- JSONL learning log with prediction and actual-result feedback

## Setup

Install Python 3.10 or newer first. On Windows, the Microsoft Store `python.exe` alias may appear even when Python is not installed, so install from python.org if the setup script cannot find a real interpreter.

```powershell
cd C:\Users\Administrator\Desktop\CluaerFPS
.\run.ps1
```

Manual setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

Optional deep-learning backend:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dl.txt
```

Then place a model in `models/` and set `model_path` in `config.json`.

## Model Notes

The app includes `models/fps_situation_model.json`, a baseline prediction model used
for visible-screen situation scoring. For better object detection, use a custom detector
trained on legitimate screenshots or public VOD frames. Recommended labels:

```text
enemy
ally
spike
smoke
flash
knocked
vehicle
cover
```

Supported model paths:

- `.pt`: loaded with Ultralytics YOLO if `ultralytics` is installed
- `.onnx`: loaded with OpenCV DNN using a YOLO-like output layout

Feedback calibration:

```powershell
.\.venv\Scripts\python tools\calibrate_situation_model.py
```

Run this after saving at least 5 actual-result feedback records in the app.

## Usage

1. Start the app.
2. Select monitor and game.
3. Fill any known context fields.
4. Press `Start Capture`.
5. Use `Save Prediction` to log the current state.
6. After the fight or round, enter the actual result and press `Save Feedback`.

Logs are stored at `data/predictions.jsonl`.

## Safety Boundary

Allowed:

- Analyzing your own captured screen
- Reviewing VODs and screenshots
- Estimating likely outcomes from visible information
- Saving prediction and feedback data

Not supported:

- Memory reading
- Packet inspection
- Anti-cheat bypass
- Aimbot, recoil macro, triggerbot, or auto movement
- Enemy position overlay from hidden information
- Any automation that plays the game for you
=======
# CluaerFPSPredictor
>>>>>>> 2386ef44bcfe27fa58409ed32ed2271d053f70b0
