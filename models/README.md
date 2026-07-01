# Model Directory

This folder contains the app's default visible-screen prediction model and optional
detector model assets.

Included files:

- `fps_situation_model.json`: baseline prediction weights used by the app.
- `labels.txt`: labels used by optional detector models.
- `yolo_dataset.yaml`: dataset config for training a detector from legitimate screenshots or public VOD frames.

Optional detector examples:

- `fps_detector.pt`
- `fps_detector.onnx`
- `labels.txt`

Set the path in `config.json`:

```json
{
  "situation_model_path": "models/fps_situation_model.json",
  "model_path": "models/fps_detector.pt",
  "model_backend": "auto"
}
```

For ONNX, `labels.txt` should contain one label per line. The detector is optional:
the app still runs with `fps_situation_model.json` when no `.pt` or `.onnx` file exists.
