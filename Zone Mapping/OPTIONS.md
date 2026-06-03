# Zone Detector — Options & Configuration

This document lists the runtime flags and the configurable constants (at the top of `zone_detector.py`) you can change to control behavior.

**Command-line flags**
- `--reset-zones` : Delete existing `zones.json` (if present) and force interactive zone drawing on next run.
- `--source <index|path>` : Override the `VIDEO_SOURCE` constant. Pass a camera index (e.g., `0`) or a video file path.

**Top-level configuration constants (edit `zone_detector.py`)**
- `VIDEO_SOURCE` : Default video input. `0` selects default webcam; otherwise supply a file path string.
- `MODEL_NAME` : Path or filename of the YOLOv8 pose model (default `yolov8n-pose.pt`). Place the model in the project root or give a path.
- `ZONES_FILE` : Path to JSON file used to store/load polygon zones (default `zones.json`).
- `LOG_FILE` : CSV file where intrusion events are appended (default `intrusion_log.csv`).
- `ALERT_COOLDOWN_SECONDS` : Minimum seconds between repeated intrusion logs for the same `(person_id, zone_name)` pair (default `30`).
- `CONFIDENCE_THRESHOLD` : Minimum model confidence for detections forwarded by `model.track()` (default `0.5`).
- `FRAME_SKIP` : Integer N to process every Nth frame. Higher values reduce CPU usage but lower temporal resolution (default `2`).
- `ANKLE_CONFIDENCE_MIN` : Minimum keypoint confidence to accept ankle coordinates (default `0.3`).

**Keypoint indices**
- Left ankle index: `15`
- Right ankle index: `16`

If a keypoint's confidence is below `ANKLE_CONFIDENCE_MIN`, the ankle is treated as unavailable for zone checks.

**Interactive zone drawing controls**
- Left-click: add polygon vertex
- ENTER: finish the current polygon; you'll be prompted to enter zone name and type (`restricted` or `normal`)
- ESC: finish drawing and save all zones to `ZONES_FILE`

**Behavior details**
- Zones have a `type`: `restricted` or `normal`. Only entries into `restricted` zones create logged intrusion events and trigger visual ALERT text; `normal` zones are visually indicated but do not generate intrusion logs.
- The script attempts an OpenCV `pointPolygonTest` check first, falling back to `shapely` if necessary.
- Tracked persons use `boxes.id` when available; otherwise anonymous persons are assigned `-1` as their ID in logs.

**Performance & tuning**
- Increase `FRAME_SKIP` to reduce CPU/GPU load.
- Lower `CONFIDENCE_THRESHOLD` to detect weaker poses (may increase false positives).
- Increase `ANKLE_CONFIDENCE_MIN` to require more confident ankle keypoints.
- For best throughput, run on a machine with GPU and ensure `ultralytics` is configured to use it.

**Troubleshooting**
- If the camera cannot be opened, try different camera indices (0, 1, ...), or verify the file path is correct.
- If no zones are detected or `zones.json` is corrupt, run with `--reset-zones` to redraw.
- If keypoints are rarely present, try lowering `CONFIDENCE_THRESHOLD` and verify the model file matches a pose-capable YOLOv8 checkpoint.

If you want, I can also add an example `zones.json` or create a small test video and demonstrate running the script — tell me which you'd prefer.
