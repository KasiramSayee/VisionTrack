# Zone Detector

Retail zone intrusion detection using YOLOv8 Pose.

**Quick overview**
- Detects people and their ankle keypoints and checks whether a foot enters user-defined polygons (zones).
- Supports interactive zone-definition, intrusion logging to CSV, and simple visual alerts.

**Files**
- [zone_detector.py](zone_detector.py#L1-L999): Main script.
- [OPTIONS.md](OPTIONS.md): Detailed runtime/configuration options.
- [yolov8n-pose.pt](yolov8n-pose.pt): Expected model file (place in project root).

**Dependencies**
Install the Python packages below (recommended in a venv):

```bash
pip install ultralytics opencv-python numpy shapely
```

**Quick start**
- Start with interactive zone setup (if you don't have `zones.json`):

```bash
python zone_detector.py
```

- Override camera or video source (example: camera index 0, or file path):

```bash
python zone_detector.py --source 0
python zone_detector.py --source "C:/path/to/video.mp4"
```

- Force re-entering zone-drawing mode (deletes existing `zones.json`):

```bash
python zone_detector.py --reset-zones
```

**Where outputs go**
- Zones are saved to `zones.json` in the project root.
- Intrusion events are appended to `intrusion_log.csv`.

**Controls (interactive and runtime)**
- During interactive zone-drawing: Left-click to add vertices, press ENTER to finish a polygon and provide name/type, press ESC to finish and save.
- During live preview: press ESC to quit.

**Notes & tips**
- The script expects the pose model file referenced by the `MODEL_NAME` constant at the top of `zone_detector.py` (default `yolov8n-pose.pt`).
- To improve performance, increase `FRAME_SKIP` in `zone_detector.py` to skip frames, or run on a machine with GPU support for `ultralytics`.
- Ankle keypoints used are indices 15 (left ankle) and 16 (right ankle) from YOLOv8 pose output.