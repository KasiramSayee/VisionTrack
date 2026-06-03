"""
Retail Zone Intrusion Detection - Single File Implementation (`zone_detector.py`)

Dependencies (install via pip):
- ultralytics
- opencv-python
- numpy
- shapely
"""

# =========================
# CONFIGURATION
# =========================
VIDEO_SOURCE = 0  # 0 for default webcam, or path string to video file
MODEL_NAME = "yolov8n-pose.pt"
ZONES_FILE = "zones.json"
LOG_FILE = "intrusion_log.csv"
ALERT_COOLDOWN_SECONDS = 30
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 2  # process every Nth frame
ANKLE_CONFIDENCE_MIN = 0.3

# =========================
# IMPORTS
# =========================
import argparse
import csv
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from ultralytics import YOLO


# =========================
# ZONE LOAD / SAVE
# =========================
def load_zones(zones_path: str):
    """Load zones from JSON file. Returns list of dicts with keys: name, type, points."""
    if not os.path.exists(zones_path):
        return []

    with open(zones_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zones = []
    for z in data:
        # normalize
        name = z.get("name", "Unnamed")
        ztype = z.get("type", "restricted")
        points = z.get("points", [])
        # ensure points are tuples of (x, y)
        points = [(int(p[0]), int(p[1])) for p in points]
        zones.append({"name": name, "type": ztype, "points": points})
    return zones


def save_zones(zones, zones_path: str):
    """Save zones list to JSON file."""
    serializable = []
    for z in zones:
        serializable.append(
            {
                "name": z["name"],
                "type": z["type"],
                "points": [[int(x), int(y)] for (x, y) in z["points"]],
            }
        )
    with open(zones_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


# =========================
# INTERACTIVE ZONE SETUP
# =========================
def draw_zone_setup(first_frame, zones_path: str):
    """
    Interactive zone drawing mode.

    - Left-click to add points for current polygon.
    - Press ENTER to finish current polygon, then enter zone name and type in terminal.
    - Press ESC to finish all zones and save to JSON.
    """

    window_name = "Zone Setup - Click to draw polygons, ENTER=finish polygon, ESC=done"
    clone = first_frame.copy()
    current_points = []
    zones = []

    instructions = [
        "Zone Setup Mode",
        "Left Click: add polygon vertex",
        "ENTER: finish polygon & define zone",
        "ESC: finish all and save",
    ]

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = clone.copy()

        # draw existing zones preview (if any)
        for z in zones:
            pts = np.array(z["points"], dtype=np.int32)
            if len(pts) >= 3:
                color = (0, 0, 255) if z["type"] == "restricted" else (0, 255, 0)
                cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)
                # semi-transparent fill
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
                cv2.putText(
                    display,
                    f"{z['name']} ({z['type']})",
                    (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # draw current polygon being defined
        for i, pt in enumerate(current_points):
            cv2.circle(display, pt, 4, (255, 255, 0), -1)
            if i > 0:
                cv2.line(display, current_points[i - 1], pt, (255, 255, 0), 2)

        # draw instructions text
        y0 = 20
        for line in instructions:
            cv2.putText(
                display,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y0 += 25

        cv2.imshow(window_name, display)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC
            # finish without adding incomplete polygon
            break
        elif key == 13:  # ENTER
            if len(current_points) >= 3:
                print("\n--- New Zone Definition ---")
                zone_name = input("Enter zone name: ").strip() or "Zone"
                while True:
                    ztype = input(
                        "Enter zone type ('restricted' or 'normal'): "
                    ).strip().lower()
                    if ztype in ("restricted", "normal"):
                        break
                    print("Invalid type. Please enter 'restricted' or 'normal'.")
                zones.append(
                    {"name": zone_name, "type": ztype, "points": list(current_points)}
                )
                current_points = []
                print(f"Zone '{zone_name}' ({ztype}) added.")
            else:
                print("Polygon must have at least 3 points.")
        # otherwise continue loop

    cv2.destroyWindow(window_name)

    if zones:
        save_zones(zones, zones_path)
        print(f"Saved {len(zones)} zones to '{zones_path}'.")
    else:
        print("No zones defined; nothing saved.")

    return zones


# =========================
# FOOT-IN-ZONES CHECK
# =========================
def _point_in_polygon_cv2(point, polygon_points):
    """
    Check if point is inside polygon using cv2.pointPolygonTest.
    Returns True if inside or on boundary, False otherwise.
    """
    if point is None:
        return False
    x, y = point
    if polygon_points is None or len(polygon_points) < 3:
        return False
    contour = np.array(polygon_points, dtype=np.int32)
    # Ensure shape Nx1x2 as expected by pointPolygonTest
    if contour.ndim == 2:
        contour = contour.reshape((-1, 1, 2))
    res = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
    return res >= 0  # inside or on edge


def _point_in_polygon_shapely(point, polygon_points):
    """Fallback: use shapely if needed."""
    if point is None:
        return False
    x, y = point
    if polygon_points is None or len(polygon_points) < 3:
        return False
    poly = ShapelyPolygon(polygon_points)
    p = ShapelyPoint(x, y)
    # within or on border (buffer(0) not necessary here)
    return p.within(poly) or p.touches(poly)


def check_foot_in_zones(left_ankle, right_ankle, zones):
    """
    Check if either ankle is inside any zones.

    Returns:
        restricted_zone_names: list of names of restricted zones containing a foot
        normal_zone_names: list of names of normal zones containing a foot
    """
    restricted_hits = set()
    normal_hits = set()

    for z in zones:
        pts = z["points"]
        name = z["name"]
        ztype = z["type"]

        # Using OpenCV first, fallback to shapely if any error
        try:
            inside_left = _point_in_polygon_cv2(left_ankle, pts)
            inside_right = _point_in_polygon_cv2(right_ankle, pts)
        except Exception:
            inside_left = _point_in_polygon_shapely(left_ankle, pts)
            inside_right = _point_in_polygon_shapely(right_ankle, pts)

        if inside_left or inside_right:
            if ztype == "restricted":
                restricted_hits.add(name)
            else:
                normal_hits.add(name)

    return list(restricted_hits), list(normal_hits)


# =========================
# INTRUSION LOGGING
# =========================
def log_intrusion(
    log_path: str,
    timestamp: float,
    person_id: int,
    zone_name: str,
    left_ankle,
    right_ankle,
):
    """
    Append an intrusion event row to CSV log.

    Columns:
        timestamp_iso, person_id, zone_name, left_ankle_x, left_ankle_y,
        right_ankle_x, right_ankle_y
    """
    ts_iso = datetime.fromtimestamp(timestamp).isoformat()

    la_x, la_y = (left_ankle if left_ankle is not None else (None, None))
    ra_x, ra_y = (right_ankle if right_ankle is not None else (None, None))

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "person_id",
                    "zone_name",
                    "left_ankle_x",
                    "left_ankle_y",
                    "right_ankle_x",
                    "right_ankle_y",
                ]
            )
        writer.writerow([ts_iso, person_id, zone_name, la_x, la_y, ra_x, ra_y])


# =========================
# OVERLAY RENDERING
# =========================
# COCO keypoint skeleton pairs for YOLOv8 Pose (standard)
SKELETON_PAIRS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (1, 6),
]


def draw_overlay(
    frame,
    zones,
    persons,
    intrusion_active: bool,
    fps: float,
):
    """
    Draw zones, skeletons, bounding boxes, person IDs, ankle dots, and alerts.

    persons: list of dicts containing:
        id, bbox (x1,y1,x2,y2), keypoints (Nx2), keypoint_scores (N),
        left_ankle, right_ankle, in_restricted (bool), in_normal (bool)
    """
    h, w = frame.shape[:2]

    # Draw zones with semi-transparent fill
    overlay = frame.copy()
    for z in zones:
        pts = np.array(z["points"], dtype=np.int32)
        if len(pts) < 3:
            continue
        color = (0, 0, 255) if z["type"] == "restricted" else (0, 255, 0)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        cv2.fillPoly(overlay, [pts], color)
        # semi-transparent
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # label at first point
        label = f"{z['name']} ({z['type']})"
        x, y = pts[0][0], pts[0][1]
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # Draw persons
    for person in persons:
        pid = person["id"]
        x1, y1, x2, y2 = person["bbox"]
        in_restricted = person.get("in_restricted", False)
        in_normal = person.get("in_normal", False)
        keypoints = person.get("keypoints")
        kps_scores = person.get("keypoint_scores")
        left_ankle = person.get("left_ankle")
        right_ankle = person.get("right_ankle")

        if in_restricted:
            box_color = (0, 0, 255)  # red
        elif in_normal:
            box_color = (0, 255, 0)  # green
        else:
            box_color = (255, 255, 255)  # white if no zone

        # Bounding box
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            box_color,
            2,
        )

        # Person ID above box
        label = f"ID {pid}"
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
            cv2.LINE_AA,
        )

        # Skeleton
        if keypoints is not None and kps_scores is not None:
            for (i, j) in SKELETON_PAIRS:
                if i < len(keypoints) and j < len(keypoints):
                    if (
                        kps_scores[i] is not None
                        and kps_scores[i] >= ANKLE_CONFIDENCE_MIN
                        and kps_scores[j] is not None
                        and kps_scores[j] >= ANKLE_CONFIDENCE_MIN
                    ):
                        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        # Ankle dots (bright yellow)
        if left_ankle is not None:
            cv2.circle(
                frame,
                (int(left_ankle[0]), int(left_ankle[1])),
                5,
                (0, 255, 255),
                -1,
            )
        if right_ankle is not None:
            cv2.circle(
                frame,
                (int(right_ankle[0]), int(right_ankle[1])),
                5,
                (0, 255, 255),
                -1,
            )

    # FPS counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Alert text
    if intrusion_active:
        alert_text = "ALERT: Zone Violated"
        (tw, th), _ = cv2.getTextSize(
            alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3
        )
        x = int((w - tw) / 2)
        y = 50
        cv2.putText(
            frame,
            alert_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return frame


# =========================
# MAIN APPLICATION
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Retail Zone Intrusion Detection using YOLOv8 Pose and ByteTrack"
    )
    parser.add_argument(
        "--reset-zones",
        action="store_true",
        help="Delete existing zones.json and re-enter zone-drawing mode",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional override for VIDEO_SOURCE (index or file path)",
    )
    args = parser.parse_args()

    # Determine actual video source
    if args.source is not None:
        try:
            # If it's an integer index
            src = int(args.source)
        except ValueError:
            src = args.source
        video_source = src
    else:
        video_source = VIDEO_SOURCE

    # Handle reset-zones flag
    if args.reset_zones and os.path.exists(ZONES_FILE):
        os.remove(ZONES_FILE)
        print(f"Removed existing zones file: {ZONES_FILE}")

    # Open video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    # Read first frame for zone setup if needed
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print("Error: Could not read first frame from video source.")
        cap.release()
        return

    # Load or define zones
    zones = load_zones(ZONES_FILE)
    if not zones:
        print("No zones found. Entering interactive zone-drawing mode.")
        zones = draw_zone_setup(first_frame, ZONES_FILE)
        # If still no zones, just continue with no zone logic
        if not zones:
            print("Warning: Running without any zones defined.")

    # Reset capture to start of stream (for files); for webcam it doesn't matter
    cap.release()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not reopen video source: {video_source}")
        return

    # Load YOLOv8 Pose model
    print(f"Loading model '{MODEL_NAME}'...")
    model = YOLO(MODEL_NAME)

    # Tracking and cooldown state
    last_log_time = {}  # key: (person_id, zone_name) -> last timestamp
    frame_idx = 0
    last_inference_results = None
    last_time = time.time()
    fps = 0.0

    window_name = "Zone Detector"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        current_time = time.time()
        dt = current_time - last_time
        if dt > 0:
            fps = 1.0 / dt
        last_time = current_time

        # Decide whether to run inference this frame
        run_inference = (frame_idx % FRAME_SKIP == 0) or (last_inference_results is None)

        if run_inference:
            try:
                results_list = model.track(
                    frame,
                    persist=True,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )
            except Exception as e:
                print(f"Error during model.track: {e}")
                break

            if not results_list:
                last_inference_results = None
            else:
                last_inference_results = results_list[0]

        persons = []
        intrusion_active = False

        if last_inference_results is not None:
            boxes = last_inference_results.boxes
            keypoints = last_inference_results.keypoints

            if boxes is not None and keypoints is not None:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
                ids = boxes.id
                if ids is not None:
                    ids = ids.cpu().numpy().astype(int) if hasattr(ids, "cpu") else ids.astype(int)
                else:
                    # If no tracking IDs, treat as -1 (anonymous)
                    ids = np.full((xyxy.shape[0],), -1, dtype=int)

                kps_xy = keypoints.xy.cpu().numpy() if hasattr(keypoints.xy, "cpu") else keypoints.xy
                kps_conf = (
                    keypoints.conf.cpu().numpy()
                    if hasattr(keypoints.conf, "cpu")
                    else keypoints.conf
                )

                for i in range(len(xyxy)):
                    pid = int(ids[i])
                    x1, y1, x2, y2 = xyxy[i]
                    kp_xy = kps_xy[i]  # shape (num_kps, 2)
                    kp_conf = kps_conf[i]  # shape (num_kps,)

                    # Get ankles if confident enough
                    left_ankle = None
                    right_ankle = None

                    if kp_conf is not None and len(kp_conf) > 16:
                        if kp_conf[15] >= ANKLE_CONFIDENCE_MIN:
                            left_ankle = (float(kp_xy[15][0]), float(kp_xy[15][1]))
                        if kp_conf[16] >= ANKLE_CONFIDENCE_MIN:
                            right_ankle = (float(kp_xy[16][0]), float(kp_xy[16][1]))

                    # If no ankles, skip zone check but still draw box/skeleton
                    in_restricted = False
                    in_normal = False
                    violated_zones = []

                    if left_ankle is not None or right_ankle is not None:
                        restricted_hits, normal_hits = check_foot_in_zones(
                            left_ankle, right_ankle, zones
                        )
                        if restricted_hits:
                            in_restricted = True
                            violated_zones = restricted_hits
                        if normal_hits:
                            in_normal = True

                    # Intrusion logging with cooldown for restricted zones
                    if in_restricted and violated_zones:
                        intrusion_active = True
                        for zone_name in violated_zones:
                            key = (pid, zone_name)
                            last_ts = last_log_time.get(key, 0)
                            if (current_time - last_ts) >= ALERT_COOLDOWN_SECONDS:
                                log_intrusion(
                                    LOG_FILE,
                                    current_time,
                                    pid,
                                    zone_name,
                                    left_ankle,
                                    right_ankle,
                                )
                                last_log_time[key] = current_time

                    persons.append(
                        {
                            "id": pid,
                            "bbox": (x1, y1, x2, y2),
                            "keypoints": kp_xy,
                            "keypoint_scores": kp_conf,
                            "left_ankle": left_ankle,
                            "right_ankle": right_ankle,
                            "in_restricted": in_restricted,
                            "in_normal": in_normal,
                            "violated_zones": violated_zones,
                        }
                    )

        # Draw overlay
        output_frame = draw_overlay(frame, zones, persons, intrusion_active, fps)

        cv2.imshow(window_name, output_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()