import cv2  # OpenCV for video I/O and drawing
import time  # Timing for periodic reset
from pathlib import Path  # Path handling
from ultralytics import YOLO  # YOLO model

# ---------------- CONFIG ----------------  # Section header
ROOT = Path(__file__).resolve().parent  # Project root (script folder)
MODEL_PATH = ROOT / "jute_bag_detector.pt"  # Model file path
DOWNLOADS_DIR = ROOT / "Downloads"  # Input videos folder
RESULTS_DIR = ROOT / "result"  # Output results folder

CONF = 0.53  # Detection confidence threshold
IMGSZ = 960  # Inference image size
IOU = 0.5  # IoU threshold for NMS/tracking

# ---------------- COUNT BOX ----------------  # Section header
BOX_X1 = 0.15  # Left bound (fraction of width)
BOX_X2 = 0.6  # Right bound (fraction of width)
BOX_Y1 = 0.50  # Top bound (fraction of height)
BOX_Y2 = 0.90  # Bottom bound (fraction of height)

# ---------------- STABILITY ----------------  # Section header
RESET_EVERY_SEC = 60  # Periodic soft reset interval

# ---------------- CROSS-VIDEO DEDUP ----------------  # Section header
CARRYOVER_WINDOW_SEC = 5  # Deduplicate only during first N seconds of a new video
CARRYOVER_STORE_LAST_SEC = 5  # Store bags visible during last N seconds of previous video
MATCH_MIN_IOU = 0.20  # Minimum bbox overlap (normalized) to treat as same bag
MATCH_MIN_HIST_SIM = 0.70  # Minimum appearance similarity (hist correlation)

# ---------------- OUTPUT ----------------  # Section header
SHOW_WINDOW = True  # Preview while processing and save annotated output
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}  # Allowed video extensions


def inside_box(cx, cy, w, h):  # Check if a point is inside the counting box
    return (w * BOX_X1 <= cx <= w * BOX_X2 and h * BOX_Y1 <= cy <= h * BOX_Y2)  # Box inclusion test

def clamp(v, lo, hi):  # Clamp value into [lo, hi]
    return max(lo, min(hi, v))


def bbox_iou(a, b):  # IoU between 2 normalized bboxes (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def bag_hist(frame, x1, y1, x2, y2):  # Simple appearance fingerprint (HSV histogram)
    h, w = frame.shape[:2]
    x1 = clamp(int(x1), 0, w - 1)
    y1 = clamp(int(y1), 0, h - 1)
    x2 = clamp(int(x2), 0, w - 1)
    y2 = clamp(int(y2), 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def hist_sim(h1, h2):  # Histogram similarity (correlation)
    if h1 is None or h2 is None:
        return -1.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def match_carryover(carryover_list, cur_bbox_norm, cur_hist):  # Find best carryover match
    best = None
    best_score = -999.0
    for item in carryover_list:
        iou = bbox_iou(item["bbox"], cur_bbox_norm)
        if iou < MATCH_MIN_IOU:
            continue
        sim = hist_sim(item["hist"], cur_hist)
        if sim < MATCH_MIN_HIST_SIM:
            continue
        score = iou + sim
        if score > best_score:
            best_score = score
            best = item
    return best


def iter_videos(folder: Path):  # List videos from a folder
    if not folder.exists():  # Guard when folder is missing
        return []  # No videos
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]  # Filter by extension
    return sorted(files)  # Sort for deterministic order


def safe_fps(cap):  # Read FPS with fallback
    fps = cap.get(cv2.CAP_PROP_FPS)  # Read FPS from capture
    if fps and fps > 0:  # Validate FPS
        return fps  # Use actual FPS
    return 25.0  # Fallback FPS


def process_video(model, video_path: Path, output_path: Path, carryover_prev):  # Process a single video
    cap = cv2.VideoCapture(str(video_path))  # Open the input video
    if not cap.isOpened():  # Check open success
        print(f"ERROR: Cannot open video: {video_path.name}")  # Report error
        return None  # Signal failure

    fps = safe_fps(cap)  # Determine FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Read width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Read height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Output codec
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))  # Output writer

    # -------- STATE --------  # Section header
    counted_ids = set()  # IDs already counted
    active_ids_in_box = set()  # IDs currently inside box
    total_count = 0  # Total bag count
    last_reset_time = time.time()  # Last reset time

    frame_idx = 0  # Frame index for time computation
    last_seen = {}  # track_id -> {'bbox_norm': (...), 'hist': ..., 'last_t': seconds}

    if hasattr(model, "reset"):  # Some YOLO trackers provide reset
        model.reset()  # Clear tracker state between videos

    while cap.isOpened():  # Main frame loop
        ret, frame = cap.read()  # Read next frame
        if not ret:  # End of video
            break  # Exit loop

        frame_idx += 1  # Increment frame index
        t_sec = frame_idx / fps  # Timestamp in seconds

        h, w, _ = frame.shape  # Frame shape

        # ---- TRACKING ----  # Section header
        results = model.track(frame, persist=True, conf=CONF, iou=IOU, imgsz=IMGSZ)  # Run tracking
        boxes = results[0].boxes  # Extract boxes

        # ---- PERIODIC SOFT RESET ----  # Section header
        now = time.time()  # Current time
        if now - last_reset_time > RESET_EVERY_SEC:  # Time to reset
            counted_ids.clear()  # Clear counted IDs
            active_ids_in_box.clear()  # Clear active IDs
            if hasattr(model, "reset"):
                model.reset()   # resets YOLO tracker IDs

            last_reset_time = now  # Update reset time
            print(f"{video_path.name}: Soft state reset")  # Log reset

        current_ids_in_box = set()  # IDs in box this frame

        if boxes is not None and boxes.id is not None:  # Validate tracking IDs
            for i in range(len(boxes)):  # Iterate detections
                cls = int(boxes.cls[i])  # Class index
                label = model.names[cls]  # Class label

                if label.lower() != "bags":  # Keep only bags
                    continue  # Skip other classes

                track_id = int(boxes.id[i])  # Tracking ID
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])  # Box coordinates

                bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)  # Normalized bbox for cross-video matching
                hfeat = bag_hist(frame, x1, y1, x2, y2)  # Appearance fingerprint
                last_seen[track_id] = {'bbox_norm': bbox_norm, 'hist': hfeat, 'last_t': t_sec}  # Update last seen

                cx = (x1 + x2) // 2  # Box center X
                cy = (y1 + y2) // 2  # Box center Y

                # ---- CHECK BOX ----  # Section header
                if inside_box(cx, cy, w, h):  # If center is inside box
                    current_ids_in_box.add(track_id)  # Track ID inside

                    # ---- COUNT WHEN A NEW ID ENTERS BOX ----  # Section header
                    if track_id not in active_ids_in_box and track_id not in counted_ids:  # New entry
                        # Cross-video de-duplication only during the first few seconds of a new clip
                        if t_sec <= CARRYOVER_WINDOW_SEC and carryover_prev:
                            matched = match_carryover(carryover_prev, bbox_norm, hfeat)
                            if matched is not None:
                                counted_ids.add(track_id)  # Mark counted to avoid later double count in this clip
                                print(f"{video_path.name}: Skipped carry-over bag at start")  # Log skip
                            else:
                                counted_ids.add(track_id)  # Mark counted
                                total_count += 1  # Increment count
                                print(f"{video_path.name}: Counted bag -> {total_count}")  # Log count
                        else:
                            counted_ids.add(track_id)  # Mark counted
                            total_count += 1  # Increment count
                            print(f"{video_path.name}: Counted bag -> {total_count}")  # Log count


                # ---- DRAW ----  # Section header
                color = (0, 255, 0) if track_id in counted_ids else (0, 0, 255)  # Green if counted
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw box
                cv2.putText(  # Draw label
                    frame,  # Target frame
                    f"ID:{track_id}",  # Text
                    (x1, y1 - 8),  # Position
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font
                    0.6,  # Font scale
                    color,  # Text color
                    2,  # Thickness
                )  # End putText

        # ---- UPDATE ACTIVE IDS ----  # Section header
        active_ids_in_box = current_ids_in_box  # Update active IDs

        # -------- UI --------  # Section header
        bx1, by1 = int(w * BOX_X1), int(h * BOX_Y1)  # Box top-left
        bx2, by2 = int(w * BOX_X2), int(h * BOX_Y2)  # Box bottom-right
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)  # Draw count box

        cv2.putText(  # Draw total count
            frame,  # Target frame
            f"Bag Count: {total_count}",  # Text
            (30, 40),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1,  # Font scale
            (0, 0, 255),  # Text color
            2,  # Thickness
        )  # End putText

        if SHOW_WINDOW:  # Show live preview
            cv2.imshow("Bag Counter (Stable + Tracking)", frame)  # Display window
            if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
                break  # Break loop

        out.write(frame)  # Save annotated frame

    carryover_out = []  # Bags to carry across clips
    video_duration_sec = frame_idx / fps  # Duration in seconds

    # Store bags that were counted and seen very recently near the end of the clip
    for tid, info in last_seen.items():
        if tid in counted_ids and (video_duration_sec - info["last_t"] <= CARRYOVER_STORE_LAST_SEC):
            carryover_out.append({"bbox": info["bbox_norm"], "hist": info["hist"]})

    cap.release()  # Release input video
    out.release()  # Release output writer
    if SHOW_WINDOW:  # Close preview window
        cv2.destroyAllWindows()  # Destroy windows

    return total_count, carryover_out  # Return count and carryover


def main():  # Program entry
    if not MODEL_PATH.exists():  # Check model exists
        print(f"ERROR: Model not found at {MODEL_PATH}")  # Report missing model
        return  # Stop execution

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output folder

    videos = iter_videos(DOWNLOADS_DIR)  # Get input videos
    if not videos:  # No videos to process
        print(f"No videos found in {DOWNLOADS_DIR}")  # Report empty input
        return  # Stop execution

    model = YOLO(str(MODEL_PATH))  # Load model

    summary_lines = ["video,total_count"]  # CSV header

    carryover_prev = []  # Cross-video carryover memory

    for video_path in videos:  # Process each video
        output_path = RESULTS_DIR / f"{video_path.stem}_result.mp4"  # Output filename
        print(f"Processing: {video_path.name}")  # Log start

        count, carryover_prev = process_video(
            model,
            video_path,
            output_path,
            carryover_prev
        )

        if count is None:  # Skip failed videos
            continue  # Next video

        summary_lines.append(f"{video_path.name},{count}")  # Append CSV row
        print(f"Saved: {output_path.name}")  # Log output

    summary_path = RESULTS_DIR / "summary.csv"  # Summary file
    summary_path.write_text("\n".join(summary_lines))  # Write summary
    print(f"Summary saved: {summary_path}")  # Log summary path


if __name__ == "__main__":  # Script entrypoint guard
    main()  # Run main
