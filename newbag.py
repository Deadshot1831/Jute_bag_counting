import cv2  # OpenCV for video I/O and drawing
import time  # Timing for periodic reset
from pathlib import Path  # Path handling
from ultralytics import YOLO  # YOLO model

# ---------------- CONFIG ----------------  # Section header
ROOT = Path(__file__).resolve().parent  # Project root (script folder)
MODEL_PATH = ROOT / "yolov11m.pt"  # Model file path
DOWNLOADS_DIR = ROOT / "Downloads"  # Input videos folder
RESULTS_DIR = ROOT / "result"  # Output results folder

CONF = 0.4  # Detection confidence threshold
IMGSZ = 640  # Inference image size
IOU = 0.5  # IoU threshold for NMS/tracking

# ---------------- ROI (PROCESS ONLY THIS REGION) ----------------
# White box ROI from your screenshot (fractions of FULL frame)
# Only this region will be passed to YOLO for detection/tracking to reduce compute.
FRAME_W = 2048
FRAME_H = 1166

ROI_X = 400      # left
ROI_Y = 430      # top
ROI_W = 940    # width  (CHANGE THIS)
ROI_H = 600   # height (CHANGE THIS)

ROI_X1 = ROI_X / FRAME_W
ROI_Y1 = ROI_Y / FRAME_H
ROI_X2 = (ROI_X + ROI_W) / FRAME_W
ROI_Y2 = (ROI_Y + ROI_H) / FRAME_H

USE_ROI = True  # set False to process full frame

# Padding around ROI (fractions of full frame). Helps tracking stability at ROI borders.
ROI_PAD_X = 0.05
ROI_PAD_Y = 0.05


# ---------------- COUNT BOX ----------------  # Section header
BOX_X1 = 0.15  # Left bound (fraction of width)
BOX_X2 = 0.7  # Right bound (fraction of width)
BOX_Y1 = 0.50  # Top bound (fraction of height)
BOX_Y2 = 0.90  # Bottom bound (fraction of height)

# ---------------- STABILITY ----------------  # Section header
RESET_EVERY_SEC = 120  # Periodic soft reset interval

# ---------------- CROSS-VIDEO DEDUP ----------------  # Section header
CARRYOVER_WINDOW_SEC = 5  # Deduplicate only during first N seconds of a new video
CARRYOVER_STORE_LAST_SEC = 5  # Store bags visible during last N seconds of previous video
MATCH_MIN_IOU = 0.20  # Minimum bbox overlap (normalized) to treat as same bag
MATCH_MIN_HIST_SIM = 0.70  # Minimum appearance similarity (hist correlation)

# ---------------- OUTPUT ----------------  # Section header
SHOW_WINDOW = False  # Preview while processing and save annotated output
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
    track_stats = {}  # track_id -> {'frames_seen': int, 'first_frame': int, 'last_frame': int}

    if hasattr(model, "reset"):  # Some YOLO trackers provide reset
        model.reset()  # Clear tracker state between videos

    while cap.isOpened():  # Main frame loop
        ret, frame = cap.read()  # Read next frame
        if not ret:  # End of video
            break  # Exit loop

        frame_idx += 1  # Increment frame index
        t_sec = frame_idx / fps  # Timestamp in seconds

        h, w, _ = frame.shape  # Frame shape

        # ---- ROI CROP (to reduce processing) ----
        H_full, W_full = h, w
        if USE_ROI:
            pad_x = int(W_full * ROI_PAD_X)
            pad_y = int(H_full * ROI_PAD_Y)
            rx1 = int(W_full * ROI_X1) - pad_x
            ry1 = int(H_full * ROI_Y1) - pad_y
            rx2 = int(W_full * ROI_X2) + pad_x
            ry2 = int(H_full * ROI_Y2) + pad_y
            rx1 = clamp(rx1, 0, W_full - 1)
            ry1 = clamp(ry1, 0, H_full - 1)
            rx2 = clamp(rx2, 0, W_full)
            ry2 = clamp(ry2, 0, H_full)
            if rx2 <= rx1 or ry2 <= ry1:
                # fallback to full frame if ROI is invalid
                rx1, ry1, rx2, ry2 = 0, 0, W_full, H_full
            proc_frame = frame[ry1:ry2, rx1:rx2]
            x_off, y_off = rx1, ry1
        else:
            proc_frame = frame
            x_off, y_off = 0, 0


        # ---- TRACKING ----  # Section header
        results = model.track(proc_frame, persist=True, conf=CONF, iou=IOU, imgsz=IMGSZ)  # Run tracking
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
                x1r, y1r, x2r, y2r = map(int, boxes.xyxy[i])  # Box coords in ROI frame

                # Convert ROI coords -> full-frame coords
                x1, y1, x2, y2 = x1r + x_off, y1r + y_off, x2r + x_off, y2r + y_off

                # ---- TRACK STATS ----  # Section header
                if track_id not in track_stats:
                    track_stats[track_id] = {
                        "frames_seen": 0,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                    }
                track_stats[track_id]["frames_seen"] += 1
                track_stats[track_id]["last_frame"] = frame_idx

                bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)  # Normalized bbox for cross-video matching
                hfeat = bag_hist(proc_frame, x1r, y1r, x2r, y2r)  # Appearance fingerprint (on ROI crop)
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

        # Draw ROI boundary (white box) for reference
        if USE_ROI:
            cv2.rectangle(frame, (x_off, y_off), (x_off + proc_frame.shape[1], y_off + proc_frame.shape[0]), (255, 255, 255), 2)

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

    return total_count, carryover_out, track_stats, fps  # Return count, carryover, stats, fps


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

        count, carryover_prev, track_stats, fps = process_video(
            model,
            video_path,
            output_path,
            carryover_prev
        )

        if count is None:  # Skip failed videos
            continue  # Next video

        summary_lines.append(f"{video_path.name},{count}")  # Append CSV row
        per_video_csv = RESULTS_DIR / f"{video_path.stem}_tracks.csv"
        lines = ["track_id,frames_seen,seconds_seen,first_frame,last_frame"]
        for tid in sorted(track_stats.keys()):
            stats = track_stats[tid]
            seconds_seen = stats["frames_seen"] / fps
            lines.append(
                f"{tid},{stats['frames_seen']},{seconds_seen:.3f},{stats['first_frame']},{stats['last_frame']}"
            )
        lines.append(f"TOTAL_BAG_COUNT,{count},,,")
        per_video_csv.write_text("\n".join(lines))
        print(f"Saved: {output_path.name}")  # Log output

    summary_path = RESULTS_DIR / "summary.csv"  # Summary file
    summary_path.write_text("\n".join(summary_lines))  # Write summary
    print(f"Summary saved: {summary_path}")  # Log summary path

if __name__ == "__main__":  # Script entrypoint guard
    main()  # Run main
