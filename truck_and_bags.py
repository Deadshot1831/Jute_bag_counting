import cv2  # OpenCV for video I/O and drawing
import time  # Timing for periodic reset
from pathlib import Path  # Path handling
from ultralytics import YOLO  # YOLO model

# ---------------- CONFIG ----------------  # Section header
ROOT = Path(__file__).resolve().parent  # Project root (script folder)
MODEL_PATH = ROOT / "yolov11m.pt"  # Model file path
DOWNLOADS_DIR = ROOT / "Downloads"  # Input videos folder
RESULTS_DIR = ROOT / "result"  # Output results folder

CONF = 0.45  # Detection confidence threshold
IMGSZ = 640  # Inference image size
IOU = 0.5  # IoU threshold for NMS/tracking

# ---------------- COUNT BOX ----------------  # Section header
BOX_X1 = 0.15  # Left bound (fraction of width)
BOX_X2 = 0.85  # Right bound (fraction of width)
BOX_Y1 = 0.50  # Top bound (fraction of height)
BOX_Y2 = 0.90  # Bottom bound (fraction of height)

# ---------------- STABILITY ----------------  # Section header
RESET_EVERY_SEC = 60  # Periodic soft reset interval

# ---------------- OUTPUT ----------------  # Section header
SHOW_WINDOW = True  # Preview while processing and save annotated output
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}  # Allowed video extensions


def inside_box(cx, cy, w, h):  # Check if a point is inside the counting box
    return (w * BOX_X1 <= cx <= w * BOX_X2 and h * BOX_Y1 <= cy <= h * BOX_Y2)  # Box inclusion test


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


def process_video(model, video_path: Path, output_path: Path):  # Process a single video
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

    if hasattr(model, "reset"):  # Some YOLO trackers provide reset
        model.reset()  # Clear tracker state between videos

    while cap.isOpened():  # Main frame loop
        ret, frame = cap.read()  # Read next frame
        if not ret:  # End of video
            break  # Exit loop

        h, w, _ = frame.shape  # Frame shape

        # ---- TRACKING ----  # Section header
        results = model.track(frame, persist=True, conf=CONF, iou=IOU, imgsz=IMGSZ)  # Run tracking
        boxes = results[0].boxes  # Extract boxes

        # ---- PERIODIC SOFT RESET ----  # Section header
        now = time.time()  # Current time
        if now - last_reset_time > RESET_EVERY_SEC:  # Time to reset
            counted_ids.clear()  # Clear counted IDs
            active_ids_in_box.clear()  # Clear active IDs
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

                cx = (x1 + x2) // 2  # Box center X
                cy = (y1 + y2) // 2  # Box center Y

                # ---- CHECK BOX ----  # Section header
                if inside_box(cx, cy, w, h):  # If center is inside box
                    current_ids_in_box.add(track_id)  # Track ID inside

                    # ---- COUNT WHEN A NEW ID ENTERS BOX ----  # Section header
                    if track_id not in active_ids_in_box and track_id not in counted_ids:  # New entry
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

    cap.release()  # Release input video
    out.release()  # Release output writer
    if SHOW_WINDOW:  # Close preview window
        cv2.destroyAllWindows()  # Destroy windows

    return total_count  # Return count


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

    for video_path in videos:  # Process each video
        output_path = RESULTS_DIR / f"{video_path.stem}_result.mp4"  # Output filename
        print(f"Processing: {video_path.name}")  # Log start
        count = process_video(model, video_path, output_path)  # Run processing
        if count is None:  # Skip failed videos
            continue  # Next video
        summary_lines.append(f"{video_path.name},{count}")  # Append CSV row
        print(f"Saved: {output_path.name}")  # Log output

    summary_path = RESULTS_DIR / "summary.csv"  # Summary file
    summary_path.write_text("\n".join(summary_lines))  # Write summary
    print(f"Summary saved: {summary_path}")  # Log summary path


if __name__ == "__main__":  # Script entrypoint guard
    main()  # Run main
