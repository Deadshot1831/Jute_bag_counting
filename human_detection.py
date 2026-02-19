import csv
import cv2
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parent
VIDEO_PATH = "Downloads/input_1.mp4"        # input video
MODEL_PATH = "yolov8s.pt"               # pretrained YOLOv8 model

OUTPUT_DIR = ROOT / "human_detection_output"  # output folder
OUTPUT_PATH = OUTPUT_DIR / f"{Path(VIDEO_PATH).stem}_human_detection.mp4"
CSV_PATH = OUTPUT_DIR / f"{Path(VIDEO_PATH).stem}_human_detection.csv"

CONF = 0.4                               # confidence threshold
IMGSZ = 640                              # inference size

# --------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load YOLOv11 model
    model = YOLO(MODEL_PATH)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    person_class_id = None
    for cls_id, name in names.items():
        if str(name).lower() == "person":
            person_class_id = int(cls_id)
            break
    class_filter = [person_class_id] if person_class_id is not None else None
    if person_class_id is not None:
        print(f"[INFO] Using class filter for person: id={person_class_id}, name={names[person_class_id]}")
    else:
        print("[WARN] Could not find a 'person' class in model names; falling back to label filtering.")

    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (w, h))

    print("[INFO] Processing video...")

    frame_idx = 0
    with CSV_PATH.open("w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_index", "human_count"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection (NOT tracking)
            results = model.predict(
                frame,
                conf=CONF,
                imgsz=IMGSZ,
                classes=class_filter,
                verbose=False
            )

            person_count = 0
            boxes = results[0].boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    label = results[0].names[cls]

                    # Only detect humans if class filter isn't available
                    if class_filter is None and str(label).lower() != "person":
                        continue

                    person_count += 1
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        "PERSON",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            csv_writer.writerow([frame_idx, person_count])
            frame_idx += 1

            # Save frame (no preview)
            writer.write(frame)

    cap.release()
    writer.release()
    print(f"[DONE] Saved output to {OUTPUT_PATH}")
    print(f"[DONE] Saved counts to {CSV_PATH}")

if __name__ == "__main__":
    main()
