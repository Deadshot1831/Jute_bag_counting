import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime, timezone
from ultralytics import YOLO
from google.cloud import storage

from utils import SERVICE_ACCOUNT_FILE, get_location
from main import main

from main_unloading_parallel import main_unloading_parallel

# ================= VIDEO SOURCE =================
SOURCE_TYPE = "local"   # "gcs" | "local"

LOCAL_VIDEO_PATH = (
    "/Users/yashwantyadav/Desktop/Computer_Vision_Task/today_task/Downloads/input_video.mp4"
)

# GCS_VIDEO_PATH = (
#     "gs://jalgaon-videos-frames/videos/"
#     "godown-no-5-external-street/cam-10/2026/02/02/"
#     "29_2026-02-02_12-11-04.mp4"
# )

GCS_VIDEO_PATH = ("gs://jalgaon-videos-frames/videos/godown-no-5-external-street/cam-10/2026/02/02/29_2026-02-02_12-01-00.mp4")
# ===============================================

# ================= CONFIG =================
MODEL_PATH = "/Users/yashwantyadav/Desktop/Computer_Vision_Task/today_task/yolov11m.pt"
CLASS_NAME = "bags"

CONF = 0.5
IOU = 0.1
IMGSZ = 640
FRAME_SKIP = 2

OUTPUT_DIR = "./frames_local-sk-2"
PROGRESS_EVERY_FRAMES = 40
SAVE_UNCOUNTED_DETECTIONS = True
# =========================================

# ---------- ROI ----------
FRAME_W = 2048
FRAME_H = 1166

ROI_X = 400
ROI_Y = 430
ROI_W = 940
ROI_H = 600

ROI_PAD_X = 0.05
ROI_PAD_Y = 0.05

ROI_X1 = ROI_X / FRAME_W
ROI_Y1 = ROI_Y / FRAME_H
ROI_X2 = (ROI_X + ROI_W) / FRAME_W
ROI_Y2 = (ROI_Y + ROI_H) / FRAME_H

# ---------- COUNT BOX ----------
BOX_X1 = 0.15
BOX_X2 = 0.70
BOX_Y1 = 0.50
BOX_Y2 = 0.90
# ============================


# ================= HELPERS =================
def start_heartbeat(msg, interval=10):
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            print(f"[HEARTBEAT] {msg}", flush=True)
            stop.wait(interval)

    threading.Thread(target=beat, daemon=True).start()
    return stop


def inside_box(cx, cy, w, h):
    return (w * BOX_X1 <= cx <= w * BOX_X2 and h * BOX_Y1 <= cy <= h * BOX_Y2)


def draw_debug_overlays(frame, roi_xyxy, w, h):
    rx1, ry1, rx2, ry2 = roi_xyxy

    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
    cv2.putText(frame, "ROI", (rx1, ry1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    bx1, by1 = int(w * BOX_X1), int(h * BOX_Y1)
    bx2, by2 = int(w * BOX_X2), int(h * BOX_Y2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
    cv2.putText(frame, "COUNT BOX", (bx1, by1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def download_gcs_video(gs_path, out_dir):
    bucket, obj = gs_path[5:].split("/", 1)
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
    blob = client.bucket(bucket).blob(obj)

    os.makedirs(out_dir, exist_ok=True)
    local = os.path.join(out_dir, os.path.basename(obj))
    blob.download_to_filename(local)

    if os.path.getsize(local) < 1024:
        raise RuntimeError("Downloaded file too small to be a valid video")

    return local


def upload_frame(local_path, gcs_path):
    bucket, obj = gcs_path.split("/", 1)
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
    client.bucket(bucket).blob(obj).upload_from_filename(local_path)


def resolve_video_source(source_type, local_path, gcs_path, tmp_dir="./_tmp"):
    if source_type == "local":
        if not os.path.exists(local_path):
            raise FileNotFoundError(local_path)
        return local_path, local_path

    if source_type == "gcs":
        os.makedirs(tmp_dir, exist_ok=True)
        local_video = download_gcs_video(gcs_path, tmp_dir)
        return local_video, gcs_path

    raise ValueError("SOURCE_TYPE must be 'local' or 'gcs'")


# ================= MAIN =================
def process_video():
    start_time = time.time()

    print("[INFO] Loading YOLO model...")
    hb = start_heartbeat("Model initializing")
    model = YOLO(MODEL_PATH, task="detect")
    hb.set()

    model(np.zeros((640, 640, 3), np.uint8), verbose=False)

    local_video, source_video_path = resolve_video_source(
        SOURCE_TYPE,
        LOCAL_VIDEO_PATH,
        GCS_VIDEO_PATH,
    )

    cap = cv2.VideoCapture(local_video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    video_name = os.path.splitext(os.path.basename(local_video))[0]
    out_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(out_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps != fps:
        fps = 10.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    annotated_path = os.path.join(out_dir, f"{video_name}_annotated.mp4")

    writer = cv2.VideoWriter(annotated_path, fourcc, fps, (frame_w, frame_h))
    assert writer.isOpened(), "VideoWriter failed to open"

    cam_id = 10
    location_name = get_location(cam_id)

    counted_ids = set()
    total_bags = 0
    roi_detection_frames = 0
    saved_frames = 0

    first_bbox = None
    first_frame_path = ""

    frame_idx = 0
    capture_dt = datetime.now(timezone.utc)

    print("[INFO] Processing frames...")
    
    
    # timing accumulators + torch sync check
    last_boxes = None   # cache detections from last inference frame
    total_inference_time = 0.0
    num_inference_calls = 0

    use_torch_sync = False
    try:
        import torch
        use_torch_sync = torch.cuda.is_available()
    except Exception:
        use_torch_sync = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # ---------- ROI COMPUTATION ----------
        pad_x = int(w * ROI_PAD_X)
        pad_y = int(h * ROI_PAD_Y)

        rx1 = max(0, int(w * ROI_X1) - pad_x)
        ry1 = max(0, int(h * ROI_Y1) - pad_y)
        rx2 = min(w, int(w * ROI_X2) + pad_x)
        ry2 = min(h, int(h * ROI_Y2) + pad_y)

        # ---------- ALWAYS DRAW ROI + COUNT BOX ----------
        draw_debug_overlays(frame, (rx1, ry1, rx2, ry2), w, h)

        # ---------- FRAME SKIP DECISION ----------
        run_inference = (frame_idx % (FRAME_SKIP + 1) == 0)

        # ---------- RUN YOLO ONLY ON SELECTED FRAMES ----------
        if run_inference:
            roi = frame[ry1:ry2, rx1:rx2]

            # GPU-safe timing: synchronize before/after if torch+cuda available
            if use_torch_sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            results = model.track(
                roi,
                persist=True,
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                verbose=False
            )

            if use_torch_sync:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            inference_time = t1 - t0
            total_inference_time += inference_time
            num_inference_calls += 1

            # update cache (guard in case results is None)
            last_boxes = results[0].boxes if results is not None else None

            # print per-inference timing and running average
            avg_ms = (total_inference_time / num_inference_calls) * 1000.0
            print(f"[INFERENCE] frame={frame_idx} time={inference_time*1000:.2f} ms | avg={avg_ms:.2f} ms over {num_inference_calls} calls")

        # ---------- DRAW CACHED DETECTIONS (NO BLINKING) ----------
        if last_boxes is not None and getattr(last_boxes, "id", None) is not None:
            for i in range(len(last_boxes)):
                if model.names[int(last_boxes.cls[i])].lower() != CLASS_NAME:
                    continue

                tid = int(last_boxes.id[i])

                x1r, y1r, x2r, y2r = map(int, last_boxes.xyxy[i])

                # ROI → full-frame projection
                x1 = x1r + rx1
                y1 = y1r + ry1
                x2 = x2r + rx1
                y2 = y2r + ry1

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # ---------- COUNT LOGIC ----------
                if inside_box(cx, cy, w, h) and tid not in counted_ids:
                    counted_ids.add(tid)
                    total_bags += 1

                    if first_bbox is None:
                        first_bbox = (x1, y1, x2, y2)

        # ---------- ALWAYS WRITE FRAME ----------
        writer.write(frame)
        frame_idx += 1

    # ---- after loop: print final timing summary ----
    if num_inference_calls > 0:
        avg_inference_ms = (total_inference_time / num_inference_calls) * 1000.0
    else:
        avg_inference_ms = 0.0

    print(f"[STATS] inference_calls={num_inference_calls} | avg_inference_time={avg_inference_ms:.2f} ms")

    if frame_idx % PROGRESS_EVERY_FRAMES == 0:
        print(f"[PROGRESS] frame={frame_idx} bags={total_bags}")

    cap.release()
    writer.release()
    end_time = time.time()
    incident_details = []
    if first_bbox:
        x1, y1, x2, y2 = first_bbox
        incident_details.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "value": str(total_bags),
            "name": "Bag Detection",
            "type": "metric",
            "class": "bags",
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })

    final_json = {
        "reference_id": os.path.basename(local_video),
        "location_details": {
            "cam_id": cam_id,
            "location_name": location_name,
        },
        "incident_details": incident_details,
        "video": {
            "source_video_path": source_video_path,
            "annotated_video_path": annotated_path,
            "capture_datetime": capture_dt.isoformat(),
        },
        "frame_path": first_frame_path,
        "incident_name": "Bag Detection",
        "total_bags": total_bags,
    }

    main(
        data=final_json,
        location_name=location_name,
        incident_type="bag_detection",
        video_file_name=os.path.basename(local_video),
        video_file_path=source_video_path,
    )

    print(f"[DONE] total_bags={total_bags} | video={annotated_path} | Time_taken = {int(end_time-start_time)}")


if __name__ == "__main__":
    process_video()
# if __name__ == "__main__":

#     video_meta = {
#         "bucket": "jalgaon-videos-frames",
#         "site": "godown-no-5-external-street",
#         "cam_id": 10,
#         "date": "2026-02-02",
#     }

#     result = main_unloading_parallel(
#         video_meta=video_meta,

#         # multiprocessing controls
#         num_parts=None,        # auto split
#         max_workers=None,      # cpu_count - 1

#         # keep pipeline same
#         incident_type="bag_detection",
#         upload_annotated=True,
#         annotated_prefix="annotated_videos",
#         cleanup_local=True,
#     )

#     print("\n===== PARALLEL RESULT =====")
#     print(f"File       : {result.get('processed_file')}")
#     print(f"Total bags : {result.get('total_bags')}")
#     print(f"GCS video  : {result.get('annotated_gs_path')}")