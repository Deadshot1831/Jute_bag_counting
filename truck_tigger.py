import argparse
from pathlib import Path

import cv2

from detecttruck import frame_has_truck, get_truck_class_ids, load_truck_model
from newbag1 import MODEL_PATH as DEFAULT_BAG_MODEL, RESULTS_DIR as DEFAULT_RESULTS_DIR
from newbag1 import run_bag_tracking_for_segment

FPS = 15
MAX_SEC = 120
WINDOW_SEC = 2
STEP_SEC = 2
TRUCK_FRAME_THRESHOLD = 1
TRUCK_CONF = 0.5


def _safe_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return fps
    return float(FPS)


def find_trigger_time(
    cap,
    truck_model,
    truck_class_ids,
    max_sec=MAX_SEC,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
    threshold=TRUCK_FRAME_THRESHOLD,
    conf=TRUCK_CONF,
):
    if step_sec <= 0:
        raise ValueError("step_sec must be > 0")

    fps = _safe_fps(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (total_frames / fps) if total_frames > 0 else max_sec

    if max_sec is None:
        scan_until = duration_sec
    else:
        scan_until = min(float(max_sec), float(duration_sec))

    t = 0.0
    while t < scan_until:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        truck_frames = 0
        frames_to_check = max(1, int(window_sec * fps))

        for _ in range(frames_to_check):
            ok, frame = cap.read()
            if not ok:
                break

            if frame_has_truck(
                model=truck_model,
                frame=frame,
                conf=conf,
                truck_class_ids=truck_class_ids,
                run_name="truck_trigger",
            ):
                truck_frames += 1

        if truck_frames >= threshold:
            return t

        t += step_sec

    return None


def run_trigger_pipeline(
    video_path,
    truck_model_path=None,
    bag_model_path=DEFAULT_BAG_MODEL,
    results_dir=DEFAULT_RESULTS_DIR,
    max_sec=MAX_SEC,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
    threshold=TRUCK_FRAME_THRESHOLD,
    truck_conf=TRUCK_CONF,
    bag_start_mode="trigger",
    show_window=False,
):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    truck_model = load_truck_model(truck_model_path)
    truck_class_ids = get_truck_class_ids(truck_model)

    trigger_time = find_trigger_time(
        cap=cap,
        truck_model=truck_model,
        truck_class_ids=truck_class_ids,
        max_sec=max_sec,
        window_sec=window_sec,
        step_sec=step_sec,
        threshold=threshold,
        conf=truck_conf,
    )
    cap.release()

    if trigger_time is None:
        return {
            "triggered": False,
            "trigger_time": None,
            "bag_start_sec": None,
            "bag_end_sec": None,
            "output_path": None,
            "bag_count": None,
        }

    if bag_start_mode not in {"trigger", "start"}:
        raise ValueError("bag_start_mode must be one of: trigger, start")

    bag_start_sec = 0.0 if bag_start_mode == "start" else trigger_time
    bag_end_sec = max_sec

    output_path, bag_count = run_bag_tracking_for_segment(
        video_path=video_path,
        start_sec=bag_start_sec,
        end_sec=bag_end_sec,
        model_path=bag_model_path,
        results_dir=results_dir,
        show_window=show_window,
    )

    return {
        "triggered": True,
        "trigger_time": trigger_time,
        "bag_start_sec": bag_start_sec,
        "bag_end_sec": bag_end_sec,
        "output_path": output_path,
        "bag_count": bag_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect truck in each 2-second window and start bag tracking from that window start."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--truck-model", default=None, help="Truck detector model path")
    parser.add_argument("--bag-model", default=str(DEFAULT_BAG_MODEL), help="Bag tracker model path")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Output folder")
    parser.add_argument("--max-sec", type=float, default=MAX_SEC, help="Max time to process")
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC, help="Detection window seconds")
    parser.add_argument("--step-sec", type=float, default=STEP_SEC, help="Window step seconds")
    parser.add_argument(
        "--threshold",
        type=int,
        default=TRUCK_FRAME_THRESHOLD,
        help="Minimum truck-positive frames required in a window to trigger",
    )
    parser.add_argument("--truck-conf", type=float, default=TRUCK_CONF, help="Truck detection confidence")
    parser.add_argument(
        "--bag-start",
        choices=["trigger", "start"],
        default="trigger",
        help="Start bag tracking from trigger time or from 0s after trigger is confirmed",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show bag tracking preview window while processing",
    )
    args = parser.parse_args()

    result = run_trigger_pipeline(
        video_path=args.video,
        truck_model_path=args.truck_model,
        bag_model_path=args.bag_model,
        results_dir=args.results_dir,
        max_sec=args.max_sec,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        threshold=args.threshold,
        truck_conf=args.truck_conf,
        bag_start_mode=args.bag_start,
        show_window=args.show_window,
    )

    if not result["triggered"]:
        print("No trigger found up to max_sec; bag tracking was not started.")
        return

    print(f"Trigger time: {result['trigger_time']:.2f}s")
    print(f"Bag tracking range: {result['bag_start_sec']:.2f}s to {result['bag_end_sec']:.2f}s")
    print(f"Bag count: {result['bag_count']}")
    print(f"Output video: {result['output_path']}")


if __name__ == "__main__":
    main()
