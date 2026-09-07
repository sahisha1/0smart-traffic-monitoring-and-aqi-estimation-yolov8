"""
Vehicle Detection, Tracking, Counting & Speed Estimation
---------------------------------------------------------
Uses YOLOv8 for detection + Ultralytics' built-in ByteTrack for
multi-object tracking (replaces a hand-rolled centroid tracker).

NOTE on estimates: AQI and speed values here are heuristic proxies for
demo purposes, not calibrated physical measurements. See README notes
in estimate_aqi() and CalibrationConfig for what would be needed to
make them accurate (camera calibration / homography, real emissions
factors, etc). Be upfront about this in an interview -- it's a design
choice, not an oversight, as long as you can explain it.
"""

import os
import math
import argparse
from dataclasses import dataclass, field

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    video_path: str = "Vehicle-Detection-Classification-and-Counting-main/Videos/video.mp4"
    model_weights: str = "yolov8n.pt"
    frame_size: tuple = (900, 500)
    conf_threshold: float = 0.3
    vehicle_classes: tuple = (2, 7)  # COCO: 2=car, 7=truck
    class_labels: dict = field(default_factory=lambda: {2: "Car", 7: "Truck"})

    # Counting lines (pixel y-coordinates in the resized frame)
    line_up: int = 400
    line_down: int = 250

    # Calibration: real-world distance between the two lines.
    # This is a rough estimate unless you've measured it against the
    # actual road/camera geometry. State this assumption out loud.
    real_distance_meters: float = 8.0
    speed_limit_kmh: float = 100.0

    output_dir: str = "detected"
    show_video: bool = True


# ============================================================
# EMISSIONS / AQI HEURISTIC
# ============================================================

def estimate_aqi(car_count: int, truck_count: int):
    """
    Rough proxy AQI based on a linear emissions-weight heuristic.
    NOT a calibrated environmental model -- trucks are weighted ~7.5x
    a car's emission factor, then scaled down and capped at 500 to
    stay in a plausible AQI-like range. Good enough to demonstrate
    trend/relative comparison across a video, not for real air-quality
    reporting.
    """
    CAR_EMISSION_RATE = 120
    TRUCK_EMISSION_RATE = 900

    total_emissions = car_count * CAR_EMISSION_RATE + truck_count * TRUCK_EMISSION_RATE
    aqi = min(int(total_emissions / 100), 500)

    if aqi < 50:
        category = "Good"
    elif aqi < 100:
        category = "Moderate"
    elif aqi < 150:
        category = "Unhealthy for Sensitive Groups"
    elif aqi < 200:
        category = "Unhealthy"
    elif aqi < 300:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"
    return aqi, category


# ============================================================
# TRACK STATE
# ============================================================

class TrackState:
    """
    Per-track bookkeeping keyed by the tracker's persistent track_id.
    Replaces the old Car class + hasattr(car, 'counted') hack.
    """

    def __init__(self, track_id: int, cy: int, frame_idx: int):
        self.track_id = track_id
        self.first_cy = cy
        self.last_cy = cy
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.counted = False       # explicit flag, always initialized
        self.direction = None      # 'up' | 'down'
        self.label = None
        self.speed_kmh = None

    def update(self, cy: int, frame_idx: int):
        self.last_cy = cy
        self.last_frame = frame_idx


class VehicleCounter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: dict[int, TrackState] = {}
        self.cnt_up = 0
        self.cnt_down = 0
        self.cnt_car = 0
        self.cnt_truck = 0
        self.speeds: dict[int, float] = {}
        self.overspeed_ids: list[int] = []

        pixel_distance = abs(cfg.line_down - cfg.line_up)
        self.meters_per_pixel = cfg.real_distance_meters / pixel_distance
        self.pixel_distance = pixel_distance

        os.makedirs(cfg.output_dir, exist_ok=True)

    def _crossed_up(self, prev_cy, cur_cy):
        # moving from below line_up to above it (y decreasing = "up" on screen)
        return prev_cy >= self.cfg.line_up > cur_cy

    def _crossed_down(self, prev_cy, cur_cy):
        return prev_cy <= self.cfg.line_down < cur_cy

    def process_detection(self, track_id, cy, label, frame_idx, fps, frame, box):
        state = self.tracks.get(track_id)
        if state is None:
            state = TrackState(track_id, cy, frame_idx)
            state.label = label
            self.tracks[track_id] = state
            return

        prev_cy = state.last_cy
        state.update(cy, frame_idx)

        if state.counted:
            return  # already counted once, don't double count

        crossed = None
        if self._crossed_up(prev_cy, cy):
            crossed = "up"
        elif self._crossed_down(prev_cy, cy):
            crossed = "down"

        if crossed is None:
            return

        state.counted = True
        state.direction = crossed
        if crossed == "up":
            self.cnt_up += 1
        else:
            self.cnt_down += 1

        if label == "Car":
            self.cnt_car += 1
        else:
            self.cnt_truck += 1

        # ---- speed estimate ----
        elapsed_frames = max(state.last_frame - state.first_frame, 1)
        time_seconds = elapsed_frames / fps if fps > 0 else 1
        real_distance = self.pixel_distance * self.meters_per_pixel
        speed_kmh = (real_distance / time_seconds) * 3.6

        if 0 < speed_kmh < 180:  # sanity clamp against tracker noise
            state.speed_kmh = round(speed_kmh, 2)
            self.speeds[track_id] = state.speed_kmh
            if speed_kmh > self.cfg.speed_limit_kmh:
                self.overspeed_ids.append(track_id)
                x1, y1, x2, y2 = box
                vehicle_img = frame[y1:y2, x1:x2]
                if vehicle_img.size > 0:
                    filename = os.path.join(
                        self.cfg.output_dir, f"overspeed_vehicle_{track_id}.jpg"
                    )
                    cv2.imwrite(filename, vehicle_img)

    def report(self):
        total_vehicles = self.cnt_car + self.cnt_truck
        aqi, category = estimate_aqi(self.cnt_car, self.cnt_truck)

        print(f"Total vehicles detected: {self.cnt_up + self.cnt_down}")
        print(f"Total overspeeding vehicles: {len(self.overspeed_ids)}")
        print("\n--- Speeds (km/h) ---")
        for vid, spd in self.speeds.items():
            flag = " --> OVERSPEEDING" if spd > self.cfg.speed_limit_kmh else ""
            print(f"Vehicle {vid}: {spd} km/h{flag}")
        print(f"Overspeeding Vehicle IDs: {self.overspeed_ids}")
        print(f"\nEstimated AQI: {aqi} ({category})")

        with open("aqi_report.txt", "w") as f:
            f.write(f"Total Vehicles: {total_vehicles}\n")
            f.write(f"Estimated AQI: {aqi}\n")
            f.write(f"Air Quality: {category}\n")
            f.write(f"Total overspeeding vehicles: {len(self.overspeed_ids)}\n")
            f.write(f"Overspeeding Vehicle IDs: {self.overspeed_ids}\n")
        print("\nAQI report saved successfully!")

        return total_vehicles, aqi, category


# ============================================================
# LIVE PLOTS
# ============================================================

class LiveGraphs:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.bars = self.ax.bar(["Cars", "Trucks", "Up", "Down"], [0, 0, 0, 0])
        self.ax.set_ylim(0, 60)

        self.fig_aqi, self.ax_aqi = plt.subplots()
        self.x_data, self.y_data = [], []
        (self.line_aqi,) = self.ax_aqi.plot([], [], "r-", label="AQI")
        self.ax_aqi.set_title("AQI Over Time")
        self.ax_aqi.set_xlabel("Frame Count")
        self.ax_aqi.set_ylabel("AQI")
        self.ax_aqi.set_ylim(0, 500)
        self.ax_aqi.legend()

    def update(self, cnt_car, cnt_truck, cnt_up, cnt_down, frame_idx):
        for bar, val in zip(self.bars, [cnt_car, cnt_truck, cnt_up, cnt_down]):
            bar.set_height(val)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        aqi, _ = estimate_aqi(cnt_car, cnt_truck)
        self.x_data.append(frame_idx)
        self.y_data.append(aqi)
        self.line_aqi.set_data(self.x_data, self.y_data)
        self.ax_aqi.set_xlim(0, max(50, len(self.x_data)))
        self.ax_aqi.set_ylim(0, max(100, max(self.y_data) + 20))
        self.fig_aqi.canvas.draw()
        self.fig_aqi.canvas.flush_events()

    def save(self):
        self.fig.savefig("vehicle_count_bar_chart.png")
        self.fig_aqi.savefig("aqi_over_time_line_graph.png")
        print("Bar chart saved as 'vehicle_count_bar_chart.png'")
        print("AQI line graph saved as 'aqi_over_time_line_graph.png'")


# ============================================================
# MAIN LOOP
# ============================================================

def run(cfg: Config):
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg.video_path}")

    model = YOLO(cfg.model_weights)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    counter = VehicleCounter(cfg)
    graphs = LiveGraphs()

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame = cv2.resize(frame, cfg.frame_size)

            # ByteTrack via Ultralytics: persist=True keeps IDs stable
            # across frames instead of re-matching centroids by hand.
            results = model.track(
                frame,
                persist=True,
                classes=list(cfg.vehicle_classes),
                conf=cfg.conf_threshold,
                verbose=False,
            )[0]

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                cls_ids = results.boxes.cls.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                for box, cls_id, track_id, conf in zip(boxes, cls_ids, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cy = (y1 + y2) // 2
                    label = cfg.class_labels.get(int(cls_id), "Vehicle")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"{label} #{track_id} {conf:.2f}",
                        (x1, y1 - 5), font, 0.5, (255, 255, 0), 1,
                    )

                    counter.process_detection(
                        track_id, cy, label, frame_idx, fps, frame, (x1, y1, x2, y2)
                    )

            # ---- overlay lines & counts ----
            w = cfg.frame_size[0]
            cv2.line(frame, (0, cfg.line_up), (w, cfg.line_up), (255, 0, 255), 2)
            cv2.line(frame, (0, cfg.line_down), (w, cfg.line_down), (255, 0, 0), 2)
            cv2.putText(frame, f"UP: {counter.cnt_up}", (10, 40), font, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"DOWN: {counter.cnt_down}", (10, 80), font, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Cars: {counter.cnt_car}", (10, 120), font, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Trucks: {counter.cnt_truck}", (10, 160), font, 0.6, (0, 255, 255), 2)

            graphs.update(counter.cnt_car, counter.cnt_truck, counter.cnt_up, counter.cnt_down, frame_idx)

            if cfg.show_video:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("h"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        graphs.save()
        counter.report()


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle detection, tracking & counting")
    p.add_argument("--video", default=Config.video_path, help="Path to input video")
    p.add_argument("--weights", default=Config.model_weights, help="YOLO weights file")
    p.add_argument("--no-display", action="store_true", help="Run headless (no cv2.imshow window)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(video_path=args.video, model_weights=args.weights, show_video=not args.no_display)
    run(cfg)
