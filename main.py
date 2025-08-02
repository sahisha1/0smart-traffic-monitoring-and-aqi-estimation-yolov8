import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from collections import deque
from matplotlib.ticker import MultipleLocator

# YOLOv8 import
from ultralytics import YOLO

plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(['Cars', 'Trucks', 'Up', 'Down'], [0, 0, 0, 0])
ax.set_ylim(0, 10)
ax.yaxis.set_major_locator(plt.MultipleLocator(2))  # Adjust as per expected range

# -------------------- CAR CLASS --------------------
class Car:
    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
        self.frames_crossed = 0
        self.cross_start_frame = None
        self.cross_end_frame = None

    def getId(self):
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn
        self.frames_crossed += 1

    def setDone(self):
        self.done = True

    def timedOut(self):
        return self.done

    def going_UP(self, mid_start, mid_end):
        if len(self.tracks) >= 2 and self.state == '0':
            if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:
                self.state = '1'
                self.dir = 'up'
                return True
        return False

    def going_DOWN(self, mid_start, mid_end):
        if len(self.tracks) >= 2 and self.state == '0':
            if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start:
                self.state = '1'
                self.dir = 'down'
                return True
        return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True

# -------------------- VARIABLES --------------------
os.makedirs('detected', exist_ok=True)

cap = cv2.VideoCapture("Vehicle-Detection-Classification-and-Counting-main/Videos/video.mp4")

# Load YOLOv8 model (will auto-download if not present)
model = YOLO("yolov8n.pt")  # change to yolov8s.pt/m/l if you need accuracy/speed tradeoff

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

font = cv2.FONT_HERSHEY_SIMPLEX

cars = []
max_p_age = 5
pid = 1

cnt_up = 0
cnt_down = 0
cnt_car = 0
cnt_truck = 0

line_up = 400
line_down = 250
up_limit = 230
down_limit = int(4.5 * (500 / 5))

real_distance_meters = 8.0  # Estimated real-world distance between lines
pixel_distance = abs(line_down - line_up)
meters_per_pixel = real_distance_meters / pixel_distance
speed_limit = 100
speeds = {}
overspeed_count = 0
overspeeding_ids = []

# -------------------- UTILS --------------------
def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def estimate_aqi(car_count, truck_count):
    car_emission_rate = 120
    truck_emission_rate = 900

    total_car_emissions = car_count * car_emission_rate
    total_truck_emissions = truck_count * truck_emission_rate
    total_emissions = total_car_emissions + total_truck_emissions

    aqi = min(int(total_emissions / 100), 500)
    if aqi < 50:
        return (aqi, "Good")
    elif aqi < 100:
        return (aqi, "Moderate")
    elif aqi < 150:
        return (aqi, "Unhealthy for Sensitive Groups")
    elif aqi < 200:
        return (aqi, "Unhealthy")
    elif aqi < 300:
        return (aqi, "Very Unhealthy")
    else:
        return (aqi, "Hazardous")

aqi_history = []
frame_count = 0

def update_live_graph(cnt_car, cnt_truck, cnt_up, cnt_down):
    estimated_aqi, _ = estimate_aqi(cnt_car, cnt_truck)
    aqi_history.append(estimated_aqi)

    # Order: car, truck, up, down
    values = [cnt_car, cnt_truck, cnt_up, cnt_down]
    for bar, val in zip(bars, values):
        bar.set_height(val)

    ax.set_ylim(0, 60)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Separate AQI Line Graph Setup
fig_aqi, ax_aqi = plt.subplots()
x_data, y_data = [], []
line_aqi, = ax_aqi.plot([], [], 'r-', label='AQI')
ax_aqi.set_title("AQI Over Time")
ax_aqi.set_xlabel("Frame Count")
ax_aqi.set_ylabel("AQI")
ax_aqi.set_ylim(0, 500)
ax_aqi.legend()
plt.ion()
fig_aqi.show()

# -------------------- MAIN LOOP --------------------
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    frame = cv2.resize(frame, (900, 500))

    # ---------------- YOLOv8 INFERENCE INSTEAD OF BACKGROUND SUBTRACTION ----------------
    results = model(frame)[0]  # single-frame result
    detected_centroids = []

    # COCO class ids: 2 = car, 7 = truck
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in [2, 7] and conf > 0.3:  # thresholding
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2
            label = "Car" if cls_id == 2 else "Truck"
            detected_centroids.append((cx, cy, x1, y1, w, h, label, conf))

    # ------------------ TRACKING / MATCHING ------------------
    for item in detected_centroids:
        cx, cy, x, y, w, h, label, conf = item
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 5), font, 0.5, (255, 255, 0), 1)

        matched = False
        for car in cars:
            if euclidean_distance(cx, cy, car.getX(), car.getY()) < 50:
                car.updateCoords(cx, cy)
                matched = True

                if car.going_UP(line_down, line_up) and not hasattr(car, 'counted'):
                    cnt_up += 1
                    if label == "Car":
                        cnt_car += 1
                    else:
                        cnt_truck += 1
                    car.counted = True

                    time_seconds = car.frames_crossed / fps if fps > 0 else 1
                    real_distance = pixel_distance * meters_per_pixel
                    speed = (real_distance / time_seconds) * 3.6
                    if 0 < speed < 180:
                        speeds[car.getId()] = round(speed, 2)
                        if speed > speed_limit:
                            overspeed_count += 1
                            overspeeding_ids.append(car.getId())
                            vehicle_img = frame[y:y+h, x:x+w]
                            filename = f"detected/overspeed_vehicle_{car.getId()}.jpg"
                            cv2.imwrite(filename, vehicle_img)

                elif car.going_DOWN(line_down, line_up) and not hasattr(car, 'counted'):
                    cnt_down += 1
                    if label == "Car":
                        cnt_car += 1
                    else:
                        cnt_truck += 1
                    car.counted = True

                    time_seconds = car.frames_crossed / fps if fps > 0 else 1
                    real_distance = pixel_distance * meters_per_pixel
                    speed = (real_distance / time_seconds) * 3.6
                    if 0 < speed < 180:
                        speeds[car.getId()] = round(speed, 2)
                        if speed > speed_limit:
                            overspeed_count += 1
                            overspeeding_ids.append(car.getId())
                            vehicle_img = frame[y:y+h, x:x+w]
                            filename = f"detected/overspeed_vehicle_{car.getId()}.jpg"
                            cv2.imwrite(filename, vehicle_img)
                break

        if not matched:
            new_car = Car(pid, cx, cy, max_p_age)
            cars.append(new_car)
            pid += 1

    for car in cars[:]:
        car.age_one()
        if car.timedOut():
            cars.remove(car)

    # ------------------ DRAW LINES & INFO ------------------
    frame = cv2.line(frame, (0, line_up), (900, line_up), (255, 0, 255), 2)
    frame = cv2.line(frame, (0, up_limit), (900, up_limit), (0, 255, 255), 2)
    frame = cv2.line(frame, (0, down_limit), (900, down_limit), (255, 0, 0), 2)
    frame = cv2.line(frame, (0, line_down), (900, line_down), (255, 0, 0), 2)

    cv2.putText(frame, f'UP: {cnt_up}', (10, 40), font, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f'DOWN: {cnt_down}', (10, 80), font, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f'Cars: {cnt_car}', (10, 120), font, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Trucks: {cnt_truck}', (10, 160), font, 0.6, (0, 255, 255), 2)

    update_live_graph(cnt_car, cnt_truck, cnt_up, cnt_down)

    # Update AQI line graph
    x_data.append(frame_count)
    y_data.append(aqi_history[-1] if aqi_history else 0)
    line_aqi.set_data(x_data, y_data)
    ax_aqi.set_xlim(0, max(50, len(x_data)))
    ax_aqi.set_ylim(0, max(100, max(y_data) + 20) if y_data else 500)
    fig_aqi.canvas.draw()
    fig_aqi.canvas.flush_events()
    frame_count += 1

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('h'):
        break

cap.release()
# -------------------- SAVE GRAPHS --------------------

# Save the final bar chart as an image
fig.savefig("vehicle_count_bar_chart.png")
print("Bar chart saved as 'vehicle_count_bar_chart.png'")

# Save the final AQI line graph as an image
fig_aqi.savefig("aqi_over_time_line_graph.png")
print("AQI line graph saved as 'aqi_over_time_line_graph.png'")
cv2.destroyAllWindows()

# -------------------- RESULTS --------------------
total_vehicles = cnt_car + cnt_truck
print(f"Total vehicles detected: {cnt_up + cnt_down}")
print(f"Total overspeeding vehicles: {overspeed_count}")
print("\n--- Speeds (km/h) ---")
for vid, spd in speeds.items():
    print(f"Vehicle {vid}: {spd} km/h")
    if spd > speed_limit:
        print(f"--> Vehicle {vid} was overspeeding!")
print(f"Overspeeding Vehicle IDs: {overspeeding_ids}")

estimated_aqi, air_quality = estimate_aqi(cnt_car, cnt_truck)
print(f"\nEstimated AQI: {estimated_aqi}")
print(f"Air Quality: {air_quality}")

with open('aqi_report.txt', 'w') as f:
    f.write(f"Total Vehicles: {total_vehicles}\n")
    f.write(f"Estimated AQI: {estimated_aqi}\n")
    f.write(f"Air Quality: {air_quality}\n")
    f.write(f"Total overspeeding vehicles: {overspeed_count}\n")
    f.write(f"Overspeeding Vehicle IDs: {overspeeding_ids}\n")

print("\nAQI report saved successfully!")
