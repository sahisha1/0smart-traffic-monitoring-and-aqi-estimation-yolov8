# smart-traffic-monitoring-and-aqi-estimation using yolov8

# Vehicle Detection, Counting, and Speed Estimation

## Description

This Python script uses the YOLOv8 object detection model to perform real-time vehicle analysis from a video file. It is designed to detect, track, and count vehicles (cars and trucks) as they pass through a predefined area.

Beyond simple counting, the script also estimates the speed of each vehicle in km/h, identifies and logs overspeeding vehicles, and generates a simple, estimated Air Quality Index (AQI) based on the volume and type of vehicles detected.

The script provides live data visualization through two `matplotlib` plots (a bar chart for counts and a line chart for AQI) and saves a final report and graph images upon completion.

## Features

  * **Vehicle Detection:** Utilizes YOLOv8 to detect 'cars' and 'trucks'.
  * **Vehicle Tracking:** Employs a simple centroid-based tracker to assign a unique ID to each vehicle and follow it across frames.
  * **Directional Counting:** Counts vehicles separately for two directions (defined as 'UP' and 'DOWN').
  * **Vehicle Classification Count:** Maintains separate counts for 'Cars' and 'Trucks'.
  * **Speed Estimation:** Calculates the speed of vehicles (in km/h) based on the time taken to cross a known real-world distance mapped to pixels.
  * **Overspeeding Detection:**
      * Flags vehicles that exceed a configurable speed limit.
      * Saves a cropped image of the overspeeding vehicle to the `detected/` directory.
      * Logs the IDs of all overspeeding vehicles.
  * **Live Data Plots:**
    1.  A `matplotlib` bar chart that updates in real-time to show the current counts for Cars, Trucks, Up, and Down.
    2.  A `matplotlib` line chart that updates in real-time to show the estimated AQI over time (frames).
  * **Final Reporting:**
      * Saves the final bar chart as `vehicle_count_bar_chart.png`.
      * Saves the final AQI line graph as `aqi_over_time_line_graph.png`.
      * Generates a `aqi_report.txt` file summarizing total vehicles, overspeeding vehicles, and the final AQI estimate.

## Requirements

The script requires the following Python libraries:

  * **OpenCV** (`opencv-python`)
  * **Ultralytics** (`ultralytics`)
  * **Matplotlib** (`matplotlib`)
  * **NumPy** (`numpy`)

You can install all dependencies using pip:

```bash
pip install opencv-python ultralytics matplotlib numpy
```

## How to Run

1.  **Install Dependencies:**
    Run the `pip install` command from the section above.

2.  **Prepare Video File:**
    The script is hardcoded to look for a video file at this path:
    `Vehicle-Detection-Classification-and-Counting-main/Videos/video.mp4`

    You must either:

      * Create this directory structure and place your video file there.
      * **OR** change the path on line 102 to point to your video file:
        ```python
        cap = cv2.VideoCapture("path/to/your/video.mp4")
        ```

3.  **Run the Script:**
    Save the code as a Python file (e.g., `main.py`) and run it from your terminal:

    ```bash
    python main.py
    ```

4.  **Stop the Script:**
    While the video feed window is active, press the **'h'** key to stop the program.

## Configuration

You can adjust the script's behavior by changing the variables in the "VARIABLES" section:

  * `model = YOLO("yolov8n.pt")`: You can use other YOLOv8 models like `yolov8s.pt` or `yolov8m.pt` for different speed/accuracy trade-offs.
  * `line_up` / `line_down`: Change the pixel (y-coordinate) values of the detection lines.
  * `real_distance_meters = 8.0`: **Crucial for speed.** Set this to the actual real-world distance (in meters) between your `line_up` and `line_down`.
  * `speed_limit = 100`: Set the speed limit in km/h for overspeeding detection.
  * `max_p_age = 5`: The number of frames a tracker can "lose" a vehicle before deleting it.

## Outputs

Upon running and stopping the script, you will find the following files in the same directory:

  * **`detected/` (Folder):**
      * Contains `.jpg` images of all vehicles that were flagged for overspeeding.
  * **`vehicle_count_bar_chart.png`:**
      * A static image of the final vehicle count bar chart.
  * **`aqi_over_time_line_graph.png`:**
      * A static image of the final AQI-over-time line graph.
  * **`aqi_report.txt`:**
      * A text file with the final summary, including total vehicles, AQI, and overspeeding IDs.

The console will also print a summary of total vehicles, overspeeding count, and a list of all vehicle speeds.
