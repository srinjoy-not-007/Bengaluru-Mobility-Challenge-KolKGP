import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLOv10
from filterpy.kalman import KalmanFilter
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import json
from functools import lru_cache
import warnings
import os

@lru_cache(maxsize=None)
def expensive_function(x):
    return x * x

# Argument parsing
parser = argparse.ArgumentParser(description='Process video paths and JSON file paths.')
parser.add_argument('input_file', type=str, help='Path to the input JSON file.')
parser.add_argument('output_file', type=str, help='Path to the output JSON file.')
parser.add_argument('--sampling_rate', type=int, default=10, help='Sampling rate for processing frames.')
args = parser.parse_args()

# Load data from the input JSON file
with open(args.input_file) as f:
    data = json.load(f)

video_paths = [video_path for _, videos in data.items() for _, video_path in videos.items()]
keys_list = list(data.keys())

# Load turning patterns
df = pd.read_csv('turning.csv')

# Adjust vehicle type names to match the desired output format
vehicle_type_mapping = {
    "bicycle": "Bicycle",
    "bus": "Bus",
    "car": "Car",
    "LCV": "LCV",
    "three-wheeler": "Three Wheeler",
    "truck": "Truck",
    "two-wheeler": "Two Wheeler"
}

def get_turning_patterns_for_region(region):
    region_data = df[df['Region'] == region]
    if not region_data.empty:
        patterns_str = region_data.iloc[0]['Turning Pattern']
        patterns_list = patterns_str.split(',')
        return patterns_list
    else:
        return []

def get_section(x,y,region):
    txt_file = os.path.join('camera', 'new', f'{region}.txt')
    Ax1, Ay1, Ax2, Ay2, Ax3, Ay3, Ax4, Ay4 = 0,0,0,0,0,0,0,0
    Bx1, By1, Bx2, By2, Bx3, By3, Bx4, By4 = 0,0,0,0,0,0,0,0
    Cx1, Cy1, Cx2, Cy2, Cx3, Cy3, Cx4, Cy4 = 0,0,0,0,0,0,0,0
    Dx1, Dy1, Dx2, Dy2, Dx3, Dy3, Dx4, Dy4 = 0,0,0,0,0,0,0,0
    Ex1, Ey1, Ex2, Ey2, Ex3, Ey3, Ex4, Ey4 = 0,0,0,0,0,0,0,0
    Fx1, Fy1, Fx2, Fy2, Fx3, Fy3, Fx4, Fy4 = 0,0,0,0,0,0,0,0
    Gx1, Gy1, Gx2, Gy2, Gx3, Gy3, Gx4, Gy4 = 0,0,0,0,0,0,0,0
    with open(txt_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        values = line.split()
        class_id = int(values[0])
        x1, y1, x2, y2, x3,y3, x4,y4 = map(float, values[1:])
        if class_id == 0:
            Ax1, Ay1, Ax2, Ay2, Ax3, Ay3, Ax4, Ay4 = x1*width, y1*height, x2*width, y2*height,x3*width, y3*height,x4*width, y4*height
        elif class_id == 1:
            Bx1, By1, Bx2, By2, Bx3, By3, Bx4, By4 = x1*width, y1*height, x2*width, y2*height,x3*width, y3*height,x4*width, y4*height
        elif class_id == 2:
            Cx1, Cy1, Cx2, Cy2, Cx3, Cy3, Cx4, Cy4 = x1*width, y1*height, x2*width, y2*height,x3*width, y3*height,x4*width, y4*height
        elif class_id == 3:
            Dx1, Dy1, Dx2, Dy2, Dx3, Dy3, Dx4, Dy4 = x1*width, y1*height, x2*width, y2*height,x3*width, y3*height,x4*width, y4*height
        elif class_id == 4:
            Ex1, Ey1, Ex2, Ey2, Ex3, Ey3, Ex4, Ey4 = x1*width, y1*height, x2*width, y2*height,x3*width, y3*height,x4*width, y4*height
        elif class_id == 5:
            Fx1, Fy1, Fx2, Fy2, Fx3, Fy3, Fx4, Fy4 = x1*width, y1*height, x2*width, y2*height,x3*width, y3*height,x4*width, y4*height
        elif class_id == 6:
            Gx1, Gy1, Gx2, Gy2, Gx3, Gy3, Gx4, Gy4 = x1*width, y1*height, x2*width, y2*height,x3*width, y3*height,x4*width, y4*height

    if x< max(Ax1,Ax2,Ax3,Ax4) and x> min(Ax1,Ax2,Ax3,Ax4) and y> min(Ay1,Ay2,Ay3,Ay4) and y< max(Ay1,Ay2,Ay3,Ay4):
        return 'A'
    elif x< max(Bx1,Bx2,Bx3,Bx4) and x> min(Bx1,Bx2,Bx3,Bx4) and y> min(By1,By2,By3,By4) and y< max(By1,By2,By3,By4):
        return 'B'
    elif x< max(Cx1,Cx2,Cx3,Cx4) and x> min(Cx1,Cx2,Cx3,Cx4) and y> min(Cy1,Cy2,Cy3,Cy4) and y< max(Cy1,Cy2,Cy3,Cy4):
        return 'C'
    elif x< max(Dx1,Dx2,Dx3,Dx4) and x> min(Dx1,Dx2,Dx3,Dx4) and y> min(Dy1,Dy2,Dy3,Dy4) and y< max(Dy1,Dy2,Dy3,Dy4):
        return 'D'
    elif x< max(Ex1,Ex2,Ex3,Ex4) and x> min(Ex1,Ex2,Ex3,Ex4) and y> min(Ey1,Ey2,Ey3,Ey4) and y< max(Ey1,Ey2,Ey3,Ey4):
        return 'E'
    elif x< max(Fx1,Fx2,Fx3,Fx4) and x> min(Fx1,Fx2,Fx3,Fx4) and y> min(Fy1,Fy2,Fy3,Fy4) and y< max(Fy1,Fy2,Fy3,Fy4):
        return 'F'
    elif x< max(Gx1,Gx2,Gx3,Gx4) and x> min(Gx1,Gx2,Gx3,Gx4) and y> min(Gy1,Gy2,Gy3,Gy4) and y< max(Gy1,Gy2,Gy3,Gy4):
        return 'G'
    return None

def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0, 0, 0, 0])  # Initial state [x, y, dx, dy]
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # Measurement matrix
                     [0, 1, 0, 0]])
    kf.P *= 1000.  # Covariance matrix
    kf.R = np.array([[5, 0], [0, 5]])  # Measurement noise
    kf.Q = np.eye(4)  # Process noise
    return kf

model = YOLOv10(r"best (1).pt")  # fine-tuned weights

class_names = ['LCV', 'bicycle', 'bus', 'car', 'three-wheeler', 'truck', 'two-wheeler']
directions = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Process each video
# Initialize the output structure
# Process each video
final_output = {}

for area, video_files in data.items():
    cam_id = f"Cam_{area}"
    final_output[cam_id] = {
        "Cumulative Counts": {},
        "Predicted Counts": {}
    }
    
    # Initialize cumulative counts
    for i in directions:
        for j in directions:
            if i != j:
                transition = f"{i}{j}"
                final_output[cam_id]["Cumulative Counts"][transition] = {
                    "Bicycle": 0, "Bus": 0, "Car": 0, "LCV": 0, 
                    "Three Wheeler": 0, "Truck": 0, "Two Wheeler": 0
                }
    
    cumulative_counts = {transition: {vt: 0 for vt in class_names} for transition in final_output[cam_id]["Cumulative Counts"]}
    
    for video_id, video_path in video_files.items():
        video_path = str(video_path)  # Ensure video_path is a string
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist.")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue
        
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video resolution

        # Initialize regions and read from the corresponding region data file
        regions = {}
        txt_file = os.path.join('camera', 'new', f'{area}.txt')

        with open(txt_file, "r") as file:
            lines = file.readlines()

        for line in lines:
            values = line.split()
            if len(values) < 9:
                print(f"Skipping line due to insufficient data: {line}")
                continue

            class_id = int(values[0])
    
            try:
                regions[directions[class_id]] = {
                    f'x{i}': float(values[2*i-1]) * width for i in range(1, 5)
                }
                regions[directions[class_id]].update({
                    f'y{i}': float(values[2*i]) * height for i in range(1, 5)
                })
            except IndexError as e:
                print(f"Error processing line: {line}. Error: {e}")
                continue


        possible_turning_patterns = get_turning_patterns_for_region(area)

        # Initialize counts dictionary
        vehicle_tracks = {}
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % int(total_frames * args.sampling_rate // 45000) != 1:
                continue

            results = model(frame)

            for result in results:
                boxes = result.boxes  
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ = boxes.cls.cpu().numpy()  
                ids = np.arange(len(boxes)) #most important to get the ids for each unique vehicle 
                
                for idx, (box, class_id) in enumerate(zip(xyxy, cls_)):
                    x_min, y_min, x_max, y_max = box
                    cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2) #center of that vehicle
                    
                    vehicle_type = class_names[int(class_id)]
                    section = get_section(cx, cy, area)
                    print(f"Detected {vehicle_type} at ({cx}, {cy}) in {section}")
                    if idx not in vehicle_tracks:
                        kf = create_kalman_filter()
                        kf.x[:2] = np.array([cx, cy])  
                        vehicle_tracks[idx] = {'kf': kf, 'section': section, 'vehicle_type': vehicle_type}
                    else:
                        # Update the Kalman filter with new measurements
                        kf = vehicle_tracks[idx]['kf']
                        kf.predict()
                        kf.update(np.array([cx, cy]))
                        cx, cy = kf.x[:2]  #corrected position

                        prev_section = vehicle_tracks[idx]['section']# Retrieve the previous section
                        
                        # Check for a section change and update the counts
                        if prev_section and prev_section != section:
                            pair = f"{prev_section}{section}"
                            reverse_pair = f"{section}{prev_section}"
                            if pair in possible_turning_patterns:
                                cumulative_counts[pair][vehicle_type] += 1
                            elif reverse_pair in possible_turning_patterns:
                                cumulative_counts[reverse_pair][vehicle_type] += 1
                        
                        vehicle_tracks[idx]['section'] = section
        cap.release()

        # Process for predicted counts
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frames_per_interval = int(1.5 * 60 * frame_rate)  # 1.5-minute interval in frames
        frame_count = 0
        vehicle_counts_intervals = {}

        for i in directions:
            for j in directions:
                if i != j:
                    vehicle_counts_intervals[f'{i}{j}'] = {vehicle: [] for vehicle in class_names}

        interval_vehicle_counts = {}

        for i in directions:
            for j in directions:
                if i != j:
                    interval_vehicle_counts[f'{i}{j}'] = {vehicle: 0 for vehicle in class_names}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            if frame_count % int(total_frames * args.sampling_rate // 45000) != 1:
                continue
            
            results = model(frame)
            
            for result in results:
                boxes = result.boxes 
                xyxy = boxes.xyxy.cpu().numpy() 
                cls_ = boxes.cls.cpu().numpy()  
                ids = np.arange(len(boxes))  
                
                for idx, (box, class_id) in enumerate(zip(xyxy, cls_)):
                    x_min, y_min, x_max, y_max = box
                    cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
                    
                    vehicle_type = class_names[int(class_id)]
                    section = get_section(cx, cy, area)
                    
                    print(f"Detected {vehicle_type} at ({cx}, {cy}) in {section}")
                    
                    if idx not in vehicle_tracks:
                        kf = create_kalman_filter()
                        kf.x[:2] = np.array([cx, cy])  # Initialize position
                        vehicle_tracks[idx] = {'kf': kf, 'section': section, 'vehicle_type': vehicle_type}
                    else:
                        kf = vehicle_tracks[idx]['kf']
                        kf.predict()
                        kf.update(np.array([cx, cy]))
                        cx, cy = kf.x[:2]  # Get the corrected position
                        
                        prev_section = vehicle_tracks[idx]['section']
                        
                        if prev_section and prev_section != section:
                            pair = f"{prev_section}{section}"
                            if pair in possible_turning_patterns:
                                interval_vehicle_counts[pair][vehicle_type] += 1
                            elif f"{section}{prev_section}" in possible_turning_patterns:
                                interval_vehicle_counts[f"{section}{prev_section}"][vehicle_type] += 1
                        
                        vehicle_tracks[idx]['section'] = section
            # Store vehicle counts for each interval
            if frame_count % frames_per_interval == 0:
                for transition, counts in vehicle_counts_intervals.items():
                    for vehicle_type in class_names:
                        counts[vehicle_type].append(
                            interval_vehicle_counts[transition][vehicle_type])
                frame_count = 0
                interval_vehicle_counts = {transition: {vehicle_type: 0 for vehicle_type in class_names} for transition in
                                        vehicle_counts_intervals.keys()}
        cap.release()

        # Update cumulative counts
        for transition, counts in cumulative_counts.items():
            for vehicle_type, count in counts.items():
                final_output[cam_id]["Cumulative Counts"][transition][vehicle_type_mapping.get(vehicle_type, vehicle_type)] += count

        # Prediction
        for transition in vehicle_counts_intervals.keys():
            if transition not in final_output[cam_id]["Predicted Counts"]:
                final_output[cam_id]["Predicted Counts"][transition] = {
                    "Bicycle": 0, "Bus": 0, "Car": 0, "LCV": 0, 
                    "Three Wheeler": 0, "Truck": 0, "Two Wheeler": 0
                }

            for vehicle_type in class_names:
                data = vehicle_counts_intervals[transition][vehicle_type]
                if len(data) < 2:
                    predicted_count = data[-1] if data else 0
                else:
                    try:
                        # Use auto_arima to automatically select the best model
                        model = auto_arima(data, start_p=0, start_q=0, max_p=3, max_q=3, m=1,
                                        start_P=0, seasonal=False, d=1, D=1, trace=True,
                                        error_action='ignore', suppress_warnings=True, stepwise=True)

                        pred = model.predict(n_periods=2)
                        predicted_count = int(pred[-1])
                    except Exception as e:
                        print(f"Error in prediction for {vehicle_type} in {transition}: {str(e)}")
                        # Fallback to simple moving average
                        predicted_count = int(np.mean(data[-3:]))  # Use last 3 observations

                # Ensure predicted_count is non-negative
                predicted_count = abs(predicted_count)
        
                final_output[cam_id]["Predicted Counts"][transition][vehicle_type_mapping.get(vehicle_type, vehicle_type)] = predicted_count

# Save results to JSON file
with open(args.output_file, 'w') as f:
    json.dump(final_output, f, indent=4)

print(f"Results saved to {args.output_file}")
