# Vehicle Detection and Tracking Project

This project leverages the YOLOv10 model for vehicle detection, Kalman filtering for object tracking, and ARIMA for traffic flow forecasting. The primary goal is to detect, track, and categorize vehicles in video feeds, analyze their movement patterns across predefined sections, and forecast future traffic flows.

## Setup Instructions
### Prerequisites
Ensure you have the following system requirements and dependencies:

GPU: NVIDIA GPU with CUDA support (CUDA 11.7 or later)
RAM: 16 GB or more
CPU: 8-core or more
Operating System: Ubuntu 20.04 or compatible

### Building the Docker Image
To build the Docker image, follow these steps:

1. Clone the Repository:
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
pip install -r requirements.txt
cd yolov10
pip install -e .
cd ..

2. Build the Docker Image:
docker build -t pritishsaha92/bmc_kolkgpconv_submission .
This command will create a Docker image named vehicle_detection_image. It uses a base image with Python 3.9 and CUDA 11.7 to ensure compatibility with the YOLOv10 model and other dependencies.

### Running the Docker Container
After building the Docker image, you can run the container with the following command:

docker run --rm --runtime=nvidia --gpus all -v <host-files-path>:<container-files-path>
<image-name>:pritishsaha92/bmc_kolkgpconv_submission:latest python3 app.py input_file.json output_file.json

The --gpus all flag enables GPU support.
The -v flag mounts the input and output directories from the host to the container.

Replace <host-path-to-input.json> with the full path to your input.json file on your host machine.

## Project Structure
app.py: Main script that loads the YOLOv10 model, processes video input, tracks vehicles using Kalman filters, counts vehicle transitions, and performs ARIMA forecasting.

requirements.txt: Contains the list of Python libraries required to run the project. This file is used during the Docker image build to install dependencies.

Dockerfile: Defines the Docker environment, installs necessary dependencies, and sets up the project for execution.

yolov10/: Directory containing YOLOv10 model files and any associated custom scripts.

## Key Components
YOLOv10 Model: Utilized for detecting various vehicle classes in video frames. The model is loaded with fine-tuned weights specified in the script.

Kalman Filter: Used to track vehicles between frames, predicting and updating vehicle positions to manage detection uncertainties.

ARIMA Forecasting: Applied to vehicle count time series data to predict future traffic flow.

Region and Turning Patterns: Vehicle movements are tracked across defined sections (regions), with transition patterns defined for traffic analysis.

## Scripts and Notebooks
app.py: The main application script that integrates detection, tracking, and forecasting. Run this script using the Docker setup for processing video files.

## Requirements
The following libraries are required to run the code:

numpy
pandas
ultralytics
filterpy
statsmodels
matplotlib
torch
torchvision
onnx
onnxruntime
pycocotools
PyYAML
scipy
onnxslim
onnxruntime-gpu
gradio
opencv-python
psutil
py-cpuinfo
huggingface-hub
safetensors
All dependencies will be installed automatically during the Docker image build process.

## Open-Source Models
YOLOv10 from the Ultralytics GitHub repository is used for vehicle detection. The model weights are fine-tuned for specific vehicle classes relevant to this project.

## System Requirements
To run this project effectively, ensure your system meets the following specifications:

GPU: NVIDIA GPU with at least 8 GB VRAM and CUDA support
CUDA: Version 11.7 or later
RAM: Minimum 16 GB
CPU: At least 8-core processor
Disk Space: Minimum 10 GB free for processing and storage

## Evaluation
The system is evaluated on its ability to detect and track various vehicle types across video feeds, count transitions between predefined regions accurately, and forecast future traffic using ARIMA models. The performance is measured based on detection accuracy, tracking precision, and forecasting reliability.
