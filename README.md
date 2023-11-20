# Car Detection Using OpenCV

This repository contains a Python script for detecting cars in video files using OpenCV's Haar Cascade Classifier. The script uses a pre-trained XML file for car detection and can process any video file.

## Contents

- `car_detection.py`: The main Python script for detecting cars in a video.
- `car_cascade.xml`: The Haar Cascade XML file used for detecting cars.
- `video1.mp4`: Sample video file for testing the car detection.
- `video2.mp4`: Another sample video file for testing.

## Requirements

- Python 3.x
- OpenCV library (`opencv-python`)
- A video file for processing (MP4 format recommended)

## Installation

Before running the script, ensure you have Python and OpenCV installed. You can install OpenCV using pip:

```bash
pip install opencv-python
```

## Usage

Run the script from the command line, passing the path of the video file you want to process as an argument:

```bash
python car_detection.py <path_to_video_file>
```

For example:

```bash
python car_detection.py video1.mp4
```

## How It Works

The script uses OpenCV's Haar Cascade Classifier to detect cars in each frame of the provided video file. Detected cars are highlighted with green rectangles.
