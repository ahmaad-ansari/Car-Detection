import cv2
import time
import sys
import logging

# Constants
CASCADE_FILE = 'car_cascade.xml'

def load_cascade(cascade_file):
    car_cascade = cv2.CascadeClassifier(cascade_file)
    if car_cascade.empty():
        raise IOError(f"Unable to load the cascade classifier from {cascade_file}")
    return car_cascade

def detect_cars(frame, car_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return cars

def draw_rectangles(frame, cars):
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def process_video(video_path, car_cascade):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: {frame_count} frames, {fps} FPS")

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cars = detect_cars(frame, car_cascade)
        draw_rectangles(frame, cars)

    end_time = time.time()
    print(f"Processing Time: {end_time - start_time:.2f} seconds")

    cap.release()

def main():
    if len(sys.argv) < 2:
        print("Usage: python object_detection_cli.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    car_cascade = load_cascade(CASCADE_FILE)
    process_video(video_path, car_cascade)

if __name__ == "__main__":
    main()
