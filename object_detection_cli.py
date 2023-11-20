import cv2
import time
import sys
import logging

# Constants
CASCADE_FILE = 'car_cascade.xml'
WINDOW_NAME = 'Car Detection'

def load_cascade(cascade_file):
    """Load the Haar cascade file."""
    car_cascade = cv2.CascadeClassifier(cascade_file)
    if car_cascade.empty():
        raise IOError(f"Unable to load the cascade classifier from {cascade_file}")
    return car_cascade

def detect_cars(frame, car_cascade):
    """Detect cars in an image frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return cars

def draw_rectangles(frame, cars):
    """Draw rectangles around detected cars."""
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def process_video(video_path, car_cascade):
    """Process the video for car detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cars = detect_cars(frame, car_cascade)
        draw_rectangles(frame, cars)

        # Save every 10th frame
        if frame_count % 10 == 0:
            cv2.imwrite(f"frame_{frame_count}.jpg", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python car_detection.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    car_cascade = load_cascade(CASCADE_FILE)

    # Start timing
    start_time = time.time()

    process_video(video_path, car_cascade)

    # End timing
    end_time = time.time()
    print(f"Processing Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
