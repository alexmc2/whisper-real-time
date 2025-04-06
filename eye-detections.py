import cv2
import dlib
import numpy as np
import pyautogui
from threading import Thread
import time


class EyeTracker:
    def __init__(self):
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

        # Eye indices for the facial landmarks
        self.right_eye_indices = list(range(36, 42))
        self.left_eye_indices = list(range(42, 48))

        # Screen size
        self.screen_width, self.screen_height = pyautogui.size()

        # Control variables
        self.running = False
        self.thread = None
        self.calibration = {
            "top": 0.3,      # Look up threshold
            "bottom": 0.7,   # Look down threshold
            "left": 0.3,     # Look left threshold
            "right": 0.7,    # Look right threshold
            "blink": 0.19    # Blink threshold (eye aspect ratio)
        }

    def eye_aspect_ratio(self, eye_points):
        """Calculate the eye aspect ratio to detect blinks"""
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])

        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye_points[0] - eye_points[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def get_gaze_ratio(self, frame, eye_points, facial_landmarks):
        """Calculate the gaze ratio (left/right)"""
        # Get the eye region as a Numpy array
        eye_region = np.array(
            [(facial_landmarks.part(i).x, facial_landmarks.part(i).y) for i in eye_points])

        # Create mask for the eye
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)

        # Extract the eye from the frame
        eye = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert eye to grayscale
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

        # Threshold to get the white part of the eye
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        # Divide the eye into left and right parts
        height, width = threshold_eye.shape
        left_side = threshold_eye[0:height, 0:width//2]
        right_side = threshold_eye[0:height, width//2:width]

        # Count white pixels in each side
        left_white = cv2.countNonZero(left_side)
        right_white = cv2.countNonZero(right_side)

        if left_white == 0:
            gaze_ratio = 1  # Looking right
        elif right_white == 0:
            gaze_ratio = 5  # Looking left
        else:
            gaze_ratio = left_white / right_white
        return gaze_ratio

    def get_vertical_ratio(self, frame, eye_points, facial_landmarks):
        """Calculate the vertical gaze ratio (up/down)"""
        # Similar to get_gaze_ratio but for vertical movement
        eye_region = np.array(
            [(facial_landmarks.part(i).x, facial_landmarks.part(i).y) for i in eye_points])

        # Create mask for the eye
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)

        # Extract the eye from the frame
        eye = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert eye to grayscale
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

        # Threshold to get the white part of the eye
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        # Divide the eye into top and bottom parts
        height, width = threshold_eye.shape
        top_side = threshold_eye[0:height//2, 0:width]
        bottom_side = threshold_eye[height//2:height, 0:width]

        # Count white pixels in each side
        top_white = cv2.countNonZero(top_side)
        bottom_white = cv2.countNonZero(bottom_side)

        if top_white == 0:
            vertical_ratio = 5  # Looking down
        elif bottom_white == 0:
            vertical_ratio = 0.1  # Looking up
        else:
            vertical_ratio = top_white / bottom_white
        return vertical_ratio

    def start(self):
        """Start the eye tracking in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.track_eyes)
            self.thread.daemon = True
            self.thread.start()
            print("Eye tracking started")

    def stop(self):
        """Stop the eye tracking"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Eye tracking stopped")

    def track_eyes(self):
        """Main eye tracking function"""
        # Start the webcam
        cap = cv2.VideoCapture(0)

        # Mouse movement speed
        move_speed = 20

        # Last blink time
        last_blink_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.detector(gray)

            for face in faces:
                # Get facial landmarks
                landmarks = self.predictor(gray, face)

                # Get eye positions
                left_eye = np.array(
                    [(landmarks.part(i).x, landmarks.part(i).y) for i in self.left_eye_indices])
                right_eye = np.array(
                    [(landmarks.part(i).x, landmarks.part(i).y) for i in self.right_eye_indices])

                # Calculate eye aspect ratio
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2

                # Detect blink
                if avg_ear < self.calibration["blink"]:
                    current_time = time.time()
                    # Check if it's a genuine blink (not too frequent)
                    if current_time - last_blink_time > 1.0:
                        print("Blink detected - left click")
                        pyautogui.click()
                        last_blink_time = current_time
                    continue  # Skip other processing during blink

                # Get horizontal gaze direction
                left_gaze = self.get_gaze_ratio(
                    frame, self.left_eye_indices, landmarks)
                right_gaze = self.get_gaze_ratio(
                    frame, self.right_eye_indices, landmarks)
                avg_gaze = (left_gaze + right_gaze) / 2

                # Get vertical gaze direction
                left_vertical = self.get_vertical_ratio(
                    frame, self.left_eye_indices, landmarks)
                right_vertical = self.get_vertical_ratio(
                    frame, self.right_eye_indices, landmarks)
                avg_vertical = (left_vertical + right_vertical) / 2

                # Current mouse position
                current_x, current_y = pyautogui.position()
                new_x, new_y = current_x, current_y

                # Horizontal movement
                if avg_gaze < self.calibration["left"]:
                    new_x = max(0, current_x - move_speed)
                elif avg_gaze > self.calibration["right"]:
                    new_x = min(self.screen_width, current_x + move_speed)

                # Vertical movement
                if avg_vertical < self.calibration["top"]:
                    new_y = max(0, current_y - move_speed)
                elif avg_vertical > self.calibration["bottom"]:
                    new_y = min(self.screen_height, current_y + move_speed)

                # Move mouse if position changed
                if new_x != current_x or new_y != current_y:
                    pyautogui.moveTo(new_x, new_y)

            # Display frame for debugging
            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

# Example of how to integrate with your main application


def add_eye_tracking_to_main():
    # Add these imports at the top of your main.py
    import argparse

    # Modify your argument parser to include eye tracking options
    parser = argparse.ArgumentParser()
    # ... your existing arguments ...
    parser.add_argument("--enable_eye_tracking", action='store_true',
                        help="Enable eye tracking for mouse control")
    args = parser.parse_args()

    # Initialize eye tracker if enabled
    eye_tracker = None
    if args.enable_eye_tracking:
        try:
            eye_tracker = EyeTracker()
            eye_tracker.start()
            print("Eye tracking initialized")
        except Exception as e:
            print(f"Error initializing eye tracking: {e}")

    # Your main application code...

    # When shutting down, stop the eye tracker if it's running
    if eye_tracker:
        eye_tracker.stop()


if __name__ == "__main__":
    tracker = EyeTracker()
    try:
        tracker.start()
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tracker.stop()
        print("Eye tracking terminated")
