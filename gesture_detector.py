import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
from twilio.rest import Client
import os
from datetime import datetime

class GestureDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize thresholds and counters
        self.blink_threshold = 0.2
        self.nod_threshold = 0.1
        self.twitch_threshold = 0.15
        self.blink_counter = 0
        self.nod_counter = 0
        self.twitch_counter = 0
        
        # Emergency blink detection
        self.consecutive_blinks = 0
        self.last_blink_time = time.time()
        self.last_nod_time = time.time()
        self.blink_timeout = 3.0  # Time window for consecutive blinks
        self.emergency_blink_count = 4  # Number of blinks to trigger emergency
        self.emergency_triggered = False
        
        # Initialize Twilio client
        self.account_sid = 'AC39ff626709438aa9595dbfb36eee793f'  # Your Account SID
        self.auth_token = '2b028e3e2bead7a23779d2f0d9099cd4'  # Your Auth Token
        self.twilio_client = Client(self.account_sid, self.auth_token)
        self.emergency_number = '+919043623005'  # Replace with your emergency contact number
        self.twilio_number = '+1 815 421 1277'    # Replace with your Twilio phone number
        
        # Cooldown period (in seconds)
        self.cooldown = 1.0
        
        # Pause state
        self.is_paused = False
        # Detection disabled after emergency
        self.detection_disabled = False

    def make_emergency_call(self):
        try:
            # Create a call
            call = self.twilio_client.calls.create(
                to=self.emergency_number,
                from_=self.twilio_number,
                twiml='<Response><Say>Emergency alert! The patient has triggered an emergency signal by blinking 4 times consecutively. Please check on them immediately.</Say></Response>'
            )
            print(f"Emergency call initiated. Call SID: {call.sid}")
            
            # Log the emergency
            self.log_emergency()
            
            # Disable further detection
            self.detection_disabled = True
            
            return True
        except Exception as e:
            print(f"Failed to make emergency call: {str(e)}")
            return False

    def log_emergency(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"Emergency triggered at {timestamp} - 4 consecutive blinks detected"
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Log to file
        with open('logs/emergency_log.txt', 'a') as f:
            f.write(log_message + '\n')
        
        print(log_message)

    def calculate_eye_aspect_ratio(self, landmarks):
        left_eye = [362, 385, 387, 263, 373, 380]
        right_eye = [33, 160, 158, 133, 153, 144]
        left_ear = self._calculate_ear(landmarks, left_eye)
        right_ear = self._calculate_ear(landmarks, right_eye)
        ear = (left_ear + right_ear) / 2
        return ear

    def _calculate_ear(self, landmarks, eye_indices):
        points = []
        for idx in eye_indices:
            point = landmarks.landmark[idx]
            points.append((point.x, point.y))
        v1 = distance.euclidean(points[1], points[5])
        v2 = distance.euclidean(points[2], points[4])
        h = distance.euclidean(points[0], points[3])
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_nod(self, landmarks):
        nose_tip = landmarks.landmark[1]
        forehead = landmarks.landmark[10]
        vertical_movement = abs(nose_tip.y - forehead.y)
        return vertical_movement > self.nod_threshold

    def detect_twitch(self, landmarks, prev_landmarks):
        if prev_landmarks is None:
            return False
        mouth_points = [61, 291, 0, 17]
        eye_points = [33, 133, 362, 263]
        mouth_movement = sum(
            distance.euclidean(
                (landmarks.landmark[i].x, landmarks.landmark[i].y),
                (prev_landmarks.landmark[i].x, prev_landmarks.landmark[i].y)
            )
            for i in mouth_points
        ) / len(mouth_points)
        eye_movement = sum(
            distance.euclidean(
                (landmarks.landmark[i].x, landmarks.landmark[i].y),
                (prev_landmarks.landmark[i].x, prev_landmarks.landmark[i].y)
            )
            for i in eye_points
        ) / len(eye_points)
        return (mouth_movement > self.twitch_threshold or 
                eye_movement > self.twitch_threshold)

    def process_frame(self, frame):
        if self.is_paused:
            cv2.putText(frame, "PAUSED - Press 'p' to resume", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        if self.detection_disabled:
            cv2.putText(frame, "EMERGENCY CALLED! Detection stopped.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            ear = self.calculate_eye_aspect_ratio(landmarks)
            current_time = time.time()
            if ear < self.blink_threshold:
                if current_time - self.last_blink_time > self.cooldown:
                    self.blink_counter += 1
                    self.last_blink_time = current_time
                    print(f"Blink detected! Count: {self.blink_counter}")
                    if current_time - self.last_blink_time < self.blink_timeout:
                        self.consecutive_blinks += 1
                        if self.consecutive_blinks >= self.emergency_blink_count and not self.emergency_triggered:
                            print("EMERGENCY TRIGGERED! Making emergency call...")
                            self.emergency_triggered = True
                            self.make_emergency_call()
                    else:
                        self.consecutive_blinks = 1
            else:
                if current_time - self.last_blink_time > self.blink_timeout:
                    self.consecutive_blinks = 0
                    self.emergency_triggered = False
            if self.detect_nod(landmarks):
                if current_time - self.last_nod_time > self.cooldown:
                    self.nod_counter += 1
                    self.last_nod_time = current_time
                    print(f"Nod detected! Count: {self.nod_counter}")
            for landmark in landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        cv2.putText(frame, f"Blinks: {self.blink_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Nods: {self.nod_counter}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Consecutive Blinks: {self.consecutive_blinks}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.emergency_triggered:
            cv2.putText(frame, "EMERGENCY CALLED!", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'p' to pause/resume", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    print("Gesture detection started.")
    print("Controls:")
    print("- Press 'p' to pause/resume detection")
    print("- Press 'q' to quit")
    print("\nEmergency System:")
    print(f"- {detector.emergency_blink_count} consecutive blinks will trigger emergency call")
    print(f"- Emergency contact: {detector.emergency_number}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = detector.process_frame(frame)
        cv2.imshow('Gesture Detection', processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            detector.is_paused = not detector.is_paused
            print("Paused" if detector.is_paused else "Resumed")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 