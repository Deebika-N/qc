import cv2
import mediapipe as mp
import numpy as np
import threading
import logging
from queue import Queue
from twilio.rest import Client
import os
import time
from datetime import datetime

# Twilio configuration (replace with your actual credentials)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'AC39ff626709438aa9595dbfb36eee793f')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '2b028e3e2bead7a23779d2f0d9099cd4')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER', '+1 815 421 1277')
TWILIO_TO_NUMBER = os.getenv('TWILIO_TO_NUMBER', '+919043623005')

# Blink detection constants
BLINK_THRESHOLD = 0.015
CONSECUTIVE_BLINKS_THRESHOLD = 4
BLINK_COOLDOWN = 1.0
BLINK_WINDOW = 30
EYE_CLOSED_SOS_TIME = 5

def send_twilio_alert(gesture_name, is_blink_emergency=False):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message_body = f"Emergency {'blink pattern' if is_blink_emergency else 'gesture'} detected: {gesture_name.upper()}!"
    # Send SMS
    client.messages.create(
        body=message_body,
        from_=TWILIO_FROM_NUMBER,
        to=TWILIO_TO_NUMBER
    )
    # Make a call (plays a simple message)
    call = client.calls.create(
        twiml=f'<Response><Say>{message_body}</Say></Response>',
        from_=TWILIO_FROM_NUMBER,
        to=TWILIO_TO_NUMBER
    )

class GestureDetector:
    def __init__(self, callback=None):
        # Hand detection setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Face mesh setup for blink detection
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.callback = callback
        self.is_running = False
        self.cap = None
        self.detection_thread = None
        self.logger = logging.getLogger("EmergencySoundTracker")
        
        # Blink detection variables
        self.total_blinks = 0
        self.blink_counter = 0
        self.blink_times = []
        self.last_blink_time = None
        self.closed_start = None
        self.eyes_closed = False
        self.blink_log = []
        self.last_alert_time = 0
        self.alert_cooldown = 15  # seconds
        
        # Define emergency gestures
        self.emergency_gestures = {
            'help': self._is_help_gesture,
            'emergency': self._is_emergency_gesture,
            'stop': self._is_stop_gesture
        }

    def _log_blink_event(self, event_type, details=""):
        """Log blink-related events"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {event_type}: {details}"
        self.blink_log.append(log_entry)
        self.logger.info(log_entry)

    def _detect_blinks(self, frame):
        """Detect and process blinks"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        
        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                landmarks = face.landmark
                
                # Get eye landmarks
                left_eye = [landmarks[i] for i in [159, 145]]
                right_eye = [landmarks[i] for i in [386, 374]]
                
                # Calculate eye distances
                left_eye_dist = abs(left_eye[0].y - left_eye[1].y)
                right_eye_dist = abs(right_eye[0].y - right_eye[1].y)
                eye_dist = (left_eye_dist + right_eye_dist) / 2
                current_time = time.time()

                # Blink Detection Logic
                if eye_dist < BLINK_THRESHOLD:  # Eyes closed
                    if not self.eyes_closed:  # Just closed eyes
                        self.eyes_closed = True
                        self.closed_start = current_time
                    elif current_time - self.closed_start >= EYE_CLOSED_SOS_TIME:
                        reason = "Eyes closed too long. Possible fatigue or emergency."
                        self._handle_blink_emergency(reason)
                        self.closed_start = None
                else:  # Eyes open
                    if self.eyes_closed:  # Just opened eyes after being closed
                        self.eyes_closed = False
                        self.total_blinks += 1
                        self.blink_times.append(current_time)
                        
                        # Log the blink
                        self._log_blink_event("Blink", f"#{self.total_blinks}")
                        
                        # Remove old blinks from window
                        self.blink_times = [t for t in self.blink_times if current_time - t <= BLINK_WINDOW]
                        
                        # Check for multiple blinks in window
                        if len(self.blink_times) >= CONSECUTIVE_BLINKS_THRESHOLD:
                            reason = f"Multiple blinks detected ({len(self.blink_times)} blinks in {BLINK_WINDOW} seconds)"
                            self._handle_blink_emergency(reason)
                            self.blink_times = []
                        
                        # Check for consecutive blinks
                        if self.last_blink_time and current_time - self.last_blink_time <= BLINK_COOLDOWN:
                            self.blink_counter += 1
                            if self.blink_counter >= CONSECUTIVE_BLINKS_THRESHOLD:
                                reason = f"Four consecutive blinks detected (blink count: {self.blink_counter})"
                                self._handle_blink_emergency(reason)
                                self.blink_counter = 0
                        else:
                            self.blink_counter = 1
                        
                        self.last_blink_time = current_time
                        
                        # Update callback with blink information
                        if self.callback:
                            self.callback({
                                'type': 'blink',
                                'total_blinks': self.total_blinks,
                                'blinks_in_window': len(self.blink_times),
                                'consecutive_blinks': self.blink_counter,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    
                    self.closed_start = None

                # Display blink information on frame
                self._display_blink_info(frame)

    def _handle_blink_emergency(self, reason):
        """Handle blink-related emergencies"""
        now = time.time()
        if now - self.last_alert_time > self.alert_cooldown:
            self._log_blink_event("EMERGENCY", reason)
            send_twilio_alert(reason, is_blink_emergency=True)
            self.last_alert_time = now

    def _display_blink_info(self, frame):
        """Display blink information on the frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Display blink statistics
        cv2.putText(frame, f"Total Blinks: {self.total_blinks}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks (30s): {len(self.blink_times)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Consecutive: {self.blink_counter}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def start(self):
        """Start the gesture detection"""
        if self.is_running:
            return
            
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        self.logger.info("Gesture detection started")

    def stop(self):
        """Stop the gesture detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        self.logger.info("Gesture detection stopped")

    def _detection_loop(self):
        """Main detection loop"""
        while self.is_running:
            success, frame = self.cap.read()
            if not success:
                continue

            # Convert to RGB for hand detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Process hand gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    for gesture_name, gesture_func in self.emergency_gestures.items():
                        if gesture_func(hand_landmarks):
                            if self.callback:
                                self.callback(gesture_name)
                            send_twilio_alert(gesture_name)

            # Process blink detection
            self._detect_blinks(frame)

            # Display the frame
            cv2.imshow('Gesture and Blink Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _is_help_gesture(self, landmarks):
        """Detect help gesture (raised hand)"""
        # Check if all fingers are extended
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
        finger_mcps = [6, 10, 14, 18]  # MCP joints
        
        for tip, mcp in zip(finger_tips, finger_mcps):
            if landmarks.landmark[tip].y >= landmarks.landmark[mcp].y:
                return False
        return True

    def _is_emergency_gesture(self, landmarks):
        """Detect emergency gesture (waving hand)"""
        wrist = landmarks.landmark[0]
        index_tip = landmarks.landmark[8]
        return abs(wrist.x - index_tip.x) > 0.2

    def _get_finger_states(self, landmarks):
        """Get the state of each finger (up/down)"""
        # Get the y-coordinates of finger tips and their corresponding PIP joints
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        finger_pips = [6, 10, 14, 18]  # Index, middle, ring, pinky PIPs
        
        # Thumb is special case
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_up = thumb_tip.x < thumb_ip.x  # Thumb is up if tip is to the left of IP
        
        # Check other fingers
        fingers = [thumb_up]
        for tip, pip in zip(finger_tips, finger_pips):
            tip_y = landmarks.landmark[tip].y
            pip_y = landmarks.landmark[pip].y
            fingers.append(tip_y < pip_y)  # Finger is up if tip is above PIP
            
        return fingers

    def _is_waving(self, landmarks):
        """Detect if hand is waving"""
        # Get wrist and index finger positions
        wrist = landmarks.landmark[0]
        index_tip = landmarks.landmark[8]
        
        # Calculate horizontal movement
        movement = abs(index_tip.x - wrist.x)
        return movement > 0.2  # Threshold for waving detection

    def _is_stop_gesture(self, landmarks):
        """Detect if hand is in stop position (palm forward)"""
        # Get palm normal vector
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        
        # Calculate palm orientation
        palm_normal = middle_tip.z - wrist.z
        return palm_normal > 0.1  # Threshold for palm forward detection 