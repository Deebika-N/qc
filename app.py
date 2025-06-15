import logging
import os
import cv2
import mediapipe as mp
import pyttsx3
import winsound
from datetime import datetime
import time
from twilio.rest import Client
import onnxruntime as ort
import numpy as np
from ui import EmergencySoundTracker
from recognizer import SpeechRecognitionEngine
from gesture_detection import GestureDetector
from scheduler import TaskScheduler
import atexit
import threading
import sounddevice as sd
import wave
import queue

# === Face Mesh Setup ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# === TTS Setup ===
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# === Twilio Config ===
TWILIO_SID = "AC5f3826ccf1abf39e25ce7cf9f15ae87e"
TWILIO_AUTH = "2092c07220e7fa7b124bc93922c60972"
TWILIO_FROM = "+15705535015"
CARETAKER_PHONE = "+918883389966"

# === Thresholds ===
BLINK_THRESHOLD = 0.015  # Adjusted for better sensitivity
NOD_MOVEMENT_THRESHOLD = 0.04
TWITCH_THRESHOLD = 0.015
EYE_CLOSED_SOS_TIME = 5
BLINK_COOLDOWN = 1.0  # Time window for consecutive blinks
CONSECUTIVE_BLINKS_THRESHOLD = 4  # Number of blinks to trigger alert
BLINK_WINDOW = 30  # Time window for counting blinks (30 seconds)
alert_cooldown = 15  # seconds

GENDERS = ["Male", "Female"]
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Anger"]

# Emergency words to detect
EMERGENCY_WORDS = ["help", "emergency", "danger", "pain", "fall", "accident"]

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("sound_tracking.log"),
            logging.StreamHandler()
        ]
    )

def save_emergency_audio(audio_data, sample_rate):
    """Save emergency audio to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emergency_audio_{timestamp}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return filename

class MonitoringSystem:
    def __init__(self, callback=None):
        self.callback = callback
        self.is_running = False
        self.cap = None
        self.blink_counter = 0
        self.blink_times = []
        self.closed_start = None
        self.nod_history = []
        self.last_alert_time = 0
        self.audio_queue = queue.Queue()
        self.recording = False
        self.emergency_audio = None
        
        try:
            self.session = ort.InferenceSession("face_attrib_net-facial-attribute-detection-float.onnx")
        except Exception as e:
            logging.error(f"Failed to load ONNX model: {e}")
            self.session = None

    def speak(self, text):
        tts.say(text)
        tts.runAndWait()

    def play_alarm(self):
        try:
            winsound.PlaySound("alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            logging.error(f"Alarm failed: {e}")

    def log_event(self, event):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("care_log.txt", "a") as f:
            f.write(f"[{timestamp}] {event}\n")
        logging.info(f"Event logged: {event}")

    def send_sms_alert(self, message):
        try:
            client = Client(TWILIO_SID, TWILIO_AUTH)
            client.messages.create(body=message, from_=TWILIO_FROM, to=CARETAKER_PHONE)
            logging.info("SMS sent to caretaker")
        except Exception as e:
            logging.error(f"SMS sending failed: {e}")

    def make_emergency_call(self, message):
        try:
            client = Client(TWILIO_SID, TWILIO_AUTH)
            call = client.calls.create(
                twiml=f'<Response><Say>{message}</Say></Response>',
                from_=TWILIO_FROM,
                to=CARETAKER_PHONE
            )
            logging.info("Emergency call initiated")
        except Exception as e:
            logging.error(f"Emergency call failed: {e}")

    def alert(self, event, make_call=False, save_audio=False):
        now = time.time()
        if now - self.last_alert_time > alert_cooldown:
            alert_type = "CALL" if make_call else "SMS"
            alert_reason = "Emergency" if make_call else "Warning"
            
            # Create detailed alert message
            alert_message = f"RA Patient Alert: {event}"
            if make_call:
                alert_message += f"\nReason: {alert_reason}"
                if "blink" in event.lower():
                    alert_message += "\nAction Required: Check patient's condition"
                elif "eye" in event.lower():
                    alert_message += "\nAction Required: Immediate attention needed"
            
            logging.warning(f"ALERT ({alert_type} - {alert_reason}): {event}")
            self.speak(event)
            self.play_alarm()
            self.send_sms_alert(alert_message)
            self.log_event(f"ALERT ({alert_type} - {alert_reason}): {event}")
            
            if save_audio and self.emergency_audio is not None:
                audio_file = save_emergency_audio(self.emergency_audio, 16000)
                self.send_sms_alert(f"Emergency audio saved: {audio_file}")
                self.log_event(f"Emergency audio saved: {audio_file}")
            
            if make_call:
                self.make_emergency_call(alert_message)
                self.log_event(f"Emergency call made: {event}")
            self.last_alert_time = now

    def detect_attributes(self, face_crop):
        if self.session is None:
            return "Unknown", "Unknown"
        try:
            resized = cv2.resize(face_crop, (128, 128))
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR) if resized.shape[2] == 3 else resized
            input_blob = bgr.astype('float32') / 255.0
            input_blob = np.transpose(input_blob, (2, 0, 1))
            input_blob = np.expand_dims(input_blob, axis=0)

            outputs = self.session.run(None, {"image": input_blob})
            gender_idx = int(np.argmax(outputs[0]))
            emotion_idx = int(np.argmax(outputs[1]))

            gender = GENDERS[gender_idx] if gender_idx < len(GENDERS) else "Unknown"
            emotion = EMOTIONS[emotion_idx] if emotion_idx < len(EMOTIONS) else "Unknown"
            return gender, emotion
        except Exception as e:
            logging.error(f"Attribute detection error: {e}")
            return "Unknown", "Unknown"

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            logging.error(f"Audio input error: {status}")
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_audio_recording(self):
        """Start recording audio"""
        self.recording = True
        self.emergency_audio = []
        self.audio_stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=16000)
        self.audio_stream.start()

    def stop_audio_recording(self):
        """Stop recording audio"""
        self.recording = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        if not self.audio_queue.empty():
            self.emergency_audio = np.concatenate([self.audio_queue.get() for _ in range(self.audio_queue.qsize())])

    def start(self):
        """Start the monitoring system"""
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.start_audio_recording()
        logging.info("Monitoring system started")
        self.log_event("=== Monitoring Session Started ===")
        self.log_event("Monitoring system started with blink detection and gesture recognition")

        # Initialize blink statistics
        self.total_blinks = 0
        self.last_blink_time = None
        self.blink_counter = 0
        self.blink_times = []
        self.closed_start = None
        self.eyes_closed = False
        self.blink_reasons = []
        self.blink_log = []  # New list to track all blinks

        while self.is_running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for face in result.multi_face_landmarks:
                    landmarks = face.landmark
                    h, w, _ = frame.shape
                    x_min = int(min(l.x for l in landmarks) * w)
                    y_min = int(min(l.y for l in landmarks) * h)
                    x_max = int(max(l.x for l in landmarks) * w)
                    y_max = int(max(l.y for l in landmarks) * h)

                    pad = 20
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(w, x_max + pad)
                    y_max = min(h, y_max + pad)

                    face_crop = frame[y_min:y_max, x_min:x_max]
                    gender, emotion = self.detect_attributes(face_crop)

                    # Get both eyes' landmarks
                    left_eye = [landmarks[i] for i in [159, 145]]
                    right_eye = [landmarks[i] for i in [386, 374]]
                    
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
                            self.alert(reason, make_call=True)
                            self.log_event(f"Long eye closure detected: {int(current_time - self.closed_start)} seconds")
                            self.blink_reasons.append(reason)
                            self.closed_start = None
                    else:  # Eyes open
                        if self.eyes_closed:  # Just opened eyes after being closed
                            self.eyes_closed = False
                            self.total_blinks += 1
                            self.blink_times.append(current_time)
                            
                            # Log the blink with timestamp
                            blink_time = datetime.now().strftime("%H:%M:%S")
                            blink_entry = f"Blink #{self.total_blinks} at {blink_time}"
                            self.blink_log.append(blink_entry)
                            self.log_event(blink_entry)
                            
                            # Remove blinks older than BLINK_WINDOW
                            self.blink_times = [t for t in self.blink_times if current_time - t <= BLINK_WINDOW]
                            
                            # Check for multiple blinks in window
                            if len(self.blink_times) >= CONSECUTIVE_BLINKS_THRESHOLD:
                                reason = f"Multiple blinks detected ({len(self.blink_times)} blinks in 30 seconds). Emergency call initiated."
                                self.alert(reason, make_call=True)
                                self.log_event(f"Multiple blinks detected: {len(self.blink_times)} blinks in 30 seconds window")
                                self.blink_reasons.append(reason)
                                self.blink_times = []
                            
                            # Check for consecutive blinks
                            if self.last_blink_time and current_time - self.last_blink_time <= BLINK_COOLDOWN:
                                self.blink_counter += 1
                                if self.blink_counter >= CONSECUTIVE_BLINKS_THRESHOLD:
                                    reason = f"Four consecutive blinks detected (blink count: {self.blink_counter}). Emergency call initiated."
                                    self.alert(reason, make_call=True)
                                    self.log_event(f"Four consecutive blinks detected: blink count = {self.blink_counter}")
                                    self.blink_reasons.append(reason)
                                    self.blink_counter = 0
                            else:
                                self.blink_counter = 1
                            
                            self.last_blink_time = current_time
                            self.log_event(f"Blink Statistics - Total: {self.total_blinks}, Window: {len(self.blink_times)}, Consecutive: {self.blink_counter}")
                        
                        self.closed_start = None

                    # Nod Detection
                    nose_y = landmarks[1].y
                    self.nod_history.append(nose_y)
                    if len(self.nod_history) == 10:
                        avg_nod = sum(self.nod_history) / 10
                        if abs(avg_nod - nose_y) > NOD_MOVEMENT_THRESHOLD:
                            self.alert("Unusual nodding pattern detected.")
                        self.nod_history.pop(0)

                    # Twitch Detection
                    brow_diff = abs(landmarks[65].y - landmarks[55].y)
                    mouth_diff = abs(landmarks[13].y - landmarks[14].y)
                    if brow_diff > TWITCH_THRESHOLD or mouth_diff > (TWITCH_THRESHOLD + 0.01):
                        self.alert("Possible facial twitch detected.")

                    # Emotion Discomfort
                    if emotion in ["Sad", "Anger"]:
                        self.alert("Emotion suggests discomfort.")

                    if self.callback:
                        self.callback({
                            'gender': gender,
                            'emotion': emotion,
                            'blinks': len(self.blink_times),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                    # Display information with enhanced visibility
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (300, 240), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                    # Display blink statistics with larger font and better contrast
                    cv2.putText(frame, f"Total Blinks: {self.total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Blinks (30s): {len(self.blink_times)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Consecutive: {self.blink_counter}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Eye Distance: {eye_dist:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Gender: {gender}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Emotion: {emotion}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display last alert reason if any
                    if self.blink_reasons:
                        last_reason = self.blink_reasons[-1]
                        cv2.putText(frame, f"Last Alert: {last_reason[:30]}...", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Display last blink time
                    if self.blink_log:
                        last_blink = self.blink_log[-1]
                        cv2.putText(frame, f"Last Blink: {last_blink}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Add timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, timestamp, (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RA Patient Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Log session summary
        self.log_event("=== Monitoring Session Summary ===")
        self.log_event(f"Total blinks detected: {self.total_blinks}")
        self.log_event("Blink Log:")
        for blink in self.blink_log:
            self.log_event(f"- {blink}")
        if self.blink_reasons:
            self.log_event("Alert reasons during session:")
            for reason in self.blink_reasons:
                self.log_event(f"- {reason}")
        self.log_event("=== Session Ended ===")

    def stop(self):
        """Stop the monitoring system"""
        self.is_running = False
        self.stop_audio_recording()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Log session summary
        self.log_event("=== Monitoring Session Summary ===")
        self.log_event(f"Total blinks detected: {self.total_blinks}")
        self.log_event("Blink Log:")
        for blink in self.blink_log:
            self.log_event(f"- {blink}")
        if self.blink_reasons:
            self.log_event("Alert reasons during session:")
            for reason in self.blink_reasons:
                self.log_event(f"- {reason}")
        self.log_event("=== Session Ended ===")

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger("EmergencySoundTracker")
    logger.info("Starting Emergency Sound Tracker application")

    try:
        # Create the main application
        app = EmergencySoundTracker()
        
        # Initialize all components
        model_path = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}. Please download it from https://alphacephei.com/vosk/models")
        
        # Initialize speech recognition
        speech_engine = SpeechRecognitionEngine(model_path)
        app.engine = speech_engine
        
        # Initialize gesture detection
        gesture_detector = GestureDetector(callback=app._on_gesture_detected)
        app.gesture_detector = gesture_detector
        
        # Initialize monitoring system
        monitor = MonitoringSystem(callback=app._on_face_recognized)
        app.face_recognizer = monitor
        
        # Initialize scheduler
        scheduler = TaskScheduler(read_aloud_callback=app._read_aloud)
        app.scheduler = scheduler
        
        # Register cleanup handlers
        atexit.register(lambda: cleanup_resources(app))
        
        # Start the main application loop
        app.mainloop()
        
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        raise

def cleanup_resources(app):
    """Cleanup all resources when the application exits"""
    logger = logging.getLogger("EmergencySoundTracker")
    
    try:
        # Stop speech recognition
        if hasattr(app, 'engine') and app.engine and app.engine.is_running:
            app.engine.stop()
            logger.info("Speech recognition stopped")
            
        # Stop gesture detection
        if hasattr(app, 'gesture_detector') and app.gesture_detector and app.gesture_detector.is_running:
            app.gesture_detector.stop()
            logger.info("Gesture detection stopped")
            
        # Stop monitoring system
        if hasattr(app, 'face_recognizer') and app.face_recognizer:
            app.face_recognizer.stop()
            logger.info("Monitoring system stopped")
            
        # Stop scheduler
        if hasattr(app, 'scheduler'):
            app.scheduler.stop()
            logger.info("Scheduler stopped")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()