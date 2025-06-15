import cv2
import numpy as np
import os
import logging
from datetime import datetime
import threading
import time
import requests
from twilio.rest import Client
import json
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Anger"]

class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", callback=None):
        self.known_faces_dir = known_faces_dir
        self.callback = callback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.is_running = False
        self.thread = None
        self.last_detection_time = None
        self.detection_cooldown = 5  # seconds
        
        # Twilio configuration
        self.twilio_account_sid = 'AC39ff626709438aa9595dbfb36eee793f'
        self.twilio_auth_token = '2b028e3e2bead7a23779d2f0d9099cd4'
        self.twilio_phone_number = '+1 815 421 1277'
        self.alert_phone_number = '+919043623005'
        self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
        
        # Emotion detection model
        self.session = None
        self.model_path = "face_attrib_net-facial-attribute-detection-float.onnx"
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Configure ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(self.model_path, session_options)
            
            # Get model input details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            raise
        
        # Create known faces directory if it doesn't exist
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
        
    def _detect_attributes(self, face_crop):
        if self.session is None:
            logger.error("Model session is None")
            return EMOTIONS[0]
            
        try:
            # Ensure face crop is valid
            if face_crop is None or face_crop.size == 0:
                logger.error("Invalid face crop")
                return EMOTIONS[0]
                
            # Debug input
            logger.info(f"Input face crop shape: {face_crop.shape}")
            
            # Preprocess image
            resized = cv2.resize(face_crop, (128, 128))
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR) if resized.shape[2] == 3 else resized
            input_blob = bgr.astype('float32') / 255.0
            input_blob = np.transpose(input_blob, (2, 0, 1))
            input_blob = np.expand_dims(input_blob, axis=0)
            
            # Debug preprocessed input
            logger.info(f"Preprocessed input shape: {input_blob.shape}")
            logger.info(f"Input range: [{input_blob.min()}, {input_blob.max()}]")
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_blob})
            
            # Debug model outputs
            logger.info(f"Model output shape: {outputs[1].shape}")
            logger.info(f"Raw emotion scores: {outputs[1][0]}")
            
            # Get emotion with highest probability
            emotion_scores = outputs[1][0]
            emotion_idx = int(np.argmax(emotion_scores))
            max_score = float(emotion_scores[emotion_idx])
            
            logger.info(f"Selected emotion index: {emotion_idx}, score: {max_score}")
            
            # Only return emotion if confidence is high enough
            if max_score > 0.5 and emotion_idx < len(EMOTIONS):
                emotion = EMOTIONS[emotion_idx]
                logger.info(f"Detected emotion: {emotion} with confidence {max_score:.2f}")
                return emotion
            else:
                logger.warning(f"Low confidence ({max_score:.2f}) or invalid index {emotion_idx}, defaulting to {EMOTIONS[0]}")
                return EMOTIONS[0]
                
        except Exception as e:
            logger.error(f"Attribute detection error: {str(e)}")
            return EMOTIONS[0]
    
    def _send_alert(self, message):
        """Send alert via Twilio"""
        try:
            # Send SMS
            self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_phone_number,
                to=self.alert_phone_number
            )
            
            # Make a call
            self.twilio_client.calls.create(
                to=self.alert_phone_number,
                from_=self.twilio_phone_number,
                twiml=f'<Response><Say>{message}</Say></Response>'
            )
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
    
    def _detection_loop(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        last_emotion_alert_time = 0
        emotion_alert_cooldown = 15  # seconds
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # For emotion detection, crop from color frame
                face_crop_color = frame[y:y+h, x:x+w]
                emotion = self._detect_attributes(face_crop_color)
                
                # Emotion discomfort alert
                if emotion in ["Sad", "Anger"] and (time.time() - last_emotion_alert_time > emotion_alert_cooldown):
                    self._send_alert(f"Alert: Detected emotion '{emotion}'. Possible discomfort.")
                    last_emotion_alert_time = time.time()
                
                # Draw rectangle and info
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Display emotion
                cv2.putText(frame, f"Emotion: {emotion}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def start(self):
        """Start face recognition"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._detection_loop)
            self.thread.start()
            logger.info("Face recognition started")
    
    def stop(self):
        """Stop face recognition"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        logger.info("Face recognition stopped") 