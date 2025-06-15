import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from datetime import datetime
import threading
import winsound
from recognizer import SpeechRecognitionEngine
from scheduler import TaskScheduler
from gesture_detection import GestureDetector
from face_recognition import FaceRecognizer
import cv2
import collections.abc

class EmergencySoundTracker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emergency Sound Tracker")
        self.geometry("1200x800")
        self.resizable(False, False)

        self.model_path = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
        self.alert_sound = os.path.join(os.path.dirname(__file__), "alert.wav")
        self.emergency_keywords = {"help", "fire", "emergency", "water", "food", "medicine"}
        
        # Initialize components
        self.engine = None
        self.gesture_detector = None
        self.face_recognizer = None
        self.scheduler = TaskScheduler(read_aloud_callback=self._read_aloud)
        self.scheduler.start()  # Start the scheduler immediately

        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        # Title Label
        title_label = tk.Label(self, text="Emergency Sound Tracker", font=("Arial", 20, "bold"), fg="#003366")
        title_label.pack(pady=10)

        # Main container
        main_container = tk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel for controls
        left_panel = tk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Right panel for log
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Speech Recognition Controls
        speech_frame = tk.LabelFrame(left_panel, text="Speech Recognition", font=("Arial", 12))
        speech_frame.pack(fill=tk.X, pady=5)

        self.start_button = tk.Button(speech_frame, text="Start Listening", width=15, bg="#4CAF50", fg="white",
                                    font=("Arial", 12), command=self._toggle_listening)
        self.start_button.pack(pady=5)

        keyword_frame = tk.Frame(speech_frame)
        keyword_frame.pack(fill=tk.X, pady=5)

        self.keyword_var = tk.StringVar()
        self.keyword_entry = tk.Entry(keyword_frame, textvariable=self.keyword_var, font=("Arial", 12), width=20)
        self.keyword_entry.pack(side=tk.LEFT, padx=5)

        self.add_button = tk.Button(keyword_frame, text="Add Keyword", bg="#2196F3", fg="white", font=("Arial", 12),
                                  command=self._add_keyword)
        self.add_button.pack(side=tk.LEFT, padx=5)

        # Blink Emergency Controls (formerly Gesture Detection)
        blink_frame = tk.LabelFrame(left_panel, text="Blink Emergency Detection", font=("Arial", 12))
        blink_frame.pack(fill=tk.X, pady=5)

        self.gesture_button = tk.Button(
            blink_frame,
            text="Start Blink Detection",
            width=20,
            bg="#9C27B0",
            fg="white",
            font=("Arial", 12),
            command=self._toggle_gesture_detection
        )
        self.gesture_button.pack(pady=5)

        # Emotion Recognition Controls (formerly Face Recognition)
        emotion_frame = tk.LabelFrame(left_panel, text="Emotion Recognition", font=("Arial", 12))
        emotion_frame.pack(fill=tk.X, pady=5)

        self.face_button = tk.Button(
            emotion_frame,
            text="Start Emotion Detection",
            width=20,
            bg="#FF9800",
            fg="white",
            font=("Arial", 12),
            command=self._toggle_face_recognition
        )
        self.face_button.pack(pady=5)

        add_face_button = tk.Button(
            emotion_frame,
            text="Add New Face",
            width=20,
            bg="#607D8B",
            fg="white",
            font=("Arial", 12),
            command=self._add_new_face
        )
        add_face_button.pack(pady=5)

        # Scheduler Controls
        scheduler_frame = tk.LabelFrame(left_panel, text="Scheduler", font=("Arial", 12))
        scheduler_frame.pack(fill=tk.X, pady=5)

        self.add_task_btn = tk.Button(
            scheduler_frame,
            text="Add Reminder",
            bg="#FF9800",
            fg="white",
            font=("Arial", 12),
            command=self._add_scheduled_task
        )
        self.add_task_btn.pack(pady=5)

        self.view_tasks_btn = tk.Button(
            scheduler_frame,
            text="View Reminders",
            bg="#9C27B0",
            fg="white",
            font=("Arial", 12),
            command=self._view_scheduled_tasks
        )
        self.view_tasks_btn.pack(pady=5)

        # Log Label and Text
        log_label = tk.Label(right_panel, text="Activity Log:", font=("Arial", 14), fg="#333")
        log_label.pack(pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, font=("Courier", 11), height=30, width=70)
        self.log_text.pack(pady=10, fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

        self._log_message("System ready. Click 'Start Listening' to begin.")

    def _toggle_listening(self):
        """Toggle speech recognition on/off"""
        if self.engine and self.engine.is_running:
            self.engine.stop()
            self.engine = None
            self.start_button.config(text="Start Listening", bg="#4CAF50")
            self._log_message("Stopped listening.")
        else:
            try:
                self.engine = SpeechRecognitionEngine(model_path=self.model_path)
                self.engine.start(self._on_text_recognized)
                self.start_button.config(text="Stop Listening", bg="#f44336")
                self._log_message("Started listening.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _toggle_gesture_detection(self):
        """Toggle blink detection on/off"""
        if self.gesture_detector and self.gesture_detector.is_running:
            self.gesture_detector.stop()
            self.gesture_detector = None
            self.gesture_button.config(text="Start Blink Detection", bg="#9C27B0")
            self._log_message("Blink detection stopped.")
        else:
            self.gesture_detector = GestureDetector(callback=self._on_gesture_detected)
            self.gesture_detector.start()
            self.gesture_button.config(text="Stop Blink Detection", bg="#f44336")
            self._log_message("Blink detection started.")

    def _toggle_face_recognition(self):
        """Toggle emotion detection on/off"""
        if self.face_recognizer and self.face_recognizer.is_running:
            self.face_recognizer.stop()
            self.face_recognizer = None
            self.face_button.config(text="Start Emotion Detection", bg="#FF9800")
            self._log_message("Emotion detection stopped.")
        else:
            self.face_recognizer = FaceRecognizer(callback=self._on_face_recognized)
            self.face_recognizer.start()
            self.face_button.config(text="Stop Emotion Detection", bg="#f44336")
            self._log_message("Emotion detection started.")

    def _add_new_face(self):
        """Add a new face to the recognition system"""
        if not self.face_recognizer:
            messagebox.showwarning("Warning", "Please start emotion recognition first.")
            return

        name = simpledialog.askstring("Add Face", "Enter person's name:")
        if not name:
            return

        try:
            # Capture frame from camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Failed to open camera")
                return

            ret, frame = cap.read()
            cap.release()

            if not ret:
                messagebox.showerror("Error", "Failed to capture image")
                return

            # Add face to recognition system
            if self.face_recognizer.add_face(name, frame):
                self._log_message(f"Added new face: {name}")
                messagebox.showinfo("Success", f"Successfully added face for {name}")
            else:
                messagebox.showerror("Error", "No face detected in the image")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add face: {str(e)}")
            self._log_message(f"Error adding face: {str(e)}", is_alert=True)

    def _on_gesture_detected(self, data):
        """Handle detected gestures and blinks"""
        print(f"[DEBUG] _on_gesture_detected received: {data} (type: {type(data)})")
        if isinstance(data, dict) or isinstance(data, collections.abc.Mapping) or (hasattr(data, 'get') and 'type' in data):  # Blink event
            blink_type = data.get('type', '')
            if blink_type == 'blink':
                total_blinks = data.get('total_blinks', 0)
                blinks_in_window = data.get('blinks_in_window', 0)
                consecutive_blinks = data.get('consecutive_blinks', 0)
                timestamp = data.get('timestamp', '')
                
                message = f"👁️ BLINK DETECTED: Total={total_blinks}, Window={blinks_in_window}, Consecutive={consecutive_blinks}"
                self._log_message(message)
                
                # Alert for multiple blinks
                if blinks_in_window >= 4 or consecutive_blinks >= 4:
                    alert_msg = f"🚨 ALERT: Multiple blinks detected! ({blinks_in_window} in window, {consecutive_blinks} consecutive)"
                    self._log_message(alert_msg, is_alert=True)
                    threading.Thread(
                        target=lambda: winsound.PlaySound(self.alert_sound, winsound.SND_FILENAME | winsound.SND_ASYNC),
                        daemon=True
                    ).start()
                    messagebox.showwarning("Blink Alert", alert_msg)
        else:  # Gesture event
            self._log_message(f"🚨 GESTURE ALERT: {str(data).upper()} detected!", is_alert=True)
            threading.Thread(
                target=lambda: winsound.PlaySound(self.alert_sound, winsound.SND_FILENAME | winsound.SND_ASYNC),
                daemon=True
            ).start()
            messagebox.showwarning("Gesture Alert", f"Detected: {str(data).upper()}")

    def _on_face_recognized(self, info):
        """Handle face recognition callback"""
        name = info.get('name', 'Unknown')
        timestamp = info.get('timestamp', '')
        message = f"Recognized: {name} at {timestamp}"
        self._log_message(message)
        
        # Play alert sound for unknown faces
        if name == "Unknown":
            if os.path.exists(self.alert_sound):
                winsound.PlaySound(self.alert_sound, winsound.SND_ASYNC)

    def _on_text_recognized(self, text):
        """Handle recognized speech"""
        self._log_message(f"Recognized: {text}")
        words = text.lower().split()
        found = self.emergency_keywords.intersection(words)
        if found:
            keyword_str = ", ".join(found)
            self._log_message(f"🚨 ALERT: {keyword_str.upper()} DETECTED!", is_alert=True)
            threading.Thread(
                target=lambda: winsound.PlaySound(self.alert_sound, winsound.SND_FILENAME | winsound.SND_ASYNC),
                daemon=True
            ).start()
            messagebox.showwarning("Emergency", f"Detected: {keyword_str.upper()}")

    def _add_keyword(self):
        """Add a new emergency keyword"""
        keyword = self.keyword_var.get().strip().lower()
        if keyword:
            self.emergency_keywords.add(keyword)
            self._log_message(f"Added keyword: {keyword}")
            self.keyword_var.set("")

    def _add_scheduled_task(self):
        """Add a new scheduled task"""
        # Create a new window for task input
        task_window = tk.Toplevel(self)
        task_window.title("Add Reminder")
        task_window.geometry("400x300")
        task_window.transient(self)
        task_window.grab_set()

        # Task name
        ttk.Label(task_window, text="Reminder:").pack(pady=5)
        task_name = ttk.Entry(task_window, width=40)
        task_name.pack(pady=5)

        # Time
        ttk.Label(task_window, text="Time (HH:MM):").pack(pady=5)
        time_entry = ttk.Entry(task_window, width=10)
        time_entry.pack(pady=5)

        # Repeat type
        ttk.Label(task_window, text="Repeat:").pack(pady=5)
        repeat_var = tk.StringVar(value="none")
        repeat_frame = ttk.Frame(task_window)
        repeat_frame.pack(pady=5)
        ttk.Radiobutton(repeat_frame, text="None", variable=repeat_var, value="none").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(repeat_frame, text="Daily", variable=repeat_var, value="daily").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(repeat_frame, text="Weekly", variable=repeat_var, value="weekly").pack(side=tk.LEFT, padx=5)

        # Days selection (for weekly)
        days_frame = ttk.LabelFrame(task_window, text="Select Days (Weekly)")
        days_frame.pack(pady=10, padx=10, fill=tk.X)
        
        day_vars = []
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, day in enumerate(days):
            var = tk.BooleanVar()
            day_vars.append(var)
            ttk.Checkbutton(days_frame, text=day, variable=var).grid(row=0, column=i, padx=2)

        def save_task():
            name = task_name.get().strip()
            time_str = time_entry.get().strip()
            repeat_type = repeat_var.get()
            
            if not name or not time_str:
                messagebox.showerror("Error", "Please fill in all fields", parent=task_window)
                return
            
            days = None
            if repeat_type == "weekly":
                days = [i for i, var in enumerate(day_vars) if var.get()]
                if not days:
                    messagebox.showerror("Error", "Please select at least one day for weekly repeat", parent=task_window)
                    return
            
            if self.scheduler.add_task(name, time_str, repeat_type, days):
                self._log_message(f"Added reminder: {name} at {time_str} (repeat: {repeat_type})")
                task_window.destroy()
            else:
                messagebox.showerror("Error", "Invalid time format. Please use HH:MM", parent=task_window)

        # Save button
        ttk.Button(task_window, text="Save", command=save_task).pack(pady=10)

    def _view_scheduled_tasks(self):
        """View and manage scheduled tasks"""
        if not self.scheduler:
            messagebox.showinfo("Info", "No reminders scheduled")
            return

        # Create a new window for task list
        tasks_window = tk.Toplevel(self)
        tasks_window.title("Scheduled Reminders")
        tasks_window.geometry("600x400")
        tasks_window.transient(self)
        tasks_window.grab_set()

        # Create Treeview
        tree = ttk.Treeview(tasks_window, columns=("Time", "Repeat", "Text"), show="headings")
        tree.heading("Time", text="Time")
        tree.heading("Repeat", text="Repeat")
        tree.heading("Text", text="Reminder")
        
        tree.column("Time", width=100)
        tree.column("Repeat", width=150)
        tree.column("Text", width=300)
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Add tasks to treeview
        for task in self.scheduler.get_tasks():
            repeat_text = task['repeat_type']
            if repeat_text == "weekly":
                days = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] for d in task['days']]
                repeat_text = f"Weekly ({', '.join(days)})"
            
            tree.insert("", tk.END, values=(
                task['time'],
                repeat_text,
                task['name']
            ))

        def remove_selected():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("Warning", "Please select a reminder to remove", parent=tasks_window)
                return
            
            item = tree.item(selected[0])
            task_name = item['values'][2]  # Get task name from the third column
            
            if self.scheduler.remove_task(task_name):
                tree.delete(selected[0])
                self._log_message(f"Removed reminder: {task_name}")
                messagebox.showinfo("Success", "Reminder removed successfully", parent=tasks_window)

        # Remove button
        ttk.Button(tasks_window, text="Remove Selected", command=remove_selected).pack(pady=5)

    def _read_aloud(self, text):
        """Handle reading text aloud"""
        self._log_message(f"Reading aloud: {text}", is_alert=True)
        threading.Thread(
            target=lambda: winsound.PlaySound(self.alert_sound, winsound.SND_FILENAME | winsound.SND_ASYNC),
            daemon=True
        ).start()

    def _log_message(self, message, is_alert=False):
        """Add a message to the log"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        tag = "alert" if is_alert else "normal"
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)

        if is_alert:
            self.log_text.tag_config("alert", foreground="red", font=("Courier", 11, "bold"))
        else:
            self.log_text.tag_config("normal", foreground="black")

        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _on_close(self):
        """Handle window close event"""
        if self.scheduler:
            self.scheduler.stop()
        if self.engine:
            self.engine.stop()
        if self.gesture_detector:
            self.gesture_detector.stop()
        if self.face_recognizer:
            self.face_recognizer.stop()
        self.destroy()