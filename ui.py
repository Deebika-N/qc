import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from datetime import datetime
import threading
import winsound
from recognizer import SpeechRecognitionEngine
from scheduler import TaskScheduler

class EmergencySoundTracker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emergency Sound Tracker")
        self.geometry("800x600")
        self.resizable(False, False)

        self.model_path = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
        self.alert_sound = os.path.join(os.path.dirname(__file__), "alert.wav")
        self.emergency_keywords = {"help", "fire", "emergency", "water", "food", "medicine"}
        self.engine = None

        self._setup_ui()
        self._setup_scheduler_ui()  # Initialize scheduler UI
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        # Title Label
        title_label = tk.Label(self, text="Emergency Sound Tracker", font=("Arial", 20, "bold"), fg="#003366")
        title_label.pack(pady=10)

        # Frame for start/stop and keyword controls
        control_frame = tk.Frame(self)
        control_frame.pack(pady=10)

        self.start_button = tk.Button(control_frame, text="Start Listening", width=15, bg="#4CAF50", fg="white",
                                    font=("Arial", 12), command=self._toggle_listening)
        self.start_button.grid(row=0, column=0, padx=10)

        self.keyword_var = tk.StringVar()
        self.keyword_entry = tk.Entry(control_frame, textvariable=self.keyword_var, font=("Arial", 12), width=20)
        self.keyword_entry.grid(row=0, column=1, padx=10)

        self.add_button = tk.Button(control_frame, text="Add Keyword", bg="#2196F3", fg="white", font=("Arial", 12),
                                  command=self._add_keyword)
        self.add_button.grid(row=0, column=2, padx=10)

        # Log Label
        log_label = tk.Label(self, text="Recognition Log:", font=("Arial", 14), fg="#333")
        log_label.pack(pady=(10, 0))

        # Scrolled text box
        self.log_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=("Courier", 11), height=20, width=90)
        self.log_text.pack(pady=10)
        self.log_text.config(state=tk.DISABLED)

        self._log_message("System ready. Click 'Start Listening' to begin.")

    def _setup_scheduler_ui(self):
        """Setup the scheduler UI components"""
        # Scheduler controls frame
        self.scheduler_frame = tk.Frame(self)
        self.scheduler_frame.pack(pady=10)
        
        # Add Reminder Button
        self.add_task_btn = tk.Button(
            self.scheduler_frame, 
            text="Add Reminder", 
            bg="#FF9800",  # Orange color to distinguish from other buttons
            fg="white",
            font=("Arial", 12),
            command=self._add_scheduled_task
        )
        self.add_task_btn.grid(row=0, column=0, padx=5)
        
        # View Reminders Button
        self.view_tasks_btn = tk.Button(
            self.scheduler_frame,
            text="View Reminders",
            bg="#9C27B0",  # Purple color
            fg="white",
            font=("Arial", 12),
            command=self._view_scheduled_tasks
        )
        self.view_tasks_btn.grid(row=0, column=1, padx=5)
        
        # Initialize scheduler
        self.scheduler = TaskScheduler(read_aloud_callback=self._read_aloud)
        self.scheduler.start()

        # Check for triggered tasks periodically
        self.after(1000, self._check_scheduled_tasks)

    def _check_scheduled_tasks(self):
        """Check for any triggered tasks (called periodically)"""
        tasks = self.scheduler.check_for_tasks()
        for task in tasks:
            self._log_message(f"REMINDER: {task['name']}", is_alert=True)
            threading.Thread(
                target=lambda: winsound.PlaySound(self.alert_sound, winsound.SND_FILENAME | winsound.SND_ASYNC),
                daemon=True
            ).start()
            messagebox.showwarning("Reminder", task['name'])
        
        # Check again in 1 second
        self.after(1000, self._check_scheduled_tasks)

    def _add_scheduled_task(self):
        """Open dialog to add a new scheduled task"""
        task_name = simpledialog.askstring("Add Reminder", "Enter reminder text:")
        if not task_name:
            return
            
        task_time = simpledialog.askstring("Add Reminder", "Enter time (HH:MM):")
        if not task_time:
            return
            
        repeat = messagebox.askyesno("Add Reminder", "Repeat daily?")
        
        try:
            # Validate time format
            datetime.datetime.strptime(task_time, "%H:%M")
            self.scheduler.add_task(task_name, task_time, repeat)
            self._log_message(f"Added reminder: {task_name} at {task_time}")
        except ValueError:
            messagebox.showerror("Error", "Invalid time format. Please use HH:MM")

    def _view_scheduled_tasks(self):
        """Show all scheduled tasks"""
        tasks = self.scheduler.get_tasks()
        if not tasks:
            messagebox.showinfo("Scheduled Reminders", "No reminders scheduled")
            return
            
        task_list = "\n".join([f"{t['name']} at {t['time']} (daily: {t['repeat']})" for t in tasks])
        messagebox.showinfo("Scheduled Reminders", task_list)

    def _read_aloud(self, text):
        """Handle reading text aloud"""
        self._log_message(f"Reading aloud: {text}", is_alert=True)
        # Currently just shows in log, can integrate with TTS here
        threading.Thread(
            target=lambda: winsound.PlaySound(self.alert_sound, winsound.SND_FILENAME | winsound.SND_ASYNC),
            daemon=True
        ).start()

    def _toggle_listening(self):
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

    def _on_text_recognized(self, text):
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

    def _log_message(self, message, is_alert=False):
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

    def _add_keyword(self):
        keyword = self.keyword_var.get().strip().lower()
        if keyword:
            self.emergency_keywords.add(keyword)
            self._log_message(f"Added keyword: {keyword}")
            self.keyword_var.set("")

    def _on_close(self):
        """Handle window closing"""
        if self.engine and self.engine.is_running:
            self.engine.stop()
        if hasattr(self, 'scheduler'):
            self.scheduler.stop()
        self.destroy()