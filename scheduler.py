import threading
import time
import datetime
import winsound
from queue import Queue
import logging
import json
import os
import pyttsx3
import subprocess
import schedule

class TaskScheduler:
    def __init__(self, read_aloud_callback=None):
        self.scheduled_tasks = []
        self.task_queue = Queue()
        self.is_running = False
        self.scheduler_thread = None
        self.read_aloud_callback = read_aloud_callback
        self.logger = logging.getLogger("EmergencySoundTracker")
        self.reminders_file = "reminders.json"
        
        # Initialize text-to-speech engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.logger.info("Text-to-speech engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize text-to-speech engine: {e}")
            self.engine = None
        
        self._load_reminders()
        self.logger.info("TaskScheduler initialized")

    def _load_reminders(self):
        """Load reminders from JSON file"""
        if os.path.exists(self.reminders_file):
            try:
                with open(self.reminders_file, 'r') as f:
                    self.scheduled_tasks = json.load(f)
                self.logger.info(f"Loaded {len(self.scheduled_tasks)} reminders from file")
            except Exception as e:
                self.logger.error(f"Error loading reminders: {e}")
                self.scheduled_tasks = []
        else:
            self.logger.info("No reminders file found, starting with empty list")

    def _save_reminders(self):
        """Save reminders to JSON file"""
        try:
            with open(self.reminders_file, 'w') as f:
                json.dump(self.scheduled_tasks, f, indent=4)
            self.logger.info("Reminders saved to file")
        except Exception as e:
            self.logger.error(f"Error saving reminders: {e}")

    def add_task(self, task_name, task_time, repeat_type="none", days=None):
        """Add a new task to the scheduler"""
        try:
            # Validate time format
            datetime.datetime.strptime(task_time, "%H:%M")
            
            task = {
                'name': task_name,
                'time': task_time,
                'repeat_type': repeat_type,  # "none", "daily", "weekly"
                'days': days,  # List of days for weekly repeat (0-6, where 0 is Monday)
                'last_triggered': None
            }
            
            self.scheduled_tasks.append(task)
            self._save_reminders()
            self.logger.info(f"Added task: {task_name} at {task_time} (repeat: {repeat_type})")
            
            # Schedule the task
            schedule.every().day.at(task_time).do(self.notify_task, task)
            return True
        except ValueError as e:
            self.logger.error(f"Invalid time format: {task_time} - {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error adding task: {e}")
            return False

    def remove_task(self, task_name):
        """Remove a task from the scheduler"""
        try:
            self.scheduled_tasks = [t for t in self.scheduled_tasks if t['name'] != task_name]
            self._save_reminders()
            self.logger.info(f"Removed task: {task_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error removing task: {e}")
            return False

    def get_tasks(self):
        """Get all scheduled tasks"""
        return self.scheduled_tasks.copy()

    def start(self):
        """Start the scheduler thread"""
        if not self.is_running:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            self.logger.info("Task scheduler started")

    def stop(self):
        """Stop the scheduler thread"""
        self.is_running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=2)
        self.logger.info("Task scheduler stopped")

    def _should_trigger(self, task):
        """Check if a task should be triggered"""
        try:
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M")
            current_day = now.weekday()  # 0-6, where 0 is Monday
            
            # Check if time matches
            if current_time != task['time']:
                return False
                
            # Check if already triggered today
            last_triggered = task.get('last_triggered')
            if last_triggered:
                last_triggered = datetime.datetime.strptime(last_triggered, "%Y-%m-%d %H:%M:%S")
                if last_triggered.date() == now.date():
                    return False
            
            # Check repeat conditions
            if task['repeat_type'] == "none":
                return True
            elif task['repeat_type'] == "daily":
                return True
            elif task['repeat_type'] == "weekly":
                return current_day in task['days']
            
            return False
        except Exception as e:
            self.logger.error(f"Error in _should_trigger: {e}")
            return False

    def _play_alert_sound(self):
        """Play alert sound using multiple methods"""
        try:
            # Method 1: Play system beep (most reliable)
            for _ in range(3):  # Play 3 beeps
                winsound.Beep(1000, 500)  # 1000Hz for 0.5 seconds
                time.sleep(0.1)  # Small pause between beeps
            self.logger.info("Played beep sound")
        except Exception as e:
            self.logger.error(f"Failed to play beep: {e}")
            try:
                # Method 2: Try playing system sound
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                self.logger.info("Played system alert sound")
            except Exception as e:
                self.logger.error(f"Failed to play system sound: {e}")
                try:
                    # Method 3: Use PowerShell to play a beep
                    subprocess.run(['powershell', '-c', '[console]::beep(1000,1000)'])
                    self.logger.info("Played PowerShell beep")
                except Exception as e:
                    self.logger.error(f"Failed to play PowerShell beep: {e}")

    def _read_aloud(self, text):
        """Read text aloud with multiple fallback methods"""
        try:
            # Method 1: Use callback if available
            if self.read_aloud_callback:
                self.read_aloud_callback(text)
                return
            
            # Method 2: Use pyttsx3
            if self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
                return
            
            # Method 3: Use PowerShell speech synthesis
            try:
                subprocess.run(['powershell', '-c', f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'])
            except Exception as e:
                self.logger.error(f"Failed to use PowerShell speech: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to read text aloud: {e}")

    def _run_scheduler(self):
        """Main scheduler loop that checks for due tasks"""
        self.logger.info("Scheduler thread started")
        while self.is_running:
            try:
                now = datetime.datetime.now()
                self.logger.debug(f"Checking tasks at {now.strftime('%H:%M:%S')}")
                
                for task in self.scheduled_tasks:
                    if self._should_trigger(task):
                        self.logger.info(f"Triggering task: {task['name']}")
                        # Update last triggered time
                        task['last_triggered'] = now.strftime("%Y-%m-%d %H:%M:%S")
                        self._save_reminders()
                        
                        # Add to queue for UI thread to handle
                        self.task_queue.put(task)
                        
                        # Play alert sound
                        self._play_alert_sound()
                        
                        # Read aloud
                        self._read_aloud(f"Reminder: {task['name']}")
                
                time.sleep(1)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1)  # Wait a bit before retrying

    def check_for_tasks(self):
        """Check if any tasks need UI attention (to be called from main thread)"""
        tasks = []
        while not self.task_queue.empty():
            tasks.append(self.task_queue.get())
        return tasks

    def notify_task(self, task):
        if not task['notified']:
            # Read the task text aloud
            self._read_aloud(task['name'])
            task['notified'] = True
            self._save_reminders()

def main():
    scheduler = TaskScheduler()
    
    # Example usage
    while True:
        print("\nTask Scheduler Menu:")
        print("1. Add new task")
        print("2. View all tasks")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            task_name = input("Enter task name: ")
            task_time = input("Enter time (HH:MM): ")
            repeat_type = input("Enter repeat type (none, daily, weekly): ")
            days = None
            if repeat_type == "weekly":
                days = [int(day) for day in input("Enter days (0-6, separated by spaces): ").split()]
            if scheduler.add_task(task_name, task_time, repeat_type, days):
                print("Task scheduled successfully!")
            
        elif choice == '2':
            print("\nScheduled Tasks:")
            for task in scheduler.get_tasks():
                print(f"Task: {task['name']} at {task['time']} (repeat: {task['repeat_type']})")
                
        elif choice == '3':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()