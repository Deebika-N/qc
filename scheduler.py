import threading
import time
import datetime
import winsound
from queue import Queue
import logging

class TaskScheduler:
    def __init__(self, read_aloud_callback=None):
        self.scheduled_tasks = []
        self.task_queue = Queue()
        self.is_running = False
        self.scheduler_thread = None
        self.read_aloud_callback = read_aloud_callback
        self.logger = logging.getLogger("EmergencySoundTracker")

    def add_task(self, task_name, task_time, repeat_daily=False):
        """Add a new task to the scheduler"""
        task = {
            'name': task_name,
            'time': task_time,
            'repeat': repeat_daily
        }
        self.scheduled_tasks.append(task)
        self.logger.info(f"Added task: {task_name} at {task_time} (repeat: {repeat_daily})")
        return True

    def remove_task(self, task_name):
        """Remove a task from the scheduler"""
        self.scheduled_tasks = [t for t in self.scheduled_tasks if t['name'] != task_name]
        self.logger.info(f"Removed task: {task_name}")
        return True

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

    def _run_scheduler(self):
        """Main scheduler loop that checks for due tasks"""
        while self.is_running:
            now = datetime.datetime.now()
            current_time = now.time()
            
            for task in self.scheduled_tasks:
                task_time = task['time']
                if isinstance(task_time, str):
                    try:
                        task_time = datetime.datetime.strptime(task_time, "%H:%M").time()
                    except ValueError:
                        continue
                
                # Check if task is due
                if (current_time.hour == task_time.hour and 
                    current_time.minute == task_time.minute and
                    current_time.second < 5):  # 5-second window to trigger
                    
                    self._trigger_task(task)
                    
                    # Handle daily repeats
                    if not task['repeat']:
                        self.scheduled_tasks.remove(task)
            
            time.sleep(1)  # Check every second

    def _trigger_task(self, task):
        """Handle task triggering"""
        self.logger.info(f"Task triggered: {task['name']}")
        
        # Add to queue for UI thread to handle
        self.task_queue.put(task)
        
        # Play alert sound
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        
        # Read aloud if callback is available
        if self.read_aloud_callback:
            self.read_aloud_callback(f"Reminder: {task['name']}")

    def check_for_tasks(self):
        """Check if any tasks need UI attention (to be called from main thread)"""
        tasks = []
        while not self.task_queue.empty():
            tasks.append(self.task_queue.get())
        return tasks