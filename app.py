import logging
from ui import EmergencySoundTracker
import atexit

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("sound_tracking.log"),
            logging.StreamHandler()
        ]
    )

    # Create the application
    app = EmergencySoundTracker()
    
    # Ensure proper cleanup on exit
    atexit.register(lambda: app.scheduler.stop() if hasattr(app, 'scheduler') else None)
    
    try:
        app.mainloop()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        if hasattr(app, 'scheduler'):
            app.scheduler.stop()
        if hasattr(app, 'engine') and app.engine and app.engine.is_running:
            app.engine.stop()
        app.destroy()

if __name__ == "__main__":
    main()