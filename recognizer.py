import queue
import json
import threading
import logging
import sounddevice as sd
from vosk import Model, KaldiRecognizer

logger = logging.getLogger("EmergencySoundTracker")

class SpeechRecognitionEngine:
    def __init__(self, model_path, samplerate=16000, device=None):
        self.samplerate = samplerate
        self.device = device
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream = None
        self.recognition_thread = None

        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.samplerate)

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio stream status: {status}")
        self.audio_queue.put(bytes(indata))

    def start(self, text_callback):
        if self.is_running:
            return
        self.is_running = True
        self.text_callback = text_callback
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.recognition_thread.start()

        self.stream = sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            device=self.device,
            dtype='int16',
            channels=1,
            callback=self.audio_callback
        )
        self.stream.__enter__()

    def _recognition_loop(self):
        while self.is_running:
            try:
                data = self.audio_queue.get(timeout=1)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        self.text_callback(text)
            except queue.Empty:
                continue

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.__exit__(None, None, None)
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=2)
