# EASEEDGE - Offline AI Health Assistant

**Team:** DEEBIKA N, DIVYA NANDHINI, SUDHARSHANA, ATHILAKSHMI

## Problem Statement

RA patients, especially those who are bedridden, often face three serious problems: constant pain and exhaustion that limit their ability to speak or move, delays in diagnosis and care, and the high cost and side effects of long-term treatment. Tech can help, but cloud-based AI raises latency issues, cost and needs internet. Regular AI running on low-end devices is often too slow to be useful.

## Our Solution

A fully offline, personal Edge AI assistant that runs on any basic device placed near the patient like a tablet, bedside screen, or small PC. It needs no internet and keeps all data private without needing any cloud connection.

## What Makes It Unique

Unlike cloud AI, Edge AI works locally, meaning no data leaves the room. Unlike standard local AI, our assistant adapts to the patient's subtle cues (like a blink or breath) and then trims unneeded parts of itself to run faster. It becomes a highly efficient, custom assistant that works smoothly on budget hardware.

A low-power "listener" stays active in the background, using very little energy. It activates the main AI only when needed, which saves power and improves speed. We also directly use device hardware for extra performance.

## Features

*   **Voice Commands**: Recognizes spoken commands and pain sounds (e.g., "water", "help").
*   **Tiny Gesture Detection**: Detects subtle gestures like blinks, nods, and facial twitches for communication.
*   **Emotion Recognition**: Identifies signs of discomfort or pain from facial expressions.
*   **Private Care Logs & Reminders**: Keeps a local log of events and manages patient reminders.
*   **Emergency SOS Alerts**: Triggers an SOS alert if the patient's eyes remain closed for an extended period.

## Tech Stack

*   **AI Runtimes**: TensorFlow Lite, ONNX Runtime (for deploying highly optimized, compact models)
*   **Visual & Audio Processing**: MediaPipe (for efficient subtle cue detection), Vosk (for offline speech recognition).
*   **Hardware Acceleration**: OpenCL, Vulkan Compute, NPU SDKs (to maximize processing speed).
*   **Application Frameworks**: Python with efficient UI libraries.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```
    Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Speech Recognition Model:**
    This application uses the Vosk Speech Recognition Toolkit. Download a model from the [Vosk models page](https://alphacephei.com/vosk/models) (the small English model is a good starting point) and extract it into the project's root directory.

4.  **Configure Twilio (Optional):**
    For SMS and call alerts, you need a Twilio account. Open `app.py` and replace the placeholder credentials with your own:
    ```python
    TWILIO_SID = "YOUR_TWILIO_SID"
    TWILIO_AUTH = "YOUR_TWILIO_AUTH_TOKEN"
    TWILIO_FROM = "YOUR_TWILIO_PHONE_NUMBER"
    CARETAKER_PHONE = "CARETAKER_PHONE_NUMBER"
    ```

## How to Run

To start the application, run the main UI file from the terminal:

```bash
python ui.py
```
The application will start, activating the camera and microphone to monitor the patient.

## Usage

*   The application runs in the background, continuously monitoring for visual and audio cues.
*   Alerts are triggered based on detected events (SOS, specific words, etc.).
*   All events are logged in `care_log.txt`.
*   Emergency audio clips are saved locally.

## Open Source Resources

This project is built upon several powerful open-source technologies:
*   [OpenCV](https://opencv.org/)
*   [MediaPipe](https://mediapipe.dev/)
*   [ONNX Runtime](https://onnxruntime.ai/)
*   [Vosk](https://alphacephei.com/vosk/)
*   [Python](https://www.python.org/) 