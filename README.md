# Tiny Gesture Detection System

This system is designed to detect small facial gestures (blinks, nods, and facial twitches) using computer vision. It's particularly useful for bedridden RA (Rheumatoid Arthritis) patients who may have limited mobility.

## Features

- Blink detection
- Nod detection
- Facial twitch detection
- Real-time visualization of facial landmarks
- Counter for detected gestures

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the gesture detection script:
```bash
python gesture_detector.py
```

2. The program will open your webcam and start detecting gestures
3. Press 'q' to quit the program

## How it Works

The system uses MediaPipe's Face Mesh to detect 468 facial landmarks. It then analyzes these landmarks to detect:

- **Blinks**: Using Eye Aspect Ratio (EAR) calculation
- **Nods**: By tracking vertical movement of facial landmarks
- **Facial Twitches**: By monitoring rapid movements of facial features

## Customization

You can adjust the sensitivity of the detection by modifying these parameters in the `GestureDetector` class:

- `blink_threshold`: Lower values make blink detection more sensitive
- `nod_threshold`: Lower values make nod detection more sensitive
- `twitch_threshold`: Lower values make twitch detection more sensitive
- `cooldown`: Time in seconds between consecutive detections

## Notes

- Ensure good lighting conditions for optimal detection
- Keep your face within the camera frame
- The system works best when the face is clearly visible and well-lit
- For bedridden patients, position the camera at an appropriate angle for comfortable interaction

## Troubleshooting

If you experience issues:

1. Check if your webcam is working properly
2. Ensure all dependencies are installed correctly
3. Verify that you have sufficient lighting
4. Make sure your face is clearly visible to the camera

## License

This project is open source and available for anyone to use and modify. 