# Video Annotation Service

An AI-powered video annotation service with **real-time and file upload capabilities** that analyzes video frames to detect eye state (Open/Closed) and posture (Straight/Hunched) using computer vision.

## Overview

This service uses **MediaPipe** for facial landmark detection and pose estimation to analyze video frames. The system offers:
- **File Upload**: Analyze pre-recorded video files with detailed frame-by-frame results
- **Real-Time Processing**: Live camera feed with instant annotations


### Key Features

**Two Processing Modes:**
- Upload video files (MP4, AVI, MOV) for thorough analysis
- Use your webcam for real-time, live annotations

**Dual Detection:**
- Eye State: Open or Closed
- Posture: Straight or Hunched

 **Real-Time Performance:**
- WebSocket-based live processing
- Low latency feedback
- Smooth annotations at ~10 FPS

### Approach & Technology Choices

**Why MediaPipe?**
- **Free & Open Source**: No API costs, runs locally
- **High Performance**: Optimized C++ backend with Python bindings
- **Robust Detection**: Industry-standard Google library used by millions
- **Real-time Capable**: Can process videos quickly
- **Good Accuracy**: State-of-the-art models for face and pose detection

**Model Details:**
- **Eye State Detection**: Uses MediaPipe Face Mesh with 468 facial landmarks. Calculates Eye Aspect Ratio (EAR) to determine if eyes are open or closed.
- **Posture Detection**: Uses MediaPipe Pose estimation with 33 body landmarks. Analyzes shoulder and hip alignment to detect hunched vs. straight posture.


## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd WellnessAtWork
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Service

Start the FastAPI server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Access the Web UI:**
- Open your browser and navigate to: `http://localhost:8000`
- You'll see a beautiful web interface with two modes:
  -  **Upload Video** tab: Upload and analyze video files
  -  **Real-Time Camera** tab: Use your webcam for live processing

**Alternative Access:**
- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Usage Guide

### Option 1: Web UI (Recommended)

1. **Start the server**: `python app.py`
2. **Open browser**: Go to `http://localhost:8000`
3. **Choose mode:**
   - **Upload Tab**: Click or drag-and-drop a video file
   - **Real-Time Tab**: Click "Start Camera" to use your webcam

### Option 2: API Usage

### Endpoint

**POST** `/annotate`

### Request

Upload a video file (.mp4, .avi, or .mov format).

#### Example cURL Command

```bash
curl -X POST "https://<your-service>.onrender.com/annotate?download=false" \
  -F "video=@path/to/your_video.mp4"
```

#### Example Python Request

```python
import requests

url = "http://localhost:8000/annotate"
files = {"video": open("path/to/your/video.mp4", "rb")}
response = requests.post(url, files=files)

print(response.json())
```

### Option 3: Real-Time WebSocket API

For real-time camera processing:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Eye State:', data.eye_state);
    console.log('Posture:', data.posture);
};
```

### Response Format

```json
{
  "video_filename": "test_video.mp4",
  "total_frames": 240,
  "labels_per_frame": {
    "0": {
      "eye_state": "Open",
      "posture": "Hunched"
    },
    "1": {
      "eye_state": "Open",
      "posture": "Hunched"
    },
    "2": {
      "eye_state": "Closed",
      "posture": "Straight"
    }
  }
}
```


## Project Structure

```
WellnessAtWork/
├── app.py                  # FastAPI application
├── eye_detector.py         # Eye state detection module
├── posture_detector.py     # Posture detection module
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## Limitations & Future Improvements

**Current Limitations:**
1. Single person detection (first person in frame)
2. Requires frontal or semi-frontal view for optimal accuracy
3. Lighting conditions affect detection quality
4. No ground truth dataset for evaluation
-

**Potential Improvements:**
1. Add confidence scores for each detection
2. Implement temporal smoothing to reduce frame-to-frame jitter
3. Add support for custom confidence thresholds
4. Provide bounding box visualization
5. Add batch processing for multiple videos
6. Implement GPU acceleration for faster processing



