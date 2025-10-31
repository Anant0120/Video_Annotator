# How to Use the Video Annotation Service

This guide provides step-by-step instructions for setting up and using the service.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python app.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Test the API

#### Using cURL

```bash
curl -X POST "http://localhost:8000/annotate" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@your_video.mp4"
```

#### Using Python

```python
import requests

url = "http://localhost:8000/annotate"
files = {"video": open("your_video.mp4", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

#### Using the Test Client

```bash
python test_client.py your_video.mp4
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with Swagger UI.

## What to Expect

### Processing Time
- Expect approximately **2-4x real-time** processing (a 1-minute video takes 2-4 minutes)

### Output Format
The API returns a JSON object with:
- `video_filename`: Name of the processed video
- `total_frames`: Total number of frames analyzed
- `labels_per_frame`: Dictionary with frame-by-frame annotations

Example:
```json
{
  "video_filename": "test.mp4",
  "total_frames": 240,
  "labels_per_frame": {
    "0": {
      "eye_state": "Open",
      "posture": "Hunched"
    }
  }
}
```

## Troubleshooting

### Server won't start
- Check that Python 3.8+ is installed
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Can't connect to API
- Ensure the server is running on port 8000
- Check firewall settings

### Poor detection accuracy
- Ensure video has good lighting
- Person should be facing camera (front or 45-degree angle)
- Avoid occlusions (glasses, hair covering eyes)

### File upload errors
- Only .mp4, .avi, or .mov files are supported
- Ensure file is not corrupted
- Check file size (processing large files may take time)

## Best Practices

1. **Video Quality**: Use well-lit videos with clear view of face and upper body
2. **Camera Angle**: Front-facing or slight angle (not side profile)
3. **Resolution**: 720p or higher recommended
4. **Duration**: Service works with any video length, but longer videos take more time

## Example Workflow

1. Record or obtain a test video (laptop camera, phone, etc.)
2. Save the video as MP4 format
3. Start the service: `python app.py`
4. In another terminal, run: `python test_client.py your_video.mp4`
5. View the results with frame-by-frame annotations

## Performance Tips

- The service processes frames sequentially
- For faster processing, ensure good hardware (faster CPU helps)
- MediaPipe is already optimized for performance
- Consider GPU acceleration for future improvements

## Cost

This service is **completely free** - no API costs, no cloud fees, runs entirely locally!

