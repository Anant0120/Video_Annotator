# Quick Start Guide

Get started with the Video Annotation Service in 3 simple steps!

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start the Server

```bash
python app.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 3: Open in Browser

Visit: **http://localhost:8000**

## 📋 What You Get

When you open the web interface, you'll see two tabs:

### 1. Upload Video Tab 
- Drag and drop or click to upload video files
- Supports MP4, AVI, MOV formats
- Get comprehensive frame-by-frame analysis
- View detailed statistics and breakdown

### 2. Real-Time Camera Tab 
- Click "Start Camera" to enable webcam
- See live annotations as you move
- Watch your eye state and posture update in real-time
- Click "Stop Camera" when done

## Try It Out

### For Upload Mode:
1. Record a short video of yourself (use your phone or laptop camera)
2. Save it as MP4
3. Drag it onto the upload zone
4. Wait for processing (2-4x video length)
5. View your results!

### For Real-Time Mode:
1. Go to "Real-Time Camera" tab
2. Click "Start Camera"
3. Allow browser camera permissions


## Tips

- **Best Lighting**: Ensure your face is well-lit for better detection
- **Camera Angle**: Face the camera directly or at a slight angle
- **Distance**: Stay 2-4 feet from the camera
- **Privacy**: All processing happens locally - no data leaves your machine!

## Troubleshooting

**Can't start camera?**
- Check browser permissions for camera access
- Try using Chrome or Firefox

**Videos won't process?**
- Ensure file format is MP4, AVI, or MOV
- Check file isn't corrupted

**Need help?**
- Check the full README.md for detailed information
- Check HOW_TO_USE.md for more examples



