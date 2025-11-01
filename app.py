from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import cv2
import json
import os
import tempfile
from pathlib import Path
from eye_detector import EyeStateDetector
from posture_detector import PostureDetector
import time
import base64
import numpy as np
from typing import Set

app = FastAPI(
    title="Video Annotation Service",
    description="AI-powered frame-by-frame video analysis for eye state and posture detection"
)

# Initialize detectors
eye_detector = EyeStateDetector()
posture_detector = PostureDetector()


# Active connections for real-time processing
active_connections: Set[WebSocket] = set()

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Annotation Service</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .tabs {
            display: flex;
            background: #f0f0f0;
            border-bottom: 2px solid #ddd;
        }
        .tab {
            flex: 1;
            padding: 20px;
            background: transparent;
            border: none;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }
        .tab:hover { background: #e0e0e0; }
        .tab.active {
            background: white;
            border-bottom: 3px solid #667eea;
            color: #667eea;
        }
        .content {
            padding: 40px;
        }
        .hidden { display: none; }
        .upload-zone {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px;
            text-align: center;
            background: #f9f9ff;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-zone:hover {
            border-color: #764ba2;
            background: #f0f0ff;
        }
        .upload-zone.dragover {
            border-color: #764ba2;
            background: #e8e8ff;
        }
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        .file-input { display: none; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 30px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 20px;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .camera-container {
            text-align: center;
        }
        #videoElement {
            width: 100%;
            max-width: 800px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            background: #000;
        }
        .annotation-panel {
            margin-top: 30px;
            padding: 30px;
            background: #f9f9f9;
            border-radius: 15px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
        }
        .annotation-card {
            flex: 1;
            min-width: 200px;
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .annotation-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .annotation-value {
            font-size: 1.8em;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
        }
        .eye-open { color: #4caf50; background: #e8f5e9; }
        .eye-closed { color: #f44336; background: #ffebee; }
        .posture-straight { color: #2196f3; background: #e3f2fd; }
        .posture-hunched { color: #ff9800; background: #fff3e0; }
        .status-indicator {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            font-weight: 600;
        }
        .status-active { background: #c8e6c9; color: #2e7d32; }
        .status-idle { background: #ffcdd2; color: #c62828; }
        .results {
            margin-top: 30px;
            padding: 30px;
            background: #f9f9f9;
            border-radius: 15px;
        }
        .result-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 10px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎥 Video Annotation Service</h1>
            <p>AI-Powered Eye State & Posture Detection</p>
        </header>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('upload')">📁 Upload Video</button>
            <button class="tab" onclick="switchTab('realtime')">📷 Real-Time Camera</button>
        </div>
        
        <div class="content">
            <!-- Upload Tab -->
            <div id="uploadTab" class="tab-content">
                <div class="upload-zone" onclick="document.getElementById('fileInput').click()" 
                     ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                    <div class="upload-icon">📤</div>
                    <h2>Drag & Drop or Click to Upload</h2>
                    <p>Supported formats: MP4, AVI, MOV</p>
                    <input type="file" id="fileInput" class="file-input" accept="video/*" onchange="handleFileSelect(event)">
                </div>
                
                <div id="uploadResults" class="results hidden">
                    <h3>Analysis Results</h3>
                    <div id="resultContent"></div>
                </div>
                
                <div id="uploadProgress" class="hidden">
                    <div class="spinner"></div>
                    <p style="text-align: center;">Processing video...</p>
                </div>
            </div>
            
            <!-- Real-Time Tab -->
            <div id="realtimeTab" class="tab-content hidden">
                <div class="camera-container">
                    <video id="videoElement" autoplay playsinline></video>
                    
                    <div class="annotation-panel">
                        <div class="annotation-card">
                            <div class="annotation-label">Eye State</div>
                            <div id="eyeState" class="annotation-value">-</div>
                        </div>
                        <div class="annotation-card">
                            <div class="annotation-label">Posture</div>
                            <div id="posture" class="annotation-value">-</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 30px;">
                        <button id="startCamera" class="btn" onclick="startCamera()">Start Camera</button>
                        <button id="stopCamera" class="btn" onclick="stopCamera()" disabled>Stop Camera</button>
                    </div>
                    
                    <div id="cameraStatus" class="status-indicator status-idle" style="margin-top: 20px;">
                        Camera: Idle
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentTab = 'upload';
        let cameraStream = null;
        let ws = null;
        let currentVideoFilename = null;
        
        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
            
            if (tab === 'upload') {
                document.querySelector('.tab').classList.add('active');
                document.getElementById('uploadTab').classList.remove('hidden');
            } else {
                document.querySelectorAll('.tab')[1].classList.add('active');
                document.getElementById('realtimeTab').classList.remove('hidden');
            }
        }
        
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                uploadAndProcess(file);
            }
        }
        
        function handleDragOver(event) {
            event.preventDefault();
            event.currentTarget.classList.add('dragover');
        }
        
        function handleDragLeave(event) {
            event.currentTarget.classList.remove('dragover');
        }
        
        function handleDrop(event) {
            event.preventDefault();
            event.currentTarget.classList.remove('dragover');
            const file = event.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) {
                uploadAndProcess(file);
            }
        }
        
        async function uploadAndProcess(file) {
            const formData = new FormData();
            formData.append('video', file);
            
            document.getElementById('uploadProgress').classList.remove('hidden');
            document.getElementById('uploadResults').classList.add('hidden');
            
            try {
                // Store the filename for download purposes
                currentVideoFilename = file.name;
                
                const response = await fetch('/annotate?download=false', {
                    method: 'POST',
                    body: formData
                    // Don't set Content-Type - browser will set it automatically with boundary
                });
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing video: ' + error.message);
            } finally {
                document.getElementById('uploadProgress').classList.add('hidden');
            }
        }
        
        async function downloadResultsAsJSON() {
            if (!currentVideoFilename) {
                alert('Please upload and process a video file first');
                return;
            }
            
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) {
                alert('Please upload a video file first');
                return;
            }
            
            const formData = new FormData();
            formData.append('video', fileInput.files[0]);
            
            try {
                const response = await fetch('/annotate?download=true', {
                    method: 'POST',
                    body: formData
                    // Don't set Content-Type - browser will set it automatically with boundary
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = currentVideoFilename.replace(/\.[^/.]+$/, '_results.json');
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                } else {
                    alert('Error downloading results');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error downloading results: ' + error.message);
            }
        }
        
        function displayResults(result) {
            document.getElementById('uploadResults').classList.remove('hidden');
            
            let html = `
                <div style="text-align: center; margin-bottom: 20px;">
                    <button class="btn" onclick="downloadResultsAsJSON()" style="background: #4caf50;">
                        📥 Download JSON Results
                    </button>
                </div>
                <div class="result-summary">
                <div class="stat-card">
                    <div class="stat-value">${result.total_frames}</div>
                    <div class="stat-label">Total Frames</div>
                </div>`;
            
            // Count statistics
            let openEyes = 0, closedEyes = 0;
            let straightPostures = 0, hunchedPostures = 0;
            
            for (let frame in result.labels_per_frame) {
                const labels = result.labels_per_frame[frame];
                if (labels.eye_state === 'Open') openEyes++;
                else if (labels.eye_state === 'Closed') closedEyes++;
                if (labels.posture === 'Straight') straightPostures++;
                else if (labels.posture === 'Hunched') hunchedPostures++;
            }
            
            html += `
                <div class="stat-card">
                    <div class="stat-value" style="color: #4caf50;">${openEyes}</div>
                    <div class="stat-label">Eyes Open</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #f44336;">${closedEyes}</div>
                    <div class="stat-label">Eyes Closed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #2196f3;">${straightPostures}</div>
                    <div class="stat-label">Straight Posture</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #ff9800;">${hunchedPostures}</div>
                    <div class="stat-label">Hunched Posture</div>
                </div>
            </div>`;
            
            document.getElementById('resultContent').innerHTML = html;
        }
        
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 1280, height: 720, facingMode: 'user' } 
                });
                
                cameraStream = stream;
                const video = document.getElementById('videoElement');
                video.srcObject = stream;
                
                document.getElementById('startCamera').disabled = true;
                document.getElementById('stopCamera').disabled = false;
                document.getElementById('cameraStatus').textContent = 'Camera: Active';
                document.getElementById('cameraStatus').className = 'status-indicator status-active';
                
                // Connect to WebSocket
                connectWebSocket();
                
            } catch (err) {
                console.error('Error accessing camera:', err);
                alert('Could not access camera. Please ensure permissions are granted.');
            }
        }
        
        function stopCamera() {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
            }
            
            const video = document.getElementById('videoElement');
            video.srcObject = null;
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            document.getElementById('startCamera').disabled = false;
            document.getElementById('stopCamera').disabled = true;
            document.getElementById('cameraStatus').textContent = 'Camera: Idle';
            document.getElementById('cameraStatus').className = 'status-indicator status-idle';
            document.getElementById('eyeState').textContent = '-';
            document.getElementById('posture').textContent = '-';
        }
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                captureFrames();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateAnnotations(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
            };
        }
        
        function captureFrames() {
            if (!cameraStream) return;
            
            const video = document.getElementById('videoElement');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            function sendFrame() {
                if (cameraStream && ws && ws.readyState === WebSocket.OPEN) {
                    ctx.drawImage(video, 0, 0);
                    const frameData = canvas.toDataURL('image/jpeg', 0.8);
                    ws.send(frameData);
                }
                if (cameraStream) {
                    setTimeout(sendFrame, 100); // ~10 FPS
                }
            }
            
            sendFrame();
        }
        
        function updateAnnotations(data) {
            const eyeState = document.getElementById('eyeState');
            const posture = document.getElementById('posture');
            
            eyeState.textContent = data.eye_state;
            // Apply appropriate CSS class based on eye state
            if (data.eye_state === 'Open') {
                eyeState.className = 'annotation-value eye-open';
            } else if (data.eye_state === 'Closed') {
                eyeState.className = 'annotation-value eye-closed';
            } else {
                // Not Detected - use a neutral/error style
                eyeState.className = 'annotation-value';
                eyeState.style.background = '#ffebee';
                eyeState.style.color = '#c62828';
            }
            
            posture.textContent = data.posture;
            // Apply appropriate CSS class based on posture
            if (data.posture === 'Straight') {
                posture.className = 'annotation-value posture-straight';
            } else if (data.posture === 'Hunched') {
                posture.className = 'annotation-value posture-hunched';
            } else {
                // Not Detected - use a neutral/error style
                posture.className = 'annotation-value';
                posture.style.background = '#fff3e0';
                posture.style.color = '#e65100';
            }
        }
    </script>
</body>
</html>
"""

@app.post("/annotate")
async def annotate_video(
    request: Request,
    video: UploadFile = File(...)
):
    """
    Accepts a video file and returns frame-by-frame analysis of eye state and posture.
    
    Args:
        video: Video file (.mp4 or .avi format)
        download: If True, returns results as downloadable JSON file
    
    Returns:
        JSON with frame-by-frame labels for eye_state and posture
    """
    temp_file_path = None
    json_file_path = None
    try:
        # Get download parameter from query string
        download = request.query_params.get("download", "false").lower() == "true"
        
        # Debug logging
        if video:
            print(f"Received upload - filename: {video.filename}, content_type: {video.content_type}, download={download}")
        else:
            print("ERROR: video parameter is None")
            raise HTTPException(status_code=400, detail="No video file provided")
        
        # Validate file type
        if not video.filename:
            print("ERROR: No filename provided")
            raise HTTPException(
                status_code=400, 
                detail="No filename provided. Please upload a video file."
            )
        
        # Check extension (case-insensitive)
        filename_lower = video.filename.lower()
        print(f"Checking extension for: {filename_lower}")
        if not filename_lower.endswith(('.mp4', '.avi', '.mov')):
            print(f"ERROR: Invalid file extension - {filename_lower}")
            raise HTTPException(
                status_code=400, 
                detail=f"Only .mp4, .avi, or .mov files are supported. Received: {video.filename}"
            )
        
        print(f"File validation passed: {video.filename}")
        
        # Create temporary file to save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_file:
            # Save uploaded video to temporary file
            content = await video.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process video
        start_time = time.time()
        result = process_video(temp_file_path)
        processing_time = time.time() - start_time
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # If download is requested, return as JSON file
        if download:
            # Create temporary JSON file
            json_file_path = tempfile.NamedTemporaryFile(
                mode='w',
                delete=False, 
                suffix='.json',
                prefix=f"{os.path.splitext(video.filename)[0]}_results_"
            ).name
            
            # Write results to JSON file
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            # Async function to clean up the file after download
            async def cleanup_file():
                import time as time_module
                if json_file_path and os.path.exists(json_file_path):
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            os.remove(json_file_path)
                            break
                        except (PermissionError, OSError) as e:
                            if attempt < max_retries - 1:
                                time_module.sleep(0.5)  # Wait 500ms before retry
                            else:
                                print(f"Warning: Could not delete JSON file: {e}")
            
            # Return file for download
            return FileResponse(
                json_file_path,
                media_type='application/json',
                filename=f"{os.path.splitext(video.filename)[0]}_results.json",
                background=cleanup_file
            )
        
        # Return JSON response
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error processing video: {str(e)}"
        print(f"Error: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )
    finally:
        # Clean up temporary files with retry logic
        import time as time_module
        
        def safe_remove(file_path, description="file"):
            if file_path and os.path.exists(file_path):
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        os.remove(file_path)
                        break
                    except (PermissionError, OSError) as e:
                        if attempt < max_retries - 1:
                            time_module.sleep(0.1)  # Wait 100ms before retry
                        else:
                            print(f"Warning: Could not delete {description}: {e}")
        
        safe_remove(temp_file_path, "temporary video file")
        # Only remove JSON file if not being downloaded 
        # (when download=True, cleanup happens in background task after file is served)
        if json_file_path and not download:
            safe_remove(json_file_path, "temporary JSON file")

def process_video(video_path: str) -> dict:
    """
    Process video frame by frame and extract labels.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video annotation results
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_filename = os.path.basename(video_path)
        
        # Calculate video duration
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        print(f"Processing video: {video_filename}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        
        labels_per_frame = {}
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect eye state
            eye_state = eye_detector.detect_eye_state(frame)
            
            # Detect posture
            posture = posture_detector.detect_posture(frame)
            
            # Store results
            labels_per_frame[str(frame_number)] = {
                "eye_state": eye_state,
                "posture": posture
            }
            
            frame_number += 1
            
            # Progress indicator
            if frame_number % 30 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")
        
        result = {
            "video_filename": video_filename,
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": round(duration_seconds, 2),
            "labels_per_frame": labels_per_frame
        }
        
        return result
    finally:
        cap.release()

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing"""
    await websocket.accept()
    active_connections.add(websocket)
    
    print("Client connected")
    
    # Reset detector states for new session (only once per connection)
    eye_detector.previous_state = None
    eye_detector.state_buffer = []
    eye_detector.ear_history.clear()
    posture_detector.previous_posture = None
    posture_detector.posture_buffer = []
    
    try:
        while True:
            # Receive base64 encoded frame
            frame_data = await websocket.receive_text()
            
            # Decode base64 to image with guards
            frame = None
            try:
                if ',' in frame_data:
                    header, encoded = frame_data.split(',', 1)
                    if encoded:
                        image_bytes = base64.b64decode(encoded)
                        if image_bytes:
                            nparr = np.frombuffer(image_bytes, np.uint8)
                            if nparr.size > 0:
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                frame = None
            
            if frame is None:
                # Send fallback rather than skipping endlessly
                await websocket.send_json({"eye_state": "Not Detected", "posture": "Not Detected"})
                continue
            
            # Process frame - maintain state across frames for temporal smoothing
            # Do NOT reset state here - it breaks temporal smoothing and hysteresis
            eye_state = eye_detector.detect_eye_state(frame)
            posture = posture_detector.detect_posture(frame)
            
            # Send results back
            result = {
                "eye_state": eye_state,
                "posture": posture
            }
            
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("Client disconnected")
        active_connections.discard(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        active_connections.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

