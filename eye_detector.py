"""
Eye State Detection Module
Uses MediaPipe facial landmarks to detect if eyes are open or closed.
"""
import os
# Suppress TensorFlow warnings and errors 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from collections import deque

# Import mediapipe before the class definition
# Suppress the audio import error by patching sys.modules
import sys

# Add doc_controls to the mock (needs to work as a decorator)
class DocControls:
    def __call__(self, *args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator
    
    def __getattr__(self, name):
        def do_not_generate_docs(func):
            return func
        return do_not_generate_docs

# Create a proper mock module for tensorflow.tools.docs
class MockTensorFlowDocsModule:
    doc_controls = DocControls()
    
    def __getattr__(self, name):
        return type(self)()  # Return another instance for any attribute

# Mock the problematic tensorflow import
class MockTensorFlowModule:
    _docs_module = MockTensorFlowDocsModule()
    
    @staticmethod
    def __getattr__(name):
        if name == 'tools':
            return MockTensorFlowModule()
        return MockTensorFlowModule()
    
    def __getattribute__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return MockTensorFlowModule()

# Pre-emptively set tensorflow in sys.modules to avoid the DLL error
if 'tensorflow' not in sys.modules:
    sys.modules['tensorflow'] = MockTensorFlowModule()
    sys.modules['tensorflow.tools'] = MockTensorFlowModule()
    sys.modules['tensorflow.tools.docs'] = MockTensorFlowDocsModule()
    
import mediapipe as mp

class EyeStateDetector:
    """
    Detects eye state (Open/Closed) using MediaPipe facial landmarks.
    
    Uses Eye Aspect Ratio (EAR) to determine if eyes are open or closed.
    """
    
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        # Increased accuracy settings
        self.face_detector = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Increased from 0.5 for better accuracy
            min_tracking_confidence=0.7    # Increased from 0.5 for better accuracy
        )
        
        # Eye landmark indices (left and right eyes)
        # Left eye indices
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye indices
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Simplified eye points for EAR calculation
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        
        # Adaptive threshold - more conservative for accuracy
        self.EAR_THRESHOLD = 0.25  # Increased from 0.2 to reduce false positives
        self.EAR_THRESHOLD_CLOSED = 0.15  # More conservative threshold for closed
        
        # Temporal smoothing for more stable predictions
        self.previous_state = None
        self.state_buffer = []
        self.buffer_size = 3  # Consider last 3 frames for stability
        self.window_states = deque(maxlen=10)  # Majority vote over last 10 frames

    def _preprocess(self, frame):
        """Improve contrast using CLAHE on luminance before RGB conversion."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (vertical_dist_1 + vertical_dist_2) / (2 * horizontal_dist)
        """
        # Calculate distances
        # Vertical distances
        vertical_dist_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_dist_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        horizontal_dist = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if horizontal_dist == 0:
            return 0.0
        
        # Calculate EAR
        ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
        return ear
    
    def detect_eye_state(self, frame):
        """
        Detect if eyes are open or closed in the given frame.
        Uses temporal smoothing for more stable and accurate results.
        
        Args:
            frame: BGR image frame
            
        Returns:
            "Open" or "Closed"
        """
        # Preprocess and convert BGR to RGB for MediaPipe
        frame_pp = self._preprocess(frame)
        rgb_frame = cv2.cvtColor(frame_pp, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detector.process(rgb_frame)
        
        current_state = None
        confidence = 0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get eye landmarks
            h, w = frame.shape[:2]
            
            # Left eye points (6 points: outer, inner, top, bottom eyes)
            left_eye_points = []
            for idx in self.LEFT_EYE_POINTS:
                landmark = face_landmarks.landmark[idx]
                x = landmark.x * w
                y = landmark.y * h
                left_eye_points.append(np.array([x, y]))
            
            # Right eye points
            right_eye_points = []
            for idx in self.RIGHT_EYE_POINTS:
                landmark = face_landmarks.landmark[idx]
                x = landmark.x * w
                y = landmark.y * h
                right_eye_points.append(np.array([x, y]))
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye_points)
            right_ear = self.calculate_ear(right_eye_points)
            
            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Determine eye state with confidence
            # Use more conservative thresholds for accuracy
            if avg_ear < self.EAR_THRESHOLD_CLOSED:
                current_state = "Closed"
                confidence = 1.0 - (avg_ear / self.EAR_THRESHOLD_CLOSED)  # Normalized confidence
            elif avg_ear > self.EAR_THRESHOLD:
                current_state = "Open"
                confidence = 1.0
            else:
                # In the ambiguous zone - use previous state if available
                if self.previous_state:
                    current_state = self.previous_state
                    confidence = 0.7  # Lower confidence for ambiguous cases
                else:
                    # Default to "Open" for conservative approach
                    current_state = "Open"
                    confidence = 0.6
        
        # If no face detected, report explicitly
        if current_state is None:
            return "Not Detected"
        
        # Temporal smoothing: maintain state if consistent (short buffer)
        if self.previous_state is not None:
            # If state changed, require confirmation over multiple frames
            if current_state != self.previous_state:
                self.state_buffer.append(current_state)
                
                # Require at least 2 out of 3 frames to agree on new state
                if len(self.state_buffer) >= self.buffer_size:
                    # Count occurrences in buffer
                    from collections import Counter
                    state_counts = Counter(self.state_buffer)
                    most_common_state = state_counts.most_common(1)[0][0]
                    
                    # Only change state if consistent (at least 2/3 agree)
                    if state_counts[most_common_state] >= 2:
                        current_state = most_common_state
                        self.previous_state = current_state
                        self.state_buffer = [current_state]
                    else:
                        # Keep previous state
                        current_state = self.previous_state
                        self.state_buffer = []
            else:
                # State hasn't changed, clear buffer
                self.state_buffer = []
                self.previous_state = current_state
        else:
            self.previous_state = current_state
        
        # Majority vote over last 10 frames to remove noise/blinks
        self.window_states.append(current_state)
        if len(self.window_states) > 0:
            # Pick most frequent state in window
            final_state = max(set(self.window_states), key=self.window_states.count)
            return final_state
        return current_state

