"""
Posture Detection Module
Uses MediaPipe pose estimation to detect if person is sitting straight or hunched.
"""
import os
# Suppress TensorFlow warnings and errors 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

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

class PostureDetector:
    """
    Detects posture (Straight/Hunched) using MediaPipe pose estimation.
    
    Analyzes the angle of the spine/back to determine posture.
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Increased confidence for more accuracy
        self.pose_detector = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # Increased from 0.5
            min_tracking_confidence=0.7     # Increased from 0.5
        )
        
        
        self.ANGLE_HUNCHED = 32.0   
        self.ANGLE_STRAIGHT = 25.0  
        # Horizontal ratio thresholds (fraction of frame width) - extremely conservative
        self.RATIO_HUNCHED_HIP = 0.15     
        self.RATIO_STRAIGHT_HIP = 0.10    
        self.RATIO_HUNCHED_FALLBACK = 0.18  
        self.RATIO_STRAIGHT_FALLBACK = 0.15 
        
        # For temporal smoothing
        self.previous_posture = None
        self.posture_buffer = []
        self.buffer_size = 5  # keep last 5 votes
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle at point2 formed by point1-point2-point3.
        
        Returns angle in degrees.
        """
        # Convert to numpy arrays
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        # Calculate vectors
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        # Normalize angle to 0-180 degrees
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def detect_posture(self, frame):
        """
        Detect if person is sitting straight or hunched.
        
        Args:
            frame: BGR image frame
            
        Returns:
            "Straight" or "Hunched"
        """
        # Preprocess: improve contrast using CLAHE on luminance
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        frame_pp = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_pp, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = self.pose_detector.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Landmarks
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            l_sh = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Require shoulders and nose; hips may be missing in tight framing
            if any(p.visibility < 0.7 for p in [nose, l_sh, r_sh]):
                return "Not Detected"
            
            # Get key body points
            # We'll use shoulder, hip, and knee landmarks to estimate spine angle
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Calculate midpoints
            shoulder_mid = np.array([
                (left_shoulder.x + right_shoulder.x) / 2 * w,
                (left_shoulder.y + right_shoulder.y) / 2 * h
            ])
            
            hips_visible = (l_hip.visibility >= 0.7 and r_hip.visibility >= 0.7)
            if hips_visible:
                hip_mid = np.array([
                    (left_hip.x + right_hip.x) / 2 * w,
                    (left_hip.y + right_hip.y) / 2 * h
                ])
            
            # Create reference point above shoulders (for angle calculation)
            ref_point = np.array([shoulder_mid[0], shoulder_mid[1] - 100])  # 100 pixels above shoulders
            
            if hips_visible:
                # Shoulder->hip vector
                vec = hip_mid - shoulder_mid
            else:
                # Fallback: shoulder->nose vector (neck orientation)
                vec = np.array([nose.x * w, nose.y * h]) - shoulder_mid
            # Angle w.r.t vertical axis (y): larger angle = more forward lean
            angle = np.degrees(np.arctan2(abs(vec[0]), max(abs(vec[1]), 1e-6)))
            
            # If spine is significantly angled (hunched forward), angle will be > threshold
            # A straight posture should have angle close to 0 or 180 degrees
            
            # Horizontal offset (normalized): hips if visible, else nose
            if hips_visible:
                horizontal_ratio = abs(hip_mid[0] - shoulder_mid[0]) / max(w, 1)
            else:
                horizontal_ratio = abs((nose.x * w) - shoulder_mid[0]) / max(w, 1)
            
            # Also check vertical position - shoulders forward = hunched
            if hips_visible:
                vertical_diff = shoulder_mid[1] - hip_mid[1]  # Negative if shoulders forward
                ratio_hunched = self.RATIO_HUNCHED_HIP
                ratio_straight = self.RATIO_STRAIGHT_HIP
            else:
                vertical_diff = 0  # No reliable hip reference; ignore this term
                ratio_hunched = self.RATIO_HUNCHED_FALLBACK
                ratio_straight = self.RATIO_STRAIGHT_FALLBACK

            # Hysteresis-aware decision
            if self.previous_posture == "Hunched":
                # stay hunched unless clearly straight
                is_hunched = not (angle < self.ANGLE_STRAIGHT and horizontal_ratio < ratio_straight)
            else:
                # switch to hunched when exceeding hunched thresholds
                is_hunched = (angle > self.ANGLE_HUNCHED) or (horizontal_ratio > ratio_hunched)
            
            current_posture = "Hunched" if is_hunched else "Straight"
            
        else:
            # No pose detected
            return "Not Detected"
        
        # Temporal smoothing: majority vote over last 5 labels
        self.posture_buffer.append(current_posture)
        if len(self.posture_buffer) > self.buffer_size:
            self.posture_buffer = self.posture_buffer[-self.buffer_size:]
        from collections import Counter
        counts = Counter(self.posture_buffer)
        voted = counts.most_common(1)[0][0]
        self.previous_posture = voted
        current_posture = voted
        
        return current_posture

