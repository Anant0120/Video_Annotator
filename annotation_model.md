# AI Models for Automated Video-Based Eye-State and Posture Annotation

## 1. Introduction

This document provides a comprehensive list of AI models suitable for analyzing a video frame-by-frame and determining two key attributes for a person visible in the video:

1. **Eye State:** Whether the person’s eyes are open or closed.  
2. **Posture:** Whether the person’s posture is straight or hunched.

The listed models include both lightweight, open-source libraries and advanced deep learning architectures. Each model is described in terms of how it can be applied to the problem, its performance characteristics, accuracy, computational requirements, and cost.

---

## 2. Eye-State Detection Models

| No. | Model Name | Mechanism | Accuracy (Approx.) | Speed | Cost | Description |
|-----|-------------|------------|--------------------|--------|------|-------------|
| 1 | **MediaPipe Face Mesh (Google)** | Detects 468 facial landmarks per frame. Eye Aspect Ratio (EAR) can be computed from landmarks to classify eye openness. | 96–99% | Real-time on CPU | Free | Ideal for video-based tasks. Tracks facial landmarks continuously and efficiently. |
| 2 | **OpenFace 2.0** | Uses Dlib-based landmark detection and a CNN for analyzing the eye region. | 94–97% | Medium (CPU OK) | Free | Classical but reliable model for eye state analysis; performs well in controlled conditions. |
| 3 | **MobileNetV2 / MobileViT**<br>*(Hugging Face: `MichalMlodawski/open-closed-eye-classification-mobilev2`)* | Lightweight CNN trained on cropped eye regions for binary open/closed classification. | 97–99% | Fast (CPU/GPU) | Free | Requires cropping eyes using landmarks from another model. High accuracy, efficient, and easy to integrate. |
| 4 | **EyeBlink (PyTorch)** | Temporal CNN trained on short clips (3–5 frames) to detect blinks. | 96–98% | Slow (GPU) | Free | Focused on blink detection but can infer eye openness over time. Useful for temporal stability. |
| 5 | **DeepBlink (Keras)** | Uses CNN + RNN sequence model to estimate eye openness probability over short frame sequences. | 97% | Medium (GPU) | Free | Captures short-term temporal dependencies for smoother predictions. |
| 6 | **YOLOv8-Pose + Custom Head** | Detects eyes and uses an auxiliary classifier to label open vs. closed. | 95–97% | Fast (GPU) | Free | High-speed model suitable for diverse head positions and lighting. |
| 7 | **OpenVINO Gaze Estimation Model** | Intel-optimized gaze estimation model that infers gaze direction and eye openness. | 95% | Fast (Intel CPU) | Free | Optimized for real-time inference on CPU devices such as laptops. |
| 8 | **CLIP / SAM-based Vision-Language Models** | Uses text prompts (e.g., “eyes open or closed?”) applied to frames. | 92–95% | Slow (GPU) | Paid (API) | Useful for experimentation or cloud inference but inefficient for per-frame classification. |

---

## 3. Posture Classification Models

| No. | Model Name | Mechanism | Accuracy (Approx.) | Speed | Cost | Description |
|-----|-------------|------------|--------------------|--------|------|-------------|
| 1 | **MediaPipe Pose (BlazePose)** | Detects 33 body landmarks per frame. Posture can be inferred from geometric angles between shoulders, hips, and neck. | 93–96% | Real-time on CPU | Free | Lightweight and well-suited for continuous video analysis. Provides stable pose tracking. |
| 2 | **OpenPose (CMU)** | CNN-based multi-person pose estimator producing 25+ body keypoints. | 94–97% | Slow (GPU required) | Free | Highly accurate but computationally heavy; suited for multi-person or high-resolution videos. |
| 3 | **YOLOv8-Pose** | Combines YOLO object detection with pose estimation head. Keypoints can be used to compute body alignment. | 93–95% | Fast (GPU) | Free | Modern and efficient for single-person posture analysis. |
| 4 | **HRNet (High-Resolution Net)** | Deep CNN preserving high-resolution features throughout to detect precise body keypoints. | 96–98% | Slow (GPU required) | Free | Excellent for research-grade accuracy; more resource-intensive. |
| 5 | **MoveNet (Google TensorFlow)** | Lightweight and fast pose estimation model. Supports single- and multi-person modes. | 92–95% | Very fast | Free | Designed for real-time pose detection in video streams; minimal latency. |
| 6 | **TimeSformer (Video Transformer)** | Processes short frame sequences to model temporal and spatial motion features. | 97–98% | Slow (GPU required) | Paid (training or hosted use) | Strong temporal modeling for video clips; not optimal for frame-by-frame inference. |
| 7 | **SlowFast (Meta)** | Dual-pathway CNN for spatial-temporal video understanding; excels at motion-based posture recognition. | 97–99% | GPU-intensive | Paid (compute cost) | High accuracy for dynamic video tasks; computationally expensive. |
| 8 | **ViViT (Google)** | Vision Transformer trained on large-scale video datasets to interpret body motion and spatial cues. | 97% | GPU-intensive | Paid | Very accurate but heavy model not suited for lightweight APIs. |
| 9 | **OpenVINO Human Pose Estimation** | Intel-optimized pose model for skeleton keypoints extraction. | 94% | Fast (CPU) | Free | Efficient for CPU-only deployment. |
| 10 | **DeepPostureNet** | CNN classifier trained on RGB posture data. | 92–94% | Medium | Free | Can be fine-tuned for specific ergonomic or frontal camera postures. |

---

## 4. Recommended Combinations for Video Analysis

| Component | Recommended Model(s) | Rationale |
|------------|----------------------|------------|
| Eye Detection and Tracking | MediaPipe Face Mesh | Provides consistent facial landmarks across frames; no GPU required. |
| Eye-State Classification | EAR (Eye Aspect Ratio) or MobileViT | High accuracy, lightweight, and interpretable. |
| Body Pose Detection | MediaPipe Pose | Efficient landmark detection suitable for continuous video. |
| Posture Classification | Shoulder–Hip or Neck–Spine Angle Rule | Simple geometric rule-based posture classification; explainable and reliable. |
| Temporal Smoothing | Rolling Median or Moving Average | Reduces label flicker across consecutive frames. |

---

## 5. Model Cost Summary

| Model | Cost Type | Notes |
|--------|------------|-------|
| MediaPipe (Face Mesh, Pose) | Free | Local processing, no license cost. |
| MobileViT / MobileNetV2 | Free | Available on Hugging Face, lightweight deployment. |
| MoveNet | Free | TensorFlow-based and efficient. |
| OpenPose | Free | GPU required for efficient inference. |
| HRNet, SlowFast, ViViT, TimeSformer | Paid (compute cost) | Require GPUs or TPUs for efficient performance. |
| OpenVINO Models | Free | Optimized for Intel hardware. |
| Cloud APIs (AWS Rekognition, Azure Video Indexer, Google Cloud Video AI) | Paid (usage-based) | Useful for scalable or production-grade pipelines. |

---

## 6. Implementation Overview

A typical pipeline for frame-by-frame video analysis can be structured as follows:

1. **Video Input:** Accept `.mp4` or `.avi` file.  
2. **Frame Extraction:** Use OpenCV to extract frames sequentially.  
3. **Facial Landmark Detection:** Apply MediaPipe Face Mesh per frame.  
4. **Eye-State Estimation:**  
   - Compute Eye Aspect Ratio (EAR) using eye landmarks.  
   - Optionally, use a CNN (MobileViT) for cropped eye regions.  
5. **Body Landmark Detection:** Apply MediaPipe Pose to each frame.  
6. **Posture Estimation:**  
   - Compute the shoulder–hip angle.  
   - Classify as “Straight” or “Hunched” based on threshold.  
7. **Temporal Smoothing:** Use moving averages or median filters to stabilize predictions.  
8. **Output:** Generate JSON structure:

```json
{
  "video_filename": "sample_video.mp4",
  "total_frames": 240,
  "labels_per_frame": {
    "0": {"eye_state": "Open", "posture": "Straight"},
    "1": {"eye_state": "Closed", "posture": "Hunched"}
  }
}
