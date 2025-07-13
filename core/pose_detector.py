"""
Pose Detection Module for AI Pushups Coach v2
"""

import logging
import mediapipe as mp
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class PoseDetector:
    """Handles pose detection using MediaPipe"""
    def __init__(self):
        self.logger = logger
        self.mp_pose = mp.solutions.pose
        # Use exact same settings as preintegration code
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=1, 
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def detect_pose(self, frame):
        """Detect pose landmarks in a frame using MediaPipe"""
        try:
            self.logger.debug("Detecting pose in frame (MediaPipe)")
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            if not results.pose_landmarks:
                return None, None
            h, w, _ = frame.shape
            # Return landmarks as a list of 33 items
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': lm.x * w,
                    'y': lm.y * h,
                    'z': lm.z * w,  # z is relative to width
                    'visibility': lm.visibility
                })
            return landmarks, results.pose_landmarks
        except Exception as e:
            self.logger.error(f"Pose detection error: {str(e)}")
            return None, None 