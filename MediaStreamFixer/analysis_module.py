import cv2
import mediapipe as mp
import numpy as np
import json
import os
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Configure logging
logger = logging.getLogger(__name__)

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global pose instance to reuse across functions
pose_instance = None

def get_pose_instance():
    """Get a reusable pose instance."""
    global pose_instance
    if pose_instance is None:
        pose_instance = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return pose_instance

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle between three points."""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    except Exception as e:
        logger.warning(f"Error calculating angle: {e}")
        return 0.0

def get_landmark_coordinates(landmarks, landmark_index: int) -> Tuple[Optional[float], Optional[float]]:
    """Get normalized coordinates for a specific landmark."""
    try:
        if landmarks and len(landmarks.landmark) > landmark_index:
            landmark = landmarks.landmark[landmark_index]
            if landmark.visibility > 0.3:  # Visibility threshold
                return landmark.x, landmark.y
        return None, None
    except Exception as e:
        logger.warning(f"Error getting landmark coordinates: {e}")
        return None, None

def validate_video_file(video_path: str) -> bool:
    """Validate that the video file can be opened and processed."""
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            logger.error(f"Video file is empty: {video_path}")
            return False
        
        logger.info(f"Video file size: {file_size} bytes")
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file with OpenCV: {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logger.error(f"Cannot read frames from video: {video_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating video file: {e}")
        return False

def process_video_frames(video_path: str, frame_processor, max_duration_seconds: int = 60):
    """Process video frames with a frame processor function."""
    cap = cv2.VideoCapture(video_path)
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit processing time
        max_frames = min(frame_count, int(fps * max_duration_seconds))
        
        # Calculate frame skip for performance
        skip_frames = max(1, frame_count // 1800)  # Process at most 1800 frames
        
        logger.info(f"Processing {max_frames} frames with skip={skip_frames}")
        
        pose = get_pose_instance()
        frame_number = 0
        processed_frames = 0
        
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if frame_number % skip_frames != 0:
                frame_number += 1
                continue
            
            try:
                # Resize frame for faster processing
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(frame_rgb)
                
                # Call the frame processor
                frame_processor(frame_number, pose_results, fps)
                
                processed_frames += 1
                
                # Log progress
                if processed_frames % 100 == 0:
                    logger.debug(f"Processed {processed_frames} frames")
                    
            except Exception as e:
                logger.warning(f"Error processing frame {frame_number}: {e}")
            
            frame_number += 1
            
    finally:
        cap.release()

def analyze_pushups(video_path: str) -> Dict:
    """Analyze pushup exercise from video."""
    logger.info(f"Starting pushup analysis for: {video_path}")
    
    if not validate_video_file(video_path):
        raise Exception("Invalid video file or cannot read video")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    results = {
        "drill_type": "pushups",
        "total_reps": 0,
        "reps": [],
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'in_pushup': False,
        'current_rep_start': None,
        'elbow_angles': [],
        'consecutive_up_frames': 0,
        'consecutive_down_frames': 0
    }
    
    def process_frame(frame_number, pose_results, fps):
        if pose_results.pose_landmarks:
            # Get key points for pushup analysis
            left_shoulder = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            left_elbow = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
            left_wrist = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
            
            right_shoulder = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
            right_elbow = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
            right_wrist = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
            
            # Check if we have at least one good arm
            left_valid = all(coord[0] is not None for coord in [left_shoulder, left_elbow, left_wrist])
            right_valid = all(coord[0] is not None for coord in [right_shoulder, right_elbow, right_wrist])
            
            if left_valid or right_valid:
                # Calculate elbow angles
                elbow_angles_frame = []
                if left_valid:
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    elbow_angles_frame.append(left_elbow_angle)
                if right_valid:
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    elbow_angles_frame.append(right_elbow_angle)
                
                avg_elbow_angle = sum(elbow_angles_frame) / len(elbow_angles_frame)
                
                # Calculate body angle (torso alignment)
                left_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
                body_angle = 180  # Default straight
                if left_hip[0] is not None and left_shoulder[0] is not None:
                    body_angle = calculate_angle(left_shoulder, left_hip, (left_hip[0], left_hip[1] - 0.1))
                
                state['elbow_angles'].append(avg_elbow_angle)
                
                # Detect pushup phases with hysteresis
                if avg_elbow_angle < 110:  # Down position
                    state['consecutive_down_frames'] += 1
                    state['consecutive_up_frames'] = 0
                    
                    if state['consecutive_down_frames'] >= 2 and not state['in_pushup']:
                        state['in_pushup'] = True
                        state['current_rep_start'] = frame_number
                        logger.debug(f"Pushup started at frame {frame_number}")
                        
                elif avg_elbow_angle > 150:  # Up position
                    state['consecutive_up_frames'] += 1
                    state['consecutive_down_frames'] = 0
                    
                    if state['consecutive_up_frames'] >= 2 and state['in_pushup']:
                        state['in_pushup'] = False
                        if state['current_rep_start'] is not None:
                            # Calculate rep metrics
                            rep_frames = state['elbow_angles'][-30:] if len(state['elbow_angles']) >= 30 else state['elbow_angles']
                            min_elbow_angle = min(rep_frames) if rep_frames else avg_elbow_angle
                            max_elbow_angle = max(rep_frames) if rep_frames else avg_elbow_angle
                            
                            rep_data = {
                                "rep_number": len(results["reps"]) + 1,
                                "start_frame": state['current_rep_start'],
                                "end_frame": frame_number,
                                "max_elbow_angle": round(max_elbow_angle, 1),
                                "min_elbow_angle": round(min_elbow_angle, 1),
                                "body_angle_at_bottom": round(body_angle, 1)
                            }
                            results["reps"].append(rep_data)
                            logger.debug(f"Pushup completed: rep {rep_data['rep_number']}")
                            state['current_rep_start'] = None
                            state['elbow_angles'] = []
    
    try:
        process_video_frames(video_path, process_frame)
        results["total_reps"] = len(results["reps"])
        
        logger.info(f"Pushup analysis complete. Found {results['total_reps']} reps")
        return results
        
    except Exception as e:
        logger.error(f"Error in pushup analysis: {e}")
        raise

def analyze_squats(video_path: str) -> Dict:
    """Analyze squat exercise from video."""
    logger.info(f"Starting squat analysis for: {video_path}")
    
    if not validate_video_file(video_path):
        raise Exception("Invalid video file or cannot read video")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    results = {
        "drill_type": "squats",
        "total_reps": 0,
        "reps": [],
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'in_squat': False,
        'current_rep_start': None,
        'knee_angles': [],
        'consecutive_up_frames': 0,
        'consecutive_down_frames': 0
    }
    
    def process_frame(frame_number, pose_results, fps):
        if pose_results.pose_landmarks:
            # Get key points for squat analysis
            left_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            left_knee = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
            left_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
            
            right_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
            right_knee = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
            right_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
            
            # Check if we have at least one good leg
            left_valid = all(coord[0] is not None for coord in [left_hip, left_knee, left_ankle])
            right_valid = all(coord[0] is not None for coord in [right_hip, right_knee, right_ankle])
            
            if left_valid or right_valid:
                # Calculate knee angles
                knee_angles_frame = []
                if left_valid:
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    knee_angles_frame.append(left_knee_angle)
                if right_valid:
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    knee_angles_frame.append(right_knee_angle)
                
                avg_knee_angle = sum(knee_angles_frame) / len(knee_angles_frame)
                
                # Calculate torso angle
                left_shoulder = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
                torso_angle = 180  # Default straight
                if left_shoulder[0] is not None and left_hip[0] is not None:
                    torso_angle = calculate_angle((left_hip[0], left_hip[1] + 0.1), left_hip, left_shoulder)
                
                state['knee_angles'].append(avg_knee_angle)
                
                # Detect squat phases with hysteresis
                if avg_knee_angle < 130:  # Down position
                    state['consecutive_down_frames'] += 1
                    state['consecutive_up_frames'] = 0
                    
                    if state['consecutive_down_frames'] >= 2 and not state['in_squat']:
                        state['in_squat'] = True
                        state['current_rep_start'] = frame_number
                        logger.debug(f"Squat started at frame {frame_number}")
                        
                elif avg_knee_angle > 160:  # Up position
                    state['consecutive_up_frames'] += 1
                    state['consecutive_down_frames'] = 0
                    
                    if state['consecutive_up_frames'] >= 2 and state['in_squat']:
                        state['in_squat'] = False
                        if state['current_rep_start'] is not None:
                            # Calculate rep metrics
                            rep_frames = state['knee_angles'][-30:] if len(state['knee_angles']) >= 30 else state['knee_angles']
                            min_knee_angle = min(rep_frames) if rep_frames else avg_knee_angle
                            
                            rep_data = {
                                "rep_number": len(results["reps"]) + 1,
                                "start_frame": state['current_rep_start'],
                                "end_frame": frame_number,
                                "min_knee_angle": round(min_knee_angle, 1),
                                "torso_angle_at_bottom": round(torso_angle, 1)
                            }
                            results["reps"].append(rep_data)
                            logger.debug(f"Squat completed: rep {rep_data['rep_number']}")
                            state['current_rep_start'] = None
                            state['knee_angles'] = []
    
    try:
        process_video_frames(video_path, process_frame)
        results["total_reps"] = len(results["reps"])
        
        logger.info(f"Squat analysis complete. Found {results['total_reps']} reps")
        return results
        
    except Exception as e:
        logger.error(f"Error in squat analysis: {e}")
        raise

def analyze_situps(video_path: str) -> Dict:
    """Analyze situp exercise from video."""
    logger.info(f"Starting situp analysis for: {video_path}")
    
    if not validate_video_file(video_path):
        raise Exception("Invalid video file or cannot read video")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    results = {
        "drill_type": "situps",
        "total_reps": 0,
        "reps": [],
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'in_situp': False,
        'current_rep_start': None,
        'consecutive_up_frames': 0,
        'consecutive_down_frames': 0
    }
    
    def process_frame(frame_number, pose_results, fps):
        if pose_results.pose_landmarks:
            # Get key points for situp analysis
            left_shoulder = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            left_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            left_knee = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
            
            # Only proceed if we have valid landmarks
            if all(coord[0] is not None for coord in [left_shoulder, left_hip, left_knee]):
                # Calculate hip angle
                hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                
                # Detect situp phases with hysteresis
                if hip_angle < 70:  # Up position
                    state['consecutive_up_frames'] += 1
                    state['consecutive_down_frames'] = 0
                    
                    if state['consecutive_up_frames'] >= 2 and not state['in_situp']:
                        state['in_situp'] = True
                        state['current_rep_start'] = frame_number
                        logger.debug(f"Situp started at frame {frame_number}")
                        
                elif hip_angle > 100:  # Down position
                    state['consecutive_down_frames'] += 1
                    state['consecutive_up_frames'] = 0
                    
                    if state['consecutive_down_frames'] >= 2 and state['in_situp']:
                        state['in_situp'] = False
                        if state['current_rep_start'] is not None:
                            rep_data = {
                                "rep_number": len(results["reps"]) + 1,
                                "start_frame": state['current_rep_start'],
                                "end_frame": frame_number,
                                "hip_angle_top": round(hip_angle, 1),
                                "hip_angle_bottom": round(hip_angle, 1)
                            }
                            results["reps"].append(rep_data)
                            logger.debug(f"Situp completed: rep {rep_data['rep_number']}")
                            state['current_rep_start'] = None
    
    try:
        process_video_frames(video_path, process_frame)
        results["total_reps"] = len(results["reps"])
        
        logger.info(f"Situp analysis complete. Found {results['total_reps']} reps")
        return results
        
    except Exception as e:
        logger.error(f"Error in situp analysis: {e}")
        raise

def analyze_chair_hold(video_path: str) -> Dict:
    """Analyze chair hold exercise from video."""
    logger.info(f"Starting chair hold analysis for: {video_path}")
    
    if not validate_video_file(video_path):
        raise Exception("Invalid video file or cannot read video")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    results = {
        "drill_type": "chair_hold",
        "total_hold_time": 0,
        "time_series_data": [],
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'hold_start_time': None,
        'consecutive_hold_frames': 0,
        'last_data_timestamp': 0
    }
    
    def process_frame(frame_number, pose_results, fps):
        if pose_results.pose_landmarks:
            # Get key points for chair hold analysis
            left_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            left_knee = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
            left_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
            
            right_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
            right_knee = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
            right_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
            
            # Check if we have at least one good leg
            left_valid = all(coord[0] is not None for coord in [left_hip, left_knee, left_ankle])
            right_valid = all(coord[0] is not None for coord in [right_hip, right_knee, right_ankle])
            
            if left_valid or right_valid:
                # Calculate angles
                knee_angles = []
                hip_angles = []
                
                if left_valid:
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    knee_angles.append(left_knee_angle)
                    
                    left_shoulder = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
                    if left_shoulder[0] is not None:
                        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                        hip_angles.append(left_hip_angle)
                
                if right_valid:
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    knee_angles.append(right_knee_angle)
                    
                    right_shoulder = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                    if right_shoulder[0] is not None:
                        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                        hip_angles.append(right_hip_angle)
                
                avg_knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else 90
                avg_hip_angle = sum(hip_angles) / len(hip_angles) if hip_angles else 90
                
                timestamp = frame_number / fps if fps > 0 else frame_number
                
                # Check if in proper chair hold position
                if 70 <= avg_knee_angle <= 120 and 70 <= avg_hip_angle <= 120:
                    state['consecutive_hold_frames'] += 1
                    
                    if state['consecutive_hold_frames'] >= 3:  # Must hold for at least 3 frames
                        if state['hold_start_time'] is None:
                            state['hold_start_time'] = timestamp
                        
                        # Sample data points (not every frame)
                        if timestamp - state['last_data_timestamp'] >= 1.0:
                            time_data = {
                                "timestamp": round(timestamp, 2),
                                "knee_angle": round(avg_knee_angle, 1),
                                "hip_angle": round(avg_hip_angle, 1)
                            }
                            results["time_series_data"].append(time_data)
                            state['last_data_timestamp'] = timestamp
                else:
                    state['consecutive_hold_frames'] = 0
                    if state['hold_start_time'] is not None:
                        results["total_hold_time"] += timestamp - state['hold_start_time']
                        state['hold_start_time'] = None
    
    try:
        process_video_frames(video_path, process_frame)
        
        # Add any remaining hold time
        if state['hold_start_time'] is not None:
            final_timestamp = results["video_info"]["duration"]
            results["total_hold_time"] += final_timestamp - state['hold_start_time']
        
        results["total_hold_time"] = round(results["total_hold_time"], 2)
        
        logger.info(f"Chair hold analysis complete. Total hold time: {results['total_hold_time']:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"Error in chair hold analysis: {e}")
        raise

def analyze_elbow_plank(video_path: str) -> Dict:
    """Analyze elbow plank exercise from video."""
    logger.info(f"Starting elbow plank analysis for: {video_path}")
    
    if not validate_video_file(video_path):
        raise Exception("Invalid video file or cannot read video")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    results = {
        "drill_type": "elbow_plank",
        "total_hold_time": 0,
        "time_series_data": [],
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'hold_start_time': None,
        'consecutive_hold_frames': 0,
        'last_data_timestamp': 0
    }
    
    def process_frame(frame_number, pose_results, fps):
        if pose_results.pose_landmarks:
            # Get key points for plank analysis
            left_shoulder = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            left_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            left_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
            
            # Only proceed if we have valid landmarks
            if all(coord[0] is not None for coord in [left_shoulder, left_hip, left_ankle]):
                # Calculate body angle (straight line from shoulder to ankle)
                body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
                
                timestamp = frame_number / fps if fps > 0 else frame_number
                
                # Check if in proper plank position (body should be relatively straight)
                if 160 <= body_angle <= 200:  # Allow some flexibility for proper plank
                    state['consecutive_hold_frames'] += 1
                    
                    if state['consecutive_hold_frames'] >= 3:  # Must hold for at least 3 frames
                        if state['hold_start_time'] is None:
                            state['hold_start_time'] = timestamp
                        
                        # Sample data points
                        if timestamp - state['last_data_timestamp'] >= 1.0:
                            time_data = {
                                "timestamp": round(timestamp, 2),
                                "body_angle": round(body_angle, 1)
                            }
                            results["time_series_data"].append(time_data)
                            state['last_data_timestamp'] = timestamp
                else:
                    state['consecutive_hold_frames'] = 0
                    if state['hold_start_time'] is not None:
                        results["total_hold_time"] += timestamp - state['hold_start_time']
                        state['hold_start_time'] = None
    
    try:
        process_video_frames(video_path, process_frame)
        
        # Add any remaining hold time
        if state['hold_start_time'] is not None:
            final_timestamp = results["video_info"]["duration"]
            results["total_hold_time"] += final_timestamp - state['hold_start_time']
        
        results["total_hold_time"] = round(results["total_hold_time"], 2)
        
        logger.info(f"Elbow plank analysis complete. Total hold time: {results['total_hold_time']:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"Error in elbow plank analysis: {e}")
        raise

def analyze_single_leg_balance(video_path: str, leg_side: str) -> Dict:
    """Analyze single leg balance exercise from video."""
    logger.info(f"Starting single leg balance analysis for: {video_path}, leg: {leg_side}")
    
    if not validate_video_file(video_path):
        raise Exception("Invalid video file or cannot read video")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    results = {
        "drill_type": "single_leg_balance",
        "leg_side": leg_side,
        "total_balance_time": 0,
        "total_fouls": 0,
        "foul_data": [],
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'balance_start_time': None,
        'last_foul_time': None,
        'foul_cooldown': 2.0,
        'consecutive_balance_frames': 0,
        'consecutive_foul_frames': 0
    }
    
    def process_frame(frame_number, pose_results, fps):
        if pose_results.pose_landmarks:
            # Get ankle positions
            left_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
            right_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
            
            # Only proceed if we have valid landmarks
            if left_ankle[0] is not None and right_ankle[0] is not None:
                timestamp = frame_number / fps if fps > 0 else frame_number
                
                # Determine if balancing on correct leg
                if leg_side == 'left':
                    standing_ankle = left_ankle
                    lifted_ankle = right_ankle
                else:
                    standing_ankle = right_ankle
                    lifted_ankle = left_ankle
                
                # Check if lifted foot is sufficiently raised (simple heuristic)
                foot_separation = abs(lifted_ankle[1] - standing_ankle[1])
                is_balanced = foot_separation > 0.05  # Threshold for detection
                
                if is_balanced:
                    state['consecutive_balance_frames'] += 1
                    state['consecutive_foul_frames'] = 0
                    
                    if state['consecutive_balance_frames'] >= 3:  # Must balance for at least 3 frames
                        if state['balance_start_time'] is None:
                            state['balance_start_time'] = timestamp
                else:
                    state['consecutive_foul_frames'] += 1
                    state['consecutive_balance_frames'] = 0
                    
                    # Foul detected
                    if state['balance_start_time'] is not None:
                        results["total_balance_time"] += timestamp - state['balance_start_time']
                        state['balance_start_time'] = None
                    
                    # Record foul if cooldown has passed and consistent foul detected
                    if (state['consecutive_foul_frames'] >= 2 and 
                        (state['last_foul_time'] is None or (timestamp - state['last_foul_time']) > state['foul_cooldown'])):
                        foul_data = {
                            "foul_number": len(results["foul_data"]) + 1,
                            "timestamp": round(timestamp, 2),
                            "frame_number": frame_number
                        }
                        results["foul_data"].append(foul_data)
                        state['last_foul_time'] = timestamp
                        logger.debug(f"Foul detected at frame {frame_number}")
    
    try:
        process_video_frames(video_path, process_frame)
        
        # Add any remaining balance time
        if state['balance_start_time'] is not None:
            final_timestamp = results["video_info"]["duration"]
            results["total_balance_time"] += final_timestamp - state['balance_start_time']
        
        results["total_fouls"] = len(results["foul_data"])
        results["total_balance_time"] = round(results["total_balance_time"], 2)
        
        logger.info(f"Single leg balance analysis complete. Balance time: {results['total_balance_time']:.2f}s, Fouls: {results['total_fouls']}")
        return results
        
    except Exception as e:
        logger.error(f"Error in single leg balance analysis: {e}")
        raise

def analyze_video(video_path: str, drill_type: str) -> Dict:
    """Main router function to call the correct analysis function."""
    logger.info(f"Analyzing video: {video_path} for drill type: {drill_type}")
    
    # Validate video file exists
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    
    # Validate video file is readable
    if not validate_video_file(video_path):
        raise Exception("Cannot read video file - it may be corrupted or in an unsupported format")
    
    try:
        start_time = time.time()
        
        # Route to appropriate analysis function
        if drill_type == 'pushups':
            result = analyze_pushups(video_path)
        elif drill_type == 'squats':
            result = analyze_squats(video_path)
        elif drill_type == 'situps':
            result = analyze_situps(video_path)
        elif drill_type == 'chair_hold':
            result = analyze_chair_hold(video_path)
        elif drill_type == 'elbow_plank':
            result = analyze_elbow_plank(video_path)
        elif drill_type == 'single_leg_balance_right':
            result = analyze_single_leg_balance(video_path, 'right')
        elif drill_type == 'single_leg_balance_left':
            result = analyze_single_leg_balance(video_path, 'left')
        else:
            raise Exception(f"Invalid drill type specified: {drill_type}")
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        return result
            
    except Exception as e:
        logger.error(f"Analysis failed for {drill_type}: {str(e)}")
        raise Exception(f"Analysis failed: {str(e)}")
