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
import tempfile
from moviepy import VideoFileClip, vfx

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

def calculate_angle(a, b, c) -> float:
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

def analyze_pushups(video_path: str, age: Optional[int] = None, weight_kg: Optional[float] = None, gender: Optional[str] = None) -> Dict:
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
        "drill_id": "pushups",
        "total_reps": 0,
        "reps": [],
        "cadence_rpm": 0,  # Reps per minute
        "avg_upward_duration": 0,  # Average time for upward phase
        "avg_downward_duration": 0,  # Average time for downward phase
        "head_neck_alignment": [],  # Head/neck position data
        "marker_path_consistency": {
            "shoulder_path": [],
            "hip_path": [],
            "consistency_score": 0
        },
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
        'head_angles': [],
        'current_rep_phases': [],
        'upward_durations': [],
        'downward_durations': [],
        'shoulder_positions': [],
        'hip_positions': [],
        'current_phase': 'down',  # Track current phase
        'phase_start_frame': None,
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
            
            # Get head/neck landmarks
            nose = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.NOSE)
            left_ear = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_EAR)
            right_ear = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_EAR)
            
            # Get hip landmarks
            left_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            right_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
            
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
                body_angle = 180  # Default straight
                if left_hip[0] is not None and left_shoulder[0] is not None and left_hip[1] is not None:
                    body_angle = calculate_angle(left_shoulder, left_hip, (left_hip[0], left_hip[1] - 0.1))
                
                # Calculate head/neck angle
                head_angle = 180  # Default neutral
                if nose[0] is not None and left_shoulder[0] is not None and left_ear[0] is not None:
                    # Calculate angle between nose, shoulder, and ear to determine head tilt
                    head_angle = calculate_angle(nose, left_shoulder, left_ear)
                
                # Track shoulder and hip positions for path analysis
                if (left_shoulder[0] is not None and right_shoulder[0] is not None and 
                    left_shoulder[1] is not None and right_shoulder[1] is not None):
                    avg_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
                    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    state['shoulder_positions'].append((frame_number, avg_shoulder_x, avg_shoulder_y))
                
                if (left_hip[0] is not None and right_hip[0] is not None and 
                    left_hip[1] is not None and right_hip[1] is not None):
                    avg_hip_x = (left_hip[0] + right_hip[0]) / 2
                    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
                    state['hip_positions'].append((frame_number, avg_hip_x, avg_hip_y))
                
                state['elbow_angles'].append(avg_elbow_angle)
                state['head_angles'].append(head_angle)
                
                # Enhanced phase detection with more precise thresholds
                if avg_elbow_angle < 110:  # Down position
                    state['consecutive_down_frames'] += 1
                    state['consecutive_up_frames'] = 0
                    
                    # Phase transition detection
                    if state['current_phase'] == 'up' and state['consecutive_down_frames'] >= 2:
                        # Transition from up to down
                        if state['phase_start_frame'] is not None:
                            phase_duration = (frame_number - state['phase_start_frame']) / fps
                            state['upward_durations'].append(phase_duration)
                            state['current_rep_phases'].append({
                                'phase': 'up',
                                'duration': phase_duration,
                                'start_frame': state['phase_start_frame'],
                                'end_frame': frame_number
                            })
                        
                        state['current_phase'] = 'down'
                        state['phase_start_frame'] = frame_number
                    
                    if state['consecutive_down_frames'] >= 2 and not state['in_pushup']:
                        state['in_pushup'] = True
                        state['current_rep_start'] = frame_number
                        state['current_phase'] = 'down'
                        state['phase_start_frame'] = frame_number
                        state['current_rep_phases'] = []
                        logger.debug(f"Pushup started at frame {frame_number}")
                        
                elif avg_elbow_angle > 150:  # Up position
                    state['consecutive_up_frames'] += 1
                    state['consecutive_down_frames'] = 0
                    
                    # Phase transition detection
                    if state['current_phase'] == 'down' and state['consecutive_up_frames'] >= 2:
                        # Transition from down to up
                        if state['phase_start_frame'] is not None:
                            phase_duration = (frame_number - state['phase_start_frame']) / fps
                            state['downward_durations'].append(phase_duration)
                            state['current_rep_phases'].append({
                                'phase': 'down',
                                'duration': phase_duration,
                                'start_frame': state['phase_start_frame'],
                                'end_frame': frame_number
                            })
                        
                        state['current_phase'] = 'up'
                        state['phase_start_frame'] = frame_number
                    
                    if state['consecutive_up_frames'] >= 2 and state['in_pushup']:
                        state['in_pushup'] = False
                        if state['current_rep_start'] is not None:
                            # Calculate rep metrics
                            rep_frames = state['elbow_angles'][-30:] if len(state['elbow_angles']) >= 30 else state['elbow_angles']
                            min_elbow_angle = min(rep_frames) if rep_frames else avg_elbow_angle
                            max_elbow_angle = max(rep_frames) if rep_frames else avg_elbow_angle
                            
                            # Calculate average head angle for this rep
                            rep_head_angles = state['head_angles'][-30:] if len(state['head_angles']) >= 30 else state['head_angles']
                            avg_head_angle = sum(rep_head_angles) / len(rep_head_angles) if rep_head_angles else head_angle
                            
                            rep_data = {
                                "rep_number": len(results["reps"]) + 1,
                                "start_frame": state['current_rep_start'],
                                "end_frame": frame_number,
                                "max_elbow_angle": round(max_elbow_angle, 1),
                                "min_elbow_angle": round(min_elbow_angle, 1),
                                "body_angle_at_bottom": round(body_angle, 1),
                                "avg_head_angle": round(avg_head_angle, 1),
                                "phases": state['current_rep_phases'].copy(),
                                "current_phase": state['current_phase'] # Add current phase to rep data
                            }
                            results["reps"].append(rep_data)
                            logger.debug(f"Pushup completed: rep {rep_data['rep_number']}")
                            state['current_rep_start'] = None
                            state['elbow_angles'] = []
                            state['head_angles'] = []
                            state['current_rep_phases'] = []
    
    try:
        process_video_frames(video_path, process_frame)
        results["total_reps"] = len(results["reps"])
        

        
        # Calculate cadence (reps per minute)
        session_duration_minutes = results["video_info"]["duration"] / 60
        if session_duration_minutes > 0:
            results["cadence_rpm"] = round(results["total_reps"] / session_duration_minutes, 1)
        
        # Calculate average phase durations
        if state['upward_durations']:
            results["avg_upward_duration"] = round(sum(state['upward_durations']) / len(state['upward_durations']), 2)
        if state['downward_durations']:
            results["avg_downward_duration"] = round(sum(state['downward_durations']) / len(state['downward_durations']), 2)
        
        # Calculate average head/neck alignment
        if state['head_angles']:
            avg_head_angle = sum(state['head_angles']) / len(state['head_angles'])
            results["head_neck_alignment"] = {
                "avg_angle": round(avg_head_angle, 1),
                "deviation_from_neutral": round(abs(avg_head_angle - 180), 1)
            }
        
        # Calculate marker path consistency
        if state['shoulder_positions'] and state['hip_positions']:
            # Calculate path consistency by measuring variance in movement
            shoulder_x_coords = [pos[1] for pos in state['shoulder_positions']]
            shoulder_y_coords = [pos[2] for pos in state['shoulder_positions']]
            hip_x_coords = [pos[1] for pos in state['hip_positions']]
            hip_y_coords = [pos[2] for pos in state['hip_positions']]
            
            # Calculate standard deviation as a measure of consistency
            shoulder_x_std = np.std(shoulder_x_coords) if len(shoulder_x_coords) > 1 else 0
            shoulder_y_std = np.std(shoulder_y_coords) if len(shoulder_y_coords) > 1 else 0
            hip_x_std = np.std(hip_x_coords) if len(hip_x_coords) > 1 else 0
            hip_y_std = np.std(hip_y_coords) if len(hip_y_coords) > 1 else 0
            
            # Consistency score (lower std = more consistent)
            max_std = max(shoulder_x_std, shoulder_y_std, hip_x_std, hip_y_std)
            consistency_score = max(0, 100 - (max_std * 1000))  # Scale to 0-100
            
            results["marker_path_consistency"] = {
                "shoulder_path": {
                    "x_std": round(shoulder_x_std, 4),
                    "y_std": round(shoulder_y_std, 4)
                },
                "hip_path": {
                    "x_std": round(hip_x_std, 4),
                    "y_std": round(hip_y_std, 4)
                },
                "consistency_score": round(consistency_score, 1)
            }
        
        # Rhythm and calories (existing logic)
        rep_durations = []
        for rep in results["reps"]:
            start = rep["start_frame"] / fps
            end = rep["end_frame"] / fps
            rep["start_time"] = round(start, 3)
            rep["end_time"] = round(end, 3)
            rep_durations.append(end - start)
        avg_rep_duration = sum(rep_durations) / len(rep_durations) if rep_durations else 0
        # Rhythm label logic
        if avg_rep_duration < 1.5:
            rhythm_label = "Fast"
        elif avg_rep_duration <= 2.5:
            rhythm_label = "Moderate"
        else:
            rhythm_label = "Slow"
        # Calories logic
        MET = 8.0
        session_duration = results["video_info"]["duration"]
        session_duration_minutes = session_duration / 60
        if weight_kg is None or age is None or gender is None:
            calories_burned_session = 0
            calories_per_hour = 0
        else:
            calories_burned_session = round((MET * weight_kg * 3.5) / 200 * session_duration_minutes)
            calories_per_hour = round((MET * weight_kg * 3.5) / 200 * 60)
        results["rhythm_label"] = rhythm_label
        results["avg_rep_duration"] = round(avg_rep_duration, 2)
        results["calories_burned_session"] = calories_burned_session
        results["calories_per_hour"] = calories_per_hour
        results["age"] = age if age is not None else None
        results["weight_kg"] = weight_kg if weight_kg is not None else None
        results["gender"] = gender if gender is not None else None
        # Add a placeholder for comparison_score (to be filled by frontend)
        results["comparison_score"] = None
        logger.info(f"Pushup analysis complete. Found {results['total_reps']} reps, Cadence: {results['cadence_rpm']} rpm")
        return results
    except Exception as e:
        logger.error(f"Error in pushup analysis: {e}")
        raise

def analyze_squats(video_path: str, age: Optional[int] = None, weight_kg: Optional[float] = None, gender: Optional[str] = None) -> Dict:
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
        "drill_id": "squats",
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
        'consecutive_down_frames': 0,
        'current_phase': 'down',  # Track current phase
        'phase_start_frame': None,
        'current_rep_phases': []  # Track phases for current rep
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
                if left_shoulder[0] is not None and left_hip[0] is not None and left_hip[1] is not None:
                    torso_angle = calculate_angle((left_hip[0], left_hip[1] + 0.1), left_hip, left_shoulder)
                
                state['knee_angles'].append(avg_knee_angle)
                
                # Detect squat phases with hysteresis
                if avg_knee_angle < 130:  # Down position
                    state['consecutive_down_frames'] += 1
                    state['consecutive_up_frames'] = 0
                    
                    # Phase transition detection
                    if state['current_phase'] == 'up' and state['consecutive_down_frames'] >= 2:
                        # Transition from up to down
                        if state['phase_start_frame'] is not None:
                            phase_duration = (frame_number - state['phase_start_frame']) / fps
                            state['current_rep_phases'].append({
                                'phase': 'up',
                                'duration': phase_duration,
                                'start_frame': state['phase_start_frame'],
                                'end_frame': frame_number
                            })
                        
                        state['current_phase'] = 'down'
                        state['phase_start_frame'] = frame_number
                    
                    if state['consecutive_down_frames'] >= 2 and not state['in_squat']:
                        state['in_squat'] = True
                        state['current_rep_start'] = frame_number
                        state['current_phase'] = 'down'
                        state['phase_start_frame'] = frame_number
                        logger.debug(f"Squat started at frame {frame_number}")
                        
                elif avg_knee_angle > 160:  # Up position
                    state['consecutive_up_frames'] += 1
                    state['consecutive_down_frames'] = 0
                    
                    # Phase transition detection
                    if state['current_phase'] == 'down' and state['consecutive_up_frames'] >= 2:
                        # Transition from down to up
                        if state['phase_start_frame'] is not None:
                            phase_duration = (frame_number - state['phase_start_frame']) / fps
                            state['current_rep_phases'].append({
                                'phase': 'down',
                                'duration': phase_duration,
                                'start_frame': state['phase_start_frame'],
                                'end_frame': frame_number
                            })
                        
                        state['current_phase'] = 'up'
                        state['phase_start_frame'] = frame_number
                    
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
                                "torso_angle_at_bottom": round(torso_angle, 1),
                                "phases": state['current_rep_phases'].copy(),
                                "current_phase": state['current_phase']  # Add current phase to rep data
                            }
                            results["reps"].append(rep_data)
                            logger.debug(f"Squat completed: rep {rep_data['rep_number']}")
                            state['current_rep_start'] = None
                            state['knee_angles'] = []
                            state['current_rep_phases'] = []
    
    try:
        process_video_frames(video_path, process_frame)
        results["total_reps"] = len(results["reps"])
        # Rhythm and calories
        rep_durations = []
        for rep in results["reps"]:
            start = rep["start_frame"] / fps
            end = rep["end_frame"] / fps
            rep["start_time"] = round(start, 3)
            rep["end_time"] = round(end, 3)
            rep_durations.append(end - start)
        avg_rep_duration = sum(rep_durations) / len(rep_durations) if rep_durations else 0
        # Rhythm label logic
        if avg_rep_duration < 1.5:
            rhythm_label = "Fast"
        elif avg_rep_duration <= 2.5:
            rhythm_label = "Moderate"
        else:
            rhythm_label = "Slow"
        # Calories logic
        MET = 5.0
        session_duration = results["video_info"]["duration"]
        session_duration_minutes = session_duration / 60
        if weight_kg is None or age is None or gender is None:
            calories_burned_session = 0
            calories_per_hour = 0
        else:
            calories_burned_session = round((MET * weight_kg * 3.5) / 200 * session_duration_minutes)
            calories_per_hour = round((MET * weight_kg * 3.5) / 200 * 60)
        results["rhythm_label"] = rhythm_label
        results["avg_rep_duration"] = round(avg_rep_duration, 2)
        results["calories_burned_session"] = calories_burned_session
        results["calories_per_hour"] = calories_per_hour
        results["age"] = age if age is not None else None
        results["weight_kg"] = weight_kg if weight_kg is not None else None
        results["gender"] = gender if gender is not None else None
        # Add a placeholder for comparison_score (to be filled by frontend)
        results["comparison_score"] = None
        logger.info(f"Squat analysis complete. Found {results['total_reps']} reps")
        return results
    except Exception as e:
        logger.error(f"Error in squat analysis: {e}")
        raise

def analyze_situps(video_path: str, age: Optional[int] = None, weight_kg: Optional[float] = None, gender: Optional[str] = None) -> Dict:
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
        "drill_id": "situps",
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
        'consecutive_down_frames': 0,
        'min_hip_angle': 180, # Initialize to a large value
        'current_phase': 'down',  # Track current phase
        'phase_start_frame': None,
        'current_rep_phases': []  # Track phases for current rep
    }
    
    UP_THRESHOLD = 60
    DOWN_THRESHOLD = 100
    MIN_UP_FRAMES = 2
    MIN_DOWN_FRAMES = 2

    state['up_frames'] = 0
    state['down_frames'] = 0

    def process_frame(frame_number, pose_results, fps):
        if pose_results.pose_landmarks:
            left_shoulder = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            left_hip = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            left_knee = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
            if all(coord[0] is not None for coord in [left_shoulder, left_hip, left_knee]):
                hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                logger.debug(f"[Situp] Frame {frame_number} hip_angle={hip_angle:.1f} in_situp={state['in_situp']} up_frames={state['up_frames']} down_frames={state['down_frames']}")
                if hip_angle < UP_THRESHOLD:
                    state['up_frames'] += 1
                    state['down_frames'] = 0
                    # Phase transition detection
                    if state['current_phase'] == 'down' and state['up_frames'] >= MIN_UP_FRAMES:
                        # Transition from down to up
                        if state['phase_start_frame'] is not None:
                            phase_duration = (frame_number - state['phase_start_frame']) / fps
                            state['current_rep_phases'].append({
                                'phase': 'down',
                                'duration': phase_duration,
                                'start_frame': state['phase_start_frame'],
                                'end_frame': frame_number
                            })
                        
                        state['current_phase'] = 'up'
                        state['phase_start_frame'] = frame_number
                elif hip_angle > DOWN_THRESHOLD:
                    state['down_frames'] += 1
                    state['up_frames'] = 0
                    # Phase transition detection
                    if state['current_phase'] == 'up' and state['down_frames'] >= MIN_DOWN_FRAMES:
                        # Transition from up to down
                        if state['phase_start_frame'] is not None:
                            phase_duration = (frame_number - state['phase_start_frame']) / fps
                            state['current_rep_phases'].append({
                                'phase': 'up',
                                'duration': phase_duration,
                                'start_frame': state['phase_start_frame'],
                                'end_frame': frame_number
                            })
                        
                        state['current_phase'] = 'down'
                        state['phase_start_frame'] = frame_number
                else:
                    state['up_frames'] = 0
                    state['down_frames'] = 0

                if state['up_frames'] >= MIN_UP_FRAMES and not state['in_situp']:
                    state['in_situp'] = True
                    state['current_rep_start'] = frame_number
                    state['current_phase'] = 'up'
                    state['phase_start_frame'] = frame_number
                    logger.debug(f"Situp started at frame {frame_number}")
                elif state['down_frames'] >= MIN_DOWN_FRAMES and state['in_situp']:
                    if state['current_rep_start'] is not None:
                        rep_data = {
                            "rep_number": len(results["reps"]) + 1,
                            "start_frame": state['current_rep_start'],
                            "end_frame": frame_number,
                            "hip_angle_top": round(hip_angle, 1),
                            "hip_angle_bottom": round(state['min_hip_angle'], 1),
                            "phases": state['current_rep_phases'].copy(),
                            "current_phase": state['current_phase']  # Add current phase to rep data
                        }
                        results["reps"].append(rep_data)
                    state['in_situp'] = False
                    state['current_rep_start'] = None
                    state['min_hip_angle'] = 180
                    state['current_rep_phases'] = []
    
    try:
        process_video_frames(video_path, process_frame)
        results["total_reps"] = len(results["reps"])
        # Rhythm and calories
        rep_durations = []
        for rep in results["reps"]:
            start = rep["start_frame"] / fps
            end = rep["end_frame"] / fps
            rep["start_time"] = round(start, 3)
            rep["end_time"] = round(end, 3)
            rep_durations.append(end - start)
        avg_rep_duration = sum(rep_durations) / len(rep_durations) if rep_durations else 0
        # Rhythm label logic
        if avg_rep_duration < 1.5:
            rhythm_label = "Fast"
        elif avg_rep_duration <= 2.5:
            rhythm_label = "Moderate"
        else:
            rhythm_label = "Slow"
        # Calories logic
        MET = 4.5
        session_duration = results["video_info"]["duration"]
        session_duration_minutes = session_duration / 60
        if weight_kg is None or age is None or gender is None:
            calories_burned_session = 0
            calories_per_hour = 0
        else:
            calories_burned_session = round((MET * weight_kg * 3.5) / 200 * session_duration_minutes)
            calories_per_hour = round((MET * weight_kg * 3.5) / 200 * 60)
        results["rhythm_label"] = rhythm_label
        results["avg_rep_duration"] = round(avg_rep_duration, 2)
        results["calories_burned_session"] = calories_burned_session
        results["calories_per_hour"] = calories_per_hour
        results["age"] = age if age is not None else None
        results["weight_kg"] = weight_kg if weight_kg is not None else None
        results["gender"] = gender if gender is not None else None
        # Add a placeholder for comparison_score (to be filled by frontend)
        results["comparison_score"] = None
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
        "time_series_data": [],  # Now a list of {start, end}
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'hold_start_time': None,
        'consecutive_hold_frames': 0
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
                is_valid = 70 <= avg_knee_angle <= 120 and 70 <= avg_hip_angle <= 140
                logger.debug(f"[ChairHold] Frame {frame_number} t={timestamp:.2f}s knee={avg_knee_angle:.1f} hip={avg_hip_angle:.1f} valid={is_valid}")
                # Check if in proper chair hold position
                if is_valid:
                    state['consecutive_hold_frames'] += 1
                    if state['consecutive_hold_frames'] >= 3:
                        if state['hold_start_time'] is None:
                            state['hold_start_time'] = timestamp
                            logger.info(f"[ChairHold] Hold segment started at {timestamp:.2f}s (frame {frame_number})")
                else:
                    state['consecutive_hold_frames'] = 0
                    if state['hold_start_time'] is not None:
                        results["time_series_data"].append({
                            "start": round(state['hold_start_time'], 2),
                            "end": round(timestamp, 2)
                        })
                        logger.info(f"[ChairHold] Hold segment ended at {timestamp:.2f}s (frame {frame_number}), duration={timestamp-state['hold_start_time']:.2f}s")
                        results["total_hold_time"] += timestamp - state['hold_start_time']
                        state['hold_start_time'] = None
            else:
                state['consecutive_hold_frames'] = 0
                if state['hold_start_time'] is not None:
                    timestamp = frame_number / fps if fps > 0 else frame_number
                    results["time_series_data"].append({
                        "start": round(state['hold_start_time'], 2),
                        "end": round(timestamp, 2)
                    })
                    logger.info(f"[ChairHold] Hold segment ended at {timestamp:.2f}s (frame {frame_number}), duration={timestamp-state['hold_start_time']:.2f}s (landmarks missing)")
                    results["total_hold_time"] += timestamp - state['hold_start_time']
                    state['hold_start_time'] = None
    try:
        process_video_frames(video_path, process_frame)
        # Add any remaining hold segment
        if state['hold_start_time'] is not None:
            final_timestamp = results["video_info"]["duration"]
            results["time_series_data"].append({
                "start": round(state['hold_start_time'], 2),
                "end": round(final_timestamp, 2)
            })
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
        "time_series_data": [],  # Now a list of {start, end}
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'hold_start_time': None,
        'consecutive_hold_frames': 0
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
                    if state['consecutive_hold_frames'] >= 3:
                        if state['hold_start_time'] is None:
                            state['hold_start_time'] = timestamp
                else:
                    state['consecutive_hold_frames'] = 0
                    if state['hold_start_time'] is not None:
                        # End of a valid hold segment
                        results["time_series_data"].append({
                            "start": round(state['hold_start_time'], 2),
                            "end": round(timestamp, 2)
                        })
                        results["total_hold_time"] += timestamp - state['hold_start_time']
                        state['hold_start_time'] = None
            else:
                # Landmarks not valid, treat as breaking the plank
                state['consecutive_hold_frames'] = 0
                if state['hold_start_time'] is not None:
                    timestamp = frame_number / fps if fps > 0 else frame_number
                    results["time_series_data"].append({
                        "start": round(state['hold_start_time'], 2),
                        "end": round(timestamp, 2)
                    })
                    results["total_hold_time"] += timestamp - state['hold_start_time']
                    state['hold_start_time'] = None
    
    try:
        process_video_frames(video_path, process_frame)
        
        # Add any remaining hold segment
        if state['hold_start_time'] is not None:
            final_timestamp = results["video_info"]["duration"]
            results["time_series_data"].append({
                "start": round(state['hold_start_time'], 2),
                "end": round(final_timestamp, 2)
            })
            results["total_hold_time"] += final_timestamp - state['hold_start_time']
        
        results["total_hold_time"] = round(results["total_hold_time"], 2)
        
        logger.info(f"Elbow plank analysis complete. Total hold time: {results['total_hold_time']:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"Error in elbow plank analysis: {e}")
        raise

def analyze_single_leg_balance(video_path: str) -> Dict:
    """Analyze single leg balance exercise from video, supporting leg switching and foul detection."""
    logger.info(f"Starting single leg balance analysis for: {video_path}")
    
    if not validate_video_file(video_path):
        raise Exception("Invalid video file or cannot read video")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    results = {
        "drill_type": "single_leg_balance",
        "total_balance_time": 0,
        "total_fouls": 0,
        "foul_data": [],
        "time_series_data": [],
        "video_info": {
            "fps": fps,
            "total_frames": frame_count,
            "duration": frame_count / fps if fps > 0 else 0
        }
    }
    
    # Analysis state
    state = {
        'segment_start_time': None,
        'segment_leg_side': None,
        'last_leg_side': None,
        'last_foul_time': None,
        'foul_cooldown': 2.0,
        'consecutive_balance_frames': 0,
        'consecutive_foul_frames': 0
    }
    
    def process_frame(frame_number, pose_results, fps):
        if pose_results.pose_landmarks:
            left_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
            right_ankle = get_landmark_coordinates(pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
            if (left_ankle[0] is not None and right_ankle[0] is not None and 
                left_ankle[1] is not None and right_ankle[1] is not None):
                timestamp = frame_number / fps if fps > 0 else frame_number
                # Determine which leg is lifted (higher y means lower on screen)
                if left_ankle[1] < right_ankle[1] - 0.05:
                    current_leg = 'right'  # Right ankle is higher (lifted)
                elif right_ankle[1] < left_ankle[1] - 0.05:
                    current_leg = 'left'   # Left ankle is higher (lifted)
                else:
                    current_leg = None  # Neither clearly lifted
                # Check if balanced (one foot clearly off the ground)
                is_balanced = current_leg is not None
                # Handle leg switching and segment tracking
                if is_balanced:
                    state['consecutive_balance_frames'] += 1
                    state['consecutive_foul_frames'] = 0
                    if state['segment_start_time'] is None or state['segment_leg_side'] != current_leg:
                        # End previous segment if any
                        if state['segment_start_time'] is not None and state['segment_leg_side'] is not None:
                            results["time_series_data"].append({
                                "start": round(state['segment_start_time'], 2),
                                "end": round(timestamp, 2),
                                "leg_side": state['segment_leg_side']
                            })
                            results["total_balance_time"] += timestamp - state['segment_start_time']
                        # Start new segment
                        state['segment_start_time'] = timestamp
                        state['segment_leg_side'] = current_leg
                        state['last_leg_side'] = current_leg
                else:
                    state['consecutive_foul_frames'] += 1
                    state['consecutive_balance_frames'] = 0
                    # End current segment if any
                    if state['segment_start_time'] is not None and state['segment_leg_side'] is not None:
                        results["time_series_data"].append({
                            "start": round(state['segment_start_time'], 2),
                            "end": round(timestamp, 2),
                            "leg_side": state['segment_leg_side']
                        })
                        results["total_balance_time"] += timestamp - state['segment_start_time']
                        state['segment_start_time'] = None
                        state['segment_leg_side'] = None
                    # Foul detection (e.g., both feet down, hopping, etc.)
                    if (state['consecutive_foul_frames'] >= 2 and 
                        (state['last_foul_time'] is None or (timestamp - state['last_foul_time']) > state['foul_cooldown'])):
                        foul_data = {
                            "foul_number": len(results["foul_data"]) + 1,
                            "timestamp": round(timestamp, 2),
                            "frame_number": frame_number,
                            "type": "foot_touch",
                            "leg_side": state['last_leg_side']
                        }
                        results["foul_data"].append(foul_data)
                        state['last_foul_time'] = timestamp
                        logger.debug(f"Foul detected at frame {frame_number}, leg: {state['last_leg_side']}")
    try:
        process_video_frames(video_path, process_frame)
        # Add any remaining segment
        if state['segment_start_time'] is not None and state['segment_leg_side'] is not None:
            final_timestamp = results["video_info"]["duration"]
            results["time_series_data"].append({
                "start": round(state['segment_start_time'], 2),
                "end": round(final_timestamp, 2),
                "leg_side": state['segment_leg_side']
            })
            results["total_balance_time"] += final_timestamp - state['segment_start_time']
        results["total_fouls"] = len(results["foul_data"])
        results["total_balance_time"] = round(results["total_balance_time"], 2)
        logger.info(f"Single leg balance analysis complete. Balance time: {results['total_balance_time']:.2f}s, Fouls: {results['total_fouls']}")
        return results
    except Exception as e:
        logger.error(f"Error in single leg balance analysis: {e}")
        raise

def correct_video_orientation(input_path: str, drill_type: str) -> str:
    """
    Corrects the orientation and inversion of the input video based on the drill type.
    Returns the path to the corrected video file in uploads/processed/.
    Deletes the original input file after correction.
    Uses moviepy (v2.0.x compatible) for all video writing and rotation.
    """
    import shutil
    import cv2
    import moviepy
    from moviepy import VideoFileClip, vfx
    logger = logging.getLogger(__name__)
    logger.info(f"[Orientation] Entered correct_video_orientation for {input_path}, drill_type={drill_type}")
    # Prevent double-processing and double '_corrected' suffix
    if os.path.basename(input_path).endswith('_corrected.mp4') and os.path.dirname(input_path).endswith('processed'):
        logger.info(f"[Orientation] File {input_path} is already processed. Returning early.")
        return input_path
    landscape_drills = {'pushups', 'situps', 'elbow_plank'}
    portrait_drills = {'squats', 'chair_hold', 'single_leg_balance_left', 'single_leg_balance_right'}
    required_landscape = drill_type in landscape_drills
    required_portrait = drill_type in portrait_drills

    # Prepare processed directory and output path
    # Store processed videos in a top-level 'processed' folder (sibling to uploads)
    base_dir = os.path.dirname(os.path.dirname(input_path))
    processed_dir = os.path.join(base_dir, 'processed')
    logger.info(f"[Orientation] Ensuring processed_dir exists: {processed_dir}")
    os.makedirs(processed_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    corrected_filename = f"{base_name}_corrected.mp4"
    corrected_path = os.path.join(processed_dir, corrected_filename)
    logger.info(f"[Orientation] Will write processed video to: {corrected_path}")

    logger.info(f"[Orientation] Using moviepy 2.0.x to process video: {input_path}")
    clip = VideoFileClip(input_path)
    logger.info(f"[DEBUG] VideoFileClip dir: {dir(clip)}")
    w, h = clip.size
    rotate_angle = 0

    # Step 3: Initial rotation to required orientation
    if required_landscape and h > w:
        rotate_angle = 90
    elif required_portrait and w > h:
        rotate_angle = -90

    if rotate_angle != 0:
        logger.info(f"[Orientation] Rotating video by {rotate_angle} degrees for orientation correction.")
        clip = clip.rotated(rotate_angle)

    # After rotation, always resize to 1280x720 for landscape drills
    if required_landscape:
        logger.info(f"[Orientation] Resizing video to 1280x720 for landscape drill.")
        clip = clip.resized((1280, 720))

    # (No backend padding for landscape drills)

    # Step 4: Pose-based inversion check (sample multiple frames)
    import random
    frame_times = []
    duration = clip.duration
    # Sample start, middle, end, and 2 random frames
    frame_times.extend([0.1, duration / 2, max(0.1, duration - 0.1)])
    if duration > 2:
        frame_times.append(random.uniform(0.5, duration - 0.5))
    if duration > 4:
        frame_times.append(random.uniform(1, duration - 1))

    pose = get_pose_instance()
    inverted = False
    found_landmarks = False

    for t in frame_times:
        frame = clip.get_frame(t)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            found_landmarks = True
            lm = results.pose_landmarks.landmark
            left_shoulder_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_ankle_y = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y
            right_ankle_y = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            shoulders_y = (left_shoulder_y + right_shoulder_y) / 2
            ankles_y = (left_ankle_y + right_ankle_y) / 2
            if shoulders_y > ankles_y:
                inverted = True
            break  # Use the first frame with landmarks

    if not found_landmarks:
        logger.warning("[Orientation] No pose landmarks found in any sampled frame. Skipping inversion check.")

    if inverted:
        logger.info(f"[Orientation] Detected inversion, applying 180-degree rotation.")
        clip = clip.rotated(180)

    # Step 6: Output corrected video using moviepy (H.264, mp4)
    logger.info(f"[Orientation] Writing final corrected video to: {corrected_path}")
    try:
        clip.write_videofile(corrected_path, codec='libx264', audio_codec='aac', fps=clip.fps)
    except Exception as e:
        logger.error(f"[Orientation] MoviePy failed to write video: {e}")
        raise
    # Ensure the file is closed before deleting the original
    clip.close()
    exists = os.path.exists(corrected_path)
    size = os.path.getsize(corrected_path) if exists else 0
    logger.info(f"[Orientation] Finished writing corrected video. File exists: {exists} Size: {size} bytes")
    if not exists:
        raise RuntimeError(f"[Orientation] Processed video was not written to {corrected_path}. Check MoviePy output and permissions.")

    # Delete the original upload (after closing all handles)
    if os.path.exists(input_path):
        try:
            os.remove(input_path)
        except Exception as e:
            logger.error(f"[Orientation] Failed to delete original upload: {e}")
    logger.info(f"[Orientation] Returning corrected video path: {corrected_path}")
    return corrected_path

def analyze_video(video_path: str, drill_type: str, age=None, weight_kg=None, gender=None) -> Dict:
    """Main router function to call the correct analysis function."""
    logger.info(f"Analyzing video: {video_path} for drill type: {drill_type}")
    
    # Validate video file exists
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    
    # Validate video file is readable
    if not validate_video_file(video_path):
        raise Exception("Cannot read video file - it may be corrupted or in an unsupported format")
    
    # --- Orientation Correction Integration ---
    corrected_path = correct_video_orientation(video_path, drill_type)
    try:
        start_time = time.time()
        
        # Route to appropriate analysis function
        if drill_type == 'pushups':
            result = analyze_pushups(corrected_path, age, weight_kg, gender)
        elif drill_type == 'squats':
            result = analyze_squats(corrected_path, age, weight_kg, gender)
        elif drill_type == 'situps':
            result = analyze_situps(corrected_path, age, weight_kg, gender)
        elif drill_type == 'chair_hold':
            result = analyze_chair_hold(corrected_path)
        elif drill_type == 'elbow_plank':
            result = analyze_elbow_plank(corrected_path)
        elif drill_type == 'single_leg_balance_right':
            result = analyze_single_leg_balance(corrected_path)
        elif drill_type == 'single_leg_balance_left':
            result = analyze_single_leg_balance(corrected_path)
        else:
            raise Exception(f"Invalid drill type specified: {drill_type}")
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        return result
    finally:
        # Clean up temp file if different from original
        if corrected_path != video_path and os.path.exists(corrected_path):
            os.remove(corrected_path)
