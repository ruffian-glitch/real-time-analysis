import cv2
import mediapipe as mp
import numpy as np
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    return angle if angle <= 180 else 360 - angle

def extract_pose_landmarks(frame) -> Optional[Dict]:
    """Extract pose landmarks from a frame using MediaPipe."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        h, w, c = frame.shape
        landmarks = {}
        
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x * w,
                'y': landmark.y * h,
                'visibility': landmark.visibility
            }
        return landmarks
    return None

def analyze_pushups(video_path: str) -> Dict:
    """Analyze push-ups using MediaPipe pose estimation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Push-up analysis variables
    reps_data = []
    current_rep = None
    pushup_stage = "up"  # "up" or "down"
    rep_count = 0
    frame_count = 0
    
    # Thresholds for push-up detection
    down_angle_threshold = 90  # Elbow angle for "down" position
    up_angle_threshold = 160   # Elbow angle for "up" position
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        landmarks = extract_pose_landmarks(frame)
        
        if landmarks:
            # Get key landmarks for push-up analysis
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            left_elbow = (landmarks[13]['x'], landmarks[13]['y'])
            right_elbow = (landmarks[14]['x'], landmarks[14]['y'])
            left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
            right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
            left_hip = (landmarks[23]['x'], landmarks[23]['y'])
            right_hip = (landmarks[24]['x'], landmarks[24]['y'])
            left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
            right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
            
            # Calculate elbow angles (use right side as primary)
            elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Calculate body alignment (shoulder-hip-ankle)
            body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            
            # Push-up state machine
            if pushup_stage == "up" and elbow_angle < down_angle_threshold:
                # Starting a new rep
                pushup_stage = "down"
                current_rep = {
                    "rep_number": rep_count + 1,
                    "start_frame": frame_count,
                    "min_elbow_angle": elbow_angle,
                    "max_elbow_angle": elbow_angle,
                    "body_angles": [body_angle]
                }
            
            elif pushup_stage == "down":
                if current_rep:
                    # Update angles during the rep
                    current_rep["min_elbow_angle"] = min(current_rep["min_elbow_angle"], elbow_angle)
                    current_rep["max_elbow_angle"] = max(current_rep["max_elbow_angle"], elbow_angle)
                    current_rep["body_angles"].append(body_angle)
                    
                    # Check if returning to up position
                    if elbow_angle > up_angle_threshold:
                        # Complete the rep
                        current_rep["end_frame"] = frame_count
                        current_rep["body_angle_at_bottom"] = np.mean(current_rep["body_angles"])
                        
                        # Calculate form score based on body alignment and elbow range
                        form_score = calculate_pushup_form_score(
                            current_rep["min_elbow_angle"],
                            current_rep["body_angle_at_bottom"]
                        )
                        current_rep["form_score"] = form_score
                        
                        # Clean up temporary data
                        del current_rep["body_angles"]
                        
                        reps_data.append(current_rep)
                        rep_count += 1
                        pushup_stage = "up"
                        current_rep = None
    
    cap.release()
    
    # Calculate overall metrics
    total_reps = len(reps_data)
    avg_form_score = np.mean([rep["form_score"] for rep in reps_data]) if reps_data else 0
    
    return {
        "drill_type": "push-ups",
        "totalReps": total_reps,
        "formScore": round(avg_form_score, 1),
        "totalTime": int(total_frames / fps) if fps > 0 else 0,
        "consistency": calculate_consistency_score(reps_data),
        "reps": reps_data
    }

def calculate_pushup_form_score(min_elbow_angle: float, body_angle: float) -> float:
    """Calculate form score for a push-up rep."""
    # Ideal elbow angle at bottom: 70-90 degrees
    elbow_score = max(0, 100 - abs(min_elbow_angle - 80) * 2)
    
    # Ideal body alignment: straight line (170-180 degrees)
    body_score = max(0, 100 - abs(body_angle - 175) * 4)
    
    # Combined score (weighted)
    total_score = (elbow_score * 0.6 + body_score * 0.4) / 10
    return round(max(1.0, min(10.0, total_score)), 1)

def analyze_squats(video_path: str) -> Dict:
    """Analyze squats using MediaPipe pose estimation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    reps_data = []
    current_rep = None
    squat_stage = "up"
    rep_count = 0
    frame_count = 0
    
    # Thresholds for squat detection
    down_angle_threshold = 120  # Knee angle for squat position
    up_angle_threshold = 160    # Knee angle for standing position
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        landmarks = extract_pose_landmarks(frame)
        
        if landmarks:
            # Get key landmarks for squat analysis
            left_hip = (landmarks[23]['x'], landmarks[23]['y'])
            right_hip = (landmarks[24]['x'], landmarks[24]['y'])
            left_knee = (landmarks[25]['x'], landmarks[25]['y'])
            right_knee = (landmarks[26]['x'], landmarks[26]['y'])
            left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
            right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            
            # Calculate knee angles (use right side as primary)
            knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Calculate hip angle for depth assessment
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Squat state machine
            if squat_stage == "up" and knee_angle < down_angle_threshold:
                # Starting a new squat
                squat_stage = "down"
                current_rep = {
                    "rep_number": rep_count + 1,
                    "start_frame": frame_count,
                    "min_knee_angle": knee_angle,
                    "hip_angles": [hip_angle]
                }
            
            elif squat_stage == "down":
                if current_rep:
                    # Update angles during the squat
                    current_rep["min_knee_angle"] = min(current_rep["min_knee_angle"], knee_angle)
                    current_rep["hip_angles"].append(hip_angle)
                    
                    # Check if returning to standing position
                    if knee_angle > up_angle_threshold:
                        # Complete the rep
                        current_rep["end_frame"] = frame_count
                        current_rep["torso_angle_at_bottom"] = np.mean(current_rep["hip_angles"])
                        
                        # Calculate form score
                        form_score = calculate_squat_form_score(
                            current_rep["min_knee_angle"],
                            current_rep["torso_angle_at_bottom"]
                        )
                        current_rep["form_score"] = form_score
                        
                        # Clean up temporary data
                        del current_rep["hip_angles"]
                        
                        reps_data.append(current_rep)
                        rep_count += 1
                        squat_stage = "up"
                        current_rep = None
    
    cap.release()
    
    # Calculate overall metrics
    total_reps = len(reps_data)
    avg_form_score = np.mean([rep["form_score"] for rep in reps_data]) if reps_data else 0
    
    return {
        "drill_type": "squats",
        "totalReps": total_reps,
        "formScore": round(avg_form_score, 1),
        "totalTime": int(total_frames / fps) if fps > 0 else 0,
        "consistency": calculate_consistency_score(reps_data),
        "reps": reps_data
    }

def calculate_squat_form_score(min_knee_angle: float, hip_angle: float) -> float:
    """Calculate form score for a squat rep."""
    # Ideal knee angle at bottom: 70-90 degrees (deeper is better)
    depth_score = max(0, 100 - max(0, min_knee_angle - 90) * 3)
    
    # Ideal hip angle: maintaining upright torso
    torso_score = max(0, 100 - abs(hip_angle - 90) * 2)
    
    # Combined score
    total_score = (depth_score * 0.7 + torso_score * 0.3) / 10
    return round(max(1.0, min(10.0, total_score)), 1)

def analyze_situps(video_path: str) -> Dict:
    """Analyze sit-ups using MediaPipe pose estimation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    reps_data = []
    current_rep = None
    situp_stage = "down"
    rep_count = 0
    frame_count = 0
    
    # Thresholds for sit-up detection
    up_angle_threshold = 60   # Hip angle for "up" position
    down_angle_threshold = 120 # Hip angle for "down" position
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        landmarks = extract_pose_landmarks(frame)
        
        if landmarks:
            # Get key landmarks for sit-up analysis
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            left_hip = (landmarks[23]['x'], landmarks[23]['y'])
            right_hip = (landmarks[24]['x'], landmarks[24]['y'])
            left_knee = (landmarks[25]['x'], landmarks[25]['y'])
            right_knee = (landmarks[26]['x'], landmarks[26]['y'])
            
            # Calculate hip angle
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Sit-up state machine
            if situp_stage == "down" and hip_angle < up_angle_threshold:
                # Starting a new sit-up
                situp_stage = "up"
                current_rep = {
                    "rep_number": rep_count + 1,
                    "start_frame": frame_count,
                    "hip_angle_top": hip_angle,
                    "hip_angles": [hip_angle]
                }
            
            elif situp_stage == "up":
                if current_rep:
                    current_rep["hip_angles"].append(hip_angle)
                    current_rep["hip_angle_top"] = min(current_rep["hip_angle_top"], hip_angle)
                    
                    # Check if returning to down position
                    if hip_angle > down_angle_threshold:
                        # Complete the rep
                        current_rep["end_frame"] = frame_count
                        current_rep["hip_angle_bottom"] = hip_angle
                        
                        # Calculate form score
                        form_score = calculate_situp_form_score(
                            current_rep["hip_angle_top"],
                            current_rep["hip_angle_bottom"]
                        )
                        current_rep["form_score"] = form_score
                        
                        # Clean up temporary data
                        del current_rep["hip_angles"]
                        
                        reps_data.append(current_rep)
                        rep_count += 1
                        situp_stage = "down"
                        current_rep = None
    
    cap.release()
    
    # Calculate overall metrics
    total_reps = len(reps_data)
    avg_form_score = np.mean([rep["form_score"] for rep in reps_data]) if reps_data else 0
    
    return {
        "drill_type": "sit-ups",
        "totalReps": total_reps,
        "formScore": round(avg_form_score, 1),
        "totalTime": int(total_frames / fps) if fps > 0 else 0,
        "consistency": calculate_consistency_score(reps_data),
        "reps": reps_data
    }

def calculate_situp_form_score(hip_angle_top: float, hip_angle_bottom: float) -> float:
    """Calculate form score for a sit-up rep."""
    # Range of motion score
    range_of_motion = abs(hip_angle_bottom - hip_angle_top)
    rom_score = min(100, range_of_motion * 2)  # Reward larger range of motion
    
    # Full sit-up completion score (reaching proper top position)
    completion_score = max(0, 100 - hip_angle_top * 2)
    
    # Combined score
    total_score = (rom_score * 0.6 + completion_score * 0.4) / 10
    return round(max(1.0, min(10.0, total_score)), 1)

def analyze_wall_sit(video_path: str) -> Dict:
    """Analyze wall sit/chair hold using MediaPipe pose estimation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    time_series_data = []
    frame_count = 0
    total_hold_time = 0
    
    # Thresholds for wall sit position
    ideal_knee_angle_min = 80
    ideal_knee_angle_max = 100
    ideal_hip_angle_min = 80
    ideal_hip_angle_max = 100
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        landmarks = extract_pose_landmarks(frame)
        
        if landmarks and frame_count % int(fps) == 0:  # Sample every second
            # Get key landmarks
            left_hip = (landmarks[23]['x'], landmarks[23]['y'])
            right_hip = (landmarks[24]['x'], landmarks[24]['y'])
            left_knee = (landmarks[25]['x'], landmarks[25]['y'])
            right_knee = (landmarks[26]['x'], landmarks[26]['y'])
            left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
            right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            
            # Calculate angles
            knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Check if in proper wall sit position
            in_position = (ideal_knee_angle_min <= knee_angle <= ideal_knee_angle_max and
                          ideal_hip_angle_min <= hip_angle <= ideal_hip_angle_max)
            
            if in_position:
                total_hold_time += 1
                
            timestamp = frame_count // int(fps)
            time_series_data.append({
                "timestamp": timestamp,
                "knee_angle": round(knee_angle, 1),
                "hip_angle": round(hip_angle, 1),
                "in_position": in_position
            })
    
    cap.release()
    
    return {
        "drill_type": "wall-sit",
        "totalHoldTime": total_hold_time,
        "formScore": calculate_wall_sit_form_score(time_series_data),
        "totalTime": int(total_frames / fps) if fps > 0 else 0,
        "consistency": calculate_hold_consistency(time_series_data),
        "time_series_data": time_series_data
    }

def calculate_wall_sit_form_score(time_series_data: List[Dict]) -> float:
    """Calculate form score for wall sit."""
    if not time_series_data:
        return 0.0
    
    # Percentage of time in correct position
    correct_positions = sum(1 for data in time_series_data if data.get("in_position", False))
    percentage_correct = (correct_positions / len(time_series_data)) * 100
    
    return round(percentage_correct / 10, 1)  # Convert to 1-10 scale

def analyze_plank(video_path: str) -> Dict:
    """Analyze elbow plank using MediaPipe pose estimation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    time_series_data = []
    frame_count = 0
    total_hold_time = 0
    
    # Ideal body alignment for plank (straight line)
    ideal_body_angle_min = 170
    ideal_body_angle_max = 185
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        landmarks = extract_pose_landmarks(frame)
        
        if landmarks and frame_count % int(fps) == 0:  # Sample every second
            # Get key landmarks
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            left_hip = (landmarks[23]['x'], landmarks[23]['y'])
            right_hip = (landmarks[24]['x'], landmarks[24]['y'])
            left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
            right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
            
            # Calculate body alignment (shoulder-hip-ankle)
            body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            
            # Check if in proper plank position
            in_position = ideal_body_angle_min <= body_angle <= ideal_body_angle_max
            
            if in_position:
                total_hold_time += 1
                
            timestamp = frame_count // int(fps)
            time_series_data.append({
                "timestamp": timestamp,
                "body_angle": round(body_angle, 1),
                "in_position": in_position
            })
    
    cap.release()
    
    return {
        "drill_type": "plank",
        "totalHoldTime": total_hold_time,
        "formScore": calculate_plank_form_score(time_series_data),
        "totalTime": int(total_frames / fps) if fps > 0 else 0,
        "consistency": calculate_hold_consistency(time_series_data),
        "time_series_data": time_series_data
    }

def calculate_plank_form_score(time_series_data: List[Dict]) -> float:
    """Calculate form score for plank."""
    if not time_series_data:
        return 0.0
    
    # Percentage of time in correct position
    correct_positions = sum(1 for data in time_series_data if data.get("in_position", False))
    percentage_correct = (correct_positions / len(time_series_data)) * 100
    
    return round(percentage_correct / 10, 1)  # Convert to 1-10 scale

def analyze_balance(video_path: str, leg_side: str = "right") -> Dict:
    """Analyze single leg balance using MediaPipe pose estimation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fouls = []
    frame_count = 0
    total_balance_time = 0
    foul_count = 0
    
    # Threshold for detecting when lifted foot touches ground
    balance_threshold = 0.05  # Difference in ankle heights
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        landmarks = extract_pose_landmarks(frame)
        
        if landmarks:
            # Get ankle positions
            left_ankle_y = landmarks[27]['y']
            right_ankle_y = landmarks[28]['y']
            
            # Determine which leg should be lifted based on leg_side
            if leg_side == "right":
                # Right leg should be lifted (higher y = lower in image)
                standing_ankle_y = left_ankle_y
                lifted_ankle_y = right_ankle_y
                is_balanced = lifted_ankle_y < standing_ankle_y - balance_threshold * frame.shape[0]
            else:
                # Left leg should be lifted
                standing_ankle_y = right_ankle_y
                lifted_ankle_y = left_ankle_y
                is_balanced = lifted_ankle_y < standing_ankle_y - balance_threshold * frame.shape[0]
            
            if is_balanced:
                total_balance_time += 1 / fps
            else:
                # Potential foul - lifted foot touching ground
                timestamp = frame_count / fps
                foul_count += 1
                fouls.append({
                    "foul_number": foul_count,
                    "timestamp": round(timestamp, 1),
                    "frame_number": frame_count
                })
    
    cap.release()
    
    return {
        "drill_type": "balance",
        "leg_side": leg_side,
        "totalBalanceTime": round(total_balance_time, 1),
        "totalFouls": len(fouls),
        "formScore": calculate_balance_form_score(total_balance_time, len(fouls), total_frames / fps),
        "totalTime": int(total_frames / fps) if fps > 0 else 0,
        "consistency": calculate_balance_consistency(fouls, total_frames / fps),
        "foul_data": fouls
    }

def calculate_balance_form_score(balance_time: float, foul_count: int, total_time: float) -> float:
    """Calculate form score for balance exercise."""
    if total_time == 0:
        return 0.0
    
    # Base score from percentage of time balanced
    balance_percentage = (balance_time / total_time) * 100
    
    # Penalty for fouls
    foul_penalty = min(50, foul_count * 10)  # Max 50% penalty
    
    final_score = max(0, balance_percentage - foul_penalty) / 10
    return round(max(1.0, min(10.0, final_score)), 1)

def calculate_consistency_score(reps_data: List[Dict]) -> int:
    """Calculate consistency score based on form score variation."""
    if len(reps_data) < 2:
        return 100
    
    form_scores = [rep.get("form_score", 0) for rep in reps_data]
    std_dev = np.std(form_scores)
    
    # Lower standard deviation = higher consistency
    consistency = max(0, 100 - (std_dev * 10))
    return int(consistency)

def calculate_hold_consistency(time_series_data: List[Dict]) -> int:
    """Calculate consistency for hold exercises."""
    if not time_series_data:
        return 0
    
    # Calculate percentage of time in correct position
    correct_positions = sum(1 for data in time_series_data if data.get("in_position", False))
    consistency = (correct_positions / len(time_series_data)) * 100
    
    return int(consistency)

def calculate_balance_consistency(fouls: List[Dict], total_time: float) -> int:
    """Calculate consistency for balance exercises."""
    if total_time == 0:
        return 0
    
    # Fewer fouls = higher consistency
    foul_rate = len(fouls) / total_time
    consistency = max(0, 100 - (foul_rate * 50))
    
    return int(consistency)

def analyze_video(video_path: str, drill_type: str) -> Dict:
    """Main analysis router function."""
    try:
        if drill_type == "push-ups":
            return analyze_pushups(video_path)
        elif drill_type == "squats":
            return analyze_squats(video_path)
        elif drill_type == "sit-ups":
            return analyze_situps(video_path)
        elif drill_type == "wall-sit":
            return analyze_wall_sit(video_path)
        elif drill_type == "plank":
            return analyze_plank(video_path)
        elif drill_type == "balance":
            return analyze_balance(video_path, "right")
        else:
            return {"error": f"Invalid drill type: {drill_type}"}
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}