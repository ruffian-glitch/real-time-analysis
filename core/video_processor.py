"""
Video Processing Module for AI Pushups Coach v2
"""

import cv2
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from .pose_detector import PoseDetector
from moviepy.editor import ImageSequenceClip
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils, pose
mp_drawing = drawing_utils
mp_pose = pose

logger = logging.getLogger(__name__)

class RepCounter:
    """
    Clean, maintainable rep counting and state machine for pushup detection.
    Centralizes thresholds, state transitions, and logging.
    """
    # Thresholds (can be moved to config if needed)
    ELBOW_UP_THRESHOLD = 145  # Arms extended
    ELBOW_DOWN_THRESHOLD = 135  # Arms bent
    BODY_ALIGNMENT_MIN = 160  # Body must be straight
    MIN_REP_DURATION = 0.15  # seconds
    MAX_REP_DURATION = 5.0   # seconds

    def __init__(self, fps=30, debug=False):
        self.fps = fps
        self.debug = debug
        self.reset()

    def reset(self):
        self.reps = []
        self.current_rep = None
        self.last_state = 'idle'
        self.frame_count = 0

    def log(self, msg):
        if self.debug:
            print(f'[RepCounter] {msg}')

    def get_state(self, elbow_angle, body_angle):
        if body_angle < self.BODY_ALIGNMENT_MIN:
            return 'invalid'
        elif elbow_angle > self.ELBOW_UP_THRESHOLD:
            return 'up'
        elif elbow_angle <= self.ELBOW_DOWN_THRESHOLD:
            return 'down'
        else:
            return 'transition'

    def process_frame(self, timestamp, elbow_angle, body_angle, form_score):
        state = self.get_state(elbow_angle, body_angle)
        self.log(f'Frame {self.frame_count}: state={state}, elbow={elbow_angle:.1f}, body={body_angle:.1f}')
        # Start of rep
        if state == 'down' and self.last_state in ['up', 'transition'] and self.current_rep is None:
            self.current_rep = {
                'start_time': timestamp,
                'start_frame': self.frame_count,
                'form_scores': [form_score]
            }
            self.log(f'Started rep at frame {self.frame_count}, time {timestamp:.2f}')
        # End of rep
        elif state == 'up' and self.last_state in ['down', 'transition'] and self.current_rep is not None:
            self.current_rep['end_time'] = timestamp
            self.current_rep['end_frame'] = self.frame_count
            self.current_rep['duration'] = self.current_rep['end_time'] - self.current_rep['start_time']
            self.current_rep['avg_form_score'] = np.mean(self.current_rep['form_scores']) if self.current_rep['form_scores'] else 0
            # Validate rep duration
            if self.MIN_REP_DURATION <= self.current_rep['duration'] <= self.MAX_REP_DURATION:
                self.reps.append(self.current_rep)
                self.log(f'Completed rep! Duration: {self.current_rep["duration"]:.2f}s, avg_form: {self.current_rep["avg_form_score"]:.1f}')
            else:
                self.log(f'Rep rejected (duration {self.current_rep["duration"]:.2f}s)')
            self.current_rep = None
        # During rep
        elif self.current_rep is not None:
            self.current_rep['form_scores'].append(form_score)
        # Reset on invalid
        if state == 'invalid':
            if self.current_rep is not None:
                self.log('Rep cancelled due to invalid state')
            self.current_rep = None
        self.last_state = state
        self.frame_count += 1

    def get_reps(self):
        return self.reps

class VideoProcessor:
    """Handles video processing and analysis with MediaPipe pose detection"""
    
    def __init__(self):
        self.logger = logger
        
        # Constants for rep detection
        self.MIN_REP_DURATION = 0.15  # seconds (match preintegration)
        self.MAX_REP_DURATION = 5.0   # seconds (match preintegration)
        
    def process_video(self, video_path: str, session_id: str) -> Dict:
        """Process uploaded video and return analysis results"""
        try:
            self.logger.info(f"Processing video: {video_path} for session: {session_id}")
            
            # Extract frames and analyze pose using MediaPipe
            frames_data = self._extract_frames_mediapipe(video_path)
            
            # Analyze exercise form
            analysis_result = self._analyze_pushups(frames_data)
            
            # Generate processed video with pose overlay
            processed_video_path = self._create_processed_video(video_path, session_id, frames_data)
            
            # Save analysis data
            analysis_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'video_path': video_path,
                'processed_video_path': processed_video_path,
                'video_url': f'/processed/{session_id}_processed.mp4',
                'analysis': analysis_result
            }
            
            # Save to file
            self._save_analysis_data(session_id, analysis_data)
            
            result = {
                'success': True,
                'video_url': f'/processed/{session_id}_processed.mp4',
                'highlights_url': f'/processed/{session_id}_highlights.mp4',
                'analysis_data': analysis_result
            }
            
            self.logger.info(f"Video processing completed for session: {session_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Video processing error: {str(e)}")
            return {
                'success': False,
                'error': f'Video processing failed: {str(e)}'
            }
    
    def _analyze_pushups(self, frames_data: List[Dict]) -> Dict:
        """Analyze pushup form and count repetitions, tracking per-rep form scores. Output matches preintegration structure."""
        if not frames_data:
            return self._get_empty_analysis()
        frame_analyses = []
        rep_breakdown = []
        rep_in_progress = False
        rep_start_frame = None
        rep_start_time = None
        rep_elbow_angles = []
        rep_body_angles = []
        rep_shoulder_hip = []
        rep_form_scores = []
        all_form_scores = []
        pushup_count = 0
        valid_reps = 0
        partial_reps = 0
        false_reps = 0
        fps = 30
        for i, frame_data in enumerate(frames_data):
            analysis = self._analyze_frame(frame_data['landmarks'], frame_data['frame'].shape[1], frame_data['frame'].shape[0])
            frame_analyses.append({
                'timestamp': frame_data['timestamp'],
                'analysis': analysis
            })
            state = analysis['state']
            elbow_angle = analysis['elbow_angle']
            body_angle = analysis['body_angle']
            shoulder_hip = analysis['shoulder_hip_alignment']
            form_score = analysis['form_score']
            all_form_scores.append(form_score)
            # Rep state machine (simple, can be enhanced with smoothing)
            if state == 'down' and not rep_in_progress:
                rep_in_progress = True
                rep_start_frame = i
                rep_start_time = frame_data['timestamp']
                rep_elbow_angles = [elbow_angle]
                rep_body_angles = [body_angle]
                rep_shoulder_hip = [shoulder_hip]
                rep_form_scores = [form_score]
            elif state == 'down' and rep_in_progress:
                rep_elbow_angles.append(elbow_angle)
                rep_body_angles.append(body_angle)
                rep_shoulder_hip.append(shoulder_hip)
                rep_form_scores.append(form_score)
            elif state == 'up' and rep_in_progress:
                # End of rep
                rep_end_frame = i
                rep_end_time = frame_data['timestamp']
                duration = rep_end_time - rep_start_time
                min_elbow = min(rep_elbow_angles) if rep_elbow_angles else 0
                avg_body = np.mean(rep_body_angles) if rep_body_angles else 0
                avg_shoulder_hip = np.mean(rep_shoulder_hip) if rep_shoulder_hip else 0
                avg_form = np.mean(rep_form_scores) if rep_form_scores else 0
                # Defensive: only add rep if rep_start_frame is not None
                if rep_start_frame is not None:
                    is_posture_good = avg_body > 95
                    is_depth_good = min_elbow < 115
                    is_form_good = avg_form >= 40
                    if is_posture_good and is_depth_good and is_form_good:
                        rep_class = 'proper'
                        valid_reps += 1
                    elif is_posture_good or is_depth_good:
                        rep_class = 'partial'
                        partial_reps += 1
                    else:
                        rep_class = 'false'
                        false_reps += 1
                    rep_breakdown.append({
                        'class': rep_class,
                        'elbow_angle': float(min_elbow),
                        'body_angle': float(avg_body),
                        'shoulder_hip_alignment': float(avg_shoulder_hip),
                        'form_score': float(avg_form),
                        'duration': float(duration),
                        'frame_start': int(rep_start_frame),
                        'frame_end': int(rep_end_frame)
                    })
                    pushup_count += 1
                rep_in_progress = False
        # Session metrics
        total_reps = valid_reps + partial_reps
        avg_form_score = np.mean(all_form_scores) if all_form_scores else 0
        rep_durations = [rep['duration'] for rep in rep_breakdown]
        avg_rep_time = np.mean(rep_durations) if rep_durations else 0
        duration = frames_data[-1]['timestamp'] if frames_data else 0
        # Output structure matches preintegration
        return {
            'rep_count': int(total_reps),
            'form_score': int(avg_form_score),
            'duration': float(duration),
            'avg_rep_time': float(avg_rep_time),
            'valid_reps': int(valid_reps),
            'partial_reps': int(partial_reps),
            'invalid_reps': int(false_reps),
            'rep_breakdown': rep_breakdown,
            'detailed_analysis': {
                'frame_analyses': frame_analyses
            }
        }
    
    def _extract_frames_mediapipe(self, video_path: str) -> List[Dict]:
        """Extract frames and pose data from video using MediaPipe"""
        frames_data = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_detector = PoseDetector()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        portrait = height > width
        print(f"[DEBUG] Video dimensions: {width}x{height}, portrait: {portrait}")
        print(f"[DEBUG] Total frames: {total_frames}, FPS: {fps}")
        last_pose_landmarks = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if portrait:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if frame_count % 30 == 0:
                    print(f"[DEBUG] Frame {frame_count}: Rotated portrait to landscape")
            landmarks, pose_landmarks = pose_detector.detect_pose(frame)
            # Convert landmarks list to dict by index for downstream compatibility
            if isinstance(landmarks, list):
                landmarks = {str(i): lm for i, lm in enumerate(landmarks)}
            if pose_landmarks is None and last_pose_landmarks is not None:
                pose_landmarks = last_pose_landmarks
                print(f"[DEBUG] Frame {frame_count}: Using last good pose landmarks")
            elif pose_landmarks is not None:
                last_pose_landmarks = pose_landmarks
                if frame_count % 30 == 0:
                    print(f"[DEBUG] Frame {frame_count}: Pose landmarks detected.")
            elif frame_count % 30 == 0:
                print(f"[DEBUG] Frame {frame_count}: No pose landmarks detected.")
            frames_data.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'landmarks': landmarks,
                'pose_landmarks': pose_landmarks,
                'frame': frame,
                'rotated': portrait
            })
            frame_count += 1
        cap.release()
        self.logger.info(f"Extracted {len(frames_data)} frames with pose data using MediaPipe")
        return frames_data
    
    def _analyze_frame(self, landmarks: Dict, frame_width: int = None, frame_height: int = None) -> Dict:
        """Analyze pose in a single frame using preintegration logic for robustness. Requires frame_width for correct thresholding."""
        default_result = {
            'state': 'invalid',
            'form_score': 0,
            'body_angle': 0.0,
            'elbow_angle': 0.0,
            'shoulder_hip_alignment': 0.0
        }
        if not landmarks or not isinstance(landmarks, dict):
            return default_result
        # Debug: print landmark keys and normalize to strings
        print(f'[DEBUG] Landmarks keys: {list(landmarks.keys())}')
        landmarks = {str(k): v for k, v in landmarks.items()}
        # Use MediaPipe indices for all keypoints
        required_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
        # Check all required landmarks are present
        if not all(str(idx) in landmarks for idx in required_indices):
            print('[DEBUG] Missing key landmarks, classified as invalid')
            return default_result
        # Build fake landmark objects for compatibility, handle None values
        class Lm: pass
        lm = {}
        for idx in required_indices:
            pt = landmarks[str(idx)]
            lm[idx] = Lm()
            lm[idx].x = pt.get('x', 0.0) if pt.get('x', 0.0) is not None else 0.0
            lm[idx].y = pt.get('y', 0.0) if pt.get('y', 0.0) is not None else 0.0
            lm[idx].visibility = pt.get('visibility', 1.0) if pt.get('visibility', 1.0) is not None else 1.0
        # Visibility check
        avg_visibility = sum(lm[idx].visibility for idx in required_indices) / len(required_indices)
        if avg_visibility < 0.4:
            print(f'[DEBUG] Frame rejected: low visibility ({avg_visibility:.2f})')
            return default_result
        # Dynamic left/right
        is_facing_right = lm[11].x < lm[12].x
        if is_facing_right:
            shoulder = (lm[11].x, lm[11].y)
            elbow = (lm[13].x, lm[13].y)
            wrist = (lm[15].x, lm[15].y)
            hip = (lm[23].x, lm[23].y)
            knee = (lm[25].x, lm[25].y)
            left_shoulder = (lm[12].x, lm[12].y)
            right_shoulder = (lm[11].x, lm[11].y)
        else:
            shoulder = (lm[12].x, lm[12].y)
            elbow = (lm[14].x, lm[14].y)
            wrist = (lm[16].x, lm[16].y)
            hip = (lm[24].x, lm[24].y)
            knee = (lm[26].x, lm[26].y)
            left_shoulder = (lm[11].x, lm[11].y)
            right_shoulder = (lm[12].x, lm[12].y)
        # Calculate angles
        def calc_angle(a, b, c):
            a, b, c = np.array(a), np.array(b), np.array(c)
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return angle if angle <= 180 else 360 - angle
        body_angle = calc_angle(shoulder, hip, knee)
        elbow_angle = calc_angle(wrist, elbow, shoulder)
        # Loosened state thresholds to match actual data
        if body_angle < 90:
            print(f'[DEBUG] Frame rejected: body_angle={body_angle:.1f} < 90 (too bent)')
            return default_result
        else:
            if elbow_angle > 110:
                state = 'up'
            elif 45 <= elbow_angle <= 110:
                state = 'down'
            elif 85 < elbow_angle <= 145:
                state = 'partial_down'
            elif 100 < elbow_angle <= 145:
                state = 'partial_up'
            else:
                print(f'[DEBUG] Frame rejected: elbow_angle={elbow_angle:.1f} not in any range')
                return default_result
        # Additional validation
        if state in ['up', 'down', 'partial_down', 'partial_up']:
            shoulder_y, hip_y = shoulder[1], hip[1]
            # Use pixel-based threshold for y-difference (as in preintegration)
            if frame_height is not None and isinstance(frame_height, int):
                shoulder_y_px = shoulder_y * frame_height if shoulder_y <= 1.0 else shoulder_y
                hip_y_px = hip_y * frame_height if hip_y <= 1.0 else hip_y
            else:
                shoulder_y_px = shoulder_y
                hip_y_px = hip_y
            try:
                y_diff = abs(float(shoulder_y_px) - float(hip_y_px))
            except Exception:
                y_diff = 0.0
            if y_diff < 10:
                print(f'[DEBUG] Frame rejected: abs(shoulder_y - hip_y)={y_diff:.1f} < 10 (too horizontal)')
                return default_result
            # Use pixel-based threshold for shoulder width (as in preintegration)
            if frame_width is not None and isinstance(frame_width, int):
                left_shoulder_x_px = left_shoulder[0] * frame_width if left_shoulder[0] <= 1.0 else left_shoulder[0]
                right_shoulder_x_px = right_shoulder[0] * frame_width if right_shoulder[0] <= 1.0 else right_shoulder[0]
                try:
                    shoulder_width = abs(float(left_shoulder_x_px) - float(right_shoulder_x_px))
                except Exception:
                    shoulder_width = 0.0
                if shoulder_width > frame_width * 0.95:
                    print(f'[DEBUG] Frame rejected: shoulder_width={shoulder_width:.2f} > {frame_width*0.95:.2f} (arms too wide)')
                    return default_result
            else:
                try:
                    shoulder_width = abs(float(left_shoulder[0]) - float(right_shoulder[0]))
                except Exception:
                    shoulder_width = 0.0
                if shoulder_width > 0.95:
                    print(f'[DEBUG] Frame rejected: shoulder_width={shoulder_width:.2f} > 0.95 (arms too wide)')
                    return default_result
        # Form score (preintegration logic)
        if state in ['up', 'down', 'partial_down', 'partial_up']:
            body_score = max(0, 100 - abs(body_angle - 180) * 2)
            if state == 'down':
                elbow_score = max(0, 100 - abs(elbow_angle - 90) * 2)
            elif state == 'up':
                elbow_score = max(0, 100 - abs(elbow_angle - 180) * 2)
            else:
                elbow_score = max(0, 100 - abs(elbow_angle - 135) * 1.5)
            score = int((body_score * 0.6 + elbow_score * 0.4))
        else:
            score = 0
        # Debug logging for first 20 frames
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        if self._debug_frame_count < 20:
            print(f'[DEBUG] Frame {self._debug_frame_count}: state={state}, elbow_angle={elbow_angle:.1f}, form_score={score}')
            self._debug_frame_count += 1
        return {
            'state': state,
            'form_score': score,
            'body_angle': body_angle,
            'elbow_angle': elbow_angle,
            'shoulder_hip_alignment': y_diff
        }
    
    def _calculate_body_angle(self, landmarks: Dict) -> float:
        """Calculate body angle (shoulder-hip-knee) with robust null checking"""
        if not landmarks:
            return 180.0
        
        # Check if all required landmarks exist
        required_landmarks = ['left_shoulder', 'left_hip', 'left_knee']
        if not all(k in landmarks for k in required_landmarks):
            return 180.0
        
        shoulder = landmarks['left_shoulder']
        hip = landmarks['left_hip']
        knee = landmarks['left_knee']
        
        # Check if any landmark data is None or missing coordinates
        if (shoulder is None or hip is None or knee is None or
            'x' not in shoulder or 'y' not in shoulder or
            'x' not in hip or 'y' not in hip or
            'x' not in knee or 'y' not in knee):
            return 180.0
        
        # Check for invalid coordinates (NaN or infinite values)
        if (np.isnan(shoulder['x']) or np.isnan(shoulder['y']) or
            np.isnan(hip['x']) or np.isnan(hip['y']) or
            np.isnan(knee['x']) or np.isnan(knee['y']) or
            np.isinf(shoulder['x']) or np.isinf(shoulder['y']) or
            np.isinf(hip['x']) or np.isinf(hip['y']) or
            np.isinf(knee['x']) or np.isinf(knee['y'])):
            return 180.0
        
        try:
            angle = self._calculate_angle(
                (shoulder['x'], shoulder['y']),
                (hip['x'], hip['y']),
                (knee['x'], knee['y'])
            )
            
            # Validate the calculated angle
            if np.isnan(angle) or np.isinf(angle):
                return 180.0
            
            return angle
        except Exception as e:
            self.logger.warning(f"Error calculating body angle: {e}")
            return 180.0
    
    def _calculate_elbow_angle(self, landmarks: Dict) -> float:
        """Calculate elbow angle (wrist-elbow-shoulder) with robust null checking"""
        if not landmarks:
            return 90.0
        
        # Check if all required landmarks exist
        required_landmarks = ['left_wrist', 'left_elbow', 'left_shoulder']
        if not all(k in landmarks for k in required_landmarks):
            return 90.0
        
        wrist = landmarks['left_wrist']
        elbow = landmarks['left_elbow']
        shoulder = landmarks['left_shoulder']
        
        # Check if any landmark data is None or missing coordinates
        if (wrist is None or elbow is None or shoulder is None or
            'x' not in wrist or 'y' not in wrist or
            'x' not in elbow or 'y' not in elbow or
            'x' not in shoulder or 'y' not in shoulder):
            return 90.0
        
        # Check for invalid coordinates (NaN or infinite values)
        if (np.isnan(wrist['x']) or np.isnan(wrist['y']) or
            np.isnan(elbow['x']) or np.isnan(elbow['y']) or
            np.isnan(shoulder['x']) or np.isnan(shoulder['y']) or
            np.isinf(wrist['x']) or np.isinf(wrist['y']) or
            np.isinf(elbow['x']) or np.isinf(elbow['y']) or
            np.isinf(shoulder['x']) or np.isinf(shoulder['y'])):
            return 90.0
        
        try:
            angle = self._calculate_angle(
                (wrist['x'], wrist['y']),
                (elbow['x'], elbow['y']),
                (shoulder['x'], shoulder['y'])
            )
            
            # Validate the calculated angle
            if np.isnan(angle) or np.isinf(angle):
                return 90.0
            
            return angle
        except Exception as e:
            self.logger.warning(f"Error calculating elbow angle: {e}")
            return 90.0
    
    def _calculate_shoulder_hip_alignment(self, landmarks: Dict) -> float:
        """Calculate shoulder-hip alignment with robust null checking"""
        if not landmarks:
            return 0.0
        
        # Check if all required landmarks exist
        required_landmarks = ['left_shoulder', 'left_hip']
        if not all(k in landmarks for k in required_landmarks):
            return 0.0
        
        shoulder = landmarks['left_shoulder']
        hip = landmarks['left_hip']
        
        # Check if any landmark data is None or missing coordinates
        if (shoulder is None or hip is None or
            'x' not in shoulder or 'y' not in shoulder or
            'x' not in hip or 'y' not in hip):
            return 0.0
        
        # Check for invalid coordinates (NaN or infinite values)
        if (np.isnan(shoulder['x']) or np.isnan(shoulder['y']) or
            np.isnan(hip['x']) or np.isnan(hip['y']) or
            np.isinf(shoulder['x']) or np.isinf(shoulder['y']) or
            np.isinf(hip['x']) or np.isinf(hip['y'])):
            return 0.0
        
        try:
            alignment = abs(shoulder['y'] - hip['y'])
            
            # Validate the calculated alignment
            if np.isnan(alignment) or np.isinf(alignment):
                return 0.0
            
            return alignment
        except Exception as e:
            self.logger.warning(f"Error calculating shoulder-hip alignment: {e}")
            return 0.0
    
    def _calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _determine_state(self, elbow_angle: float, body_angle: float) -> str:
        """Determine pushup state based on angles"""
        if body_angle < 160:  # Body not straight
            return 'invalid'
        elif elbow_angle > 145:  # Arms extended
            return 'up'
        elif elbow_angle <= 135:  # Arms bent
            return 'down'
        else:
            return 'transition'
    
    def _calculate_form_score(self, elbow_angle: float, body_angle: float, alignment: float) -> int:
        """Calculate form score using exact preintegration logic."""
        # Calculate posture score: penalize deviation from 180°
        posture_score = max(0, 100 - abs(body_angle - 180) * 2)
        
        # Calculate depth score: penalize both too shallow and too deep
        if elbow_angle < 70:  # Too deep
            depth_score = max(0, 100 - (70 - elbow_angle) * 3)
        elif elbow_angle > 110:  # Too shallow
            depth_score = max(0, 100 - (elbow_angle - 110) * 2)
        else:  # Good depth range (70-110°)
            depth_score = max(0, 100 - abs(elbow_angle - 90) * 1.5)
        
        # Weighted sum: 60% posture, 40% depth (exactly as in preintegration)
        overall_score = int((posture_score * 0.6 + depth_score * 0.4))
        
        return overall_score
    
    def _detect_repetitions(self, frame_analyses: List[Dict]) -> List[Dict]:
        """Detect pushup repetitions from frame analyses, tracking per-rep form scores."""
        reps = []
        current_rep = None
        for i, frame_analysis in enumerate(frame_analyses):
            state = frame_analysis['analysis']['state']
            if state == 'down' and current_rep is None:
                # Start of a rep
                current_rep = {
                    'start_time': frame_analysis['timestamp'],
                    'start_frame': i,
                    'form_scores': []
                }
            elif state == 'up' and current_rep is not None:
                # End of a rep
                current_rep['end_time'] = frame_analysis['timestamp']
                current_rep['end_frame'] = i
                current_rep['duration'] = current_rep['end_time'] - current_rep['start_time']
                current_rep['avg_form_score'] = np.mean(current_rep['form_scores']) if current_rep['form_scores'] else 0
                # Validate rep duration
                if self.MIN_REP_DURATION <= current_rep['duration'] <= self.MAX_REP_DURATION:
                    reps.append(current_rep)
                current_rep = None
            elif current_rep is not None:
                # During a rep
                current_rep['form_scores'].append(frame_analysis['analysis']['form_score'])
        return reps
    
    def _calculate_metrics(self, frame_analyses: List[Dict], reps: List[Dict]) -> Dict:
        """Calculate overall metrics"""
        if not frame_analyses:
            return self._get_empty_metrics()
        
        # Calculate average form score
        form_scores = [fa['analysis']['form_score'] for fa in frame_analyses]
        avg_form_score = np.mean(form_scores)
        
        # Calculate rep metrics
        rep_durations = [rep['duration'] for rep in reps]
        avg_rep_time = np.mean(rep_durations) if rep_durations else 0
        
        return {
            'form_score': int(avg_form_score),
            'avg_rep_time': round(avg_rep_time, 2),
            'total_duration': frame_analyses[-1]['timestamp'],
            'rep_count': len(reps)
        }
    
    def _generate_feedback(self, metrics: Dict, frame_analyses: List[Dict]) -> Dict:
        """Generate form feedback and recommendations"""
        issues = []
        recommendations = []
        
        # Analyze form issues
        if metrics['form_score'] < 70:
            issues.append("Poor overall form - focus on technique")
            recommendations.append("Practice with proper form before increasing reps")
        
        if metrics['avg_rep_time'] < 2.0:
            issues.append("Reps too fast - sacrificing form for speed")
            recommendations.append("Slow down and focus on controlled movement")
        
        if metrics['avg_rep_time'] > 6.0:
            issues.append("Reps too slow - may indicate fatigue")
            recommendations.append("Take breaks between sets to maintain quality")
        
        # Check for specific form issues in frame analyses
        body_angles = [fa['analysis']['body_angle'] for fa in frame_analyses]
        elbow_angles = [fa['analysis']['elbow_angle'] for fa in frame_analyses]
        
        if any(angle < 160 for angle in body_angles):
            issues.append("Body not maintaining straight line")
            recommendations.append("Keep your body in a straight line from head to heels")
        
        if any(angle < 80 for angle in elbow_angles):
            issues.append("Elbows bending too much")
            recommendations.append("Don't go too deep - maintain proper elbow angle")
        
        if not issues:
            issues.append("Good form overall!")
            recommendations.append("Keep up the great work and gradually increase difficulty")
        
        return {
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _create_processed_video(self, video_path: str, session_id: str, frames_data: List[Dict]) -> str:
        """Create processed video with pose overlay using MoviePy for browser compatibility, using existing rep data for overlay."""
        try:
            output_path = f"processed/{session_id}_processed.mp4"
            os.makedirs("processed", exist_ok=True)
            
            # Get original video properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            portrait = height > width
            cap.release()
            
            print(f"[DEBUG] Creating video: {total_frames} frames at {fps} FPS")
            
            # Create a mapping from frame number to frames_data
            frame_data_map = {frame_data['frame_number']: frame_data for frame_data in frames_data}
            
            # Analyze all frames to get rep data
            frame_analyses = []
            for frame_data in frames_data:
                analysis = self._analyze_frame(frame_data['landmarks'], frame_data['frame'].shape[1], frame_data['frame'].shape[0])
                frame_analyses.append({
                    'timestamp': frame_data['timestamp'],
                    'analysis': analysis
                })
            
            # Get detected reps using existing logic
            detected_reps = self._detect_repetitions(frame_analyses)
            print(f"[DEBUG] Detected {len(detected_reps)} reps for overlay")
            
            # Create a set of frames where reps end (for overlay counter)
            rep_end_frames = set()
            for rep in detected_reps:
                rep_end_frames.add(rep['end_frame'])
                print(f"[DEBUG] Rep ends at frame {rep['end_frame']}")
            
            # --- Per-frame overlay logic using existing rep data ---
            current_rep_count = 0
            processed_frames = []
            for frame_idx in range(total_frames):
                if frame_idx in frame_data_map:
                    frame_data = frame_data_map[frame_idx]
                    frame = frame_data['frame']
                    pose_landmarks = frame_data.get('pose_landmarks', None)
                    landmarks = frame_data.get('landmarks', None)
                    
                    # Analyze state for this frame
                    frame_analysis = self._analyze_frame(landmarks, frame.shape[1], frame.shape[0]) if landmarks else {'state': 'idle'}
                    state = frame_analysis.get('state', 'idle')
                    
                    # Increment rep count if this frame is where a rep ends
                    if frame_idx in rep_end_frames:
                        current_rep_count += 1
                        print(f"[DEBUG] Frame {frame_idx}: Rep {current_rep_count} completed")
                    
                    # Draw overlays
                    if pose_landmarks is not None:
                        frame = self._draw_pose_overlay(frame, pose_landmarks)
                    frame = self._draw_metrics_overlay(frame, current_rep_count, 0, state)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frames.append(frame_rgb)
                else:
                    print(f"[DEBUG] Frame {frame_idx} not in frame_data_map - this should not happen")
            
            # Write video using MoviePy for browser compatibility
            print(f'[DEBUG] Writing processed video to: {output_path}')
            print(f'[DEBUG] Processed {len(processed_frames)} frames, expected {total_frames}')
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(processed_frames, fps=fps)
            clip.write_videofile(output_path, codec='libx264', audio=False)
            print(f'[DEBUG] Processed video written: {output_path}')
            return output_path
        except Exception as e:
            self.logger.error(f"Error creating processed video: {str(e)}")
            return video_path  # Return original if processing fails
    
    def _draw_pose_overlay(self, frame: np.ndarray, pose_landmarks) -> np.ndarray:
        """Draw pose overlay with white lines and cyan ring keypoints, all lines joined to keypoints."""
        if pose_landmarks is None:
            return frame
        h, w = frame.shape[:2]
        # Colors
        LINE_COLOR = (255, 255, 255)  # White
        KEYPOINT_COLOR = (255, 255, 0)  # Cyan (OpenCV uses BGR, so (255,255,0) is cyan)
        # Draw lines (connections)
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = pose_landmarks.landmark[start_idx]
            end = pose_landmarks.landmark[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 3)
        # Draw keypoints as cyan rings
        for landmark in pose_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 7, KEYPOINT_COLOR, 2)  # Ring (not filled)
            cv2.circle(frame, (x, y), 3, (255,255,255), -1)  # Small white center dot for clarity
        return frame

    def _draw_metrics_overlay(self, frame: np.ndarray, rep_count: int = 0, form_score: int = 0, state: str = 'idle') -> np.ndarray:
        """Draw a clean metrics box showing rep count and state (status below rep), adjusted for layout."""
        h, w = frame.shape[:2]
        # Box settings
        BOX_WIDTH = 210
        BOX_HEIGHT = 80
        BOX_MARGIN = 18
        BOX_PADDING = 12
        box_x = BOX_MARGIN
        box_y = BOX_MARGIN
        # Draw solid white rectangle with black border
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + BOX_WIDTH, box_y + BOX_HEIGHT), (255,255,255), -1)
        cv2.rectangle(overlay, (box_x, box_y), (box_x + BOX_WIDTH, box_y + BOX_HEIGHT), (0,0,0), 2)
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale = 0.6
        value_scale = 1.0
        label_thick = 1
        value_thick = 2
        # Draw title
        cv2.putText(overlay, 'AI PUSHUP COACH', (box_x + BOX_PADDING, box_y + 22), font, label_scale, (0,0,0), label_thick, cv2.LINE_AA)
        # Draw rep count (large and bold)
        cv2.putText(overlay, f'REPS: {rep_count}', (box_x + BOX_PADDING, box_y + 50), font, value_scale, (0,0,0), value_thick, cv2.LINE_AA)
        # Draw state/status below rep count
        cv2.putText(overlay, f'STATUS: {state.upper()}', (box_x + BOX_PADDING, box_y + 72), font, label_scale, (0,0,0), label_thick, cv2.LINE_AA)
        # Blend overlay
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame
    
    def _save_analysis_data(self, session_id: str, analysis_data: Dict):
        """Save analysis data to JSON file"""
        try:
            os.makedirs("processed", exist_ok=True)
            file_path = f"processed/{session_id}_analysis.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Analysis data saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving analysis data: {str(e)}")
    
    def _get_empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'rep_count': 0,
            'form_score': 0,
            'duration': 0,
            'avg_rep_time': 0,
            'form_issues': ['No pose data detected'],
            'recommendations': ['Ensure your full body is visible in the video']
        }
    
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'form_score': 0,
            'avg_rep_time': 0,
            'total_duration': 0,
            'rep_count': 0
        } 