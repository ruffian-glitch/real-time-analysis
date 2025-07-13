"""
Form Analysis Module for AI Pushups Coach v2
"""

import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

class FormAnalyzer:
    """Analyzes pushup form and provides feedback"""
    
    def __init__(self):
        self.logger = logger
        self.rep_count = 0
        self.current_state = 'idle'
        self.last_state = 'idle'
        self.state_transitions = []
        self.rep_in_progress = False  # Track if we're in the middle of a rep
        
        # Timing tracking for rep validation
        self.rep_start_time = None
        self.frame_count = 0
        self.fps = 30  # Estimated FPS for timing calculations
        
        # Optimized constants for real-time pushup analysis
        # Based on fitness research and empirical testing
        self.BODY_ALIGNMENT_MIN = 150  # Minimum body alignment for valid pushup (was 120)
        self.BODY_ALIGNMENT_OPTIMAL = 170  # Optimal body alignment threshold
        
        # Elbow angle thresholds for state detection
        self.ELBOW_UP_THRESHOLD = 150  # Arms extended (was 145)
        self.ELBOW_DOWN_THRESHOLD = 120  # Arms bent for down position (was 135)
        self.ELBOW_PARTIAL_UP = 140  # Partial up threshold (was 145)
        
        # Rep counting thresholds
        self.MIN_REP_DURATION = 0.5  # Minimum time for a valid rep (seconds)
        self.MAX_REP_DURATION = 8.0  # Maximum time for a valid rep (seconds)
        
        # Form scoring thresholds
        self.FORM_SCORE_THRESHOLD = 50  # Minimum form score for valid rep (was 40)
        
        # Depth scoring thresholds
        self.ELBOW_OPTIMAL_MIN = 80  # Optimal depth range (was 70)
        self.ELBOW_OPTIMAL_MAX = 100  # Optimal depth range (was 110)
        self.ELBOW_TOO_DEEP = 70  # Too deep threshold (was 70)
        self.ELBOW_TOO_SHALLOW = 110  # Too shallow threshold (was 110)
        
        # MediaPipe pose landmark indices
        self.LANDMARK_INDICES = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            a = np.array([a['x'], a['y']])
            b = np.array([b['x'], b['y']])
            c = np.array([c['x'], c['y']])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        except Exception as e:
            self.logger.error(f"Angle calculation error: {str(e)}")
            return 180
    
    def analyze_form(self, pose_data):
        """Analyze pushup form from pose data"""
        try:
            if not pose_data:
                return {
                    'state': 'invalid',
                    'form_score': 0,
                    'rep_count': self.rep_count,
                    'message': 'No pose data available',
                    'issues': ['No pose detected']
                }
            
            # Helper function to get landmark by index
            def get_landmark(idx):
                if idx < len(pose_data):
                    return pose_data[idx]
                return {'x': 0, 'y': 0, 'z': 0, 'visibility': 0}
            
            # Calculate body alignment (shoulder-hip-ankle)
            body_line_l = self.calculate_angle(
                get_landmark(self.LANDMARK_INDICES['left_shoulder']),
                get_landmark(self.LANDMARK_INDICES['left_hip']),
                get_landmark(self.LANDMARK_INDICES['left_ankle'])
            )
            body_line_r = self.calculate_angle(
                get_landmark(self.LANDMARK_INDICES['right_shoulder']),
                get_landmark(self.LANDMARK_INDICES['right_hip']),
                get_landmark(self.LANDMARK_INDICES['right_ankle'])
            )
            body_line = (body_line_l + body_line_r) / 2
            
            # Calculate elbow angle (for depth)
            elbow_l = self.calculate_angle(
                get_landmark(self.LANDMARK_INDICES['left_wrist']),
                get_landmark(self.LANDMARK_INDICES['left_elbow']),
                get_landmark(self.LANDMARK_INDICES['left_shoulder'])
            )
            elbow_r = self.calculate_angle(
                get_landmark(self.LANDMARK_INDICES['right_wrist']),
                get_landmark(self.LANDMARK_INDICES['right_elbow']),
                get_landmark(self.LANDMARK_INDICES['right_shoulder'])
            )
            min_elbow_angle = min(elbow_l, elbow_r)
            
            # Determine state based on optimized thresholds
            if body_line < self.BODY_ALIGNMENT_MIN:
                state = 'invalid'
            elif min_elbow_angle > self.ELBOW_UP_THRESHOLD:
                state = 'up'
            elif min_elbow_angle <= self.ELBOW_DOWN_THRESHOLD:
                state = 'down'
            elif min_elbow_angle <= self.ELBOW_PARTIAL_UP:
                state = 'partial_up'
            else:
                state = 'partial_down'
            
            # Debug log
            self.logger.info(f"[DEBUG] frame: state={state}, elbow_angle={min_elbow_angle:.1f}, alignment={body_line:.1f}, rep_count={self.rep_count}")
            if state == 'invalid':
                self.logger.info(f"[DEBUG] Invalid reason: body_line={body_line:.1f} (threshold: {self.BODY_ALIGNMENT_MIN}), elbow_angle={min_elbow_angle:.1f}")
                # Log landmark visibility for debugging
                left_shoulder_vis = get_landmark(self.LANDMARK_INDICES['left_shoulder'])['visibility']
                left_hip_vis = get_landmark(self.LANDMARK_INDICES['left_hip'])['visibility']
                left_ankle_vis = get_landmark(self.LANDMARK_INDICES['left_ankle'])['visibility']
                self.logger.info(f"[DEBUG] Landmark visibility - shoulder: {left_shoulder_vis:.2f}, hip: {left_hip_vis:.2f}, ankle: {left_ankle_vis:.2f}")
            
            # Handle rep counting
            self.handle_rep_counting(state)
            
            # Calculate form score with optimized thresholds
            # Posture score: penalize deviation from optimal alignment
            posture_score = max(0, 100 - abs(body_line - 180) * 1.5)
            
            # Improved depth scoring with optimized thresholds
            if min_elbow_angle < self.ELBOW_TOO_DEEP:  # Too deep
                depth_score = max(0, 100 - (self.ELBOW_TOO_DEEP - min_elbow_angle) * 2.5)
            elif min_elbow_angle > self.ELBOW_TOO_SHALLOW:  # Too shallow
                depth_score = max(0, 100 - (min_elbow_angle - self.ELBOW_TOO_SHALLOW) * 1.5)
            else:  # Good depth range (70-110°)
                # Optimal range gets bonus points
                if self.ELBOW_OPTIMAL_MIN <= min_elbow_angle <= self.ELBOW_OPTIMAL_MAX:
                    depth_score = max(0, 100 - abs(min_elbow_angle - 90) * 1.0)
                else:
                    depth_score = max(0, 100 - abs(min_elbow_angle - 90) * 1.5)
            
            # Overall score (60% posture, 40% depth)
            form_score = int((posture_score * 0.6 + depth_score * 0.4))
            
            # Generate feedback
            feedback = self.generate_feedback(state, form_score, min_elbow_angle, body_line)
            issues = self.identify_issues(state, form_score, min_elbow_angle, body_line)
            
            return {
                'state': state,
                'form_score': form_score,
                'rep_count': self.rep_count,
                'message': feedback,
                'issues': issues,
                'metrics': {
                    'elbow_angle': min_elbow_angle,
                    'body_alignment': body_line,
                    'posture_score': posture_score,
                    'depth_score': depth_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Form analysis error: {str(e)}")
            return {
                'state': 'invalid',
                'form_score': 0,
                'rep_count': self.rep_count,
                'message': 'Analysis error occurred',
                'issues': ['Analysis failed']
            }
    
    def handle_rep_counting(self, current_state):
        """Handle rep counting logic with timing validation"""
        self.frame_count += 1
        
        # Start of rep: transition from up/partial_up to down
        if current_state == 'down' and self.last_state in ['up', 'partial_up'] and not self.rep_in_progress:
            self.rep_in_progress = True
            self.rep_start_time = self.frame_count / self.fps
            self.logger.info(f"Rep started: {self.last_state} -> {current_state} at frame {self.frame_count}")
        
        # End of rep: transition from down/partial_up to up
        elif current_state == 'up' and self.last_state in ['down', 'partial_up'] and self.rep_in_progress:
            # Calculate rep duration
            if self.rep_start_time is not None:
                rep_duration = (self.frame_count / self.fps) - self.rep_start_time
                
                # Validate rep duration
                if self.MIN_REP_DURATION <= rep_duration <= self.MAX_REP_DURATION:
                    self.rep_count += 1
                    self.logger.info(f"Rep completed! Total: {self.rep_count} (duration: {rep_duration:.2f}s, {self.last_state} -> {current_state})")
                else:
                    self.logger.info(f"Rep rejected - duration {rep_duration:.2f}s outside valid range [{self.MIN_REP_DURATION}, {self.MAX_REP_DURATION}]")
            else:
                self.logger.info(f"Rep completed without timing data")
                self.rep_count += 1
            
            self.rep_in_progress = False
            self.rep_start_time = None
        
        # Reset if we go to invalid state
        elif current_state == 'invalid':
            if self.rep_in_progress:
                self.logger.info(f"Rep cancelled due to invalid state")
            self.rep_in_progress = False
            self.rep_start_time = None
        
        self.last_state = current_state
    
    def generate_feedback(self, state, form_score, elbow_angle, body_line):
        """Generate user-friendly feedback"""
        if state == 'invalid':
            return "Please position yourself in a proper pushup stance"
        elif form_score >= 80:
            return "Excellent form! Keep it up!"
        elif form_score >= 60:
            return "Good form, minor adjustments needed"
        elif form_score >= 40:
            return "Form needs improvement, focus on alignment"
        else:
            return "Form needs significant improvement"
    
    def identify_issues(self, state, form_score, elbow_angle, body_line):
        """Identify specific form issues"""
        issues = []
        
        if body_line < self.BODY_ALIGNMENT_OPTIMAL:
            issues.append("Keep your body straight")
        
        if elbow_angle > self.ELBOW_DOWN_THRESHOLD:
            issues.append("Go deeper - aim for 90° elbow angle")
        elif elbow_angle < self.ELBOW_TOO_DEEP:
            issues.append("Don't go too deep - protect your joints")
        
        if form_score < self.FORM_SCORE_THRESHOLD:
            issues.append("Focus on proper form before speed")
        
        return issues
    
    def reset_rep_count(self):
        """Reset rep counter"""
        self.rep_count = 0
        self.current_state = 'idle'
        self.last_state = 'idle'
        self.state_transitions = []
        self.rep_in_progress = False
        self.rep_start_time = None
        self.frame_count = 0 