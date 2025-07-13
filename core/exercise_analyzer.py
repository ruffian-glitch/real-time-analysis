"""
Modular Exercise Analyzer for AI Pushups Coach v2
Supports multiple exercise types with extensible design
"""

import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExerciseMetrics:
    """Standard metrics for any exercise"""
    rep_count: int
    form_score: int
    duration: float
    avg_rep_time: float
    form_issues: List[str]
    recommendations: List[str]
    detailed_analysis: Dict

class ExerciseAnalyzer(ABC):
    """Abstract base class for exercise analyzers"""
    
    def __init__(self):
        # No pose detection initialization needed - handled by video processor
        pass
    
    @abstractmethod
    def analyze_frames(self, frames_data: List[Dict]) -> ExerciseMetrics:
        """Analyze frames and return exercise metrics"""
        pass
    
    @abstractmethod
    def analyze_frame(self, landmarks: Dict) -> Dict:
        """Analyze a single frame"""
        pass
    
    @abstractmethod
    def detect_repetitions(self, frame_analyses: List[Dict]) -> List[Dict]:
        """Detect exercise repetitions"""
        pass
    
    def extract_landmarks(self, yolo_keypoints: List[Dict]) -> Dict:
        """Extract relevant landmarks from YOLO pose detection"""
        # This method is kept for compatibility but landmarks are now passed directly
        # from the video processor in the correct format
        return yolo_keypoints[0] if yolo_keypoints else {}
    
    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_metrics(self, frame_analyses: List[Dict], reps: List[Dict]) -> Dict:
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
    
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'form_score': 0,
            'avg_rep_time': 0,
            'total_duration': 0,
            'rep_count': 0
        }

class PushupAnalyzer(ExerciseAnalyzer):
    """Pushup-specific exercise analyzer"""
    
    def __init__(self):
        super().__init__()
        self.POSTURE_THRESHOLD = 95
        self.DEPTH_THRESHOLD = 115
        self.MIN_REP_DURATION = 0.15
        self.MAX_REP_DURATION = 5.0
    
    def analyze_frames(self, frames_data: List[Dict]) -> ExerciseMetrics:
        """Analyze pushup frames and return metrics"""
        if not frames_data:
            return self._get_empty_pushup_metrics()
        
        # Analyze each frame
        frame_analyses = []
        for frame_data in frames_data:
            analysis = self.analyze_frame(frame_data['landmarks'])
            frame_analyses.append({
                'timestamp': frame_data['timestamp'],
                'analysis': analysis
            })
        
        # Detect pushup repetitions
        reps = self.detect_repetitions(frame_analyses)
        
        # Calculate overall metrics
        metrics = self.calculate_metrics(frame_analyses, reps)
        
        # Generate form feedback
        feedback = self._generate_feedback(metrics, frame_analyses)
        
        return ExerciseMetrics(
            rep_count=len(reps),
            form_score=metrics['form_score'],
            duration=frames_data[-1]['timestamp'] if frames_data else 0,
            avg_rep_time=metrics['avg_rep_time'],
            form_issues=feedback['issues'],
            recommendations=feedback['recommendations'],
            detailed_analysis={
                'repetitions': reps,
                'frame_analyses': frame_analyses,
                'metrics': metrics
            }
        )
    
    def analyze_frame(self, landmarks: Dict) -> Dict:
        """Analyze pushup pose in a single frame"""
        if not landmarks:
            return {'state': 'unknown', 'form_score': 0}
        
        # Calculate key angles and measurements
        body_angle = self._calculate_body_angle(landmarks)
        elbow_angle = self._calculate_elbow_angle(landmarks)
        shoulder_hip_alignment = self._calculate_shoulder_hip_alignment(landmarks)
        
        # Determine pushup state
        state = self._determine_state(elbow_angle, body_angle)
        
        # Calculate form score
        form_score = self._calculate_form_score(elbow_angle, body_angle, shoulder_hip_alignment)
        
        return {
            'state': state,
            'form_score': form_score,
            'body_angle': body_angle,
            'elbow_angle': elbow_angle,
            'shoulder_hip_alignment': shoulder_hip_alignment
        }
    
    def detect_repetitions(self, frame_analyses: List[Dict]) -> List[Dict]:
        """Detect pushup repetitions from frame analyses"""
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
    
    def _calculate_body_angle(self, landmarks: Dict) -> float:
        """Calculate body angle (shoulder-hip-knee)"""
        if not all(k in landmarks for k in ['left_shoulder', 'left_hip', 'left_knee']):
            return 180.0
        
        shoulder = landmarks['left_shoulder']
        hip = landmarks['left_hip']
        knee = landmarks['left_knee']
        
        return self.calculate_angle(
            (shoulder['x'], shoulder['y']),
            (hip['x'], hip['y']),
            (knee['x'], knee['y'])
        )
    
    def _calculate_elbow_angle(self, landmarks: Dict) -> float:
        """Calculate elbow angle (wrist-elbow-shoulder)"""
        if not all(k in landmarks for k in ['left_wrist', 'left_elbow', 'left_shoulder']):
            return 90.0
        
        wrist = landmarks['left_wrist']
        elbow = landmarks['left_elbow']
        shoulder = landmarks['left_shoulder']
        
        return self.calculate_angle(
            (wrist['x'], wrist['y']),
            (elbow['x'], elbow['y']),
            (shoulder['x'], shoulder['y'])
        )
    
    def _calculate_shoulder_hip_alignment(self, landmarks: Dict) -> float:
        """Calculate shoulder-hip alignment"""
        if not all(k in landmarks for k in ['left_shoulder', 'left_hip']):
            return 0.0
        
        shoulder = landmarks['left_shoulder']
        hip = landmarks['left_hip']
        
        return abs(shoulder['y'] - hip['y'])
    
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
        """Calculate pushup form score (0-100) using posture and depth scores"""
        # Posture score: penalize deviation from straight body
        posture_score = max(0, 100 - abs(body_angle - 180) * 2)
        # Depth score: penalize too shallow or too deep
        if elbow_angle < 70:  # Too deep
            depth_score = max(0, 100 - (70 - elbow_angle) * 3)
        elif elbow_angle > 110:  # Too shallow
            depth_score = max(0, 100 - (elbow_angle - 110) * 2)
        else:  # Good depth range (70-110°)
            depth_score = max(0, 100 - abs(elbow_angle - 90) * 1.5)
        # Weighted sum for overall form score
        overall_score = int((posture_score * 0.6 + depth_score * 0.4))
        return max(0, min(100, overall_score))
    
    def _generate_feedback(self, metrics: Dict, frame_analyses: List[Dict]) -> Dict:
        """Generate pushup-specific feedback"""
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
        
        # Check for specific form issues
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
    
    def _get_empty_pushup_metrics(self) -> ExerciseMetrics:
        """Return empty pushup metrics"""
        return ExerciseMetrics(
            rep_count=0,
            form_score=0,
            duration=0,
            avg_rep_time=0,
            form_issues=['No pose data detected'],
            recommendations=['Ensure your full body is visible in the video'],
            detailed_analysis={}
        )

class SquatAnalyzer(ExerciseAnalyzer):
    """Squat-specific exercise analyzer (for future use)"""
    
    def __init__(self):
        super().__init__()
        self.MIN_REP_DURATION = 0.5
        self.MAX_REP_DURATION = 8.0
    
    def analyze_frames(self, frames_data: List[Dict]) -> ExerciseMetrics:
        """Analyze squat frames and return metrics"""
        # Placeholder implementation for squats
        return ExerciseMetrics(
            rep_count=0,
            form_score=0,
            duration=0,
            avg_rep_time=0,
            form_issues=['Squat analysis not yet implemented'],
            recommendations=['Coming soon!'],
            detailed_analysis={}
        )
    
    def analyze_frame(self, landmarks: Dict) -> Dict:
        """Analyze squat pose in a single frame"""
        return {'state': 'unknown', 'form_score': 0}
    
    def detect_repetitions(self, frame_analyses: List[Dict]) -> List[Dict]:
        """Detect squat repetitions"""
        return []

class ExerciseAnalyzerFactory:
    """Factory for creating exercise analyzers"""
    
    _analyzers = {
        'pushup': PushupAnalyzer,
        'squat': SquatAnalyzer,
        # Add more exercise types here
    }
    
    @classmethod
    def create_analyzer(cls, exercise_type: str) -> ExerciseAnalyzer:
        """Create an exercise analyzer for the specified type"""
        if exercise_type not in cls._analyzers:
            raise ValueError(f"Unsupported exercise type: {exercise_type}")
        
        return cls._analyzers[exercise_type]()
    
    @classmethod
    def get_supported_exercises(cls) -> List[str]:
        """Get list of supported exercise types"""
        return list(cls._analyzers.keys())
    
    @classmethod
    def register_analyzer(cls, exercise_type: str, analyzer_class: type):
        """Register a new exercise analyzer"""
        cls._analyzers[exercise_type] = analyzer_class 