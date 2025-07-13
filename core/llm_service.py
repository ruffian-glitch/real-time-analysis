"""
LLM Service for AI Pushups Coach v2
Handles both regular mode (post-analysis coaching) and real-time mode (live feedback)
"""

import os
import json
import re
import requests
import logging
from typing import Dict, List, Tuple, Optional
from rag_retriever import retrieve_chunks, simple_rag_search

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service for pushup coaching and feedback"""
    
    def __init__(self):
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not set. LLM functionality will be disabled.")
        
        # Chat history for regular mode
        self.chat_history = []
        
        # Real-time feedback templates
        self.feedback_templates = self._load_feedback_templates()
    
    def _load_feedback_templates(self) -> Dict:
        """Load feedback templates for real-time mode"""
        templates_dir = os.path.join(os.path.dirname(__file__), '..', 'response_templates')
        templates = {}
        
        if os.path.exists(templates_dir):
            for filename in os.listdir(templates_dir):
                if filename.endswith('.txt'):
                    template_name = filename.replace('.txt', '')
                    try:
                        with open(os.path.join(templates_dir, filename), 'r', encoding='utf-8') as f:
                            templates[template_name] = f.read().strip()
                    except Exception as e:
                        logger.error(f"Error loading template {filename}: {e}")
        
        return templates
    
    def query_gemini(self, prompt: str, context: str = "", stream: bool = False) -> str:
        """Query Gemini API"""
        if not self.gemini_api_key:
            return "LLM service not available. Please set GEMINI_API_KEY environment variable."
        
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": self.gemini_api_key
            }
            
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            data = {
                "contents": [{"parts": [{"text": full_prompt}]}]
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            return result["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error: {str(e)}"
    
    def regular_mode_coaching(self, user_question: str, analysis_data: Dict) -> Tuple[str, Optional[Tuple[float, float]]]:
        """
        Handle regular mode coaching (post-analysis)
        Returns: (response_text, video_timestamps)
        """
        # Check if this is a rep-specific query
        if self._is_rep_query(user_question):
            return self._handle_rep_query(user_question, analysis_data)
        
        # Check if this is a metric query
        if self._is_metric_query(user_question):
            return self._handle_metric_query(user_question, analysis_data)
        
        # General coaching query
        return self._handle_general_coaching(user_question, analysis_data)
    
    def real_time_feedback(self, current_state: str, form_score: int, 
                          elbow_angle: float, body_alignment: float,
                          rep_count: int, issues: List[str]) -> str:
        """
        Generate real-time feedback for live pushup session
        """
        # Get relevant knowledge base chunks
        context_chunks = retrieve_chunks("pushup form correction tips", top_k=2)
        knowledge_context = "\n".join([chunk['text'] for chunk in context_chunks])
        
        # Build current state context
        state_context = f"""
Current Pushup State:
- State: {current_state}
- Form Score: {form_score}/100
- Elbow Angle: {elbow_angle:.1f}°
- Body Alignment: {body_alignment:.1f}°
- Rep Count: {rep_count}
- Issues: {', '.join(issues) if issues else 'None'}

Knowledge Base Context:
{knowledge_context}
"""
        
        # Generate appropriate feedback based on state
        if current_state == 'invalid':
            prompt = "The user is in an invalid pushup position. Provide a brief, encouraging tip to help them get into proper form."
        elif current_state == 'down' and form_score < 60:
            prompt = "The user is in the down position with poor form. Give a specific, actionable tip to improve their form."
        elif current_state == 'up' and rep_count > 0:
            prompt = "The user just completed a rep. Give brief, motivating feedback and a tip for the next rep."
        else:
            prompt = "The user is doing pushups. Provide a brief, encouraging tip to maintain good form."
        
        response = self.query_gemini(prompt, state_context)
        
        # Keep response concise for real-time display
        if len(response) > 150:
            response = response[:147] + "..."
        
        return response
    
    def _is_rep_query(self, user_message: str) -> bool:
        """Check if user is asking about a specific rep"""
        msg = user_message.lower()
        rep_keywords = [
            'best rep', 'worst rep', 'show me rep', 'show me my rep', 'play rep', 'play my rep',
            'pushup #', 'push-up #', 'rep #', 'rep number', 'rep no', 'my best', 'my worst', 
            'lowest rep', 'highest rep', 'where did i do best', 'where did i do worst'
        ]
        
        for kw in rep_keywords:
            if kw in msg:
                return True
        
        # Match numbered rep queries
        if re.search(r'rep\s*(?:#|number|no\.?|my)?\s*\d+', msg):
            return True
        if re.search(r'push[- ]?up\s*(?:#|number|no\.?|my)?\s*\d+', msg):
            return True
        
        return False
    
    def _is_metric_query(self, user_message: str) -> bool:
        """Check if user is asking about metrics"""
        msg = user_message.lower()
        metric_keywords = [
            'power output', 'calories', 'calorie', 'percentile', 'score', 'time under tension',
            'push-up count', 'pushup count', 'body alignment', 'rhythm', 'tempo', 'depth',
            'rep count', 'how many reps', 'how many pushups', 'my stats', 'my metrics',
            'my report', 'my results', 'my performance'
        ]
        return any(kw in msg for kw in metric_keywords)
    
    def _handle_rep_query(self, user_message: str, analysis_data: Dict) -> Tuple[str, Optional[Tuple[float, float]]]:
        """Handle rep-specific queries with video timestamps"""
        rep_breakdown = analysis_data.get('rep_breakdown', [])
        fps = analysis_data.get('fps', 30)
        msg = user_message.lower()
        
        # Determine which rep to show
        rep_idx = None
        label = None
        
        if "best" in msg and ("rep" in msg or "pushup" in msg):
            valid_reps = [i for i, rep in enumerate(rep_breakdown) if rep.get('class') in ['proper', 'partial']]
            if valid_reps:
                rep_idx = max(valid_reps, key=lambda i: rep_breakdown[i].get('form_score', -1))
                label = 'best'
        elif "worst" in msg and ("rep" in msg or "pushup" in msg):
            valid_reps = [i for i, rep in enumerate(rep_breakdown) if rep.get('class') in ['proper', 'partial']]
            if valid_reps:
                rep_idx = min(valid_reps, key=lambda i: rep_breakdown[i].get('form_score', float('inf')))
                label = 'worst'
        else:
            # Check for numbered rep
            nth_match = re.search(r'rep\s*(?:#|number|no\.?|my)?\s*(\d+)', msg)
            if not nth_match:
                nth_match = re.search(r'push[- ]?up\s*(?:#|number|no\.?|my)?\s*(\d+)', msg)
            
            if nth_match:
                n = int(nth_match.group(1))
                if 1 <= n <= len(rep_breakdown):
                    rep_idx = n - 1
                    label = f'#{n}'
        
        if rep_idx is not None and rep_idx < len(rep_breakdown):
            rep = rep_breakdown[rep_idx]
            start_time = rep.get('frame_start', 0) / fps
            end_time = min(rep.get('frame_end', 0) / fps, start_time + 3.0)  # Max 3 seconds
            
            # Generate response with video control
            response = f"Playing your {label} rep (Rep {rep_idx + 1})!\n"
            response += f"Form score: {rep.get('form_score', '-')}, "
            response += f"Depth: {rep.get('elbow_angle', '-'):.1f}°, "
            response += f"Duration: {rep.get('duration', '-'):.2f}s."
            
            return response, (start_time, end_time)
        
        return "Sorry, I couldn't find that rep in your session. Try asking for a rep number within your set!", None
    
    def _handle_metric_query(self, user_message: str, analysis_data: Dict) -> Tuple[str, None]:
        """Handle metric-related queries"""
        context = f"""
User's Pushup Session Data:
- Total Reps: {analysis_data.get('total_reps', 'N/A')}
- Valid Reps: {analysis_data.get('valid_reps', 'N/A')}
- Session Duration: {analysis_data.get('duration', 'N/A')} seconds
- Form Score: {analysis_data.get('form_score', 'N/A')}
- Body Alignment: {analysis_data.get('body_alignment_percent', 'N/A')}%
- Pushup Depth: {analysis_data.get('pushup_depth_label', 'N/A')}
- Rhythm: {analysis_data.get('rhythm_tempo_label', 'N/A')}
- Calories Burned: {analysis_data.get('calories_burned', 'N/A')}
- Power Output: {analysis_data.get('power_output', 'N/A')} Watts
"""
        
        prompt = f"Based on the user's pushup session data above, answer their question: {user_message}"
        response = self.query_gemini(prompt, context)
        return response, None
    
    def _handle_general_coaching(self, user_question: str, analysis_data: Dict) -> Tuple[str, None]:
        """Handle general coaching questions"""
        # Get relevant knowledge base chunks
        context_chunks = retrieve_chunks(user_question, top_k=3)
        knowledge_context = "\n".join([chunk['text'] for chunk in context_chunks])
        
        # Build analysis context
        analysis_context = f"""
User's Pushup Session Data:
- Total Reps: {analysis_data.get('total_reps', 'N/A')}
- Valid Reps: {analysis_data.get('valid_reps', 'N/A')}
- Form Score: {analysis_data.get('form_score', 'N/A')}
- Body Alignment: {analysis_data.get('body_alignment_percent', 'N/A')}%
- Pushup Depth: {analysis_data.get('pushup_depth_label', 'N/A')}

Knowledge Base Context:
{knowledge_context}
"""
        
        system_prompt = """
You are "K-AI," a friendly, encouraging, and knowledgeable push-up coach. 
Provide clear, simple, and motivating fitness advice based on the user's data and knowledge base.
Always reference the user's actual numbers and trends in your answer.
"""
        
        full_prompt = f"{system_prompt}\n\nUser Question: {user_question}\nCoach:"
        response = self.query_gemini(full_prompt, analysis_context)
        
        # Add to chat history
        self.chat_history.append({'role': 'user', 'text': user_question})
        self.chat_history.append({'role': 'coach', 'text': response})
        
        # Keep history manageable
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        return response, None
    
    def get_motivational_feedback(self, rep_count: int, form_score: int) -> str:
        """Get motivational feedback based on performance"""
        if rep_count == 0:
            return "Great start! Focus on proper form and you'll be crushing pushups in no time! 💪"
        elif rep_count == 1:
            return "First rep down! That's the hardest one. Keep going! 🔥"
        elif form_score >= 80:
            return f"Excellent form on rep {rep_count}! You're a pushup machine! 🚀"
        elif form_score >= 60:
            return f"Good work on rep {rep_count}! Keep that form strong! 💪"
        else:
            return f"Rep {rep_count} complete! Focus on form - quality over quantity! 🎯"
    
    def get_form_correction(self, issues: List[str]) -> str:
        """Get specific form correction advice"""
        if not issues:
            return "Perfect form! Keep it up! ✨"
        
        # Get relevant knowledge base chunks for form correction
        context_chunks = retrieve_chunks("pushup form correction", top_k=2)
        knowledge_context = "\n".join([chunk['text'] for chunk in context_chunks])
        
        issues_text = ", ".join(issues)
        prompt = f"The user has these form issues: {issues_text}. Provide a brief, specific tip to help them improve."
        
        response = self.query_gemini(prompt, knowledge_context)
        
        # Keep it concise for real-time display
        if len(response) > 120:
            response = response[:117] + "..."
        
        return response 