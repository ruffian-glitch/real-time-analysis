#!/usr/bin/env python3
"""
AI Pushups Coach v2 - Main Flask Application
Modular, real-time capable pushup analysis system
"""

import os
import sys
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, send_from_directory, send_file, abort
from werkzeug.utils import secure_filename
import mimetypes
import re

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from config.settings import Config
from core.video_processor import VideoProcessor
from core.pose_detector import PoseDetector
from core.form_analyzer import FormAnalyzer
from core.llm_service import LLMService
from services.ai_coach import AICoach
from services.realtime_processor import RealtimeProcessor
from services.tts_service import TTSService
from api.routes import api_bp
from utils.file_handler import FileHandler

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

# Initialize services
file_handler = FileHandler()
video_processor = VideoProcessor()
pose_detector = PoseDetector()
form_analyzer = FormAnalyzer()
llm_service = LLMService()
ai_coach = AICoach()
realtime_processor = RealtimeProcessor()
tts_service = TTSService()

@app.route('/', methods=['GET'])
def index():
    """Landing page with ChatGPT-inspired interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload for analysis"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = file_handler.save_upload(file, session_id, filename)
        
        # Process video
        result = video_processor.process_video(upload_path, session_id)
        
        if result['success']:
            # Save analysis data to file
            file_handler.save_analysis_data(session_id, result)
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'video_url': result['video_url'],
                'highlights_url': result.get('highlights_url'),
                'analysis_data': result['analysis_data']
            })
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/realtime/start', methods=['POST'])
def start_realtime():
    """Start real-time analysis session"""
    try:
        data = request.get_json()
        session_id = str(uuid.uuid4())
        
        # Initialize real-time session
        result = realtime_processor.start_session(session_id, data.get('camera_type', 'front'))
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'websocket_url': f"ws://{request.host}/ws/realtime/{session_id}"
        })
        
    except Exception as e:
        app.logger.error(f"Realtime start error: {str(e)}")
        return jsonify({'error': 'Failed to start real-time session'}), 500

@app.route('/realtime/stop', methods=['POST'])
def stop_realtime():
    """Stop real-time analysis session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id:
            realtime_processor.stop_session(session_id)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'No session ID provided'}), 400
            
    except Exception as e:
        app.logger.error(f"Realtime stop error: {str(e)}")
        return jsonify({'error': 'Failed to stop real-time session'}), 500

@app.route('/live/start', methods=['POST'])
def start_live_processing():
    """Start live video processing from camera"""
    try:
        data = request.get_json()
        camera_index = data.get('camera_index', 0)
        session_id = str(uuid.uuid4())
        
        # Start live processing
        success = video_processor.start_live_processing(
            camera_index=camera_index,
            callback=lambda result: _handle_live_result(session_id, result)
        )
        
        if success:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': f'Live processing started on camera {camera_index}'
            })
        else:
            return jsonify({'error': 'Failed to start live processing'}), 500
            
    except Exception as e:
        app.logger.error(f"Live processing start error: {str(e)}")
        return jsonify({'error': 'Failed to start live processing'}), 500

@app.route('/live/stop', methods=['POST'])
def stop_live_processing():
    """Stop live video processing"""
    try:
        video_processor.stop_live_processing()
        return jsonify({
            'success': True,
            'message': 'Live processing stopped'
        })
        
    except Exception as e:
        app.logger.error(f"Live processing stop error: {str(e)}")
        return jsonify({'error': 'Failed to stop live processing'}), 500

@app.route('/live/status', methods=['GET'])
def get_live_status():
    """Get current live processing status"""
    try:
        return jsonify({
            'success': True,
            'is_processing': video_processor.is_live_processing,
            'message': 'Live processing active' if video_processor.is_live_processing else 'Live processing inactive'
        })
        
    except Exception as e:
        app.logger.error(f"Live status error: {str(e)}")
        return jsonify({'error': 'Failed to get live status'}), 500

def _handle_live_result(session_id: str, result: dict):
    """Handle live processing results (callback function)"""
    try:
        # Log the result for debugging
        app.logger.info(f"Live result for session {session_id}: {result}")
        
        # Here you could:
        # 1. Send results via WebSocket to frontend
        # 2. Store results in memory/database
        # 3. Trigger real-time feedback
        
        # For now, just log the analysis
        if 'analysis' in result:
            analysis = result['analysis']
            app.logger.info(f"Form score: {analysis.get('form_score', 0)}, State: {analysis.get('state', 'unknown')}")
            
    except Exception as e:
        app.logger.error(f"Error handling live result: {str(e)}")

@app.route('/results/<session_id>')
def results(session_id):
    """Display analysis results"""
    try:
        # Load analysis data
        analysis_data = file_handler.load_analysis_data(session_id)
        
        if analysis_data:
            return render_template('results.html', 
                                 analysis_data=analysis_data,
                                 session_id=session_id)
        else:
            return render_template('error.html', 
                                 message="Analysis data not found"), 404
            
    except Exception as e:
        app.logger.error(f"Results error: {str(e)}")
        return render_template('error.html', 
                             message="Failed to load results"), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle AI chat requests"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id')
        analysis_data = data.get('analysis_data', {})
        
        # Get AI response
        response = ai_coach.get_response(message, analysis_data, session_id)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Chat failed'}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Stream AI chat responses"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id')
        analysis_data = data.get('analysis_data', {})
        
        def generate():
            for chunk in ai_coach.get_streaming_response(message, analysis_data, session_id):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        app.logger.error(f"Chat stream error: {str(e)}")
        return jsonify({'error': 'Chat stream failed'}), 500

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate audio using TTS service
        audio_data = tts_service.synthesize(text)
        
        if audio_data:
            # Return audio as base64 encoded data
            import base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            return jsonify({
                'success': True,
                'audio_data': audio_b64,
                'format': 'wav'
            })
        else:
            return jsonify({'error': 'Failed to generate audio'}), 500
            
    except Exception as e:
        app.logger.error(f"TTS error: {str(e)}")
        return jsonify({'error': 'TTS failed'}), 500

# --- Real-time single-frame analysis endpoint for frontend ---
@app.route('/api/realtime_analyze_frame', methods=['POST'])
def realtime_analyze_frame():
    """Analyze a single frame for real-time pushup detection"""
    try:
        # Accept base64 encoded frame
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame provided'}), 400
        
        import base64
        import cv2
        import numpy as np
        
        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        # Decode base64 to image
        npimg = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect pose using MediaPipe
        landmarks, pose_landmarks = pose_detector.detect_pose(frame)
        h, w = frame.shape[:2]
        
        if landmarks is None:
            return jsonify({
                'state': 'invalid',
                'score': 0,
                'landmarks': None,
                'frame_width': w,
                'frame_height': h,
                'message': 'No pose detected'
            })
        
        # Analyze pushup form
        analysis = form_analyzer.analyze_form(landmarks)
        metrics = analysis.get('metrics', {}) if analysis else {}
        
        if analysis is None:
            return jsonify({
                'state': 'invalid',
                'score': 0,
                'landmarks': None,
                'frame_width': w,
                'frame_height': h,
                'message': 'Analysis failed'
            })
        
        # Extract key metrics for overlay
        response = {
            'state': analysis.get('state', 'invalid'),
            'score': analysis.get('form_score', 0),
            'landmarks': landmarks,
            'frame_width': w,
            'frame_height': h,
            'message': analysis.get('feedback', ''),
            'rep_count': analysis.get('rep_count', 0),
            'issues': analysis.get('issues', [])
        }
        # Add all metrics fields to the response
        response.update(metrics)
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Real-time analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/api/llm/coaching', methods=['POST'])
def llm_coaching():
    """Regular mode coaching endpoint (post-analysis)"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question']
        analysis_data = data.get('analysis_data', {})
        
        # Get coaching response
        response, video_timestamps = llm_service.regular_mode_coaching(question, analysis_data)
        
        return jsonify({
            'success': True,
            'response': response,
            'video_timestamps': video_timestamps
        })
        
    except Exception as e:
        app.logger.error(f"LLM coaching error: {str(e)}")
        return jsonify({'error': 'Coaching failed'}), 500

@app.route('/api/llm/realtime_feedback', methods=['POST'])
def realtime_feedback():
    """Real-time feedback endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Generate real-time feedback
        feedback = llm_service.real_time_feedback(
            current_state=data.get('state', 'invalid'),
            form_score=data.get('form_score', 0),
            elbow_angle=data.get('elbow_angle', 0),
            body_alignment=data.get('body_alignment', 0),
            rep_count=data.get('rep_count', 0),
            issues=data.get('issues', [])
        )
        
        return jsonify({
            'success': True,
            'feedback': feedback
        })
        
    except Exception as e:
        app.logger.error(f"Real-time feedback error: {str(e)}")
        return jsonify({'error': 'Feedback generation failed'}), 500

@app.route('/api/llm/motivational', methods=['POST'])
def motivational_feedback():
    """Get motivational feedback"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        feedback = llm_service.get_motivational_feedback(
            rep_count=data.get('rep_count', 0),
            form_score=data.get('form_score', 0)
        )
        
        return jsonify({
            'success': True,
            'feedback': feedback
        })
        
    except Exception as e:
        app.logger.error(f"Motivational feedback error: {str(e)}")
        return jsonify({'error': 'Feedback generation failed'}), 500

@app.route('/api/llm/form_correction', methods=['POST'])
def form_correction():
    """Get form correction advice"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        correction = llm_service.get_form_correction(
            issues=data.get('issues', [])
        )
        
        return jsonify({
            'success': True,
            'correction': correction
        })
        
    except Exception as e:
        app.logger.error(f"Form correction error: {str(e)}")
        return jsonify({'error': 'Correction generation failed'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/processed/<path:filename>')
def serve_processed_file(filename):
    abs_path = os.path.abspath(os.path.join('processed', filename))
    print(f"[DEBUG] Serving file: {abs_path} | Exists: {os.path.exists(abs_path)} | Size: {os.path.getsize(abs_path) if os.path.exists(abs_path) else 'N/A'}")
    response = send_from_directory('processed', filename, mimetype='video/mp4')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message="Internal server error"), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
    
    print("🚀 AI Pushups Coach v2 Starting...")
    print(f"📁 Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"📁 Processed folder: {Config.PROCESSED_FOLDER}")
    print(f"🌐 Server URL: http://127.0.0.1:{Config.PORT}")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True
    ) 