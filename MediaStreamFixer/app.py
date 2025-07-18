import os
import json
import logging
import time
import threading
from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from google import genai
from google.genai import types
from analysis_module import analyze_video, correct_video_orientation

# Set up logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create processed directory
processed_dir = os.path.join(os.path.dirname(app.config['UPLOAD_FOLDER']), 'processed')
os.makedirs(processed_dir, exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'default-key')
client = None

def initialize_gemini_client(api_key=None):
    """Initialize or reinitialize the Gemini client with the provided API key."""
    global client
    key_to_use = api_key or GEMINI_API_KEY

    if key_to_use and key_to_use != 'default-key':
        try:
            client = genai.Client(api_key=key_to_use)
            logger.info("Gemini API client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API client: {e}")
            client = None
            return False
    else:
        client = None
        logger.warning("Gemini API key not configured - chat functionality will be disabled")
        return False

# Initialize on startup
initialize_gemini_client()

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v', '3gp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global dictionary to store analysis results temporarily
analysis_results_cache = {}

def analyze_video_with_timeout(video_path, drill_type, result_id, timeout_seconds=180, age=None, weight_kg=None, gender=None):
    """Run video analysis with timeout in a separate thread."""
    def target():
        try:
            logger.info(f"Starting analysis for {drill_type} on {video_path}")
            result = analyze_video(video_path, drill_type, age=age, weight_kg=weight_kg, gender=gender)
            analysis_results_cache[result_id] = {'status': 'completed', 'result': result}
            logger.info(f"Analysis completed for {result_id}")
        except Exception as e:
            logger.error(f"Analysis failed for {result_id}: {str(e)}")
            analysis_results_cache[result_id] = {'status': 'error', 'error': str(e)}

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()

    # Wait for completion or timeout
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logger.error(f"Analysis timed out for {result_id}")
        analysis_results_cache[result_id] = {'status': 'error', 'error': 'Analysis timed out - video may be too long or complex'}
        return analysis_results_cache[result_id]

    return analysis_results_cache.get(result_id, {'status': 'error', 'error': 'Unknown error occurred'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/api/analysis/<filename>')
def get_analysis_data(filename):
    """Get analysis data by filename."""
    try:
        json_path = os.path.join(processed_dir, filename)
        if not os.path.exists(json_path):
            return jsonify({"error": "Analysis data not found"}), 404
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error loading analysis data: {str(e)}")
        return jsonify({"error": "Failed to load analysis data"}), 500

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/api/settings/gemini-key', methods=['POST'])
def update_gemini_key():
    try:
        data = request.get_json()
        if not data or 'api_key' not in data:
            return jsonify({"error": "API key is required"}), 400

        api_key = data['api_key'].strip()
        if not api_key:
            return jsonify({"error": "API key cannot be empty"}), 400

        # Test the API key by initializing the client
        if initialize_gemini_client(api_key):
            # Store in environment for this session
            os.environ['GEMINI_API_KEY'] = api_key
            global GEMINI_API_KEY
            GEMINI_API_KEY = api_key

            logger.info("Gemini API key updated successfully")
            return jsonify({"success": True, "message": "API key updated successfully"})
        else:
            return jsonify({"error": "Invalid API key or failed to connect to Gemini"}), 400

    except Exception as e:
        logger.error(f"Error updating Gemini API key: {str(e)}")
        return jsonify({"error": f"Failed to update API key: {str(e)}"}), 500

@app.route('/api/settings/gemini-key', methods=['GET'])
def get_gemini_key_status():
    try:
        has_key = GEMINI_API_KEY and GEMINI_API_KEY != 'default-key'
        is_working = client is not None

        return jsonify({
            "has_key": has_key,
            "is_working": is_working,
            "masked_key": f"sk-...{GEMINI_API_KEY[-4:]}" if has_key else None
        })
    except Exception as e:
        logger.error(f"Error getting API key status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def handle_analysis():
    try:
        logger.info("Analysis request received")

        # Check if video file is present
        if 'video' not in request.files:
            logger.error("No video file provided")
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        drill_type = request.form.get('drill_type')
        age = request.form.get('age', type=int)
        weight = request.form.get('weight', type=float)
        gender = request.form.get('gender', type=str)

        logger.info(f"Received drill_type: {drill_type}")
        logger.info(f"Received video file: {video_file.filename}")
        logger.info(f"Video file size: {video_file.content_length if hasattr(video_file, 'content_length') else 'unknown'}")

        # Validate inputs
        if not drill_type:
            logger.error("No drill type specified")
            return jsonify({"error": "No drill type specified"}), 400

        if video_file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(video_file.filename):
            logger.error(f"Invalid file type: {video_file.filename}")
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        # Save video file
        filename = secure_filename(video_file.filename)
        # Add timestamp to filename to avoid conflicts
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            logger.info(f"Saving video to: {video_path}")
            video_file.save(video_path)
            logger.info(f"Video saved successfully")
        except Exception as e:
            logger.error(f"Failed to save video file: {str(e)}")
            return jsonify({"error": "Failed to save video file"}), 500

        # Verify file was saved and is accessible
        if not os.path.exists(video_path):
            logger.error(f"Video file not found after saving: {video_path}")
            return jsonify({"error": "Video file could not be saved"}), 500

        # Check file size
        file_size = os.path.getsize(video_path)
        logger.info(f"Video file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

        if file_size == 0:
            logger.error("Video file is empty")
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({"error": "Video file is empty or corrupted"}), 400

        # --- Orientation Correction Integration ---
        try:
            corrected_path = correct_video_orientation(video_path, drill_type)
            # Store processed video in top-level 'processed' folder
            corrected_filename = os.path.relpath(corrected_path, processed_dir)
        except Exception as e:
            logger.error(f"Error correcting video orientation: {str(e)}")
            return jsonify({"error": f"Failed to correct video orientation: {str(e)}"}), 500

        # Analyze video with timeout
        try:
            logger.info("Starting video analysis with timeout...")
            result_id = f"analysis_{timestamp}"

            # Initialize result in cache
            analysis_results_cache[result_id] = {'status': 'processing'}

            # Run analysis with timeout using the corrected video
            analysis_result = analyze_video_with_timeout(corrected_path, drill_type, result_id, timeout_seconds=180, age=age, weight_kg=weight, gender=gender)

            # Store processed video path in results for the results page
            if analysis_result['status'] == 'completed':
                analysis_result['result']['video_path'] = corrected_filename
                # Save JSON result in the same top-level processed folder
                json_base = os.path.splitext(os.path.basename(corrected_filename))[0]
                json_filename = f"{json_base}.json"
                json_path = os.path.join(processed_dir, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result['result'], f, ensure_ascii=False, indent=2)
                analysis_result['result']['json_path'] = json_filename

            # Clean up cache
            if result_id in analysis_results_cache:
                del analysis_results_cache[result_id]

            if analysis_result['status'] == 'completed':
                logger.info("Video analysis completed successfully")
                result = analysis_result['result']
                # Use a simple ID-based redirect instead of passing all data in URL
                result['redirect_url'] = f"/results?id={json_filename}"
                return jsonify(result)
            else:
                logger.error(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")
                return jsonify({"error": analysis_result.get('error', 'Analysis failed')}), 500

        except Exception as e:
            logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Unexpected error in handle_analysis: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        logger.info("Chat request received")

        if not client:
            logger.error("Gemini API client not configured")
            return jsonify({"error": "Gemini API key is not configured on the server."}), 500

        data = request.get_json()
        if not data:
            logger.error("No JSON data provided")
            return jsonify({"error": "No JSON data provided"}), 400

        user_question = data.get('question')
        analysis_json = data.get('analysis_data')

        if not user_question or not analysis_json:
            logger.error("Missing question or analysis data")
            return jsonify({"error": "Missing question or analysis data"}), 400

        logger.info(f"Processing chat question: {user_question}")

        # Enhanced prompt with specific guidance for new metrics
        drill_type = analysis_json.get('drill_type', '')
        
        if drill_type == 'pushups':
            # Dictionary of video-related keywords that should trigger video playback
            video_keywords = {
                'show', 'display', 'play', 'watch', 'see', 'view', 'demonstrate',
                'where', 'when', 'at what time', 'at what point', 'during which',
                'highlight', 'point out', 'mark', 'indicate', 'locate',
                'video', 'clip', 'segment', 'moment', 'instance', 'frame',
                'timeline', 'timestamp', 'timecode', 'position', 'spot',
                'visual', 'visually', 'appears', 'looks like', 'can see',
                'observe', 'notice', 'spot', 'identify', 'find'
            }
            
            # Check if user is asking for video demonstration
            user_words = set(user_question.lower().split())
            wants_video = any(keyword in user_words for keyword in video_keywords)
            
            prompt = f"""You are an expert AI fitness coach specializing in pushup analysis. Answer the user's question based on the comprehensive performance data provided.

**Performance Data:**
```json
{json.dumps(analysis_json, indent=2)}
```

**Available Pushup Metrics:**
- **Cadence**: {analysis_json.get('cadence_rpm', 'N/A')} reps per minute
- **Phase Timing**: Upward: {analysis_json.get('avg_upward_duration', 'N/A')}s, Downward: {analysis_json.get('avg_downward_duration', 'N/A')}s
- **Head/Neck**: {analysis_json.get('head_neck_alignment', {}).get('avg_angle', 'N/A')}° (deviation: {analysis_json.get('head_neck_alignment', {}).get('deviation_from_neutral', 'N/A')}°)
- **Movement Consistency**: {analysis_json.get('marker_path_consistency', {}).get('consistency_score', 'N/A')}/100
- **Rep Details**: Each rep includes phase breakdowns and timing data

**User Question:** {user_question}

**Video Keywords Detected:** {'Yes' if wants_video else 'No'} - User {'is' if wants_video else 'is NOT'} asking for video demonstration

**CRITICAL:** If Video Keywords Detected = No, DO NOT mention any video timestamps, video segments, or "you can review this in the video" in your response. Provide data analysis only.

**Instructions:**
- Start with a motivational line like a real coach would
- Keep response to 4-5 lines maximum
- Focus on 1-2 key insights from the data
- Be specific and actionable with concrete numbers when available
- Use encouraging but direct coaching language
- For consistency questions, mention the actual consistency score (0-100)
- For phase timing questions, provide both upward and downward durations
- For head position questions, mention the angle and deviation from neutral
- For cadence questions, provide the exact reps per minute
- For detailed rep analysis, break down the phases with timing

**IMPORTANT - Video Playback Rules:**
- ONLY suggest video playback if user explicitly asks to see/watch/show something in the video
- Video keywords include: show, display, play, watch, see, view, demonstrate, where, when, highlight, point out, video, clip, segment, moment, instance, frame, timeline, timestamp, timecode, position, spot, visual, visually, appears, looks like, can see, observe, notice, spot, identify, find
- If user asks for data analysis only (no video keywords), provide text-only response with numbers and insights
- If user asks for video demonstration, include specific timestamps and rep numbers for reference
- NEVER mention video timestamps or "you can review this in the video" unless user explicitly asks for video
- For data questions (cadence, consistency, head position, etc.), provide numbers and analysis only
- Only mention video segments when user uses video-related keywords

**Answer:**"""
        else:
            prompt = f"""You are an expert AI fitness coach. Your task is to answer the user's question based on the performance data provided.

**Performance Data:**
```json
{json.dumps(analysis_json, indent=2)}
```

**User Question:** {user_question}

**Instructions:**
- Start with a motivational line like a real coach would
- Keep response to 4-5 lines maximum
- Focus on 1-2 key insights from the data
- Be specific and actionable
- Use encouraging but direct coaching language
- Don't repeat data points, interpret them

**Answer:**"""

        logger.info(f"Sending prompt to Gemini API")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        def generate_stream():
            if response.text:
                # Simulate streaming by yielding one word at a time
                for word in response.text.split():
                    yield word + ' '
                    time.sleep(0.03)  # Simulate typing delay
            else:
                yield "[No response from AI service]"

        return Response(stream_with_context(generate_stream()), mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error in chat handler: {str(e)}", exc_info=True)
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

@app.route('/api/update_metrics', methods=['POST'])
def update_metrics():
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'Missing filename'}), 400
        json_path = os.path.join('processed', filename)
        if not os.path.exists(json_path):
            return jsonify({'error': 'File not found'}), 404
        # Load existing JSON
        with open(json_path, 'r') as f:
            analysis_data = json.load(f)
        # Update fields if present in request
        for key in ['age', 'weight_kg', 'gender', 'calories_burned_session', 'calories_per_hour', 'comparison_score', 'power_per_rep', 'total_power_output']:
            if key in data:
                analysis_data[key] = data[key]
        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        return jsonify(analysis_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({"status": "ok"})

@app.route('/api/debug/videos')
def debug_videos():
    """Debug endpoint to check video files."""
    try:
        processed_files = []
        if os.path.exists(processed_dir):
            for file in os.listdir(processed_dir):
                file_path = os.path.join(processed_dir, file)
                if os.path.isfile(file_path):
                    processed_files.append({
                        'name': file,
                        'size': os.path.getsize(file_path),
                        'path': file_path
                    })
        
        upload_files = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                if os.path.isfile(file_path):
                    upload_files.append({
                        'name': file,
                        'size': os.path.getsize(file_path),
                        'path': file_path
                    })
        
        return jsonify({
            'processed_dir': processed_dir,
            'processed_files': processed_files,
            'upload_dir': app.config['UPLOAD_FOLDER'],
            'upload_files': upload_files
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<path:filename>')
def serve_video(filename):
    """Serve uploaded video files."""
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@app.route('/processed/<path:filename>')
def serve_processed_video(filename):
    """Serve processed video files."""
    try:
        file_path = os.path.join(processed_dir, filename)
        logger.info(f"Serving processed video: {file_path}")
        logger.info(f"File exists: {os.path.exists(file_path)}")
        if os.path.exists(file_path):
            logger.info(f"File size: {os.path.getsize(file_path)} bytes")
        return send_file(file_path)
    except FileNotFoundError:
        logger.error(f"Processed file not found: {filename}")
        return jsonify({"error": "Processed file not found"}), 404
    except Exception as e:
        logger.error(f"Error serving processed video {filename}: {str(e)}")
        return jsonify({"error": f"Error serving file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)