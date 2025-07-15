import os
import json
import logging
import time
import threading
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from google import genai
from google.genai import types
from analysis_module import analyze_video

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

def analyze_video_with_timeout(video_path, drill_type, result_id, timeout_seconds=180):
    """Run video analysis with timeout in a separate thread."""
    def target():
        try:
            logger.info(f"Starting analysis for {drill_type} on {video_path}")
            result = analyze_video(video_path, drill_type)
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

        # Analyze video with timeout
        try:
            logger.info("Starting video analysis with timeout...")
            result_id = f"analysis_{timestamp}"

            # Initialize result in cache
            analysis_results_cache[result_id] = {'status': 'processing'}

            # Run analysis with timeout
            analysis_result = analyze_video_with_timeout(video_path, drill_type, result_id, timeout_seconds=180)

            # Keep the video file for the results page
            # Store video path in results for the results page
            if analysis_result['status'] == 'completed':
                analysis_result['result']['video_path'] = filename

            # Clean up cache
            if result_id in analysis_results_cache:
                del analysis_results_cache[result_id]

            if analysis_result['status'] == 'completed':
                logger.info("Video analysis completed successfully")
                result = analysis_result['result']
                # Add redirect URL to the response
                result['redirect_url'] = f"/results?data={json.dumps(result)}"
                return jsonify(result)
            else:
                logger.error(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")
                return jsonify({"error": analysis_result.get('error', 'Analysis failed')}), 500

        except Exception as e:
            logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
            # Clean up the uploaded file in case of error
            if os.path.exists(video_path):
                os.remove(video_path)
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

        # Construct the prompt for the Gemini API (RAG)
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

        # Send request to Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            logger.info("Received response from Gemini API")
            return jsonify({"response": response.text})
        else:
            logger.error("Empty response from Gemini API")
            return jsonify({"error": "No response from AI service"}), 500

    except Exception as e:
        logger.error(f"Error in chat handler: {str(e)}", exc_info=True)
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def serve_video(filename):
    """Serve uploaded video files."""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)