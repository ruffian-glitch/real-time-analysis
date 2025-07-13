# AI Pushups Coach v2

A modern, real-time capable pushup analysis system with AI coaching, built with Flask and featuring a ChatGPT-inspired interface.

## 🚀 Features

### Core Features
- **Video Upload & Analysis**: Upload pushup videos for detailed form analysis
- **Real-time Analysis**: Live camera feed with instant pushup detection and feedback
- **AI Coaching**: Personalized coaching using Gemini 2.5 Pro LLM
- **Form Scoring**: Comprehensive form evaluation with detailed metrics
- **Rep Counting**: Accurate pushup repetition counting
- **Voice Feedback**: Text-to-speech coaching during real-time sessions

### UI/UX Features
- **ChatGPT-inspired Interface**: Modern, clean design with large search bar
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Modals**: Seamless camera switching and analysis controls
- **Interactive Results**: Rich results page with video playback and chat
- **Keyboard Shortcuts**: Enhanced user experience with keyboard controls

### Technical Features
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **WebSocket Support**: Real-time communication for live analysis
- **MediaPipe Integration**: Advanced pose detection and tracking
- **Error Handling**: Comprehensive error handling and user feedback
- **Session Management**: Persistent session data and state management

## 📁 Project Structure

```
ai_pushups_coach_v2/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── config/
│   └── settings.py       # Configuration settings
├── core/
│   ├── video_processor.py    # Video processing logic
│   ├── pose_detector.py      # MediaPipe pose detection
│   └── form_analyzer.py      # Pushup form analysis
├── services/
│   ├── ai_coach.py           # AI coaching service
│   ├── realtime_processor.py # Real-time analysis
│   └── tts_service.py        # Text-to-speech service
├── api/
│   └── routes.py             # API endpoints
├── utils/
│   └── file_handler.py       # File management utilities
├── static/
│   ├── app.js               # Main application JavaScript
│   ├── chatgpt_ui.js        # ChatGPT-inspired UI
│   ├── realtime.js          # Real-time analysis JavaScript
│   └── KproSportLogo.svg    # Logo
├── templates/
│   ├── index.html           # Landing page
│   ├── results.html         # Results page
│   └── error.html           # Error page
├── uploads/                 # Uploaded videos
├── processed/               # Processed videos and data
└── temp/                    # Temporary files
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for video processing)
- Webcam (for real-time analysis)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai_pushups_coach_v2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=your_secret_key_here
   FLASK_DEBUG=True
   ```

5. **Install FFmpeg**
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 🎯 Usage

### Video Upload Analysis
1. Click "Upload Video" on the landing page
2. Select or drag & drop a pushup video
3. Wait for analysis to complete
4. View detailed results with form scoring and AI coaching

### Real-time Analysis
1. Click "Start Real-Time Analysis"
2. Grant camera permissions
3. Get into pushup position
4. Receive instant feedback and coaching

### AI Chat Interface
1. After analysis, use the chat interface
2. Ask questions about your form, technique, or progress
3. Get personalized coaching advice

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key for AI coaching
- `OPENAI_API_KEY`: OpenAI API key (alternative AI service)
- `SECRET_KEY`: Flask secret key for sessions
- `FLASK_DEBUG`: Enable/disable debug mode
- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 5000)

### Settings
Key settings can be modified in `config/settings.py`:
- Video processing parameters
- Pose detection confidence thresholds
- Form scoring criteria
- Real-time analysis settings

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=.
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - Landing page
- `POST /upload` - Upload video for analysis
- `GET /results/<session_id>` - View analysis results

### Real-time Endpoints
- `POST /realtime/start` - Start real-time session
- `POST /realtime/stop` - Stop real-time session
- `WS /ws/realtime/<session_id>` - WebSocket for real-time data

### API Endpoints
- `POST /api/chat` - AI chat interface
- `POST /api/chat/stream` - Streaming chat responses
- `POST /api/tts` - Text-to-speech conversion
- `GET /health` - Health check

## 🔒 Security

- File upload validation and sanitization
- Session management with secure cookies
- API rate limiting (configurable)
- Input validation and sanitization
- Secure file handling

## 🚀 Deployment

### Production Deployment
1. Set `FLASK_DEBUG=False`
2. Configure production database
3. Set up reverse proxy (nginx)
4. Use Gunicorn for WSGI server
5. Configure SSL certificates

### Docker Deployment
```bash
# Build image
docker build -t ai-pushups-coach .

# Run container
docker run -p 5000:5000 ai-pushups-coach
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe for pose detection
- Google Gemini for AI coaching
- Tailwind CSS for styling
- Flask community for the web framework

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## 🔄 Version History

### v2.0.0 (Current)
- Complete UI redesign with ChatGPT-inspired interface
- Real-time analysis capabilities
- Modular architecture
- Enhanced AI coaching
- Voice feedback system

### v1.0.0 (Previous)
- Basic video upload and analysis
- Simple form detection
- Basic AI responses 