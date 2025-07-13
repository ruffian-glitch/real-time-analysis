"""
Configuration settings for AI Pushups Coach v2
"""

import os
from pathlib import Path

class Config:
    """Main configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', 5000))
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
    TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    TEMPLATES_FOLDER = os.path.join(BASE_DIR, 'templates')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
    
    # AI and API settings
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # MediaPipe settings
    POSE_DETECTION_CONFIDENCE = 0.5
    POSE_TRACKING_CONFIDENCE = 0.5
    POSE_MODEL_COMPLEXITY = 1
    
    # Pushup analysis settings
    POSTURE_THRESHOLD = 95
    DEPTH_THRESHOLD = 115
    FORM_SCORE_THRESHOLD = 40
    MIN_REP_DURATION = 0.15
    MAX_REP_DURATION = 5.0
    
    # Real-time settings
    REALTIME_FPS = 30
    REALTIME_BUFFER_SIZE = 5
    REALTIME_MAX_SESSIONS = 10
    
    # TTS settings
    TTS_ENABLED = True
    TTS_VOICE = 'en-US-Neural2-F'  # Google Cloud TTS voice
    TTS_SPEED = 1.0
    
    # WebSocket settings
    WEBSOCKET_PING_INTERVAL = 20
    WEBSOCKET_PING_TIMEOUT = 20
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(BASE_DIR, 'logs', 'app.log')
    
    # Database settings (for future use)
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///pushups.db')
    
    # Cache settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Security settings
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.PROCESSED_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
        
        # Set up logging
        import logging
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
            handlers=[
                logging.FileHandler(Config.LOG_FILE),
                logging.StreamHandler()
            ]
        )

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 