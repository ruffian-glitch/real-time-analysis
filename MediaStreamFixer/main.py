import logging
import os
import sys

# Add the MediaStreamFixer directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for OpenCV
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

from app import app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting AI Fitness Coach application...")
        
        # Log environment info
        logger.info(f"Python version: {os.sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Check for required directories
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
            logger.info("Created uploads directory")
        
        if not os.path.exists('templates'):
            logger.warning("Templates directory not found")
        
        if not os.path.exists('static'):
            logger.warning("Static directory not found")
        
        # Start the Flask application
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == '__main__':
    main()