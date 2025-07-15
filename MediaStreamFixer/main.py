import logging
import os
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
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == '__main__':
    main()
