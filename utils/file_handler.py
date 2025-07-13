"""
File handling utilities for AI Pushups Coach v2
"""

import os
import json
import shutil
from pathlib import Path
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file operations for uploads, processing, and storage"""
    
    def __init__(self, upload_folder=None, processed_folder=None, temp_folder=None):
        self.upload_folder = upload_folder or 'uploads'
        self.processed_folder = processed_folder or 'processed'
        self.temp_folder = temp_folder or 'temp'
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        for folder in [self.upload_folder, self.processed_folder, self.temp_folder]:
            os.makedirs(folder, exist_ok=True)
    
    def save_upload(self, file, session_id, filename):
        """Save uploaded file and return the path"""
        try:
            # Secure the filename
            safe_filename = secure_filename(filename)
            
            # Create unique filename with session ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{session_id}_{timestamp}_{safe_filename}"
            
            # Save file
            file_path = os.path.join(self.upload_folder, unique_filename)
            file.save(file_path)
            
            logger.info(f"File saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving upload: {str(e)}")
            raise
    
    def get_processed_path(self, session_id, filename):
        """Get path for processed file"""
        return os.path.join(self.processed_folder, f"{session_id}_{filename}")
    
    def save_analysis_data(self, session_id, analysis_data):
        """Save analysis data to JSON file"""
        try:
            filename = f"{session_id}_analysis.json"
            file_path = os.path.join(self.processed_folder, filename)
            
            # Add metadata
            analysis_data['session_id'] = session_id
            analysis_data['created_at'] = datetime.now().isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis data saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving analysis data: {str(e)}")
            raise
    
    def load_analysis_data(self, session_id):
        """Load analysis data from JSON file"""
        try:
            filename = f"{session_id}_analysis.json"
            file_path = os.path.join(self.processed_folder, filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"Analysis data not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Analysis data loaded: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading analysis data: {str(e)}")
            return None
    
    def get_video_url(self, session_id, filename):
        """Get URL for processed video"""
        return f"/processed/{session_id}_{filename}"
    
    def cleanup_session(self, session_id):
        """Clean up all files for a session"""
        try:
            # Find all files for this session
            session_files = []
            
            for folder in [self.upload_folder, self.processed_folder, self.temp_folder]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        if filename.startswith(session_id):
                            session_files.append(os.path.join(folder, filename))
            
            # Delete files
            for file_path in session_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete file {file_path}: {str(e)}")
            
            logger.info(f"Cleanup completed for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_file_size(self, file_path):
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Error getting file size: {str(e)}")
            return 0
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def is_valid_video_file(self, filename):
        """Check if file is a valid video file"""
        allowed_extensions = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def get_temp_path(self, session_id, filename):
        """Get temporary file path"""
        return os.path.join(self.temp_folder, f"{session_id}_{filename}")
    
    def cleanup_temp_files(self, max_age_hours=24):
        """Clean up old temporary files"""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            
            if os.path.exists(self.temp_folder):
                for filename in os.listdir(self.temp_folder):
                    file_path = os.path.join(self.temp_folder, filename)
                    if os.path.isfile(file_path):
                        file_time = os.path.getmtime(file_path)
                        if file_time < cutoff_time:
                            try:
                                os.remove(file_path)
                                logger.info(f"Cleaned up temp file: {file_path}")
                            except Exception as e:
                                logger.warning(f"Could not delete temp file {file_path}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error during temp cleanup: {str(e)}")
    
    def get_session_files(self, session_id):
        """Get all files associated with a session"""
        files: dict[str, str | None] = {
            'upload': None,
            'processed': None,
            'analysis': None,
            'highlights': None
        }
        
        try:
            # Check upload folder
            for filename in os.listdir(self.upload_folder):
                if filename.startswith(session_id):
                    files['upload'] = os.path.join(self.upload_folder, filename)
                    break
            
            # Check processed folder
            for filename in os.listdir(self.processed_folder):
                if filename.startswith(session_id):
                    if filename.endswith('_analysis.json'):
                        files['analysis'] = os.path.join(self.processed_folder, filename)
                    elif filename.endswith('_highlights.mp4'):
                        files['highlights'] = os.path.join(self.processed_folder, filename)
                    elif filename.endswith('_processed.mp4'):
                        files['processed'] = os.path.join(self.processed_folder, filename)
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting session files: {str(e)}")
            return files 