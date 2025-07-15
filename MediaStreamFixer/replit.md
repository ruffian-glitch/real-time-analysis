# AI Fitness Coach - System Architecture

## Overview

This application is an AI-powered fitness coach that analyzes workout videos using computer vision and provides intelligent feedback through a chat interface. The system combines MediaPipe pose estimation, video analysis, and AI-powered conversation to deliver personalized fitness coaching.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Vanilla JavaScript with Bootstrap for UI
- **Key Components**:
  - Video upload interface with drill type selection
  - Real-time analysis progress tracking
  - Interactive chat interface for AI feedback
  - Data visualization for workout metrics
- **Communication**: REST API calls to Flask backend

### Backend Architecture
- **Framework**: Flask web application
- **Core Components**:
  - Video upload and processing handler
  - MediaPipe-based pose analysis engine
  - AI chat integration with Google Gemini
  - RESTful API endpoints for frontend communication
- **Processing**: Multi-threaded video analysis with timeout handling

### Data Storage Solutions
- **File Storage**: Local filesystem for uploaded videos (`uploads/` directory)
- **Session Data**: In-memory storage for analysis results
- **Configuration**: Environment variables for API keys and settings
- **Database**: Currently file-based, designed to accommodate future database integration

## Key Components

### 1. Video Analysis Module (`analysis_module.py`)
- **Purpose**: Core computer vision processing using MediaPipe
- **Key Features**:
  - Pose estimation and landmark detection
  - Angle calculation between body joints
  - Exercise-specific form analysis
  - Frame-by-frame metric extraction
- **Architecture Decision**: Reusable pose instance for performance optimization

### 2. Flask Application (`app.py`)
- **Purpose**: Web server and API endpoint management
- **Key Features**:
  - File upload handling with security validation
  - Integration with Google Gemini API
  - Error handling and logging
  - CORS and proxy support for deployment
- **Architecture Decision**: Modular design separating concerns between web handling and analysis

### 3. Main Entry Point (`main.py`)
- **Purpose**: Application bootstrap and configuration
- **Key Features**:
  - Environment setup and validation
  - Directory creation and health checks
  - Development server configuration

### 4. Frontend Interface
- **Templates**: HTML templates with Bootstrap styling
- **Static Assets**: CSS and JavaScript for user interaction
- **Key Features**:
  - Responsive design for mobile and desktop
  - Real-time feedback during video processing
  - Interactive chat interface

## Data Flow

1. **Video Upload**: User selects drill type and uploads video file
2. **Analysis Processing**: Backend processes video using MediaPipe pose estimation
3. **Data Generation**: System creates structured JSON with frame-by-frame analysis
4. **Visualization**: Frontend displays metrics, graphs, and video player
5. **AI Interaction**: User asks questions about their form and performance
6. **Contextual Response**: AI provides feedback using analysis data as context

## External Dependencies

### Core Libraries
- **MediaPipe**: Google's pose estimation and computer vision framework
- **OpenCV**: Video processing and frame manipulation
- **NumPy**: Mathematical calculations for angle and distance measurements
- **Flask**: Web framework for API endpoints
- **Google Gemini API**: Large language model for AI chat functionality

### Frontend Dependencies
- **Bootstrap**: UI framework for responsive design
- **Font Awesome**: Icon library for user interface elements

### Development Dependencies
- **Logging**: Built-in Python logging for debugging and monitoring
- **Threading**: Concurrent processing for video analysis
- **Werkzeug**: WSGI utilities for file handling and security

## Deployment Strategy

### Development Setup
- **Local Development**: Flask development server with debug mode
- **Port Configuration**: Default port 5000 with host binding to 0.0.0.0
- **File Handling**: Local uploads directory with 100MB file size limit

### Production Considerations
- **Proxy Support**: ProxyFix middleware for reverse proxy deployment
- **Security**: Secure filename handling and file extension validation
- **Logging**: Comprehensive logging to both console and files
- **Environment Variables**: API keys and configuration via environment variables

### Scalability Design
- **Modular Architecture**: Separate analysis module for easy scaling
- **Thread Safety**: Concurrent processing support for multiple users
- **Database Ready**: Architecture designed to support future database integration (likely PostgreSQL with Drizzle ORM)

## Key Architectural Decisions

### MediaPipe Integration
- **Problem**: Need accurate pose estimation for exercise analysis
- **Solution**: MediaPipe Pose with optimized configuration for real-time processing
- **Rationale**: Industry-standard solution with good performance and accuracy balance

### AI Chat with RAG Pattern
- **Problem**: Need context-aware fitness coaching conversations
- **Solution**: Retrieval-Augmented Generation using analysis data as context
- **Rationale**: Provides personalized feedback based on actual performance data

### Modular Design
- **Problem**: Complex system with multiple responsibilities
- **Solution**: Separate modules for analysis, web handling, and AI integration
- **Rationale**: Maintainable code with clear separation of concerns

### File-based Storage
- **Problem**: Need to store videos and analysis results
- **Solution**: Local filesystem with structured JSON output
- **Rationale**: Simple implementation with clear path to database migration

### Error Handling Strategy
- **Problem**: Video processing can fail in various ways
- **Solution**: Comprehensive logging and graceful error handling
- **Rationale**: Provides debugging information while maintaining user experience