# AI Fitness Coach - Replit Architecture Guide

## Overview

This is a full-stack AI Fitness Coach application that allows users to upload workout videos and receive detailed analysis of their form and performance. The system uses computer vision to analyze body movements and provides personalized feedback through an AI-powered chat interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **UI Library**: Radix UI components with shadcn/ui styling system
- **Styling**: Tailwind CSS with custom CSS variables for theming
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: TanStack Query for server state management
- **Charts**: Recharts for data visualization

### Backend Architecture
- **Runtime**: Node.js with Express.js
- **Language**: TypeScript with ES modules
- **AI Analysis**: Python with MediaPipe for computer vision
- **AI Chat**: Gemini 2.5 Pro for intelligent coaching responses
- **Database**: In-memory storage for fast prototyping
- **File Upload**: Multer for handling video uploads
- **Data Validation**: Zod schemas for runtime type checking

## Key Components

### Database Schema
- **Users**: Basic user authentication (username, password)
- **Videos**: Stores uploaded video metadata (filename, drill type, processing status)
- **Analysis Results**: JSON storage of computer vision analysis data
- **Chat Messages**: Conversational AI responses linked to video analysis

### Frontend Components
- **DrillSelection**: UI for selecting workout types (push-ups, squats, etc.)
- **VideoUpload**: File upload interface with validation
- **ProcessingSection**: Progress tracking during video analysis
- **ResultsSection**: Display of analysis results with video player
- **ChatInterface**: AI-powered chat for asking questions about performance
- **PerformanceChart**: Data visualization of form scores over time

### API Endpoints
- `POST /api/videos/upload` - Upload video files
- `POST /api/videos/:id/analyze` - Trigger video analysis
- `GET /api/videos/:id/analysis` - Retrieve analysis results
- `POST /api/videos/:id/chat` - Chat with AI about video performance

## Data Flow

1. **Upload Phase**: User selects drill type and uploads video file
2. **Processing Phase**: Backend processes video using real MediaPipe computer vision analysis
3. **Analysis Phase**: Python module analyzes video frame-by-frame using pose estimation to generate:
   - Overall metrics (total reps, form score, consistency)
   - Per-rep breakdown with timestamps and form scores
   - Drill-specific measurements (angles, depth, alignment)
4. **Results Phase**: Frontend displays video player with real analysis visualization
5. **Chat Phase**: Gemini 2.5 Pro AI provides intelligent coaching responses with video navigation

## External Dependencies

### Database
- **Neon Database**: Serverless PostgreSQL for production
- **Drizzle ORM**: Type-safe database queries and migrations
- **Connection**: Uses `@neondatabase/serverless` for edge-compatible connections

### UI Components
- **Radix UI**: Accessible component primitives
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide Icons**: Consistent icon library
- **Recharts**: Chart library for performance visualization

### Development Tools
- **Vite**: Fast build tool with HMR
- **TypeScript**: Static typing for better developer experience
- **ESLint/Prettier**: Code formatting and linting (implied by structure)

## Deployment Strategy

### Build Process
- **Frontend**: Vite builds to `dist/public` directory
- **Backend**: ESBuild bundles server code to `dist/index.js`
- **Database**: Drizzle migrations in `migrations/` directory

### Environment Configuration
- **Development**: Uses `tsx` for TypeScript execution
- **Production**: Compiled JavaScript with `NODE_ENV=production`
- **Database**: Requires `DATABASE_URL` environment variable

### File Structure
```
├── client/          # Frontend React application
├── server/          # Backend Express application
├── shared/          # Shared TypeScript definitions
├── migrations/      # Database migration files
└── uploads/         # Video file storage (development)
```

### Key Features
- **Real Computer Vision**: MediaPipe pose estimation for accurate form analysis
- **AI-Powered Coaching**: Gemini 2.5 Pro provides intelligent, personalized feedback
- **Video Navigation**: AI can direct users to specific moments in their workout videos
- **Real-time Analysis**: Frame-by-frame pose tracking with angle calculations
- **Responsive Design**: Mobile-first UI with Tailwind CSS
- **Type Safety**: End-to-end TypeScript with shared schemas
- **Hybrid Architecture**: Python computer vision + Node.js server + React frontend

The application is structured as a monorepo with clear separation between frontend, backend, and shared code, making it easy to develop and deploy as a cohesive full-stack application.

## Recent Updates (January 2025)
- ✅ Implemented real computer vision analysis using MediaPipe pose estimation
- ✅ Integrated Gemini 2.5 Pro for intelligent AI coaching and chat responses
- ✅ Created comprehensive analysis module supporting all 6 exercise types
- ✅ Added video navigation commands from AI chat responses
- ✅ Built hybrid Python/Node.js architecture for optimal performance