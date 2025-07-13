/**
 * Real-time Analysis Module for AI Pushups Coach v2
 * Handles camera access, pose detection, and real-time feedback
 */

class RealtimeAnalyzer {
    constructor() {
        this.cameraStream = null;
        this.currentCamera = 'user'; // 'user' for front camera, 'environment' for back
        this.isActive = false;
        this.voiceEnabled = false;
        this.poseDetector = null;
        this.analysisInterval = null;
        this.repCount = 0;
        this.currentState = 'idle';
        this.formScore = 0;
        
        // Stats tracking
        this.stats = {
            repCount: 0,
            formScore: 0,
            currentState: 'idle',
            lastFeedback: null
        };
        
        this.initializePoseDetector();
    }

    async initializePoseDetector() {
        try {
            // Load MediaPipe Pose if available
            if (typeof MediaPipePose !== 'undefined') {
                this.poseDetector = new MediaPipePose({
                    locateFile: (file) => {
                        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
                    }
                });
                
                this.poseDetector.setOptions({
                    modelComplexity: 1,
                    smoothLandmarks: true,
                    enableSegmentation: false,
                    smoothSegmentation: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });
                
                this.poseDetector.onResults((results) => {
                    this.handlePoseResults(results);
                });
            }
        } catch (error) {
            console.warn('MediaPipe Pose not available, using backend processing:', error);
        }
    }

    async startCamera(cameraType = 'user') {
        try {
            // Stop existing stream if any
            await this.stopCamera();
            
            this.currentCamera = cameraType;
            
            // Get camera stream
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: cameraType,
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                },
                audio: false
            });
            
            this.cameraStream = stream;
            
            // Set video element source
            const videoElement = document.getElementById('cameraFeed');
            if (videoElement) {
                videoElement.srcObject = stream;
                videoElement.play();
            }
            
            this.isActive = true;
            
            // Start analysis loop
            this.startAnalysisLoop();
            
            return true;
            
        } catch (error) {
            console.error('Failed to start camera:', error);
            throw new Error(`Camera access failed: ${error.message}`);
        }
    }

    async stopCamera() {
        this.isActive = false;
        
        // Stop analysis loop
        if (this.analysisInterval) {
            clearInterval(this.analysisInterval);
            this.analysisInterval = null;
        }
        
        // Stop camera stream
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => track.stop());
            this.cameraStream = null;
        }
        
        // Reset video element
        const videoElement = document.getElementById('cameraFeed');
        if (videoElement) {
            videoElement.srcObject = null;
        }
        
        // Reset stats
        this.resetStats();
    }

    async switchCamera() {
        const newCamera = this.currentCamera === 'user' ? 'environment' : 'user';
        await this.startCamera(newCamera);
    }

    startAnalysisLoop() {
        // Send frames to backend for analysis
        this.analysisInterval = setInterval(() => {
            if (this.isActive && this.cameraStream) {
                this.captureAndSendFrame();
            }
        }, 1000 / 15); // 15 FPS for better performance
    }

    async captureAndSendFrame() {
        const videoElement = document.getElementById('cameraFeed');
        if (!videoElement || videoElement.videoWidth === 0) return;
        
        try {
            // Create canvas to capture frame
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            
            // Draw video frame to canvas
            context.drawImage(videoElement, 0, 0);
            
            // Convert to base64
            const base64Frame = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to backend
            await this.sendFrameToBackend(base64Frame);
            
        } catch (error) {
            console.error('Frame capture error:', error);
        }
    }

    async sendFrameToBackend(base64Frame) {
        try {
            const response = await fetch('/api/realtime_analyze_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frame: base64Frame,
                    enable_feedback: true  // Enable LLM feedback
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.handleBackendData(data);
            } else {
                console.error('Backend analysis failed:', response.status);
            }
        } catch (error) {
            console.error('Failed to send frame to backend:', error);
        }
    }

    handlePoseResults(results) {
        // Process MediaPipe pose results (if using client-side detection)
        if (results.poseLandmarks) {
            const poseData = this.extractPoseData(results.poseLandmarks);
            this.analyzePose(poseData);
        }
    }

    extractPoseData(landmarks) {
        // Extract relevant pose data from MediaPipe landmarks
        const poseData = {
            shoulders: {
                left: { x: landmarks[11].x, y: landmarks[11].y, z: landmarks[11].z },
                right: { x: landmarks[12].x, y: landmarks[12].y, z: landmarks[12].z }
            },
            elbows: {
                left: { x: landmarks[13].x, y: landmarks[13].y, z: landmarks[13].z },
                right: { x: landmarks[14].x, y: landmarks[14].y, z: landmarks[14].z }
            },
            wrists: {
                left: { x: landmarks[15].x, y: landmarks[15].y, z: landmarks[15].z },
                right: { x: landmarks[16].x, y: landmarks[16].y, z: landmarks[16].z }
            },
            hips: {
                left: { x: landmarks[23].x, y: landmarks[23].y, z: landmarks[23].z },
                right: { x: landmarks[24].x, y: landmarks[24].y, z: landmarks[24].z }
            },
            knees: {
                left: { x: landmarks[25].x, y: landmarks[25].y, z: landmarks[25].z },
                right: { x: landmarks[26].x, y: landmarks[26].y, z: landmarks[26].z }
            }
        };
        
        return poseData;
    }

    analyzePose(poseData) {
        // Analyze pose for pushup detection
        const analysis = {
            bodyAngle: this.calculateBodyAngle(poseData),
            elbowAngle: this.calculateElbowAngle(poseData),
            shoulderHipAlignment: this.calculateShoulderHipAlignment(poseData),
            state: this.determineState(poseData),
            formScore: this.calculateFormScore(poseData)
        };
        
        this.updateStats(analysis);
        this.updateUI(analysis);
    }

    calculateBodyAngle(poseData) {
        // Calculate body angle (shoulder-hip-knee)
        const shoulder = {
            x: (poseData.shoulders.left.x + poseData.shoulders.right.x) / 2,
            y: (poseData.shoulders.left.y + poseData.shoulders.right.y) / 2
        };
        
        const hip = {
            x: (poseData.hips.left.x + poseData.hips.right.x) / 2,
            y: (poseData.hips.left.y + poseData.hips.right.y) / 2
        };
        
        const knee = {
            x: (poseData.knees.left.x + poseData.knees.right.x) / 2,
            y: (poseData.knees.left.y + poseData.knees.right.y) / 2
        };
        
        return this.calculateAngle(shoulder, hip, knee);
    }

    calculateElbowAngle(poseData) {
        // Calculate elbow angle (wrist-elbow-shoulder)
        const wrist = {
            x: (poseData.wrists.left.x + poseData.wrists.right.x) / 2,
            y: (poseData.wrists.left.y + poseData.wrists.right.y) / 2
        };
        
        const elbow = {
            x: (poseData.elbows.left.x + poseData.elbows.right.x) / 2,
            y: (poseData.elbows.left.y + poseData.elbows.right.y) / 2
        };
        
        const shoulder = {
            x: (poseData.shoulders.left.x + poseData.shoulders.right.x) / 2,
            y: (poseData.shoulders.left.y + poseData.shoulders.right.y) / 2
        };
        
        return this.calculateAngle(wrist, elbow, shoulder);
    }

    calculateAngle(a, b, c) {
        // Calculate angle between three points
        const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
        let angle = Math.abs(radians * 180.0 / Math.PI);
        if (angle > 180.0) angle = 360 - angle;
        return angle;
    }

    calculateShoulderHipAlignment(poseData) {
        // Calculate shoulder-hip alignment
        const shoulderY = (poseData.shoulders.left.y + poseData.shoulders.right.y) / 2;
        const hipY = (poseData.hips.left.y + poseData.hips.right.y) / 2;
        return Math.abs(shoulderY - hipY);
    }

    determineState(poseData) {
        const bodyAngle = this.calculateBodyAngle(poseData);
        const elbowAngle = this.calculateElbowAngle(poseData);
        
        if (bodyAngle < 90) return 'invalid';
        if (elbowAngle > 145) return 'up';
        if (elbowAngle <= 135) return 'down';
        if (elbowAngle <= 145) return 'partial_up';
        return 'partial_down';
    }

    calculateFormScore(poseData) {
        const bodyAngle = this.calculateBodyAngle(poseData);
        const elbowAngle = this.calculateElbowAngle(poseData);
        const alignment = this.calculateShoulderHipAlignment(poseData);
        
        let score = 100;
        
        // Penalize poor body alignment
        if (bodyAngle < 170) score -= (180 - bodyAngle) * 2;
        
        // Penalize poor elbow angles
        if (elbowAngle < 80 || elbowAngle > 160) score -= 20;
        
        // Penalize poor shoulder-hip alignment
        if (alignment > 0.1) score -= alignment * 100;
        
        return Math.max(0, Math.min(100, score));
    }

    handleBackendData(data) {
        // Handle data from backend analysis
        const analysis = {
            state: data.state || 'idle',
            formScore: data.score || 0,
            landmarks: data.landmarks || null,
            // Extract all metrics from backend response
            body_alignment: data.body_alignment,
            elbow_angle: data.elbow_angle,
            posture_score: data.posture_score,
            depth_score: data.depth_score,
            rep_count: data.rep_count,
            message: data.message,
            issues: data.issues || []
        };
        
        this.updateStats(analysis);
        this.updateUI(analysis);
        
        // Handle rep counting logic
        this.handleRepCounting(analysis);
        
        // Handle feedback - use LLM feedback if available, otherwise fallback
        let feedback = data.feedback; // Backend LLM feedback
        if (!feedback) {
            feedback = this.generateFeedback(analysis);
        }
        
        if (feedback && feedback !== this.stats.lastFeedback) {
            this.showFeedback(feedback);
            this.stats.lastFeedback = feedback;
        }
        
        // Request additional LLM feedback periodically
        this.requestLLMFeedback(analysis);
    }

    handleRepCounting(analysis) {
        const currentState = analysis.state;
        const previousState = this.currentState;
        
        // Use backend rep count instead of frontend logic
        if (analysis.rep_count !== undefined) {
            this.repCount = analysis.rep_count;
            this.stats.repCount = this.repCount;
            this.updateStatsDisplay();
        }
        
        this.currentState = currentState;
    }

    updateStats(analysis) {
        this.stats = {
            ...this.stats,
            ...analysis
        };
        
        // Update UI elements
        this.updateStatsDisplay();
    }

    updateStatsDisplay() {
        const repCountEl = document.getElementById('repCount');
        const formScoreEl = document.getElementById('formScore');
        const currentStateEl = document.getElementById('currentState');
        const intensityBar = document.getElementById('intensityFill');
        
        if (repCountEl) repCountEl.textContent = this.stats.repCount;
        if (formScoreEl) formScoreEl.textContent = Math.round(this.stats.formScore);
        if (currentStateEl) currentStateEl.textContent = this.stats.currentState;
        // Animate intensity bar based on elbow angle (depth)
        if (intensityBar && this.stats.elbow_angle) {
            // Map elbow angle (60-160) to bar height (100%-10%)
            const angle = this.stats.elbow_angle;
            const percent = Math.max(10, Math.min(100, 100 - ((angle - 60) / 100) * 90));
            intensityBar.style.height = percent + '%';
            intensityBar.style.background = `linear-gradient(to top,#22c55e 0%,#f43f5e 100%)`;
        }
    }

    updateUI(analysis) {
        // Update pose overlay
        this.updatePoseOverlay(analysis);
        
        // Update feedback area
        this.updateFeedbackArea(analysis);
    }

    updatePoseOverlay(analysis) {
        const overlay = document.getElementById('poseOverlay');
        if (!overlay) return;
        overlay.innerHTML = '';
        const videoElem = document.getElementById('cameraFeed');
        if (!videoElem) return;
        // Get displayed size of the video element
        const elemW = videoElem.clientWidth;
        const elemH = videoElem.clientHeight;
        // Get backend frame size
        const frameW = analysis.frame_width || videoElem.videoWidth;
        const frameH = analysis.frame_height || videoElem.videoHeight;
        // Debug log
        console.log('[DEBUG] videoElem.videoWidth:', videoElem.videoWidth, 'videoElem.videoHeight:', videoElem.videoHeight);
        console.log('[DEBUG] clientWidth:', elemW, 'clientHeight:', elemH);
        console.log('[DEBUG] backend frame_width:', frameW, 'frame_height:', frameH);
        // Calculate scale and offset for object-fit: contain
        let scale, offsetX, offsetY;
        if (elemW / elemH > frameW / frameH) {
            // Letterbox left/right
            scale = elemH / frameH;
            offsetX = (elemW - frameW * scale) / 2;
            offsetY = 0;
        } else {
            // Letterbox top/bottom
            scale = elemW / frameW;
            offsetX = 0;
            offsetY = (elemH - frameH * scale) / 2;
        }
        overlay.style.position = 'absolute';
        overlay.style.left = videoElem.offsetLeft + 'px';
        overlay.style.top = videoElem.offsetTop + 'px';
        overlay.style.width = elemW + 'px';
        overlay.style.height = elemH + 'px';
        // Use displayed size for canvas
        const w = elemW;
        const h = elemH;
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        const ctx = canvas.getContext('2d');
        // Draw only keypoints as large red dots with names
        if (analysis.landmarks && frameW && frameH) {
            // Draw all 32 keypoints and full pose skeleton
            // Standard MediaPipe pose connections
            const POSE_CONNECTIONS = [
                [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
                [9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32]
            ];
            // Map: index -> {x, y}
            const scaledLandmarks = [];
            for (let i = 0; i < 33; i++) {
                const pt = analysis.landmarks[i] || {};
                const scaledX = pt.x * scale + offsetX;
                const mirroredX = w - scaledX;
                const scaledY = pt.y * scale + offsetY;
                scaledLandmarks.push({ x: mirroredX, y: scaledY });
            }
            // Draw lines
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.9;
            POSE_CONNECTIONS.forEach(([a, b]) => {
                if (scaledLandmarks[a] && scaledLandmarks[b]) {
                    ctx.beginPath();
                    ctx.moveTo(scaledLandmarks[a].x, scaledLandmarks[a].y);
                    ctx.lineTo(scaledLandmarks[b].x, scaledLandmarks[b].y);
                    ctx.stroke();
                }
            });
            // Draw keypoints
            scaledLandmarks.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 7, 0, 2 * Math.PI);
                ctx.fillStyle = '#06e6f7';
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
            ctx.globalAlpha = 1.0;
        }
        overlay.appendChild(canvas);
        // If recording, composite video + overlay
        if (this.isRecording && this.recordCtx && videoElem) {
            this.recordCtx.drawImage(videoElem, 0, 0, w, h);
            this.recordCtx.drawImage(canvas, 0, 0, w, h);
        }
    }

    drawPoseSkeleton(ctx, landmarks, w, h) {
        // Define MediaPipe pose connections (subset for clarity)
        const connections = [
            ['left_shoulder','right_shoulder'], ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
            ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
            ['left_shoulder','left_hip'], ['right_shoulder','right_hip'],
            ['left_hip','right_hip'], ['left_hip','left_knee'], ['left_knee','left_ankle'],
            ['right_hip','right_knee'], ['right_knee','right_ankle']
        ];
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 4;
        ctx.globalAlpha = 0.9;
        // Draw lines
        connections.forEach(([a, b]) => {
            if (landmarks[a] && landmarks[b]) {
                ctx.beginPath();
                ctx.moveTo(landmarks[a].x * w / 640, landmarks[a].y * h / 480);
                ctx.lineTo(landmarks[b].x * w / 640, landmarks[b].y * h / 480);
                ctx.stroke();
            }
        });
        // Draw keypoints
        Object.values(landmarks).forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * w / 640, pt.y * h / 480, 8, 0, 2 * Math.PI);
            ctx.fillStyle = '#06e6f7';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
        ctx.globalAlpha = 1.0;
    }

    drawElbowAngles(ctx, landmarks, w, h, metrics) {
        // Draw elbow angles near elbows
        if (!metrics) return;
        const left = landmarks['left_elbow'];
        const right = landmarks['right_elbow'];
        if (left && metrics.elbow_angle) {
            ctx.font = 'bold 22px Arial';
            ctx.fillStyle = '#2563eb';
            ctx.fillText(`${Math.round(metrics.elbow_angle)}°`, left.x * w / 640 + 18, left.y * h / 480);
        }
        if (right && metrics.elbow_angle) {
            ctx.font = 'bold 22px Arial';
            ctx.fillStyle = '#2563eb';
            ctx.fillText(`${Math.round(metrics.elbow_angle)}°`, right.x * w / 640 + 18, right.y * h / 480);
        }
    }

    updateFeedbackArea(analysis) {
        // Show only the latest feedback message in the message banner
        const messageBanner = document.getElementById('messageBanner');
        if (!messageBanner) return;
        const feedback = this.generateFeedback(analysis);
        if (feedback && feedback !== this.stats.lastFeedback) {
            messageBanner.textContent = feedback;
            this.stats.lastFeedback = feedback;
        }
    }

    generateFeedback(analysis) {
        const state = analysis.state;
        const formScore = analysis.formScore;
        
        if (state === 'invalid') {
            return 'Please get into pushup position';
        }
        
        if (formScore < 60) {
            return 'Focus on maintaining proper form';
        }
        
        if (state === 'down') {
            return 'Good depth! Now push back up';
        }
        
        if (state === 'up') {
            return 'Great! Maintain this position';
        }
        
        return null;
    }

    showFeedback(message) {
        const feedbackArea = document.getElementById('feedbackArea');
        if (!feedbackArea) return;
        
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'bg-blue-50 border-l-4 border-blue-400 p-4 mb-2 animate-pulse';
        feedbackDiv.innerHTML = `
            <div class="flex">
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="ml-3">
                    <p class="text-sm text-blue-700">${message}</p>
                </div>
            </div>
        `;
        
        feedbackArea.appendChild(feedbackDiv);
        feedbackArea.scrollTop = feedbackArea.scrollHeight;
        
        // Remove old feedback after 3 seconds
        setTimeout(() => {
            if (feedbackDiv.parentNode) {
                feedbackDiv.remove();
            }
        }, 3000);
    }
    
    async requestLLMFeedback(analysis) {
        // Request LLM feedback every 10 frames (about every 2-3 seconds)
        this.llmFeedbackCounter = (this.llmFeedbackCounter || 0) + 1;
        if (this.llmFeedbackCounter % 10 !== 0) return;
        
        try {
            const response = await fetch('/api/llm/realtime_feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    state: analysis.state,
                    form_score: analysis.formScore,
                    elbow_angle: analysis.elbow_angle,
                    body_alignment: analysis.body_alignment,
                    rep_count: analysis.rep_count,
                    issues: analysis.issues
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.feedback) {
                    this.showFeedback(data.feedback);
                }
            }
        } catch (error) {
            console.error('LLM feedback request failed:', error);
        }
    }

    async playAudioFeedback(text) {
        try {
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    voice: 'default'
                })
            });

            if (response.ok) {
                const data = await response.json();
                const audio = new Audio(data.audio_url);
                await audio.play();
            }
        } catch (error) {
            console.error('TTS error:', error);
        }
    }

    toggleVoice() {
        this.voiceEnabled = !this.voiceEnabled;
        return this.voiceEnabled;
    }

    resetStats() {
        this.stats = {
            repCount: 0,
            formScore: 0,
            currentState: 'idle',
            lastFeedback: null
        };
        this.repCount = 0;
        this.currentState = 'idle';
        this.updateStatsDisplay();
    }

    getStats() {
        return { ...this.stats };
    }

    // --- Recording logic ---
    startRecording() {
        if (this.mediaRecorder) return;
        const videoElem = document.getElementById('cameraFeed');
        const overlayElem = document.getElementById('poseOverlay');
        if (!videoElem) return;
        // Create a canvas to composite video + overlay
        this.recordCanvas = document.createElement('canvas');
        this.recordCanvas.width = videoElem.videoWidth;
        this.recordCanvas.height = videoElem.videoHeight;
        this.recordCtx = this.recordCanvas.getContext('2d');
        // Capture stream from canvas
        this.recordStream = this.recordCanvas.captureStream(30);
        this.mediaRecorder = new MediaRecorder(this.recordStream, { mimeType: 'video/webm' });
        this.recordedChunks = [];
        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.recordedChunks.push(e.data);
        };
        this.mediaRecorder.onstop = () => {
            const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'pushup_session.webm';
            a.click();
        };
        this.mediaRecorder.start();
        this.isRecording = true;
    }
    stopRecording() {
        if (this.mediaRecorder) {
            this.mediaRecorder.stop();
            this.mediaRecorder = null;
            this.isRecording = false;
        }
    }

    // --- Overlay drawing improvements ---
    updatePoseOverlay(analysis) {
        const overlay = document.getElementById('poseOverlay');
        if (!overlay) return;
        overlay.innerHTML = '';
        const videoElem = document.getElementById('cameraFeed');
        if (!videoElem) return;
        // Get displayed size of the video element
        const elemW = videoElem.clientWidth;
        const elemH = videoElem.clientHeight;
        // Get backend frame size
        const frameW = analysis.frame_width || videoElem.videoWidth;
        const frameH = analysis.frame_height || videoElem.videoHeight;
        // Debug log
        console.log('[DEBUG] videoElem.videoWidth:', videoElem.videoWidth, 'videoElem.videoHeight:', videoElem.videoHeight);
        console.log('[DEBUG] clientWidth:', elemW, 'clientHeight:', elemH);
        console.log('[DEBUG] backend frame_width:', frameW, 'frame_height:', frameH);
        // Calculate scale and offset for object-fit: contain
        let scale, offsetX, offsetY;
        if (elemW / elemH > frameW / frameH) {
            // Letterbox left/right
            scale = elemH / frameH;
            offsetX = (elemW - frameW * scale) / 2;
            offsetY = 0;
        } else {
            // Letterbox top/bottom
            scale = elemW / frameW;
            offsetX = 0;
            offsetY = (elemH - frameH * scale) / 2;
        }
        overlay.style.position = 'absolute';
        overlay.style.left = videoElem.offsetLeft + 'px';
        overlay.style.top = videoElem.offsetTop + 'px';
        overlay.style.width = elemW + 'px';
        overlay.style.height = elemH + 'px';
        // Use displayed size for canvas
        const w = elemW;
        const h = elemH;
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        const ctx = canvas.getContext('2d');
        // Draw only keypoints as large red dots with names
        if (analysis.landmarks && frameW && frameH) {
            // Draw all 32 keypoints and full pose skeleton
            // Standard MediaPipe pose connections
            const POSE_CONNECTIONS = [
                [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
                [9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32]
            ];
            // Map: index -> {x, y}
            const scaledLandmarks = [];
            for (let i = 0; i < 33; i++) {
                const pt = analysis.landmarks[i] || {};
                const scaledX = pt.x * scale + offsetX;
                const mirroredX = w - scaledX;
                const scaledY = pt.y * scale + offsetY;
                scaledLandmarks.push({ x: mirroredX, y: scaledY });
            }
            // Draw lines
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.9;
            POSE_CONNECTIONS.forEach(([a, b]) => {
                if (scaledLandmarks[a] && scaledLandmarks[b]) {
                    ctx.beginPath();
                    ctx.moveTo(scaledLandmarks[a].x, scaledLandmarks[a].y);
                    ctx.lineTo(scaledLandmarks[b].x, scaledLandmarks[b].y);
                    ctx.stroke();
                }
            });
            // Draw keypoints
            scaledLandmarks.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 7, 0, 2 * Math.PI);
                ctx.fillStyle = '#06e6f7';
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
            ctx.globalAlpha = 1.0;
        }
        overlay.appendChild(canvas);
        // If recording, composite video + overlay
        if (this.isRecording && this.recordCtx && videoElem) {
            this.recordCtx.drawImage(videoElem, 0, 0, w, h);
            this.recordCtx.drawImage(canvas, 0, 0, w, h);
        }
    }

    drawPoseSkeleton(ctx, landmarks, w, h) {
        // Define MediaPipe pose connections (subset for clarity)
        const connections = [
            ['left_shoulder','right_shoulder'], ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
            ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
            ['left_shoulder','left_hip'], ['right_shoulder','right_hip'],
            ['left_hip','right_hip'], ['left_hip','left_knee'], ['left_knee','left_ankle'],
            ['right_hip','right_knee'], ['right_knee','right_ankle']
        ];
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 4;
        ctx.globalAlpha = 0.9;
        // Draw lines
        connections.forEach(([a, b]) => {
            if (landmarks[a] && landmarks[b]) {
                ctx.beginPath();
                ctx.moveTo(landmarks[a].x * w / 640, landmarks[a].y * h / 480);
                ctx.lineTo(landmarks[b].x * w / 640, landmarks[b].y * h / 480);
                ctx.stroke();
            }
        });
        // Draw keypoints
        Object.values(landmarks).forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * w / 640, pt.y * h / 480, 8, 0, 2 * Math.PI);
            ctx.fillStyle = '#06e6f7';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
        ctx.globalAlpha = 1.0;
    }

    drawElbowAngles(ctx, landmarks, w, h, metrics) {
        // Draw elbow angles near elbows
        if (!metrics) return;
        const left = landmarks['left_elbow'];
        const right = landmarks['right_elbow'];
        if (left && metrics.elbow_angle) {
            ctx.font = 'bold 22px Arial';
            ctx.fillStyle = '#2563eb';
            ctx.fillText(`${Math.round(metrics.elbow_angle)}°`, left.x * w / 640 + 18, left.y * h / 480);
        }
        if (right && metrics.elbow_angle) {
            ctx.font = 'bold 22px Arial';
            ctx.fillStyle = '#2563eb';
            ctx.fillText(`${Math.round(metrics.elbow_angle)}°`, right.x * w / 640 + 18, right.y * h / 480);
        }
    }

    // --- Intensity bar animation ---
    updateStatsDisplay() {
        const repCountEl = document.getElementById('repCount');
        const formScoreEl = document.getElementById('formScore');
        const currentStateEl = document.getElementById('currentState');
        const intensityBar = document.getElementById('intensityFill');
        if (repCountEl) repCountEl.textContent = this.stats.repCount;
        if (formScoreEl) formScoreEl.textContent = Math.round(this.stats.formScore);
        if (currentStateEl) currentStateEl.textContent = this.stats.state || this.stats.currentState;
        // Animate intensity bar based on elbow angle (depth)
        if (intensityBar && this.stats.elbow_angle) {
            // Map elbow angle (60-160) to bar height (100%-10%)
            const angle = this.stats.elbow_angle;
            const percent = Math.max(10, Math.min(100, 100 - ((angle - 60) / 100) * 90));
            intensityBar.style.height = percent + '%';
            intensityBar.style.background = `linear-gradient(to top,#22c55e 0%,#f43f5e 100%)`;
        }
    }

    // --- Improved rep counting logic ---
    handleRepCounting(analysis) {
        const currentState = analysis.state;
        const previousState = this.currentState;
        // Debug log
        console.log(`[RepCount] Prev: ${previousState}, Curr: ${currentState}, Alignment: ${analysis.body_alignment}`);
        // Only count rep if transition is from down to up and body alignment is good
        if (previousState === 'down' && currentState === 'up' && analysis.body_alignment > 160) {
            this.repCount++;
            this.stats.repCount = this.repCount;
            this.updateStatsDisplay();
            console.log(`[RepCount] Incremented! Total: ${this.repCount}`);
        }
        this.currentState = currentState;
    }

    // --- Add record session button to UI ---
    static addRecordButton() {
        let btn = document.getElementById('recordSessionBtn');
        if (!btn) {
            btn = document.createElement('button');
            btn.id = 'recordSessionBtn';
            btn.textContent = 'Record Session';
            btn.style.position = 'absolute';
            btn.style.bottom = '40px';
            btn.style.right = '40px';
            btn.style.background = '#2563eb';
            btn.style.color = '#fff';
            btn.style.padding = '14px 28px';
            btn.style.borderRadius = '1rem';
            btn.style.fontWeight = '600';
            btn.style.fontSize = '1.1rem';
            btn.style.boxShadow = '0 2px 8px rgba(0,0,0,0.10)';
            btn.style.border = 'none';
            btn.style.outline = 'none';
            btn.style.cursor = 'pointer';
            btn.style.transition = 'background 0.2s';
            document.body.appendChild(btn);
        }
        btn.onclick = () => {
            const analyzer = window.realtimeAnalyzerInstance;
            if (!analyzer) return;
            if (!analyzer.isRecording) {
                analyzer.startRecording();
                btn.textContent = 'Stop Recording';
                btn.style.background = '#dc2626';
            } else {
                analyzer.stopRecording();
                btn.textContent = 'Record Session';
                btn.style.background = '#2563eb';
            }
        };
    }
}

// Export for use in other modules
window.RealtimeAnalyzer = RealtimeAnalyzer; 