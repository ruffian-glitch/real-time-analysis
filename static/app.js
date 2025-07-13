/**
 * Main Application JavaScript for AI Pushups Coach v2
 * Handles general app functionality and utilities
 */

class PushupsCoachApp {
    constructor() {
        this.currentSessionId = null;
        this.analysisData = null;
        this.realtimeAnalyzer = null;
        this.chatInterface = null;
        this.liveProcessor = null;
        
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.initializeComponents();
        this.setupErrorHandling();
    }

    setupEventListeners() {
        // Global error handling
        window.addEventListener('error', (e) => this.handleGlobalError(e));
        window.addEventListener('unhandledrejection', (e) => this.handlePromiseRejection(e));
        
        // Page visibility changes
        document.addEventListener('visibilitychange', () => this.handleVisibilityChange());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }

    initializeComponents() {
        // Initialize real-time analyzer if on real-time page
        if (document.getElementById('cameraFeed')) {
            this.realtimeAnalyzer = new RealtimeAnalyzer();
        }
        
        // Initialize chat interface if on results page
        if (document.getElementById('chatContainer')) {
            this.chatInterface = new ChatInterface();
        }
        
        // Initialize live processor if on live page
        if (document.getElementById('liveVideoContainer')) {
            this.liveProcessor = new LiveVideoProcessor();
        }
    }

    setupErrorHandling() {
        // Override console.error to also show user-friendly messages
        const originalError = console.error;
        console.error = (...args) => {
            originalError.apply(console, args);
            this.showUserFriendlyError(args.join(' '));
        };
    }

    handleGlobalError(error) {
        console.error('Global error:', error);
        this.showUserFriendlyError('An unexpected error occurred. Please refresh the page.');
    }

    handlePromiseRejection(event) {
        console.error('Unhandled promise rejection:', event.reason);
        this.showUserFriendlyError('A network error occurred. Please check your connection.');
    }

    handleVisibilityChange() {
        if (document.hidden) {
            // Page is hidden, pause real-time analysis if active
            if (this.realtimeAnalyzer && this.realtimeAnalyzer.isActive) {
                this.realtimeAnalyzer.pauseAnalysis();
            }
        } else {
            // Page is visible, resume analysis if it was paused
            if (this.realtimeAnalyzer && this.realtimeAnalyzer.isActive) {
                this.realtimeAnalyzer.resumeAnalysis();
            }
        }
    }

    handleKeyboardShortcuts(event) {
        // Ctrl/Cmd + Enter to submit forms
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            const activeElement = document.activeElement;
            if (activeElement && activeElement.tagName === 'TEXTAREA') {
                event.preventDefault();
                this.submitActiveForm();
            }
        }
        
        // Escape to close modals
        if (event.key === 'Escape') {
            this.closeActiveModals();
        }
        
        // Space to pause/resume video
        if (event.key === ' ' && event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {
            event.preventDefault();
            this.toggleVideoPlayback();
        }
    }

    submitActiveForm() {
        const activeElement = document.activeElement;
        if (activeElement && activeElement.form) {
            const submitButton = activeElement.form.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.click();
            }
        }
    }

    closeActiveModals() {
        const modals = document.querySelectorAll('.modal-overlay:not(.hidden)');
        modals.forEach(modal => {
            const closeButton = modal.querySelector('[id*="close"]');
            if (closeButton) {
                closeButton.click();
            }
        });
    }

    toggleVideoPlayback() {
        const videos = document.querySelectorAll('video');
        videos.forEach(video => {
            if (video.paused) {
                video.play();
            } else {
                video.pause();
            }
        });
    }

    showUserFriendlyError(message) {
        // Create or update error notification
        let notification = document.getElementById('errorNotification');
        
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'errorNotification';
            notification.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 transform transition-transform duration-300';
            notification.innerHTML = `
                <div class="flex items-center space-x-2">
                    <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <span>${message}</span>
                    <button onclick="this.parentElement.parentElement.remove()" class="ml-2 hover:text-gray-200">
                        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                        </svg>
                    </button>
                </div>
            `;
            document.body.appendChild(notification);
        } else {
            notification.querySelector('span').textContent = message;
        }
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification && notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    showSuccessMessage(message) {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 transform transition-transform duration-300';
        notification.innerHTML = `
            <div class="flex items-center space-x-2">
                <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                </svg>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-2 hover:text-gray-200">
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                    </svg>
                </button>
            </div>
        `;
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification && notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    }

    // Utility functions
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // API helpers
    async apiRequest(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const finalOptions = {
            ...defaultOptions,
            ...options,
            headers: {
                ...defaultOptions.headers,
                ...options.headers,
            },
        };

        try {
            const response = await fetch(url, finalOptions);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Session management
    setSessionData(sessionId, data) {
        this.currentSessionId = sessionId;
        this.analysisData = data;
        
        // Store in sessionStorage for persistence
        sessionStorage.setItem('pushupsCoachSession', JSON.stringify({
            sessionId,
            data,
            timestamp: Date.now()
        }));
    }

    getSessionData() {
        const stored = sessionStorage.getItem('pushupsCoachSession');
        if (stored) {
            const parsed = JSON.parse(stored);
            // Check if session is still valid (24 hours)
            if (Date.now() - parsed.timestamp < 24 * 60 * 60 * 1000) {
                this.currentSessionId = parsed.sessionId;
                this.analysisData = parsed.data;
                return parsed;
            }
        }
        return null;
    }

    clearSessionData() {
        this.currentSessionId = null;
        this.analysisData = null;
        sessionStorage.removeItem('pushupsCoachSession');
    }

    // Navigation helpers
    navigateToResults(sessionId) {
        window.location.href = `/results/${sessionId}`;
    }

    navigateToHome() {
        window.location.href = '/';
    }

    // Loading states
    showLoading(element, message = 'Loading...') {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (!element) return;
        
        element.disabled = true;
        element.dataset.originalText = element.textContent;
        element.innerHTML = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            ${message}
        `;
    }

    hideLoading(element) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (!element) return;
        
        element.disabled = false;
        if (element.dataset.originalText) {
            element.textContent = element.dataset.originalText;
            delete element.dataset.originalText;
        }
    }

    // Video utilities
    createVideoPlayer(videoElement, options = {}) {
        const player = {
            element: videoElement,
            isPlaying: false,
            currentTime: 0,
            duration: 0,
            
            play() {
                this.element.play();
                this.isPlaying = true;
            },
            
            pause() {
                this.element.pause();
                this.isPlaying = false;
            },
            
            seek(time) {
                this.element.currentTime = time;
            },
            
            setPlaybackRate(rate) {
                this.element.playbackRate = rate;
            },
            
            onTimeUpdate(callback) {
                this.element.addEventListener('timeupdate', callback);
            },
            
            onEnded(callback) {
                this.element.addEventListener('ended', callback);
            },
            
            onLoadedMetadata(callback) {
                this.element.addEventListener('loadedmetadata', callback);
            }
        };
        
        // Initialize
        if (options.autoplay) player.play();
        if (options.playbackRate) player.setPlaybackRate(options.playbackRate);
        
        return player;
    }

    // Chart utilities
    createChart(canvas, data, options = {}) {
        const ctx = canvas.getContext('2d');
        
        // Simple chart implementation
        const chart = {
            canvas,
            ctx,
            data,
            options: {
                width: canvas.width,
                height: canvas.height,
                padding: 20,
                ...options
            },
            
            draw() {
                this.clear();
                this.drawAxes();
                this.drawData();
                this.drawLabels();
            },
            
            clear() {
                this.ctx.clearRect(0, 0, this.options.width, this.options.height);
            },
            
            drawAxes() {
                this.ctx.strokeStyle = '#e5e7eb';
                this.ctx.lineWidth = 1;
                
                // X-axis
                this.ctx.beginPath();
                this.ctx.moveTo(this.options.padding, this.options.height - this.options.padding);
                this.ctx.lineTo(this.options.width - this.options.padding, this.options.height - this.options.padding);
                this.ctx.stroke();
                
                // Y-axis
                this.ctx.beginPath();
                this.ctx.moveTo(this.options.padding, this.options.padding);
                this.ctx.lineTo(this.options.padding, this.options.height - this.options.padding);
                this.ctx.stroke();
            },
            
            drawData() {
                if (!this.data || this.data.length === 0) return;
                
                this.ctx.strokeStyle = '#3b82f6';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                
                const maxValue = Math.max(...this.data);
                const minValue = Math.min(...this.data);
                const range = maxValue - minValue || 1;
                
                this.data.forEach((value, index) => {
                    const x = this.options.padding + (index / (this.data.length - 1)) * (this.options.width - 2 * this.options.padding);
                    const y = this.options.height - this.options.padding - ((value - minValue) / range) * (this.options.height - 2 * this.options.padding);
                    
                    if (index === 0) {
                        this.ctx.moveTo(x, y);
                    } else {
                        this.ctx.lineTo(x, y);
                    }
                });
                
                this.ctx.stroke();
            },
            
            drawLabels() {
                this.ctx.fillStyle = '#6b7280';
                this.ctx.font = '12px Arial';
                this.ctx.textAlign = 'center';
                
                // X-axis labels
                this.data.forEach((value, index) => {
                    const x = this.options.padding + (index / (this.data.length - 1)) * (this.options.width - 2 * this.options.padding);
                    const y = this.options.height - this.options.padding + 15;
                    this.ctx.fillText(index + 1, x, y);
                });
            }
        };
        
        chart.draw();
        return chart;
    }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new PushupsCoachApp();
    // Update metrics display from analysisData if present
    if (typeof analysisData !== 'undefined' && analysisData) {
        const pushupCountEl = document.getElementById('pushupCount');
        const formScoreEl = document.getElementById('formScore');
        const durationEl = document.getElementById('duration');
        const avgRepTimeEl = document.getElementById('avgRepTime');
        if (pushupCountEl) pushupCountEl.textContent = analysisData.rep_count ?? 0;
        if (formScoreEl) formScoreEl.textContent = (analysisData.form_score ?? 0) + '%';
        if (durationEl) durationEl.textContent = (analysisData.duration ? analysisData.duration.toFixed(1) : 0) + 's';
        if (avgRepTimeEl) avgRepTimeEl.textContent = (analysisData.avg_rep_time ? analysisData.avg_rep_time.toFixed(1) : 0) + 's';
    }
    const realtimeBtn = document.getElementById('realtimeModeBtn');
    if (!realtimeBtn) return;

    let stream = null;
    let modal = null;
    let videoElem = null;
    let repOverlay = null;
    let statusOverlay = null;
    let intensityBar = null;
    let messageBanner = null;
    let angleGauge = null;
    let formScoreBadge = null;
    let stopBtn = null;
    let animationFrameId = null;

    function cleanupCameraUI() {
        console.log('[Realtime] Cleaning up camera UI');
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        if (modal) modal.remove();
        modal = null;
        videoElem = null;
        repOverlay = null;
        statusOverlay = null;
        intensityBar = null;
        messageBanner = null;
        angleGauge = null;
        formScoreBadge = null;
        stopBtn = null;
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
    }

    function createGaugeSVG(value, max, color) {
        const radius = 28, stroke = 6, norm = Math.max(0, Math.min(1, value / max));
        const circ = 2 * Math.PI * radius;
        const offset = circ * (1 - norm);
        return `<svg width="64" height="64" viewBox="0 0 64 64">
            <circle cx="32" cy="32" r="${radius}" stroke="#e5e7eb" stroke-width="${stroke}" fill="none"/>
            <circle cx="32" cy="32" r="${radius}" stroke="${color}" stroke-width="${stroke}" fill="none" stroke-linecap="round" stroke-dasharray="${circ}" stroke-dashoffset="${offset}"/>
            <text x="32" y="38" text-anchor="middle" font-size="20" font-weight="bold" fill="${color}">${Math.round(value)}</text>
        </svg>`;
    }

    async function startCameraUI() {
        console.log('[Realtime] Start Real-Time Analysis button clicked');
        
        // Create modal container
        modal = document.createElement('div');
        modal.id = 'realtimeTestModal';
        modal.setAttribute('style', 'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.7);');
        modal.innerHTML = `
        <div style="position:relative;width:100vw;height:100vh;max-width:100vw;max-height:100vh;background:#fff;border-radius:0;box-shadow:0 8px 40px rgba(0,0,0,0.2);overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <video id="cameraFeed" autoplay playsinline muted style="position:absolute;left:0;top:0;width:100%;height:100%;object-fit:contain;background:#000;"></video>
            <div id="poseOverlay" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;"></div>
            <div id="intensityBar" style="position:absolute;left:0;top:0;height:100%;width:32px;display:flex;flex-direction:column;justify-content:flex-end;">
                <div id="intensityFill" style="width:100%;border-bottom-right-radius:1rem;border-bottom-left-radius:1rem;height:20%;background:linear-gradient(to top,#22c55e 0%,#f43f5e 100%);box-shadow:0 0 12px 2px #22c55e55;transition:height 0.3s;"></div>
            </div>
            <div id="repStatusOverlay" style="position:absolute;top:32px;left:40px;display:flex;flex-direction:column;align-items:flex-start;gap:8px;background:rgba(255,255,255,0.92);border-radius:1rem;padding:20px 28px;box-shadow:0 4px 24px rgba(0,0,0,0.10);backdrop-filter:blur(4px);">
                <div id="repCount" style="font-size:2.2rem;font-weight:700;color:#2563eb;text-shadow:0 1px 4px #0001;">0</div>
                <div id="currentState" style="font-size:1.1rem;font-weight:500;color:#16a34a;">Idle</div>
            </div>
            <div id="messageBanner" style="position:absolute;top:32px;left:50%;transform:translateX(-50%);width:38vw;max-width:440px;background:rgba(255,255,255,0.96);border-radius:1rem;box-shadow:0 4px 24px rgba(0,0,0,0.10);padding:16px 28px;text-align:center;font-size:1.05rem;font-weight:600;color:#1e293b;border:2px solid #bfdbfe;backdrop-filter:blur(4px);"></div>
            <div id="feedbackArea" style="position:absolute;top:120px;left:50%;transform:translateX(-50%);width:38vw;max-width:440px;max-height:200px;overflow-y:auto;"></div>
            <div style="position:absolute;bottom:40px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:28px;">
                <div id="angleGauge" style="background:rgba(255,255,255,0.96);border-radius:50%;box-shadow:0 2px 8px #0001;padding:8px;min-width:64px;min-height:64px;display:flex;align-items:center;justify-content:center;"></div>
                <div id="formScore" style="background:linear-gradient(90deg,#2563eb 0%,#8b5cf6 100%);color:#fff;font-size:1.05rem;font-weight:600;border-radius:2rem;padding:14px 28px;box-shadow:0 2px 8px #0001;">Form: 0</div>
            </div>
            <button id="stopCameraBtn" style="position:absolute;top:32px;right:40px;background:#dc2626;color:#fff;padding:14px 28px;border-radius:1rem;font-weight:600;font-size:1.1rem;box-shadow:0 2px 8px rgba(0,0,0,0.10);border:none;outline:none;cursor:pointer;transition:background 0.2s;">Stop Analysis</button>
        </div>
        `;
        document.body.appendChild(modal);
        console.log('[Realtime] Modal appended to DOM');
        
        // Get references to elements
        videoElem = modal.querySelector('#cameraFeed');
        videoElem.style.transform = 'scaleX(-1)'; // Flip video horizontally
        repOverlay = modal.querySelector('#repCount');
        statusOverlay = modal.querySelector('#currentState');
        intensityBar = modal.querySelector('#intensityFill');
        messageBanner = modal.querySelector('#messageBanner');
        angleGauge = modal.querySelector('#angleGauge');
        formScoreBadge = modal.querySelector('#formScore');
        stopBtn = modal.querySelector('#stopCameraBtn');

        // Initialize RealtimeAnalyzer
        const realtimeAnalyzer = new RealtimeAnalyzer();
        window.realtimeAnalyzerInstance = realtimeAnalyzer;
        RealtimeAnalyzer.addRecordButton();
        
        // Start camera and analysis
        try {
            await realtimeAnalyzer.startCamera('user');
            console.log('[Realtime] Camera and analysis started');
            
            // Update message banner
            if (messageBanner) {
                messageBanner.textContent = 'Real-time analysis active - Get into pushup position!';
            }
            
        } catch (err) {
            cleanupCameraUI();
            app.showUserFriendlyError('Camera access denied or not available. Please allow camera access to use real-time analysis.');
            console.error('[Realtime] Camera access error:', err);
            return;
        }

        // Stop button handler
        stopBtn.onclick = () => {
            console.log('[Realtime] Stop Analysis clicked');
            realtimeAnalyzer.stopCamera();
            cleanupCameraUI();
        };
    }

    realtimeBtn.addEventListener('click', startCameraUI);
});

// Export for use in other modules
window.PushupsCoachApp = PushupsCoachApp;

/**
 * Live Video Processor Class
 * Handles real-time video processing from camera
 */
class LiveVideoProcessor {
    constructor() {
        this.isActive = false;
        this.currentSessionId = null;
        this.cameraIndex = 0;
        this.analysisResults = [];
        this.updateInterval = null;
        
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.container = document.getElementById('liveVideoContainer');
        this.videoElement = document.getElementById('liveVideo');
        this.startButton = document.getElementById('startLiveBtn');
        this.stopButton = document.getElementById('stopLiveBtn');
        this.statusElement = document.getElementById('liveStatus');
        this.analysisElement = document.getElementById('liveAnalysis');
        this.cameraSelect = document.getElementById('cameraSelect');
        
        if (!this.container) {
            console.error('Live video container not found');
            return;
        }
    }

    setupEventListeners() {
        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.startProcessing());
        }
        
        if (this.stopButton) {
            this.stopButton.addEventListener('click', () => this.stopProcessing());
        }
        
        if (this.cameraSelect) {
            this.cameraSelect.addEventListener('change', (e) => {
                this.cameraIndex = parseInt(e.target.value);
            });
        }
    }

    async startProcessing() {
        try {
            if (this.isActive) {
                app.showUserFriendlyError('Live processing is already active');
                return;
            }

            app.showLoading(this.startButton, 'Starting...');
            
            const response = await app.apiRequest('/live/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    camera_index: this.cameraIndex
                })
            });

            if (response.success) {
                this.isActive = true;
                this.currentSessionId = response.session_id;
                this.updateStatus('Live processing active', 'success');
                this.startResultsPolling();
                app.showSuccessMessage('Live processing started successfully');
                
                // Update UI
                if (this.startButton) this.startButton.disabled = true;
                if (this.stopButton) this.stopButton.disabled = false;
            } else {
                throw new Error(response.error || 'Failed to start live processing');
            }
            
        } catch (error) {
            console.error('Error starting live processing:', error);
            app.showUserFriendlyError('Failed to start live processing: ' + error.message);
        } finally {
            app.hideLoading(this.startButton);
        }
    }

    async stopProcessing() {
        try {
            if (!this.isActive) {
                app.showUserFriendlyError('Live processing is not active');
                return;
            }

            app.showLoading(this.stopButton, 'Stopping...');
            
            const response = await app.apiRequest('/live/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.success) {
                this.isActive = false;
                this.currentSessionId = null;
                this.updateStatus('Live processing stopped', 'info');
                this.stopResultsPolling();
                app.showSuccessMessage('Live processing stopped');
                
                // Update UI
                if (this.startButton) this.startButton.disabled = false;
                if (this.stopButton) this.stopButton.disabled = true;
            } else {
                throw new Error(response.error || 'Failed to stop live processing');
            }
            
        } catch (error) {
            console.error('Error stopping live processing:', error);
            app.showUserFriendlyError('Failed to stop live processing: ' + error.message);
        } finally {
            app.hideLoading(this.stopButton);
        }
    }

    startResultsPolling() {
        // Poll for live analysis results every 2 seconds
        this.updateInterval = setInterval(async () => {
            if (!this.isActive) return;
            
            try {
                const response = await app.apiRequest('/live/status');
                if (response.success && response.is_processing) {
                    // Update analysis display with latest results
                    this.updateAnalysisDisplay();
                }
            } catch (error) {
                console.error('Error polling live status:', error);
            }
        }, 2000);
    }

    stopResultsPolling() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    updateStatus(message, type = 'info') {
        if (this.statusElement) {
            this.statusElement.textContent = message;
            this.statusElement.className = `text-sm font-medium ${
                type === 'success' ? 'text-green-600' : 
                type === 'error' ? 'text-red-600' : 
                'text-blue-600'
            }`;
        }
    }

    updateAnalysisDisplay() {
        if (!this.analysisElement) return;
        
        // This would be updated with real analysis data from the backend
        const mockAnalysis = {
            form_score: Math.floor(Math.random() * 100),
            state: ['up', 'down', 'transition'][Math.floor(Math.random() * 3)],
            rep_count: Math.floor(Math.random() * 10),
            confidence: (Math.random() * 0.5 + 0.5).toFixed(2)
        };
        
        this.analysisElement.innerHTML = `
            <div class="grid grid-cols-2 gap-4">
                <div class="bg-white p-4 rounded-lg shadow">
                    <h3 class="text-lg font-semibold text-gray-800">Form Score</h3>
                    <p class="text-3xl font-bold text-blue-600">${mockAnalysis.form_score}%</p>
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h3 class="text-lg font-semibold text-gray-800">Current State</h3>
                    <p class="text-xl font-semibold text-green-600 capitalize">${mockAnalysis.state}</p>
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h3 class="text-lg font-semibold text-gray-800">Rep Count</h3>
                    <p class="text-3xl font-bold text-purple-600">${mockAnalysis.rep_count}</p>
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h3 class="text-lg font-semibold text-gray-800">Confidence</h3>
                    <p class="text-xl font-semibold text-orange-600">${mockAnalysis.confidence}</p>
                </div>
            </div>
        `;
    }

    // Cleanup method
    destroy() {
        this.stopProcessing();
        this.stopResultsPolling();
    }
}

// Export for use in other modules
window.LiveVideoProcessor = LiveVideoProcessor; 