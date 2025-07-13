/**
 * ChatGPT-Inspired UI for AI Pushups Coach v2
 * Handles the main interface interactions
 */

class ChatGPTUI {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.currentSessionId = null;
        this.analysisData = null;
        this.isRealtimeActive = false;
    }

    initializeElements() {
        // Main elements
        this.mainSearchInput = document.getElementById('mainSearchInput');
        this.uploadVideoBtn = document.getElementById('uploadVideoBtn');
        this.realtimeModeBtn = document.getElementById('realtimeModeBtn');
        this.uploadModal = document.getElementById('uploadModal');
        this.closeModal = document.getElementById('closeModal');
        this.processingModal = document.getElementById('processingModal');
        this.resultsContainer = document.getElementById('resultsContainer');

        // Upload elements
        this.uploadZone = document.getElementById('uploadZone');
        this.uploadDefault = document.getElementById('uploadDefault');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.browseFiles = document.getElementById('browseFiles');
        this.videoUpload = document.getElementById('videoUpload');
        this.uploadFileName = document.getElementById('uploadFileName');

        // Processing elements
        this.processingMessage = document.getElementById('processingMessage');
        this.processingProgress = document.getElementById('processingProgress');

        // Quick action buttons
        this.quickActionButtons = document.querySelectorAll('.bg-gray-100');
    }

    bindEvents() {
        // Upload button click
        this.uploadVideoBtn?.addEventListener('click', () => this.openUploadModal());

        // Modal controls
        this.closeModal?.addEventListener('click', () => this.closeUploadModal());
        
        this.uploadModal?.addEventListener('click', (e) => {
            if (e.target === this.uploadModal) this.closeUploadModal();
        });

        // File upload
        this.browseFiles?.addEventListener('click', () => this.videoUpload?.click());
        this.videoUpload?.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));

        // Drag and drop
        this.uploadZone?.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadZone?.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadZone?.addEventListener('drop', (e) => this.handleDrop(e));

        // Quick actions
        this.quickActionButtons.forEach(btn => {
            btn.addEventListener('click', (e) => this.handleQuickAction(e.target.textContent));
        });

        // Search input (for future chat functionality)
        this.mainSearchInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleSearchQuery(e.target.value);
        });
    }

    // Upload Modal Functions
    openUploadModal() {
        this.uploadModal?.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        this.resetUploadState();
    }

    closeUploadModal() {
        this.uploadModal?.classList.add('hidden');
        document.body.style.overflow = 'auto';
        this.resetUploadState();
    }

    resetUploadState() {
        this.uploadDefault?.classList.remove('hidden');
        this.uploadProgress?.classList.add('hidden');
        this.uploadZone?.classList.remove('dragover');
        this.videoUpload.value = '';
        this.uploadFileName.textContent = '';
    }

    // File Upload Functions
    handleDragOver(e) {
        e.preventDefault();
        this.uploadZone?.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadZone?.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadZone?.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelect(files[0]);
        }
    }

    handleFileSelect(file) {
        if (!file) return;

        if (!this.isVideoFile(file)) {
            this.showError('Please select a valid video file (MP4, WebM, MOV, AVI)');
            return;
        }

        this.showUploadProgress(file);
        this.uploadFile(file);
    }

    isVideoFile(file) {
        const videoTypes = ['video/mp4', 'video/webm', 'video/mov', 'video/avi', 'video/quicktime'];
        return videoTypes.includes(file.type) || file.name.match(/\.(mp4|webm|mov|avi)$/i);
    }

    showUploadProgress(file) {
        this.uploadDefault?.classList.add('hidden');
        this.uploadProgress?.classList.remove('hidden');
        this.uploadFileName.textContent = file.name;
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('video', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                this.handleUploadSuccess(data);
            } else {
                const errorData = await response.json();
                this.showError(errorData.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Upload failed. Please try again.');
        }
    }

    handleUploadSuccess(data) {
        this.closeUploadModal();
        this.currentSessionId = data.session_id;
        this.analysisData = data.analysis_data;
        this.showProcessingModal();
        this.startProcessing(data.session_id);
    }

    // Processing Functions
    showProcessingModal() {
        this.processingModal?.classList.remove('hidden');
        this.updateProcessingMessage('Initializing analysis...');
        this.updateProcessingProgress(10);
    }

    hideProcessingModal() {
        this.processingModal?.classList.add('hidden');
    }

    updateProcessingMessage(message) {
        if (this.processingMessage) {
            this.processingMessage.textContent = message;
        }
    }

    updateProcessingProgress(percent) {
        if (this.processingProgress) {
            this.processingProgress.style.width = `${percent}%`;
        }
    }

    async startProcessing(sessionId) {
        const progressMessages = [
            'Analyzing video frames...',
            'Detecting pose landmarks...',
            'Counting pushup reps...',
            'Evaluating form quality...',
            'Generating AI insights...',
            'Preparing your report...'
        ];

        let messageIndex = 0;
        let progress = 10;

        // Update progress messages
        const messageInterval = setInterval(() => {
            if (messageIndex < progressMessages.length) {
                this.updateProcessingMessage(progressMessages[messageIndex]);
                progress += 15;
                this.updateProcessingProgress(Math.min(progress, 90));
                messageIndex++;
            }
        }, 2000);

        // Poll for completion
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/progress/${sessionId}`);
                if (response.ok) {
                    const progressData = await response.json();
                    
                    if (progressData.status === 'completed') {
                        clearInterval(messageInterval);
                        clearInterval(pollInterval);
                        this.updateProcessingMessage('Analysis complete!');
                        this.updateProcessingProgress(100);
                        
                        setTimeout(() => {
                            this.hideProcessingModal();
                            this.showResults();
                        }, 1000);
                    } else if (progressData.status === 'error') {
                        clearInterval(messageInterval);
                        clearInterval(pollInterval);
                        this.showError(progressData.message || 'Processing failed');
                        this.hideProcessingModal();
                    }
                }
            } catch (error) {
                console.error('Progress polling error:', error);
            }
        }, 1000);
    }

    // Results Functions
    showResults() {
        // Hide main content and show results
        document.querySelector('main')?.classList.add('hidden');
        this.resultsContainer?.classList.remove('hidden');
        
        // Load results content
        this.loadResultsContent();
    }

    async loadResultsContent() {
        if (!this.analysisData) return;

        try {
            // Redirect to results page
            window.location.href = `/results/${this.currentSessionId}`;
        } catch (error) {
            console.error('Error loading results:', error);
            this.showError('Failed to load results');
        }
    }

    goBack() {
        this.resultsContainer?.classList.add('hidden');
        document.querySelector('main')?.classList.remove('hidden');
        this.resetState();
    }

    resetState() {
        this.currentSessionId = null;
        this.analysisData = null;
        this.isRealtimeActive = false;
    }

    // Utility Functions
    handleQuickAction(action) {
        console.log('Quick action:', action);
        // Could open a chat interface or show specific information
    }

    handleSearchQuery(query) {
        console.log('Search query:', query);
        // Could open chat interface with the query
    }

    showError(message) {
        // Simple error notification
        alert(message);
    }
}

// Initialize the UI when DOM is loaded
let chatGPTUI;
document.addEventListener('DOMContentLoaded', () => {
    chatGPTUI = new ChatGPTUI();
}); 