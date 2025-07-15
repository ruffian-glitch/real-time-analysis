let analysisData = null;
let processingStartTime = null;

let selectedDrill = null;
let selectedFile = null;

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const sendChatBtn = document.getElementById('send-chat');
    const chatInput = document.getElementById('chat-input');
    const videoFileInput = document.getElementById('video-file');
    const drillOptions = document.querySelectorAll('.drill-option');

    // Setup drill selection
    drillOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Remove selected class from all options
            drillOptions.forEach(opt => opt.classList.remove('selected'));
            // Add selected class to clicked option
            this.classList.add('selected');
            selectedDrill = this.getAttribute('data-drill');
            updateAnalyzeButton();
        });
    });

    // Setup file input
    videoFileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            showFileSelected(file);
            updateAnalyzeButton();
        }
    });

    uploadForm.addEventListener('submit', handleVideoUpload);
    sendChatBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    // Health check on page load
    checkServerHealth();
});

function showFileSelected(file) {
    const placeholder = document.getElementById('upload-placeholder');
    const fileSelected = document.getElementById('file-selected');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    
    placeholder.classList.add('d-none');
    fileSelected.classList.remove('d-none');
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
}

function updateAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyze-btn');
    if (selectedDrill && selectedFile) {
        analyzeBtn.disabled = false;
    } else {
        analyzeBtn.disabled = true;
    }
}

async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('Server health check:', data);
    } catch (error) {
        console.warn('Server health check failed:', error);
    }
}

async function handleVideoUpload(e) {
    e.preventDefault();
    
    if (!selectedDrill || !selectedFile) {
        showError('Please select a drill type and upload a video file.');
        return;
    }
    
    const formData = new FormData();
    formData.append('drill_type', selectedDrill);
    formData.append('video', selectedFile);
    
    console.log('Upload started:', { 
        drillType: selectedDrill, 
        fileName: selectedFile.name, 
        fileSize: selectedFile.size,
        fileType: selectedFile.type 
    });
    
    // Validate file size (100MB limit)
    if (selectedFile.size > 100 * 1024 * 1024) {
        showError('Video file is too large. Maximum size is 100MB.');
        return;
    }
    
    // Validate file type
    const allowedTypes = [
        'video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 
        'video/x-msvideo', 'video/x-matroska', 'video/x-ms-wmv', 
        'video/x-flv', 'video/webm', 'video/mp4v-es', 'video/3gpp'
    ];
    
    if (!allowedTypes.includes(selectedFile.type) && !selectedFile.name.toLowerCase().match(/\.(mp4|avi|mov|mkv|wmv|flv|webm|m4v|3gp)$/)) {
        showError('Invalid file type. Please upload a video file (MP4, AVI, MOV, MKV, WMV, FLV, WEBM, M4V, 3GP).');
        return;
    }
    
    // Check if file is actually a video by trying to create a video element
    try {
        await validateVideoFile(selectedFile);
    } catch (error) {
        showError('Invalid video file. Please ensure the file is a valid video.');
        return;
    }
    
    showProcessing();
    
    const startTime = Date.now();
    
    try {
        console.log('Sending request to /api/analyze');
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', Object.fromEntries(response.headers.entries()));
        
        const result = await response.json();
        console.log('Response data:', result);
        
        const processingTime = (Date.now() - startTime) / 1000;
        console.log(`Processing completed in ${processingTime.toFixed(2)} seconds`);
        
        if (response.ok) {
            analysisData = result;
            showResults(result);
        } else {
            showError(result.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        const processingTime = (Date.now() - startTime) / 1000;
        console.log(`Processing failed after ${processingTime.toFixed(2)} seconds`);
        
        if (error.name === 'AbortError') {
            showError('Request timed out. Please try with a shorter video or try again.');
        } else {
            showError('Network error. Please check your connection and try again.');
        }
    }
}

async function validateVideoFile(file) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        
        video.onloadedmetadata = () => {
            URL.revokeObjectURL(video.src);
            resolve();
        };
        
        video.onerror = () => {
            URL.revokeObjectURL(video.src);
            reject(new Error('Invalid video file'));
        };
        
        video.src = URL.createObjectURL(file);
    });
}

async function sendChatMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();
    
    if (!message || !analysisData) {
        return;
    }
    
    console.log('Sending chat message:', message);
    
    // Add user message to chat
    addChatMessage(message, 'user');
    chatInput.value = '';
    
    // Disable input while processing
    chatInput.disabled = true;
    document.getElementById('send-chat').disabled = true;
    
    // Show typing indicator
    const typingId = addChatMessage('AI is analyzing your question...', 'ai', true);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: message,
                analysis_data: analysisData
            })
        });
        
        const result = await response.json();
        console.log('Chat response:', result);
        
        // Remove typing indicator
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
        
        if (response.ok) {
            addChatMessage(result.text_response, 'ai');
            
            // Handle any actions
            if (result.action && result.action.type === 'play_segment') {
                const actionMessage = `📹 Video segment reference: Frame ${result.action.start_frame} to ${result.action.end_frame}`;
                addChatMessage(actionMessage, 'system');
            }
        } else {
            addChatMessage(result.error || 'Sorry, I encountered an error. Please try again.', 'ai');
        }
    } catch (error) {
        console.error('Chat error:', error);
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
        addChatMessage('Sorry, I encountered a network error. Please try again.', 'ai');
    } finally {
        // Re-enable input
        chatInput.disabled = false;
        document.getElementById('send-chat').disabled = false;
        chatInput.focus();
    }
}

function addChatMessage(message, sender, isTyping = false) {
    const chatMessages = document.getElementById('chat-messages');
    const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `chat-message ${sender}`;
    
    if (isTyping) {
        messageDiv.innerHTML = `
            <div class="message-content typing">
                <span class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </span>
                ${message}
            </div>
        `;
    } else {
        const icon = sender === 'user' ? 'fas fa-user' : sender === 'ai' ? 'fas fa-robot' : 'fas fa-info';
        messageDiv.innerHTML = `
            <div class="message-content">
                <i class="${icon}"></i>
                ${message}
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

function showProcessing() {
    console.log('Showing processing state');
    processingStartTime = new Date();
    
    // Start progress animation
    let progress = 0;
    const progressBar = document.getElementById('progress-bar');
    const processingStatus = document.getElementById('processing-status');
    
    const steps = [
        { progress: 20, text: 'Analyzing video orientation...' },
        { progress: 40, text: 'Detecting body landmarks...' },
        { progress: 60, text: 'Tracking movement patterns...' },
        { progress: 80, text: 'Calculating form metrics...' },
        { progress: 100, text: 'Generating insights...' }
    ];
    
    let stepIndex = 0;
    const progressInterval = setInterval(() => {
        if (stepIndex < steps.length) {
            const step = steps[stepIndex];
            progressBar.style.width = step.progress + '%';
            processingStatus.textContent = step.text;
            stepIndex++;
        } else {
            clearInterval(progressInterval);
        }
    }, 800);
    
    // Store interval ID for cleanup
    window.processingInterval = progressInterval;
    
    document.getElementById('upload-section').classList.add('d-none');
    document.getElementById('error-section').classList.add('d-none');
    document.getElementById('results-section').classList.add('d-none');
    document.getElementById('processing-section').classList.remove('d-none');
}

function showError(message) {
    console.log('Showing error:', message);
    
    // Clear progress interval
    if (window.processingInterval) {
        clearInterval(window.processingInterval);
        window.processingInterval = null;
    }
    
    document.getElementById('error-message').textContent = message;
    document.getElementById('upload-section').classList.add('d-none');
    document.getElementById('processing-section').classList.add('d-none');
    document.getElementById('results-section').classList.add('d-none');
    document.getElementById('error-section').classList.remove('d-none');
}

function resetForm() {
    // Clear progress interval
    if (window.processingInterval) {
        clearInterval(window.processingInterval);
        window.processingInterval = null;
    }
    
    // Reset form state
    selectedDrill = null;
    selectedFile = null;
    analysisData = null;
    
    // Reset UI
    document.querySelectorAll('.drill-option').forEach(option => {
        option.classList.remove('selected');
    });
    
    document.getElementById('upload-placeholder').classList.remove('d-none');
    document.getElementById('file-selected').classList.add('d-none');
    document.getElementById('video-file').value = '';
    document.getElementById('analyze-btn').disabled = true;
    
    // Show upload section
    document.getElementById('upload-section').classList.remove('d-none');
    document.getElementById('processing-section').classList.add('d-none');
    document.getElementById('error-section').classList.add('d-none');
    document.getElementById('results-section').classList.add('d-none');
}

function showErrorDetails() {
    const errorDetails = document.getElementById('error-details');
    errorDetails.classList.toggle('d-none');
}

function showResults(data) {
    console.log('Showing results:', data);
    
    // Clear progress interval
    if (window.processingInterval) {
        clearInterval(window.processingInterval);
        window.processingInterval = null;
    }
    
    document.getElementById('upload-section').classList.add('d-none');
    document.getElementById('processing-section').classList.add('d-none');
    document.getElementById('error-section').classList.add('d-none');
    document.getElementById('results-section').classList.remove('d-none');
    
    displayAnalysisResults(data);
    
    // Clear previous chat messages
    document.getElementById('chat-messages').innerHTML = '';
    
    // Enable chat
    document.getElementById('chat-input').disabled = false;
    document.getElementById('send-chat').disabled = false;
    
    // Add welcome message
    const drillName = data.drill_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    const processingTime = processingStartTime ? ((Date.now() - processingStartTime.getTime()) / 1000).toFixed(1) : 'unknown';
    
    addChatMessage(`Great! I've analyzed your ${drillName} video in ${processingTime} seconds. Ask me anything about your performance!`, 'ai');
}

function displayAnalysisResults(data) {
    const summaryDiv = document.getElementById('analysis-summary');
    const detailedDiv = document.getElementById('detailed-results');
    
    // Display summary
    const drillName = data.drill_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    let summaryHtml = `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h6><i class="fas fa-chart-line me-2"></i>${drillName} Analysis</h6>
            <small class="text-muted">${data.video_info.total_frames} frames @ ${data.video_info.fps.toFixed(1)} FPS</small>
        </div>
    `;
    
    if (data.total_reps !== undefined) {
        summaryHtml += `
            <div class="row">
                <div class="col-6">
                    <div class="stat-card">
                        <div class="stat-number">${data.total_reps}</div>
                        <div class="stat-label">Total Reps</div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="stat-card">
                        <div class="stat-number">${data.video_info.duration.toFixed(1)}s</div>
                        <div class="stat-label">Duration</div>
                    </div>
                </div>
            </div>
        `;
    } else if (data.total_hold_time !== undefined) {
        summaryHtml += `
            <div class="row">
                <div class="col-6">
                    <div class="stat-card">
                        <div class="stat-number">${data.total_hold_time.toFixed(1)}s</div>
                        <div class="stat-label">Hold Time</div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="stat-card">
                        <div class="stat-number">${data.video_info.duration.toFixed(1)}s</div>
                        <div class="stat-label">Duration</div>
                    </div>
                </div>
            </div>
        `;
    } else if (data.total_balance_time !== undefined) {
        const efficiency = data.video_info.duration > 0 ? (data.total_balance_time / data.video_info.duration * 100).toFixed(1) : 0;
        summaryHtml += `
            <div class="row">
                <div class="col-4">
                    <div class="stat-card">
                        <div class="stat-number">${data.total_balance_time.toFixed(1)}s</div>
                        <div class="stat-label">Balance Time</div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="stat-card">
                        <div class="stat-number">${data.total_fouls}</div>
                        <div class="stat-label">Fouls</div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="stat-card">
                        <div class="stat-number">${efficiency}%</div>
                        <div class="stat-label">Efficiency</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    summaryDiv.innerHTML = summaryHtml;
    
    // Display detailed results
    let detailedHtml = '<h6>Detailed Analysis</h6>';
    
    if (data.reps && data.reps.length > 0) {
        detailedHtml += '<div class="table-responsive"><table class="table table-sm table-striped">';
        detailedHtml += '<thead><tr><th>Rep</th><th>Frames</th><th>Metrics</th></tr></thead><tbody>';
        
        data.reps.forEach(rep => {
            let metrics = '';
            if (rep.min_elbow_angle !== undefined) {
                metrics = `Elbow: ${rep.min_elbow_angle}° - ${rep.max_elbow_angle}°`;
            } else if (rep.min_knee_angle !== undefined) {
                metrics = `Knee: ${rep.min_knee_angle}°`;
            } else if (rep.hip_angle_top !== undefined) {
                metrics = `Hip: ${rep.hip_angle_top}°`;
            }
            
            detailedHtml += `
                <tr>
                    <td><strong>${rep.rep_number}</strong></td>
                    <td>${rep.start_frame} - ${rep.end_frame}</td>
                    <td><small>${metrics}</small></td>
                </tr>
            `;
        });
        
        detailedHtml += '</tbody></table></div>';
    } else if (data.foul_data && data.foul_data.length > 0) {
        detailedHtml += '<div class="table-responsive"><table class="table table-sm table-striped">';
        detailedHtml += '<thead><tr><th>Foul</th><th>Time</th><th>Frame</th></tr></thead><tbody>';
        
        data.foul_data.forEach(foul => {
            detailedHtml += `
                <tr>
                    <td><strong>${foul.foul_number}</strong></td>
                    <td>${foul.timestamp}s</td>
                    <td>${foul.frame_number}</td>
                </tr>
            `;
        });
        
        detailedHtml += '</tbody></table></div>';
    } else if (data.time_series_data && data.time_series_data.length > 0) {
        detailedHtml += `
            <div class="alert alert-info">
                <i class="fas fa-chart-area me-2"></i>
                <strong>${data.time_series_data.length} data points</strong> captured during your ${data.drill_type.replace(/_/g, ' ')}.
                <br><small>Use the AI chat to ask specific questions about your performance.</small>
            </div>
        `;
    } else {
        detailedHtml += '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle me-2"></i>No detailed metrics available for this exercise type.</div>';
    }
    
    detailedDiv.innerHTML = detailedHtml;
}

function resetForm() {
    console.log('Resetting form');
    document.getElementById('upload-form').reset();
    document.getElementById('upload-section').classList.remove('d-none');
    document.getElementById('processing-section').classList.add('d-none');
    document.getElementById('error-section').classList.add('d-none');
    document.getElementById('results-section').classList.add('d-none');
    analysisData = null;
    processingStartTime = null;
}
