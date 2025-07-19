// Only declare analysisData once at the top
let analysisData = null;
let processingStartTime = null;

let selectedDrill = null;
let selectedFile = null;

document.addEventListener('DOMContentLoaded', function() {
    // All code that references DOM elements should be inside here
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleVideoUpload);
    }
    const sendChatBtn = document.getElementById('send-chat');
    if (sendChatBtn) {
        sendChatBtn.addEventListener('click', sendChatMessage);
    }
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
    const videoFileInput = document.getElementById('video-file');
    if (videoFileInput) {
        videoFileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                showFileSelected(file);
                updateAnalyzeButton();
            }
        });
    }
    const drillOptions = document.querySelectorAll('.drill-option');
    if (drillOptions && drillOptions.length > 0) {
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
    }
    // Health check on page load
    checkServerHealth();
    // Add event listeners for new fields to update the Analyze button
    ['age','gender','weight'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', updateAnalyzeButton);
            el.addEventListener('change', updateAnalyzeButton);
        }
    });
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
    const age = document.getElementById('age')?.value;
    const gender = document.getElementById('gender')?.value;
    const weight = document.getElementById('weight')?.value;
    if (selectedDrill && selectedFile && age && gender && weight) {
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

    const age = document.getElementById('age')?.value;
    const gender = document.getElementById('gender')?.value;
    const weight = document.getElementById('weight')?.value;

    if (!selectedDrill || !selectedFile || !age || !gender || !weight) {
        showError('Please select a drill type, upload a video file, and enter your age, gender, and weight.');
        return;
    }

    const formData = new FormData();
    formData.append('drill_type', selectedDrill);
    formData.append('video', selectedFile);
    formData.append('age', age);
    formData.append('weight', weight);
    formData.append('gender', gender);

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

    // Show processing screen immediately
    showProcessing();

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.redirect_url) {
            window.location.href = result.redirect_url;
        } else if (response.ok) {
            analysisData = result;
            showResults(result);
        } else {
            showError(result.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        showError('Network error. Please check your connection and try again.');
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
    
    // Store the user question for later reference
    const userQuestion = message.toLowerCase();
    
    // Add user message to chat
    addChatMessage(message, 'user');
    chatInput.value = '';
    
    // Scroll to bottom after adding user message
    scrollChatToBottom();
    
    // Disable input while processing
    chatInput.disabled = true;
    document.getElementById('send-chat').disabled = true;
    
    // Show typing indicator
    const typingId = addChatMessage('AI is analyzing your question...', 'ai', true);
    scrollChatToBottom();
    
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
            addChatMessage(result.response || result.text_response, 'ai');
            // Scroll to bottom after adding AI response
            scrollChatToBottom();
            // Handle any actions - only play video if explicitly requested
            if (result.action && result.action.type === 'play_segment') {
                // Check if user actually asked for video demonstration
                const videoKeywords = [
                    'show', 'display', 'play', 'watch', 'see', 'view', 'demonstrate',
                    'where', 'when', 'at what time', 'at what point', 'during which',
                    'highlight', 'point out', 'mark', 'indicate', 'locate',
                    'video', 'clip', 'segment', 'moment', 'instance', 'frame',
                    'timeline', 'timestamp', 'timecode', 'position', 'spot',
                    'visual', 'visually', 'appears', 'looks like', 'can see',
                    'observe', 'notice', 'spot', 'identify', 'find'
                ];
                
                const wantsVideo = videoKeywords.some(keyword => userQuestion.includes(keyword));
                
                if (wantsVideo) {
                    const actionMessage = `📹 Video segment reference: ` +
                        (result.action.start_time !== undefined && result.action.end_time !== undefined
                            ? `Time ${result.action.start_time.toFixed(2)}s to ${result.action.end_time.toFixed(2)}s`
                            : result.action.start_frame !== undefined && result.action.end_frame !== undefined
                            ? `Frame ${result.action.start_frame} to ${result.action.end_frame}`
                            : '');
                    addChatMessage(actionMessage, 'system');
                    scrollChatToBottom();
                    // --- Video seek/play logic ---
                    const videoElem = document.querySelector('#chat-video-container video');
                    if (videoElem && result.action.start_time !== undefined) {
                        videoElem.currentTime = result.action.start_time;
                        videoElem.play();
                        // Remove any previous pause handler
                        if (videoElem._pauseHandler) {
                            videoElem.removeEventListener('timeupdate', videoElem._pauseHandler);
                        }
                        if (result.action.end_time !== undefined) {
                            videoElem._pauseHandler = function() {
                                if (videoElem.currentTime >= result.action.end_time) {
                                    videoElem.pause();
                                    videoElem.removeEventListener('timeupdate', videoElem._pauseHandler);
                                }
                            };
                            videoElem.addEventListener('timeupdate', videoElem._pauseHandler);
                        }
                    }
                } else {
                    // User didn't ask for video, just show the reference without playing
                    const actionMessage = `📹 Reference: ` +
                        (result.action.start_time !== undefined && result.action.end_time !== undefined
                            ? `Time ${result.action.start_time.toFixed(2)}s to ${result.action.end_time.toFixed(2)}s`
                            : result.action.start_frame !== undefined && result.action.end_frame !== undefined
                            ? `Frame ${result.action.start_frame} to ${result.action.end_frame}`
                            : '');
                    addChatMessage(actionMessage, 'system');
                    scrollChatToBottom();
                }
            }
        } else {
            addChatMessage(result.error || 'Sorry, I encountered an error. Please try again.', 'ai');
            scrollChatToBottom();
        }
    } catch (error) {
        console.error('Chat error:', error);
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
        addChatMessage('Sorry, I encountered a network error. Please try again.', 'ai');
        scrollChatToBottom();
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
    
    // Immediate scroll to ensure message is visible
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

function scrollChatToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        console.log('Scrolling chat to bottom, current scrollHeight:', chatMessages.scrollHeight);
        
        // Use smooth scrolling for better UX
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
        
        // Fallback for older browsers or if smooth scrolling fails
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
        
        // Ensure chat input is focused after scrolling
        setTimeout(() => {
            const chatInput = document.getElementById('chat-input');
            if (chatInput && !chatInput.disabled) {
                chatInput.focus();
            }
        }, 300);
    } else {
        console.error('Chat messages container not found');
    }
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
    console.log('Resetting form');
    document.getElementById('upload-form').reset();
    document.getElementById('upload-section').classList.remove('d-none');
    document.getElementById('processing-section').classList.add('d-none');
    document.getElementById('error-section').classList.add('d-none');
    document.getElementById('results-section').classList.add('d-none');
    analysisData = null;
    processingStartTime = null;
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
    scrollChatToBottom();

    // --- Set up main video player ---
    const mainVideoContainer = document.getElementById('main-video-container');
    if (mainVideoContainer) {
        mainVideoContainer.innerHTML = '';
        
        // Determine video orientation for main player
        const landscapeDrills = ['pushups', 'elbow_plank', 'situps'];
        const portraitDrills = ['squats', 'chair_hold', 'single_leg_balance', 'single_leg_balance_left', 'single_leg_balance_right'];
        const drillType = (data.drill_type || '').toLowerCase();
        
        // Main video should be larger
        let width = '640px', height = '360px';
        if (portraitDrills.includes(drillType)) {
            width = '400px';
            height = '640px';
        }
        
        // Build video URL for main player
        let videoPath = data.video_path;
        if (videoPath) {
            // If videoPath is a full URL, use as is; otherwise, prefix with /processed/
            let videoUrl = videoPath.startsWith('http') ? videoPath : `/processed/${videoPath}`;
            console.log('Setting up main video with URL:', videoUrl);
            
            const videoElem = document.createElement('video');
            videoElem.src = videoUrl;
            videoElem.controls = true;
            videoElem.style.width = width;
            videoElem.style.height = height;
            videoElem.style.borderRadius = '16px';
            videoElem.style.background = '#000';
            videoElem.style.boxShadow = '0 4px 20px rgba(0,0,0,0.15)';
            videoElem.style.maxWidth = '100%';
            videoElem.style.maxHeight = '70vh';
            videoElem.id = 'main-video-player';
            
            // Add error handling
            videoElem.onerror = function() {
                console.error('Failed to load main video:', videoUrl);
                mainVideoContainer.innerHTML = '<div class="alert alert-warning">Failed to load video. Please try again.</div>';
            };
            
            mainVideoContainer.appendChild(videoElem);
        } else {
            console.error('No video path found in analysis data:', data);
            mainVideoContainer.innerHTML = '<div class="alert alert-warning">Processed video not available.</div>';
        }
    }

    // --- Add video player to chat/results right side (smaller version) ---
    // Find the chat column (col-md-6) and add a container for the video if not present
    const chatCol = document.querySelector('#results-section .col-md-6:last-child .p-4');
    if (chatCol) {
        let videoContainer = document.getElementById('chat-video-container');
        if (!videoContainer) {
            videoContainer = document.createElement('div');
            videoContainer.id = 'chat-video-container';
            videoContainer.style.display = 'flex';
            videoContainer.style.justifyContent = 'center';
            videoContainer.style.alignItems = 'flex-start';
            videoContainer.style.marginBottom = '1.5rem';
            chatCol.insertBefore(videoContainer, chatCol.firstChild);
        }
        videoContainer.innerHTML = '';
        
        // Smaller video for chat section
        let width = '320px', height = '180px';
        if (portraitDrills.includes(drillType)) {
            width = '200px';
            height = '320px';
        }
        
        // Build video URL for chat video
        if (videoPath) {
            let videoUrl = videoPath.startsWith('http') ? videoPath : `/processed/${videoPath}`;
            const videoElem = document.createElement('video');
            videoElem.src = videoUrl;
            videoElem.controls = true;
            videoElem.style.width = width;
            videoElem.style.height = height;
            videoElem.style.borderRadius = '12px';
            videoElem.style.background = '#000';
            videoElem.style.boxShadow = '0 2px 12px rgba(0,0,0,0.1)';
            videoElem.style.marginBottom = '1rem';
            videoElem.id = 'chat-video-player';
            videoContainer.appendChild(videoElem);
        } else {
            videoContainer.innerHTML = '<div class="alert alert-warning">Processed video not available.</div>';
        }
    }
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

// Add this function to update backend JSON after user input/metrics change
async function updateMetricsOnBackend({ filename, age, weight, gender, caloriesSession, caloriesHour, comparisonScore }) {
    try {
        await fetch('/api/update_metrics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename,
                age,
                weight_kg: weight,
                gender,
                calories_burned_session: caloriesSession,
                calories_per_hour: caloriesHour,
                comparison_score: comparisonScore
            })
        });
    } catch (err) {
        console.error('Failed to update backend metrics:', err);
    }
}
window.updateMetricsOnBackend = updateMetricsOnBackend;
