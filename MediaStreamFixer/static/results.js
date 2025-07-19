const { useState, useEffect, useRef } = React;

// Mock analysis data for development
const mockAnalysisData = {
    drill_type: "squats",
    total_reps: 24,
    reps: [
        { rep_number: 1, start_frame: 1, end_frame: 15, min_knee_angle: 76.2, torso_angle_at_bottom: 167.5 },
        { rep_number: 2, start_frame: 35, end_frame: 54, min_knee_angle: 72.4, torso_angle_at_bottom: 166.6 },
        { rep_number: 3, start_frame: 73, end_frame: 92, min_knee_angle: 67.1, torso_angle_at_bottom: 168.0 },
        // ... more reps
    ],
    video_info: {
        fps: 29.92,
        total_frames: 923,
        duration: 30.8
    }
};

const VideoPlayer = ({ analysisData, currentSegment, setCurrentSegment, onSegmentEnd }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const segmentHandlerRef = useRef(null);
  const programmaticSeek = useRef(false);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas) return;

    // Set canvas size to match video
    const resizeCanvas = () => {
      const rect = video.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize MediaPipe Pose
    let pose = null;
    let poseDetectionActive = false;

    const initPoseDetection = async () => {
      try {
        // Check if MediaPipe is available globally
        if (typeof window.Pose === 'undefined') {
          console.error('MediaPipe Pose not loaded');
          return;
        }

        pose = new window.Pose({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
          }
        });

        pose.setOptions({
          modelComplexity: 1, // Use higher complexity for better accuracy
          smoothLandmarks: false, // Disable smoothing for more responsive tracking
          enableSegmentation: false,
          smoothSegmentation: false,
          minDetectionConfidence: 0.7, // Higher confidence for better accuracy
          minTrackingConfidence: 0.7
        });

        pose.onResults(onPoseResults);
        poseDetectionActive = true;
        console.log('MediaPipe Pose initialized successfully');
      } catch (error) {
        console.error('Failed to initialize MediaPipe Pose:', error);
      }
    };

    const onPoseResults = (results) => {
      if (!canvas || !poseDetectionActive) return;
      
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      if (results.poseLandmarks) {
        // Scale landmarks to canvas size
        const landmarks = results.poseLandmarks.map(landmark => ({
          x: landmark.x * canvas.width,
          y: landmark.y * canvas.height,
          z: landmark.z
        }));

        // Draw pose connections
        if (window.POSE_CONNECTIONS) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.lineWidth = 2;
          window.POSE_CONNECTIONS.forEach(connection => {
            const [start, end] = connection;
            if (landmarks[start] && landmarks[end]) {
              ctx.beginPath();
              ctx.moveTo(landmarks[start].x, landmarks[start].y);
              ctx.lineTo(landmarks[end].x, landmarks[end].y);
              ctx.stroke();
            }
          });
        }

        // Draw landmarks
        ctx.fillStyle = 'rgba(0, 0, 255, 0.8)';
        landmarks.forEach(landmark => {
          ctx.beginPath();
          ctx.arc(landmark.x, landmark.y, 2, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    };

    // Process video frames for pose detection
    const processVideoFrame = async () => {
      if (!pose || !poseDetectionActive || video.paused || video.ended) return;
      
      try {
        // Use requestAnimationFrame for better performance and timing
        requestAnimationFrame(async () => {
          try {
            // Create a canvas to capture the current video frame
            const frameCanvas = document.createElement('canvas');
            const frameCtx = frameCanvas.getContext('2d');
            frameCanvas.width = video.videoWidth;
            frameCanvas.height = video.videoHeight;
            
            // Draw current video frame to canvas
            frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
            
            // Send frame to MediaPipe
            await pose.send({ image: frameCanvas });
          } catch (error) {
            console.error('Error processing video frame:', error);
          }
        });
      } catch (error) {
        console.error('Error in processVideoFrame:', error);
      }
    };

    // Start pose detection when video loads
    const handleLoadedData = () => {
      // Disabled pose detection for now
      // if (!poseDetectionActive) {
      //   initPoseDetection();
      // }
    };

    // Process frames during playback
    const handleTimeUpdate = () => {
      // Disabled pose detection for now
      // if (poseDetectionActive) {
      //   processVideoFrame();
      // }
    };

    video.addEventListener('loadeddata', handleLoadedData);
    video.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      video.removeEventListener('loadeddata', handleLoadedData);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      poseDetectionActive = false;
    };
  }, [analysisData]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // Clean up previous handler
    if (video._segmentHandler) {
      console.log('[Segment] Removing previous handler');
      video.removeEventListener('timeupdate', video._segmentHandler);
      video._segmentHandler = null;
    }

    if (currentSegment) {
      console.log('[Segment] Setting segment:', currentSegment);
      programmaticSeek.current = true; // Set flag before seeking
      video.currentTime = currentSegment.startTime;
      const handler = () => {
        console.log('[Segment] timeupdate:', video.currentTime, 'endTime:', currentSegment.endTime);
        if (video.currentTime >= currentSegment.endTime) {
          console.log('[Segment] Pausing at end:', video.currentTime, '>=', currentSegment.endTime);
          video.pause();
          video.removeEventListener('timeupdate', handler);
          video._segmentHandler = null;
          setCurrentSegment(null);
          if (onSegmentEnd) onSegmentEnd();
        }
      };
      video._segmentHandler = handler;
      video.addEventListener('timeupdate', handler);
      console.log('[Segment] Handler attached');
      video.play();
    }

    return () => {
      if (video._segmentHandler) {
        console.log('[Segment] Cleaning up handler');
        video.removeEventListener('timeupdate', video._segmentHandler);
        video._segmentHandler = null;
      }
    };
  }, [currentSegment]);

  // Remove handler and clear segment only on true manual interaction
  const handleManual = (e) => {
    if (programmaticSeek.current) {
      // Ignore the first seeked event after programmatic seek
      programmaticSeek.current = false;
      console.log('[Segment] Ignoring programmatic seeked event');
      return;
    }
    const video = videoRef.current;
    if (e && e.isTrusted && video._segmentHandler) {
      console.log('[Segment] Manual interaction, removing handler');
      video.removeEventListener('timeupdate', video._segmentHandler);
      video._segmentHandler = null;
      if (currentSegment) setCurrentSegment(null);
    }
  };

  // Check if this is a rep-based drill
  const isRepBasedDrill = ['pushups', 'squats', 'situps'].includes(analysisData?.drill_type);

  return (
    <div className="video-container">
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <video
          ref={videoRef}
          className="video-player"
          controls
          src={analysisData?.video_path ? `/processed/${analysisData.video_path}` : ''}
          onSeeked={handleManual}
        />
        {/* Pose detection canvas - disabled for now */}
        {/* <canvas 
          ref={canvasRef} 
          style={{ 
            position: 'absolute', 
            top: 0, 
            left: 0, 
            width: '100%', 
            height: '100%',
            pointerEvents: 'none',
            zIndex: 5
          }}
        /> */}
        {isRepBasedDrill && analysisData?.reps && (
          <RepCounterOverlay 
            repData={analysisData} 
            videoRef={videoRef} 
          />
        )}
      </div>
    </div>
  );
};

const MetricsDashboard = ({ analysisData }) => {
    const getMainLeg = () => {
        if (Array.isArray(analysisData?.time_series_data) && analysisData.time_series_data.length > 0) {
            const counts = {};
            for (const seg of analysisData.time_series_data) {
                counts[seg.leg_side] = (counts[seg.leg_side] || 0) + (seg.end - seg.start);
            }
            let maxLeg = null, maxTime = 0;
            for (const leg in counts) {
                if (counts[leg] > maxTime) {
                    maxLeg = leg;
                    maxTime = counts[leg];
                }
            }
            return maxLeg ? maxLeg.charAt(0).toUpperCase() + maxLeg.slice(1) : 'None';
        }
        return 'None';
    };

    return (
        <div className="metrics-section">
            <h2>Performance Metrics</h2>
            <div className="metrics-grid">
                {['pushups','squats','situps'].includes(analysisData?.drill_type) ? (
                    <>
                        <div className="metric-card">
                            <div className="metric-value">{analysisData?.total_reps || 0}</div>
                            <div className="metric-label">Total Reps</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">{(analysisData?.video_info?.duration || 0).toFixed(1)}s</div>
                            <div className="metric-label">Duration</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">{analysisData?.calories_per_hour ?? 0}</div>
                            <div className="metric-label">Calories<br/><span className="metric-sublabel">Est. calories burned per hour (kcal/hr)</span></div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">{analysisData?.rhythm_label || '-'}</div>
                            <div className="metric-label">Rhythm<br/><span className="metric-sublabel">{analysisData?.avg_rep_duration ? `${analysisData.avg_rep_duration}s/rep` : ''}</span></div>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="metric-card">
                            <div className="metric-value">{analysisData?.total_reps || analysisData?.total_hold_time || analysisData?.total_balance_time || 0}</div>
                            <div className="metric-label">
                                {analysisData?.total_reps ? 'Total Reps' : 'Hold Time (s)'}
                            </div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">{analysisData?.video_info?.duration?.toFixed(1) || '0'}s</div>
                            <div className="metric-label">Duration</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">{analysisData?.calories_per_hour ?? 0}</div>
                            <div className="metric-label">Calories<br/><span className="metric-sublabel">Est. calories burned per hour (kcal/hr)</span></div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">{analysisData?.video_info?.fps?.toFixed(1) || '30'}</div>
                            <div className="metric-label">FPS</div>
                        </div>
                        {['single_leg_balance','single_leg_balance_left','single_leg_balance_right'].includes(analysisData?.drill_type) && (
                            <div className="metric-card">
                                <div className="metric-value">{getMainLeg()}</div>
                                <div className="metric-label">Main Ankle Raised</div>
                            </div>
                        )}
                    </>
                )}
            </div>
            {analysisData?.reps && (
                <div>
                    <h3>Detailed Rep Analysis</h3>
                    <table className="rep-details-table">
                        <thead>
                            <tr>
                                <th>Rep</th>
                                <th>Frames</th>
                                <th>Metrics</th>
                            </tr>
                        </thead>
                        <tbody>
                            {analysisData.reps.slice(0, 8).map((rep) => (
                                <tr key={rep.rep_number}>
                                    <td>{rep.rep_number}</td>
                                    <td>{rep.start_frame} - {rep.end_frame}</td>
                                    <td>
                                        {rep.min_knee_angle && `Knee: ${rep.min_knee_angle}°`}
                                        {rep.min_elbow_angle && `Elbow: ${rep.min_elbow_angle}°`}
                                        {rep.hip_angle_top && `Hip: ${rep.hip_angle_top}°`}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

const PerformanceChart = ({ analysisData }) => {
    const chartRef = useRef(null);
    const chartInstance = useRef(null);

    useEffect(() => {
        if (chartRef.current && analysisData?.reps) {
            const ctx = chartRef.current.getContext('2d');
            if (chartInstance.current) {
                chartInstance.current.destroy();
            }
            const labels = analysisData.reps.map(rep => `Rep ${rep.rep_number}`);
            const data = analysisData.reps.map(rep => 
                rep.min_knee_angle || rep.min_elbow_angle || rep.hip_angle_top || 0
            );
            chartInstance.current = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Form Consistency',
                        data: data,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    }
                }
            });
        }
        return () => {
            if (chartInstance.current) {
                chartInstance.current.destroy();
            }
        };
    }, [analysisData]);

    return (
        <div className="charts-section">
            <h2>Performance Trends</h2>
            <div className="chart-container">
                <canvas ref={chartRef}></canvas>
            </div>
        </div>
    );
};

const ChatInterface = ({ analysisData, isOpen, onClose, onVideoSegment, isVideoPlaying }) => {
    const [messages, setMessages] = useState([
        {
            type: 'ai',
            content: `Great! I've analyzed your ${analysisData?.drill_type || 'exercise'} video. Ask me anything about your performance!`
        }
    ]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [chatSize, setChatSize] = useState({ width: 380, height: 500 });
    const chatPanelRef = useRef(null);
    const isResizing = useRef(false);

    const sendMessage = async () => {
        if (!inputValue.trim()) return;
        const userMessage = { type: 'user', content: inputValue };
        setMessages(prev => [...prev, userMessage, { type: 'ai', content: '' }]);
        setInputValue('');
        setIsLoading(true);
        
        // Add a small delay to show typing animation
        await new Promise(resolve => setTimeout(resolve, 500));
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: inputValue,
                    analysis_data: analysisData
                })
            });
            if (!response.body) throw new Error('No response body');
            const reader = response.body.getReader();
            let aiMessage = '';
            let done = false;
            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                if (value) {
                    aiMessage += new TextDecoder().decode(value);
                    setMessages(msgs => {
                        const updated = [...msgs];
                        updated[updated.length - 1].content = aiMessage;
                        return updated;
                    });
                }
            }
            // Parse AI response for rep information and extract video segments
            // Only play video if user explicitly asked for it
            const videoKeywords = [
                'show', 'display', 'play', 'watch', 'see', 'view', 'demonstrate',
                'where', 'when', 'at what time', 'at what point', 'during which',
                'highlight', 'point out', 'mark', 'indicate', 'locate',
                'video', 'clip', 'segment', 'moment', 'instance', 'frame',
                'timeline', 'timestamp', 'timecode', 'position', 'spot',
                'visual', 'visually', 'appears', 'looks like', 'can see',
                'observe', 'notice', 'spot', 'identify', 'find'
            ];
            
            const userWantsVideo = videoKeywords.some(keyword => inputValue.toLowerCase().includes(keyword));
            
            if (userWantsVideo) {
                const segment = parseRepFromResponse(aiMessage, analysisData);
                if (segment && onVideoSegment) {
                    console.log('Playing video segment:', segment);
                    onVideoSegment(segment);
                }
            }
        } catch (error) {
            setMessages(msgs => [...msgs, { type: 'ai', content: 'Sorry, I encountered an error. Please try again.' }]);
        }
        setIsLoading(false);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const handleResizeStart = (e) => {
        e.preventDefault();
        isResizing.current = true;
        if (chatPanelRef.current) {
            chatPanelRef.current.classList.add('resizing');
        }
        document.addEventListener('mousemove', handleResizeMove);
        document.addEventListener('mouseup', handleResizeEnd);
    };

    const handleResizeMove = (e) => {
        if (!isResizing.current || !chatPanelRef.current) return;
        
        const rect = chatPanelRef.current.getBoundingClientRect();
        const newWidth = Math.max(300, Math.min(600, rect.right - e.clientX));
        const newHeight = Math.max(400, Math.min(window.innerHeight * 0.8, rect.bottom - e.clientY));
        
        setChatSize({ width: newWidth, height: newHeight });
    };

    const handleResizeEnd = () => {
        isResizing.current = false;
        if (chatPanelRef.current) {
            chatPanelRef.current.classList.remove('resizing');
        }
        document.removeEventListener('mousemove', handleResizeMove);
        document.removeEventListener('mouseup', handleResizeEnd);
    };

    useEffect(() => {
        return () => {
            document.removeEventListener('mousemove', handleResizeMove);
            document.removeEventListener('mouseup', handleResizeEnd);
        };
    }, []);

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        const chatMessages = chatPanelRef.current?.querySelector('.chat-messages');
        if (chatMessages) {
            chatMessages.scrollTo({
                top: chatMessages.scrollHeight,
                behavior: 'smooth'
            });
        }
    }, [messages]);

    // Function to parse rep information from AI response and extract video segments
    const parseRepFromResponse = (aiMessage, analysisData) => {
        if (!analysisData || !analysisData.reps || !analysisData.video_info) {
            return null;
        }

        const message = aiMessage.toLowerCase();
        const reps = analysisData.reps;
        const fps = analysisData.video_info.fps;

        // Look for specific rep numbers
        const repMatch = message.match(/(\d+)(?:st|nd|rd|th)?\s*rep/);
        if (repMatch) {
            const repNumber = parseInt(repMatch[1]);
            const rep = reps.find(r => r.rep_number === repNumber);
            if (rep) {
                const startTime = rep.start_time !== undefined ? rep.start_time : rep.start_frame / fps;
                const endTime = rep.end_time !== undefined ? rep.end_time : rep.end_frame / fps;
                return {
                    startTime,
                    endTime
                };
            }
        }

        // Look for "best rep" or "worst rep"
        if (message.includes('best rep') || message.includes('strongest') || message.includes('excellent')) {
            // Find rep with best metrics (lowest angle for pushups/squats, highest for other exercises)
            let bestRep = reps[0];
            let bestScore = reps[0]?.min_knee_angle || reps[0]?.min_elbow_angle || 0;
            
            for (const rep of reps) {
                const score = rep.min_knee_angle || rep.min_elbow_angle || 0;
                if (analysisData.drill_type === 'pushups' || analysisData.drill_type === 'squats') {
                    // Lower angle is better for pushups/squats
                    if (score < bestScore) {
                        bestScore = score;
                        bestRep = rep;
                    }
                } else {
                    // Higher angle is better for other exercises
                    if (score > bestScore) {
                        bestScore = score;
                        bestRep = rep;
                    }
                }
            }
            
            if (bestRep) {
                const startTime = bestRep.start_time !== undefined ? bestRep.start_time : bestRep.start_frame / fps;
                const endTime = bestRep.end_time !== undefined ? bestRep.end_time : bestRep.end_frame / fps;
                return {
                    startTime,
                    endTime
                };
            }
        }

        if (message.includes('worst rep') || message.includes('weakest') || message.includes('poor')) {
            // Find rep with worst metrics (highest angle for pushups/squats, lowest for other exercises)
            let worstRep = reps[0];
            let worstScore = reps[0]?.min_knee_angle || reps[0]?.min_elbow_angle || 0;
            
            for (const rep of reps) {
                const score = rep.min_knee_angle || rep.min_elbow_angle || 0;
                if (analysisData.drill_type === 'pushups' || analysisData.drill_type === 'squats') {
                    // Higher angle is worse for pushups/squats
                    if (score > worstScore) {
                        worstScore = score;
                        worstRep = rep;
                    }
                } else {
                    // Lower angle is worse for other exercises
                    if (score < worstScore) {
                        worstScore = score;
                        worstRep = rep;
                    }
                }
            }
            
            if (worstRep) {
                const startTime = worstRep.start_time !== undefined ? worstRep.start_time : worstRep.start_frame / fps;
                const endTime = worstRep.end_time !== undefined ? worstRep.end_time : worstRep.end_frame / fps;
                return {
                    startTime,
                    endTime
                };
            }
        }

        // Look for "first rep" or "last rep"
        if (message.includes('first rep')) {
            const firstRep = reps[0];
            if (firstRep) {
                const startTime = firstRep.start_time !== undefined ? firstRep.start_time : firstRep.start_frame / fps;
                const endTime = firstRep.end_time !== undefined ? firstRep.end_time : firstRep.end_frame / fps;
                return {
                    startTime,
                    endTime
                };
            }
        }

        if (message.includes('last rep')) {
            const lastRep = reps[reps.length - 1];
            if (lastRep) {
                const startTime = lastRep.start_time !== undefined ? lastRep.start_time : lastRep.start_frame / fps;
                const endTime = lastRep.end_time !== undefined ? lastRep.end_time : lastRep.end_frame / fps;
                return {
                    startTime,
                    endTime
                };
            }
        }

        return null;
    };

    return (
        <div 
            ref={chatPanelRef}
            className={`chat-panel floating ${isOpen ? 'open' : ''} ${isVideoPlaying ? 'video-playing' : ''}`} 
            style={{
                display: isOpen ? 'flex' : 'none',
                width: `${chatSize.width}px`,
                height: `${chatSize.height}px`
            }}
        >
            <div 
                className="chat-resize-handle" 
                onMouseDown={handleResizeStart}
                title="Drag to resize"
            ></div>
            <div className="chat-header">
                <h3 style={{display:'flex',alignItems:'center',gap:'0.5rem'}}>
                  <span style={{fontSize:'1.5rem'}}>🤖</span> AI Coach Chat
                </h3>
                <button className="chat-close" onClick={onClose}>×</button>
            </div>
            <div className="chat-messages">
                {messages.map((message, index) => (
                    <div key={index} className={`chat-message ${message.type}`}>
                        {message.type === 'ai' && (
                          <span className="avatar" style={{marginRight:'0.7rem',verticalAlign:'middle',display:'inline-block'}}>
                            <span role="img" aria-label="AI">🤖</span>
                          </span>
                        )}
                        <span>
                            {message.content}
                            {isLoading && index === messages.length - 1 && message.type === 'ai' && message.content === '' && (
                                <>
                                    <span style={{opacity: 0.7, fontSize: '0.9em'}}>AI is typing</span>
                                    <span className="typing-indicator">
                                        <span></span><span></span><span></span>
                                    </span>
                                </>
                            )}
                        </span>
                    </div>
                ))}
            </div>
            <div className="chat-input-container">
                <input
                    className="chat-input"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about your form, best rep, improvements..."
                    disabled={isLoading}
                />
                <button 
                    className="chat-send" 
                    onClick={sendMessage}
                    disabled={isLoading}
                >
                    Send
                </button>
            </div>
        </div>
    );
};

// RepCounterOverlay: shows real-time rep count and up/down state
function RepCounterOverlay({ repData, videoRef }) {
  const overlayRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    
    console.log('RepCounterOverlay useEffect called:', { video, overlay, repData });
    
    if (!video || !overlay) {
      console.log('RepCounterOverlay: Missing video or overlay element');
      return;
    }

    const updateOverlay = () => {
      const currentTime = video.currentTime;
      let repCount = 0;
      let status = 'READY';
      
      // Check if this is a rep-based drill
      const drill = (repData?.drill_type || '').toLowerCase();
      if (["pushups", "situps", "squats"].includes(drill) && repData?.reps && repData.reps.length > 0) {
        for (let i = 0; i < repData.reps.length; ++i) {
          const rep = repData.reps[i];
          if (currentTime >= rep.end_time) {
            repCount++;
          }
          if (currentTime >= rep.start_time && currentTime <= rep.end_time) {
            const mid = (rep.start_time + rep.end_time) / 2;
            status = currentTime < mid ? 'DOWN' : 'UP';
            repCount = rep.rep_number - 1; // Show completed reps
          }
        }
        
        // Always show the overlay for rep-based drills
        overlay.innerHTML = `
          <div style="
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            font-weight: bold;
            line-height: 1.2;
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
          ">
            <div>Rep: ${repCount}</div>
            <div>Status: ${status}</div>
          </div>
        `;
        overlay.style.display = 'block';
        
        // Debug logging
        console.log('RepCounterOverlay updated:', { currentTime, repCount, status, drill });
      } else {
        // Hide overlay for non-rep-based drills
        overlay.style.display = 'none';
        console.log('RepCounterOverlay hidden - not a rep-based drill or no reps data');
      }
    };

    const handleTimeUpdate = () => updateOverlay();
    const handleSeeked = () => updateOverlay();
    const handleLoadedData = () => updateOverlay();
    const handlePlay = () => updateOverlay();
    const handlePause = () => updateOverlay();

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('seeked', handleSeeked);
    video.addEventListener('loadeddata', handleLoadedData);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    // Handle fullscreen changes
    const handleFullscreenChange = () => {
      if (document.fullscreenElement === video) {
        // Video is in fullscreen, move overlay to fullscreen container
        const fullscreenContainer = document.fullscreenElement.parentElement;
        if (fullscreenContainer && overlay.parentElement !== fullscreenContainer) {
          fullscreenContainer.appendChild(overlay);
        }
      } else {
        // Video exited fullscreen, move overlay back to video container
        const videoContainer = video.parentElement;
        if (videoContainer && overlay.parentElement !== videoContainer) {
          videoContainer.appendChild(overlay);
        }
      }
      updateOverlay();
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);

    // Initial update
    updateOverlay();

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('seeked', handleSeeked);
      video.removeEventListener('loadeddata', handleLoadedData);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [repData, videoRef]);

  return (
    <div 
      ref={overlayRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 1000,
        pointerEvents: 'none',
        display: 'block'
      }}
    />
  );
}

const ResultsPage = () => {
    const [analysisData, setAnalysisData] = useState(null);
    const [isChatOpen, setIsChatOpen] = useState(false);
    const [currentVideoSegment, setCurrentVideoSegment] = useState(null);
    const [isVideoPlaying, setIsVideoPlaying] = useState(false);
    const [segmentTriggerId, setSegmentTriggerId] = useState(0); // Add trigger id
    // Add a ref to pass down for manual control
    const manualControlRef = useRef(null);

    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const idParam = urlParams.get('id');
        
        if (idParam) {
            // Load analysis data from API endpoint
            fetch(`/api/analysis/${idParam}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Loaded analysis data:', data);
                    console.log('Video path:', data.video_path);
                    setAnalysisData(data);
                    // Make analysis data available globally for frame-based status detection
                    window.analysisData = data;
                })
                .catch(e => {
                    console.error('Error loading analysis data:', e);
                    setAnalysisData(mockAnalysisData);
                });
        } else {
            console.log('No id param, using mock data');
            setAnalysisData(mockAnalysisData);
        }
    }, []);

    // Modified: handleVideoSegment no longer uses a timeout
    const handleVideoSegment = (segment) => {
        // Only set if different
        if (
            !currentVideoSegment ||
            currentVideoSegment.startTime !== segment.startTime ||
            currentVideoSegment.endTime !== segment.endTime
        ) {
            setCurrentVideoSegment(segment);
            setIsVideoPlaying(true);
            setIsChatOpen(false); // Hide chat when video starts playing
            setSegmentTriggerId(id => id + 1); // Increment trigger id
        }
    };

    // New: callback for when segment playback is done
    const handleSegmentEnd = () => {
        setIsVideoPlaying(false);
        setTimeout(() => {
            setIsChatOpen(true);
        }, 500); // Small delay before reopening chat
    };

    const toggleChat = () => {
        console.log('Toggling chat, current state:', isChatOpen);
        setIsChatOpen(!isChatOpen);
    };

    return (
        <div className="results-root">
            <div className="container">
                <div className="header">
                    <h1>🏋️ Analysis Results</h1>
                    <p>Your {analysisData?.drill_type} analysis is complete!</p>
                </div>
                <div className="video-section">
                    <h2>📹 Video Analysis</h2>
                    <VideoPlayer
                        analysisData={analysisData}
                        currentSegment={currentVideoSegment}
                        setCurrentSegment={setCurrentVideoSegment}
                        onSegmentEnd={handleSegmentEnd} // Pass callback
                    />
                </div>
                <MetricsDashboard analysisData={analysisData} />
                <PerformanceChart analysisData={analysisData} />
            </div>
            <ChatInterface
                analysisData={analysisData}
                isOpen={isChatOpen && !isVideoPlaying}
                onClose={() => setIsChatOpen(false)}
                onVideoSegment={handleVideoSegment}
                isVideoPlaying={isVideoPlaying}
            />
            <button 
                className={`chat-button ${isChatOpen ? 'expanded' : ''} ${isVideoPlaying ? 'hidden' : ''}`}
                onClick={toggleChat}
                title="AI Coach Chat"
                style={{ display: isVideoPlaying ? 'none' : 'block' }}
            >
                💬
            </button>
        </div>
    );
};

ReactDOM.render(<ResultsPage />, document.getElementById('root')); 