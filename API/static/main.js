// Tab management
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabId).style.display = 'block';
    event.currentTarget.classList.add('active');

    // Stop processing if switching away
    if (tabId !== 'camera-tab') {
        stopCamera();
    }
    if (tabId !== 'video-tab') {
        const videoEl = document.getElementById('videoElement');
        if (videoEl) videoEl.pause();
    }
}

// Update file name UI
function updateFileMessage(input) {
    if (input.files && input.files[0]) {
        input.previousElementSibling.textContent = 'Đã chọn: ' + input.files[0].name;
    }
}

// ============================================
// #1. TAB ẢNH (Image Mode)
// ============================================
async function handleImageUpload(e) {
    e.preventDefault();
    const input = document.getElementById('imageInput');
    if (!input.files || !input.files[0]) return;

    const formData = new FormData();
    formData.append('image_name', input.files[0]);

    document.getElementById('uploadbanner').style.display = 'none';
    document.getElementById('image-loading').style.display = 'block';

    try {
        const response = await fetch('/image', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        // Show results
        document.getElementById('image-loading').style.display = 'none';
        document.getElementById('image-result').style.display = 'block';
        
        document.getElementById('result-img').src = data.result_url;
        const audioEl = document.getElementById('image-audio');
        audioEl.src = data.audio_url;
        audioEl.play();

        document.getElementById('image-classes').textContent = 
            data.detected_names.length > 0 ? data.detected_names.join(', ') : 'Không có';

    } catch (err) {
        alert("Lỗi khi kết nối tới máy chủ.");
        resetImageTab();
    }
}

function resetImageTab() {
    document.getElementById('uploadbanner').style.display = 'block';
    document.getElementById('image-result').style.display = 'none';
    document.getElementById('image-loading').style.display = 'none';
    document.getElementById('uploadbanner').reset();
    document.querySelector('#uploadbanner .file-message').textContent = 'Kéo thả ảnh vào đây hoặc bấm để chọn';
}

// ============================================
// #COMMON: AI Real-Time processing & Voice
// ============================================
let lastSpoken = {};
function speakDetectedClasses(classes) {
    if (!classes || classes.length === 0) return;
    
    // Only speak in supported browsers
    if (!('speechSynthesis' in window)) return;

    const now = Date.now();
    classes.forEach(cls => {
        // Cooldown 10s (10000ms) để không bị spam quá trình đọc voice liên tục
        if (!lastSpoken[cls] || now - lastSpoken[cls] > 10000) {
            lastSpoken[cls] = now;
            const textToSpeak = cls;
            const utterThis = new SpeechSynthesisUtterance(textToSpeak);
            utterThis.lang = 'vi-VN';
            window.speechSynthesis.speak(utterThis);
        }
    });
}

function adjustCanvasSize(videoEl, canvasEl) {
    if(videoEl.videoWidth === 0) return;
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
}

let isProcessingFrame = false;

// Process a single frame wrapper
async function processVideoFrame(videoEl, overlayCanvas) {
    if (isProcessingFrame || videoEl.paused || videoEl.ended) return;
    isProcessingFrame = true;

    // Grab frame
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = videoEl.videoWidth;
    tempCanvas.height = videoEl.videoHeight;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(videoEl, 0, 0, tempCanvas.width, tempCanvas.height);
    
    const base64Image = tempCanvas.toDataURL('image/jpeg', 0.6); // 60% quality is enough for detection

    try {
        const response = await fetch('/detect_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });
        const data = await response.json();
        
        adjustCanvasSize(videoEl, overlayCanvas);
        
        // Draw boxes
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        if (data.detections && data.detections.length > 0) {
            let highestConfidenceDet = data.detections[0];

            data.detections.forEach(det => {
                const [x1, y1, x2, y2] = det.box;
                const width = x2 - x1;
                const height = y2 - y1;

                // Cập nhật biển báo có độ tự tin cao nhất
                if (det.confidence > highestConfidenceDet.confidence) {
                    highestConfidenceDet = det;
                }

                // Draw bounding box
                ctx.strokeStyle = '#22c55e'; // green 500
                ctx.lineWidth = Math.max(3, overlayCanvas.width / 300);
                ctx.strokeRect(x1, y1, width, height);

                // Draw label background
                ctx.fillStyle = '#22c55e';
                const label = `[${det.class_name}] ${det.meaning} ${(det.confidence * 100).toFixed(0)}%`;
                ctx.font = `${Math.max(16, overlayCanvas.width / 50)}px Inter`;
                const textWidth = ctx.measureText(label).width;
                ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);

                // Draw label text
                ctx.fillStyle = '#fff';
                ctx.fillText(label, x1 + 5, y1 - 5);
            });

            // Chỉ đọc ý nghĩa của biển báo có độ tự tin cao nhất
            speakDetectedClasses([highestConfidenceDet.meaning]);
        }

    } catch (err) {
        console.error("Frame processing error", err);
    }

    isProcessingFrame = false;
}


// ============================================
// #2. TAB VIDEO (Video Mode)
// ============================================
let videoInterval = null;

function handleVideoFile(input) {
    const file = input.files[0];
    if (!file) return;

    document.getElementById('video-upload-area').style.display = 'none';
    document.getElementById('video-player-container').style.display = 'flex';
    document.getElementById('video-player-container').style.flexDirection = 'column';
    
    const videoEl = document.getElementById('videoElement');
    const overlay = document.getElementById('videoOverlay');
    
    const objectUrl = URL.createObjectURL(file);
    videoEl.src = objectUrl;

    videoEl.onplay = () => {
        // Start processing loop when video plays
        if(videoInterval) clearInterval(videoInterval);
        videoInterval = setInterval(() => {
            processVideoFrame(videoEl, overlay);
        }, 150); // ~6-7 FPS
    };

    videoEl.onpause = () => clearInterval(videoInterval);
    videoEl.onended = () => clearInterval(videoInterval);
}

function resetVideoTab() {
    const videoEl = document.getElementById('videoElement');
    videoEl.pause();
    videoEl.src = "";
    if(videoInterval) clearInterval(videoInterval);
    
    const overlay = document.getElementById('videoOverlay');
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    document.getElementById('video-upload-area').style.display = 'flex';
    document.getElementById('video-player-container').style.display = 'none';
    document.getElementById('videoInput').value = "";
    document.querySelector('#video-upload-area .file-message').textContent = 'Kéo thả Video vào đây hoặc bấm để chọn';
}

// ============================================
// #3. TAB CAMERA (Realtime Mode)
// ============================================
let cameraStream = null;
let cameraInterval = null;

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 1280, height: 720 } });
        cameraStream = stream;
        
        const videoEl = document.getElementById('webcamElement');
        const overlay = document.getElementById('webcamOverlay');
        
        videoEl.srcObject = stream;
        
        document.getElementById('btn-start-camera').style.display = 'none';
        document.getElementById('btn-stop-camera').style.display = 'inline-flex';
        document.getElementById('camera-container').style.display = 'flex';
        document.getElementById('camera-processing-status').style.display = 'flex';

        videoEl.onplay = () => {
            if(cameraInterval) clearInterval(cameraInterval);
            cameraInterval = setInterval(() => {
                processVideoFrame(videoEl, overlay);
            }, 100); // ~10 FPS for camera is smoother
        };
        
    } catch (err) {
        alert("Không thể truy cập camera: " + err.message);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    if (cameraInterval) {
        clearInterval(cameraInterval);
        cameraInterval = null;
    }
    
    document.getElementById('webcamElement').srcObject = null;
    
    document.getElementById('btn-start-camera').style.display = 'inline-flex';
    document.getElementById('btn-stop-camera').style.display = 'none';
    document.getElementById('camera-container').style.display = 'none';
    document.getElementById('camera-processing-status').style.display = 'none';
    
    const overlay = document.getElementById('webcamOverlay');
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}
