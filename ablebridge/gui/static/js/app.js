/**
 * AbleBridge Web UI — JavaScript
 * Socket.IO client for real-time communication with the engine.
 */

// ── State ───────────────────────────────────────────────────────────────────
let socket = null;
let currentText = '';
let shiftActive = false;
let eventLog = [];

// ── Socket.IO Connection ──────────────────────────────────────────────────────
function initSocket() {
    const namespace = '/';
    const url = window.location.origin;
    socket = io(url + namespace, { transports: ['websocket', 'polling'] });

    socket.on('connect', () => {
        console.log('[Socket] Connected');
        document.getElementById('engineStatus').textContent = '🟢';
        updateStatus();
    });

    socket.on('disconnect', () => {
        console.log('[Socket] Disconnected');
        document.getElementById('engineStatus').textContent = '🔴';
    });

    socket.on('connected', (data) => {
        document.getElementById('sessionId').textContent = `Session: ${data.version}`;
    });

    socket.on('engine_event', (event) => {
        handleEngineEvent(event);
    });

    socket.on('predictions', (data) => {
        renderPredictions(data.predictions);
    });
}

// ── Keyboard ─────────────────────────────────────────────────────────────────
function initKeyboard() {
    const keyboard = document.getElementById('keyboard');
    if (!keyboard) return;

    keyboard.addEventListener('click', (e) => {
        const key = e.target.closest('.key');
        if (!key) return;

        const keyValue = key.dataset.key;
        handleKeyPress(keyValue);
    });

    // Physical keyboard support
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        if (e.key === 'Shift') {
            shiftActive = !shiftActive;
            document.querySelectorAll('.key').forEach(k => {
                if (k.dataset.key.length === 1) {
                    k.textContent = shiftActive ? k.dataset.key.toUpperCase() : k.dataset.key.toLowerCase();
                }
            });
        } else if (e.key === 'Backspace') {
            handleKeyPress('BACKSPACE');
        } else if (e.key === ' ') {
            e.preventDefault();
            handleKeyPress(' ');
        } else if (e.key.length === 1) {
            handleKeyPress(shiftActive ? e.key.toUpperCase() : e.key.toLowerCase());
        }
    });
}

function handleKeyPress(key) {
    if (key === 'SHIFT') {
        shiftActive = !shiftActive;
        return;
    }

    if (key === 'BACKSPACE') {
        if (currentText.length > 0) {
            currentText = currentText.slice(0, -1);
            updateTextDisplay();
            fetchPredictions();
        }
        return;
    }

    if (key === ' ') {
        key = ' ';
    }

    currentText += key;
    updateTextDisplay();
    fetchPredictions();

    // Send to engine
    if (socket && socket.connected) {
        socket.emit('key_press', { key, current_text: currentText });
    }
}

// ── Text Display ───────────────────────────────────────────────────────────────
function updateTextDisplay() {
    const display = document.getElementById('textDisplay');
    if (!display) return;

    if (currentText.trim() === '') {
        display.innerHTML = '<span class="placeholder-text">Your message will appear here...</span>';
    } else {
        display.textContent = currentText;
    }
}

function clearText() {
    currentText = '';
    updateTextDisplay();
    clearPredictions();
}

function copyText() {
    if (currentText) {
        navigator.clipboard.writeText(currentText).then(() => {
            showToast('Copied to clipboard!');
        });
    }
}

function speakText() {
    if (!currentText.trim()) return;

    fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: currentText }),
    }).then(() => {
        document.getElementById('speakBtn').textContent = '🔊 Speaking...';
        setTimeout(() => {
            document.getElementById('speakBtn').textContent = '🔊 Speak';
        }, 2000);
    });
}

// ── Predictions ───────────────────────────────────────────────────────────────
function fetchPredictions() {
    if (!currentText.trim()) {
        clearPredictions();
        return;
    }

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context: currentText }),
    })
    .then(r => r.json())
    .then(data => renderPredictions(data.predictions || []))
    .catch(() => {});
}

function renderPredictions(predictions) {
    const bar = document.getElementById('predictionsBar');
    if (!bar) return;

    bar.innerHTML = predictions
        .filter(p => p.confidence > 0.05)
        .map((p, i) => `
            <button class="prediction-chip" onclick="applyPrediction('${escapeHtml(p.text)}')" 
                    style="opacity:${Math.min(1, 0.5 + p.confidence * 0.5)}">
                ${escapeHtml(p.text)}${p.confidence > 0.5 ? ' ✓' : ''}
            </button>
        `).join('');
}

function clearPredictions() {
    const bar = document.getElementById('predictionsBar');
    if (bar) bar.innerHTML = '';
}

function applyPrediction(word) {
    currentText += (currentText && !currentText.endsWith(' ') ? ' ' : '') + word + ' ';
    updateTextDisplay();
    fetchPredictions();
}

// ── Quick Actions ─────────────────────────────────────────────────────────────
function quickAction(text, isEmergency = false) {
    currentText = text;
    updateTextDisplay();
    speakText();

    if (isEmergency) {
        // Emergency: also dispatch high-priority output
        fetch('/api/input', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'emergency', value: text }),
        });
    }
}

// ── Status ─────────────────────────────────────────────────────────────────────
function updateStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(status => {
            // Update channel status
            Object.entries(status.inputs || {}).forEach(([ch, info]) => {
                const el = document.getElementById(`ch-${ch}`);
                if (!el) return;
                const badge = el.querySelector('.channel-badge');
                if (badge) {
                    const state = info.state || 'stopped';
                    badge.className = 'channel-badge';
                    if (state === 'running') {
                        badge.classList.add('status-active');
                        badge.textContent = `${Math.round((info.confidence || 0) * 100)}%`;
                    } else if (state === 'error') {
                        badge.classList.add('status-error');
                        badge.textContent = 'Error';
                    } else {
                        badge.classList.add('status-off');
                        badge.textContent = 'Off';
                    }
                }
            });

            // Update AI status
            const aiInfo = status.ai || {};
            document.getElementById('intentStatus').textContent = aiInfo.intent || 'None';
            document.getElementById('predictionStatus').textContent = aiInfo.prediction || 'None';

            // Update session
            if (status.session_id) {
                document.getElementById('sessionId').textContent = 
                    `Session: ${status.session_id.slice(-8)}`;
            }
        })
        .catch(err => console.error('Status update failed:', err));
}

// ── Engine Events ─────────────────────────────────────────────────────────────
function handleEngineEvent(event) {
    const { type, payload_type, payload, timestamp } = event;

    eventLog.unshift({ type, payload_type, payload, timestamp });
    if (eventLog.length > 100) eventLog.pop();

    // Update log UI if visible
    const logContent = document.getElementById('logContent');
    if (logContent && logContent.style.display !== 'none') {
        const entry = eventLog.slice(0, 20).map(e =>
            `<div class="log-entry ${getLogClass(e.type)}">[${new Date(e.timestamp * 1000).toLocaleTimeString()}] ${e.type}: ${e.payload_type}</div>`
        ).join('');
        logContent.innerHTML = entry;
    }

    // React to specific events
    if (type === 'INTENT_RESOLVED' && payload) {
        if (payload.response) {
            currentText = payload.response;
            updateTextDisplay();
        }
    }
}

function getLogClass(type) {
    if (type.includes('INPUT')) return 'input';
    if (type.includes('OUTPUT')) return 'output';
    if (type.includes('INTENT')) return 'intent';
    if (type.includes('ERROR')) return 'error';
    return '';
}

// ── Log Toggle ────────────────────────────────────────────────────────────────
function toggleLog() {
    const content = document.getElementById('logContent');
    const toggle = document.getElementById('logToggle');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.textContent = '▲';
    } else {
        content.style.display = 'none';
        toggle.textContent = '▼';
    }
}

// ── Calibration ────────────────────────────────────────────────────────────────
function openCalibration() {
    document.getElementById('calibrationModal').style.display = 'flex';
}

function closeCalibration() {
    document.getElementById('calibrationModal').style.display = 'none';
}

function startCalibration() {
    fetch('/api/calibrate', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.calibrated) {
                showToast('Calibration complete!');
            } else {
                showToast('Calibration failed. Check camera.');
            }
        });
    closeCalibration();
}

// ── Settings ──────────────────────────────────────────────────────────────────
function openSettings() {
    document.getElementById('settingsModal').style.display = 'flex';
    // Load profiles
    fetch('/api/profile')
        .then(r => r.json())
        .then(data => {
            const select = document.getElementById('profileSelect');
            select.innerHTML = data.profiles.map(p =>
                `<option value="${p}" ${p === data.current ? 'selected' : ''}>${p}</option>`
            ).join('');
        });
}

function closeSettings() {
    document.getElementById('settingsModal').style.display = 'none';
}

function switchProfile(profileId) {
    fetch('/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: profileId }),
    }).then(() => {
        showToast(`Profile switched to: ${profileId}`);
        updateStatus();
    });
}

function setTTSSpeed(wpm) {
    document.getElementById('ttsSpeedValue').textContent = `${wpm} WPM`;
    // Note: Actual TTS rate setting requires engine API
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('ablebridge-theme', theme);
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(message) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
        background: var(--text-primary); color: var(--bg-primary);
        padding: 10px 20px; border-radius: 99px; font-size: 0.9rem;
        z-index: 9999; animation: fadeIn 0.2s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2500);
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Apply saved theme
    const savedTheme = localStorage.getItem('ablebridge-theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
        document.getElementById('themeSelect').value = savedTheme;
    }

    initSocket();
    initKeyboard();

    // Periodic status refresh
    setInterval(updateStatus, 5000);
});
