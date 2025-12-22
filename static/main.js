/**
 * Agentic Shell - Client JavaScript
 * 
 * This client handles the web interface for the Agentic Shell application,
 * including terminal emulation, chat interface, voice input, and Socket.IO
 * communication with the backend server.
 */

// Configure marked.js for markdown rendering
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false
});

// Initialize Socket.IO connection
const socket = io();

// Detect color scheme preference
const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

// Terminal theme configuration
const darkTheme = {
    background: '#1a1a1a',
    foreground: '#e0e0e0',
    cursor: '#4a9eff',
    cursorAccent: '#1a1a1a',
    selectionBackground: '#3a5a7a',
    black: '#1a1a1a',
    red: '#e74c3c',
    green: '#5fb76f',
    yellow: '#f39c12',
    blue: '#4a9eff',
    magenta: '#9b59b6',
    cyan: '#3498db',
    white: '#e0e0e0',
    brightBlack: '#555555',
    brightRed: '#ff6b6b',
    brightGreen: '#7ed68d',
    brightYellow: '#ffc952',
    brightBlue: '#6db1ff',
    brightMagenta: '#b77dd4',
    brightCyan: '#5dade2',
    brightWhite: '#ffffff'
};

const lightTheme = {
    background: '#f5f5f5',
    foreground: '#2a2a2a',
    cursor: '#0066cc',
    cursorAccent: '#f5f5f5',
    selectionBackground: '#b3d9ff',
    black: '#2a2a2a',
    red: '#c0392b',
    green: '#27ae60',
    yellow: '#f39c12',
    blue: '#0066cc',
    magenta: '#8e44ad',
    cyan: '#16a085',
    white: '#666666',
    brightBlack: '#7f8c8d',
    brightRed: '#e74c3c',
    brightGreen: '#2ecc71',
    brightYellow: '#f1c40f',
    brightBlue: '#3498db',
    brightMagenta: '#9b59b6',
    brightCyan: '#1abc9c',
    brightWhite: '#000000'
};

// Initialize xterm.js terminal with appropriate theme
const term = new Terminal({
    cursorBlink: true,
    fontSize: 14,
    fontFamily: 'Courier New, monospace',
    theme: isDarkMode ? darkTheme : lightTheme
});

// Load and configure fit addon for responsive terminal sizing
const fitAddon = new FitAddon.FitAddon();
term.loadAddon(fitAddon);
term.open(document.getElementById('terminal-container'));
fitAddon.fit();

// Resize terminal when window size changes
window.addEventListener('resize', () => fitAddon.fit());

// Forward terminal input to server via Socket.IO
term.onData(data => socket.emit('pty_input', { 'input': data }));

// Only forward non-agent terminal output
socket.on('pty_output', data => {
    if (!data.fromAgent) {
        term.write(data.output);
    }
});

// DOM Element References
const els = {
    agentBar: document.getElementById('agent-bar'),
    agentInput: document.getElementById('agent-input'),
    approvalButtons: document.getElementById('approval-buttons'),
    sendBtn: document.getElementById('send-btn'),
    modeToggleBtn: document.getElementById('mode-toggle-btn'),
    modelToggleBtn: document.getElementById('model-toggle-btn'),
    loadingContainer: document.getElementById('loading-container'),
    voiceBtn: document.getElementById('voice-btn'),
    languageSelect: document.getElementById('language-select'),
    chatMessages: document.getElementById('chat-messages'),
    clearChatBtn: document.getElementById('clear-chat-btn')
};

/**
 * Add a message to the chat panel
 * @param {string} content - Message content
 * @param {string} type - Message type ('agent' or 'user')
 * @param {string} icon - Emoji icon for the message
 */
const addChatMessage = (content, type = 'agent', icon = '🤖') => {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    // Display "AGENT" for all non-user messages
    const displayName = type === 'user' ? 'USER' : 'AGENT';
    headerDiv.innerHTML = `<span class="message-icon">${icon}</span><span>${displayName}</span>`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (type === 'user') {
        contentDiv.textContent = content;
    } else {
        // Preprocess content: remove markdown code fence wrappers if present
        let processedContent = content.trim();
        
        // Check if content is wrapped in ```markdown or similar code fence
        const codeFencePattern = /^```(?:markdown|md)?\s*\n([\s\S]*?)\n```$/;
        const match = processedContent.match(codeFencePattern);
        
        if (match) {
            // Extract the actual markdown content from inside the code fence
            processedContent = match[1].trim();
        }
        
        // Render the markdown with GitHub styling
        contentDiv.className = 'message-content markdown-body';
        contentDiv.innerHTML = marked.parse(processedContent);
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    els.chatMessages.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    
    return messageDiv;
};

// Socket event handlers for different message types
socket.on('agent_message', data => {
    addChatMessage(data.message, 'agent', '🤖');
});

// Application State
let currentMode = 'ask';
let currentModel = 'gemini';
let isAgentWorking = false;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// Mode Configuration
const modes = {
    task: { text: 'Task Mode', placeholder: 'Enter a task for the agent...' },
    ask: { text: 'Ask Mode', placeholder: 'Ask a question...' },
    auto: { text: 'Auto Mode', placeholder: 'Enter a task to auto-execute...' },
    iterative: { text: 'Iterative Mode', placeholder: 'Enter a task for adaptive execution...' }
};

const modeOrder = ['ask', 'task', 'auto', 'iterative'];

/**
 * Get terminal context from the last 20 lines of visible output
 * @returns {string} Terminal context string
 */
const getTerminalContext = () => {
    const buffer = term.buffer.active;
    const lines = [];
    for (let i = Math.max(0, buffer.cursorY - 20); i <= buffer.cursorY; i++) {
        const line = buffer.getLine(i);
        if (line) {
            lines.push(line.translateToString(true));
        }
    }
    return lines.join('\n');
};

/**
 * Update send button enabled state based on input and agent state
 */
const updateSendButton = () => {
    els.sendBtn.disabled = !els.agentInput.value.trim() || isAgentWorking;
};

/**
 * Show or hide the agent control bar
 * @param {boolean} visible - Whether the control bar should be visible
 */
const setControlBarVisibility = (visible) => {
    if (visible) {
        els.agentBar.style.visibility = 'visible';
        els.agentBar.style.pointerEvents = 'auto';
    } else {
        els.agentBar.style.visibility = 'hidden';
        els.agentBar.style.pointerEvents = 'none';
    }
};

/**
 * Update mode toggle button display based on current mode
 */
const updateModeToggle = () => {
    const mode = modes[currentMode];
    els.modeToggleBtn.querySelector('.toggle-text').textContent = mode.text;
    els.modeToggleBtn.setAttribute('data-mode', currentMode);
    els.agentInput.placeholder = mode.placeholder;
};

/**
 * Update model toggle button display based on current model
 */
const updateModelToggle = () => {
    els.modelToggleBtn.setAttribute('data-model', currentModel);
    els.modelToggleBtn.querySelector('.model-text').textContent = 
        currentModel === 'gemini' ? 'Gemini' : 'Qwen';
};

/**
 * Update voice button UI based on recording state
 * @param {boolean} recording - Whether currently recording
 */
const updateVoiceButton = (recording) => {
    els.voiceBtn.classList.toggle('recording', recording);
    els.voiceBtn.innerHTML = recording 
        ? '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>'
        : '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>';
};

/**
 * Get full language code from short language identifier
 * @param {string} lang - Short language code (e.g., 'en', 'hi')
 * @returns {string} Full language code (e.g., 'en-IN', 'hi-IN')
 */
const getLanguageCode = lang => {
    const languageCodes = {
        'en': 'en-IN', 'hi': 'hi-IN', 'bn': 'bn-IN', 'te': 'te-IN',
        'mr': 'mr-IN', 'ta': 'ta-IN', 'gu': 'gu-IN', 'kn': 'kn-IN',
        'ml': 'ml-IN', 'pa': 'pa-IN', 'or': 'or-IN', 'as': 'as-IN'
    };
    return languageCodes[lang] || 'en-IN';
};

// Event Handlers

// Agent prompt form submission
els.agentBar.addEventListener('submit', e => {
    e.preventDefault();
    const prompt = els.agentInput.value.trim();
    
    if (prompt && !isAgentWorking) {
        // Add user message to chat
        addChatMessage(prompt, 'user', '👤');
        
        // Send prompt to server
        socket.emit('agent_prompt', { 
            prompt: prompt,
            context: getTerminalContext(),
            mode: currentMode,
            model: currentModel,
            language: els.languageSelect.value 
        });
        
        // Clear input and update button state
        els.agentInput.value = '';
        updateSendButton();
    }
});

// Input field change handler
els.agentInput.addEventListener('input', updateSendButton);

// Mode toggle button click handler
els.modeToggleBtn.addEventListener('click', () => {
    const currentIndex = modeOrder.indexOf(currentMode);
    const nextIndex = (currentIndex + 1) % modeOrder.length;
    currentMode = modeOrder[nextIndex];
    updateModeToggle();
});

// Model toggle button click handler
els.modelToggleBtn.addEventListener('click', () => {
    currentModel = currentModel === 'gemini' ? 'qwen' : 'gemini';
    updateModelToggle();
});

// Approval and cancellation button handlers
document.getElementById('approve-btn').addEventListener('click', () => {
    socket.emit('user_approval', { approved: true });
});

document.getElementById('reject-btn').addEventListener('click', () => {
    socket.emit('user_approval', { approved: false });
});

document.getElementById('cancel-btn').addEventListener('click', () => {
    socket.emit('cancel_agent');
});

// Language selection change handler
els.languageSelect.addEventListener('change', () => {
    els.voiceBtn.disabled = els.languageSelect.value === 'off';
});

// Clear chat button handler
els.clearChatBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear the chat?')) {
        els.chatMessages.innerHTML = '';
    }
});

// Socket Event Handlers

// Show/hide approval buttons based on server state
socket.on('show_approval_buttons', data => {
    els.approvalButtons.style.display = data.show ? 'flex' : 'none';
    setControlBarVisibility(!data.show);
});

// Update UI based on agent working state
socket.on('agent_working', data => {
    isAgentWorking = data.working;
    els.loadingContainer.style.display = isAgentWorking ? 'flex' : 'none';
    setControlBarVisibility(!isAgentWorking);
    updateSendButton();
});

// Handle transcription results from voice input
socket.on('transcription_result', data => {
    if (data.text) {
        els.agentInput.value = data.text;
        els.agentInput.focus();
        updateSendButton();
    }
});

// Handle transcription errors
socket.on('transcription_error', data => {
    console.error('Transcription error:', data.error);
});

// Voice Recording Functions

/**
 * Start recording audio from the microphone
 */
const startRecording = async () => {
    if (els.languageSelect.value === 'off' || isRecording) {
        return;
    }
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 48000,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        const options = { mimeType: 'audio/webm;codecs=opus' };
        mediaRecorder = MediaRecorder.isTypeSupported(options.mimeType) 
            ? new MediaRecorder(stream, options)
            : new MediaRecorder(stream);
        
        audioChunks = [];
        
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const reader = new FileReader();
            
            reader.onloadend = () => {
                socket.emit('transcribe_audio', {
                    audio: reader.result,
                    language: getLanguageCode(els.languageSelect.value)
                });
            };
            
            reader.readAsDataURL(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        isRecording = true;
        updateVoiceButton(true);
    } catch (error) {
        console.error('Microphone error:', error);
        alert('Could not access microphone. Please check permissions.');
    }
};

/**
 * Stop recording audio
 */
const stopRecording = () => {
    if (!isRecording || !mediaRecorder) {
        return;
    }
    
    mediaRecorder.stop();
    isRecording = false;
    updateVoiceButton(false);
};

// Voice button toggle event
els.voiceBtn.addEventListener('click', e => {
    e.preventDefault();
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// Initialize application state
updateSendButton();
updateModeToggle();
updateModelToggle();
els.voiceBtn.disabled = els.languageSelect.value === 'off';