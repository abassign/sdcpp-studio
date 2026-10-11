// init.js - Inizializzazione
// ===========================

// Carica la configurazione del server incluso il percorso dinamico dei modelli
async function loadServerConfig() {
    try {
        const res = await fetch('/api/config/settings');
        const data = await res.json();
        // Salva il percorso dei modelli come variabile globale
        window.modelsDir = data.models_dir_resolved || '/app/models';
        console.log('--> [Config] Models directory:', window.modelsDir);
        return window.modelsDir;
    } catch (e) {
        console.error('Failed to load server config:', e);
        window.modelsDir = '/app/models'; // Fallback
        return window.modelsDir;
    }
}

async function initApp() {
    // Load the server configuration FIRST
    await loadServerConfig();

    // Slider listeners
    if (sliderW) sliderW.oninput = () => { valW.innerText = sliderW.value; };
    if (sliderH) sliderH.oninput = () => { valH.innerText = sliderH.value; };
    if (sliderSteps) {
        sliderSteps.oninput = () => {
            valSteps.innerText = sliderSteps.value;
            // Dynamically adjust step based on value
            const val = parseInt(sliderSteps.value);
            if (val < 10) {
                sliderSteps.step = 1;
            } else if (val < 20) {
                sliderSteps.step = 2;
            } else {
                sliderSteps.step = 5;
            }
        };
    }
    if (sliderCfg) sliderCfg.oninput = () => { valCfg.innerText = sliderCfg.value; };
    if (sliderTopK) sliderTopK.oninput = () => { valTopK.innerText = sliderTopK.value; };
    if (sliderTemp) sliderTemp.oninput = () => { valTemp.innerText = sliderTemp.value; };
    if (sliderStrength) sliderStrength.oninput = () => { valStrength.innerText = sliderStrength.value; };
    if (sliderMix) sliderMix.oninput = () => { valMix.innerText = sliderMix.value; };
    if (sliderClip && valClip) sliderClip.oninput = () => { valClip.innerText = sliderClip.value; };

    const sliderNoise = document.getElementById('noise-level');
    const valNoise = document.getElementById('val-noise');
    if (sliderNoise && valNoise) {
        sliderNoise.oninput = () => { valNoise.innerText = Math.round(350 * Math.pow(sliderNoise.value / 100, 2)); };
    }

    const sliderScale = document.getElementById('scale-factor');
    const valScale = document.getElementById('val-scale');
    if (sliderScale && valScale) {
        sliderScale.oninput = () => { valScale.innerText = sliderScale.value; };
    }


    // Use the dynamic models path from the configuration
    if (loraFolderInput) loraFolderInput.value = window.modelsDir + "/loras";

    await scanLoras();
    addLoraSlot();
    addLoraSlot();

    await initModelSelector();
    await initVaeSelector();
    await updateSystemBadge();

    switchTab('t2i', document.querySelector('.nav-btn.active'));

    // Log mode button state is already handled by onclick="toggleLogMode()" in HTML

    // Start console observers for both log outputs
    startBothObservers();

    // Start periodic UI validation (check every 1s)
    setInterval(() => {
        if (window.validateInputs) window.validateInputs();
    }, 1000);

    // Initialize semantic tagging controls
    if (typeof addSemanticControls === 'function') {
        addSemanticControls();
    }
}

async function updateSystemBadge() {
    const badge = document.getElementById('gpu-badge');
    if (!badge) return;
    try {
        const res = await fetch('/api/gpu-type');
        const data = await res.json();
        if (data.gpu_type === 'NVIDIA') {
            badge.innerText = 'NVIDIA';
            badge.style.color = '#4ade80';
        } else if (data.gpu_type === 'ROCm') {
            badge.innerText = 'ROCm';
            badge.style.color = '#fbbf24';
        } else {
            badge.innerText = 'Unknown';
            badge.style.color = '#aaa';
        }
    } catch (e) {
        console.error("GPU Type Error", e);
        badge.innerText = 'Error';
        badge.style.color = '#ef4444';
    }
}

// Log mode: 'STATUS' (model logs) or 'SYSTEM' (server logs)
let currentLogMode = 'STATUS';
let consoleObserver = null;   // Observer for console div (model logs)
let systemConsoleObserver = null; // Observer for system console div
let logEventSource = null;   // SSE for server logs
let logTimeout = null;       // Timeout for SSE connection

function updateStatusFromActiveConsole() {
    const activeConsoleId = currentLogMode === 'SYSTEM' ? 'system-console' : 'console';
    const consoleEl = document.getElementById(activeConsoleId);
    const statusText = document.getElementById('status-text');
    if (!consoleEl || !statusText) return;

    const lines = consoleEl.innerText.trim().split('\n');
    const lastLine = lines[lines.length - 1];
    statusText.innerText = lastLine || (currentLogMode === 'SYSTEM' ? 'Connecting to server logs...' : 'Ready');
}

function startBothObservers() {
    stopBothObservers();

    const consoleEl = document.getElementById('console');
    const systemConsoleEl = document.getElementById('system-console');

    const observerCallback = () => {
        updateStatusFromActiveConsole();
    };

    if (consoleEl) {
        consoleObserver = new MutationObserver(observerCallback);
        consoleObserver.observe(consoleEl, { childList: true, subtree: true });
    }

    if (systemConsoleEl) {
        systemConsoleObserver = new MutationObserver(observerCallback);
        systemConsoleObserver.observe(systemConsoleEl, { childList: true, subtree: true });
    }

    updateStatusFromActiveConsole();
}

function stopBothObservers() {
    if (consoleObserver) {
        consoleObserver.disconnect();
        consoleObserver = null;
    }
    if (systemConsoleObserver) {
        systemConsoleObserver.disconnect();
        systemConsoleObserver = null;
    }
}

function toggleLogMode() {
    console.log('toggleLogMode called, current mode:', currentLogMode);
    const btn = document.getElementById('btn-log-mode');
    const statusText = document.getElementById('status-text');
    const consoleEl = document.getElementById('console');
    const systemConsoleEl = document.getElementById('system-console');
    const drawer = document.getElementById('system-log-drawer');

    if (!btn || !statusText || !consoleEl || !systemConsoleEl) {
        console.error('Missing required elements for toggleLogMode');
        return;
    }

    if (currentLogMode === 'STATUS') {
        // Switch to SYSTEM mode (show server logs)
        currentLogMode = 'SYSTEM';
        btn.innerText = 'SYSTEM';
        btn.classList.add('active');
        statusText.classList.add('system-mode');

        consoleEl.classList.add('hidden');
        systemConsoleEl.classList.remove('hidden');
        consoleEl.style.display = 'none';
        systemConsoleEl.style.display = 'block';

        console.log('Switched to SYSTEM mode, starting SSE...');
        startLogStreaming();
    } else {
        // Switch to STATUS mode (show model logs)
        currentLogMode = 'STATUS';
        btn.innerText = 'STATUS';
        btn.classList.remove('active');
        statusText.classList.remove('system-mode');

        systemConsoleEl.classList.add('hidden');
        consoleEl.classList.remove('hidden');
        systemConsoleEl.style.display = 'none';
        consoleEl.style.display = 'block';

        console.log('Switched to STATUS mode, stopping SSE...');
        stopLogStreaming();
    }
    updateStatusFromActiveConsole();
}

function startLogStreaming() {
    if (logEventSource) {
        logEventSource.close();
    }

    try {
        logEventSource = new EventSource('/api/logs/stream');

        logEventSource.onmessage = function (event) {
            const systemConsoleEl = document.getElementById('system-console');
            if (systemConsoleEl) {
                const newLine = event.data;
                const div = document.createElement('div');
                div.textContent = newLine;
                systemConsoleEl.appendChild(div);

                // Keep only last 1000 lines to avoid memory leak
                if (systemConsoleEl.childNodes.length > 1000) {
                    systemConsoleEl.removeChild(systemConsoleEl.firstChild);
                }

                // Auto scroll
                if (systemConsoleEl.scrollHeight - systemConsoleEl.clientHeight <= systemConsoleEl.scrollTop + 100) {
                    systemConsoleEl.scrollTop = systemConsoleEl.scrollHeight;
                }
            }
        };

        logEventSource.onerror = function (error) {
            console.error('Log streaming error:', error);
            // Auto-reconnect after 3 seconds
            clearTimeout(logTimeout);
            logTimeout = setTimeout(() => {
                if (currentLogMode === 'SYSTEM') {
                    startLogStreaming();
                }
            }, 3000);

            // Close broken connection
            if (logEventSource) {
                logEventSource.close();
                logEventSource = null;
            }
        };
    } catch (e) {
        console.error('Failed to start log streaming:', e);
    }
}

function stopLogStreaming() {
    if (logEventSource) {
        logEventSource.close();
        logEventSource = null;
    }
    if (logTimeout) {
        clearTimeout(logTimeout);
        logTimeout = null;
    }
}

window.initApp = initApp;
window.toggleLogMode = toggleLogMode;
console.log('Init loaded');
