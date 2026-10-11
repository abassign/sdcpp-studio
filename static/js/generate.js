// generate.js - Generation logic
// ===============================

let isGenerating = false;
window.isProcessing = false;
let abortController = null;

function initGenerate() {
    if (btnGenerate) {
        btnGenerate.onclick = async function () {
            if (window.isProcessing) {
                if (confirm("Abort current process?")) {
                    // 1. Tell backend to kill process
                    fetch('/api/generate/stop', { method: 'POST' });
                    // 2. Abort client stream
                    if (abortController) abortController.abort();
                    // 3. Reset UI immediately
                    stopGeneration();
                    updateStatusGen("Aborted by User", "warning");
                }
                return;
            }

            // Single unified handler - always uses sdcpp backend
            handleGeneration();
        };
    }
    
    // Edit Image button
    const btnEditImage = document.getElementById('btn-edit-image');
    if (btnEditImage) {
        btnEditImage.onclick = function () {
            if (!window.isProcessing) {
                // Auto-select qwen_image_vae for Qwen Image Edit mode
                autoSelectQwenImageVAE();
                // For Edit Image button, use edit backend (i2i_edit mode)
                handleGeneration('edit');
            }
        };
    }

    if (btnStop) {
        btnStop.onclick = function () {
            fetch('/api/generate/stop', { method: 'POST' });
            stopGeneration();
        };
    }

    if (btnExit) {
        btnExit.onclick = function () { exitButtonClicked(); };
    }

    // Watch for prompt changes to ask about keeping IA metadata
    setupPendingIAWatch();
}

function setGenerationState(processing) {
    window.isProcessing = processing;

    if (processing) {
        if (btnGenerate) {
            btnGenerate.innerText = "STOP";
            btnGenerate.title = "Click to Abort";
            btnGenerate.classList.add("btn-stop-active");
            btnGenerate.disabled = false;
            btnGenerate.style.opacity = "1";
            btnGenerate.style.cursor = "pointer";
            btnGenerate.classList.remove('hidden');
        }
        const btnEditImage = document.getElementById('btn-edit-image');
        if (btnEditImage) btnEditImage.classList.add('hidden');
        updateStatusGen("Processing...", "info");
    } else {
        if (btnGenerate) {
            if (window.currentMode === 'i2t') btnGenerate.innerText = "ANALYZE IMAGE";
            else if (window.currentMode === 'i2i') btnGenerate.innerText = "TRANSFORM IMAGE";
            else btnGenerate.innerText = "GENERATE";
            btnGenerate.classList.remove("btn-stop-active");
        }
        const btnEditImage = document.getElementById('btn-edit-image');
        if (btnEditImage) btnEditImage.classList.add('hidden');
    }

    // Update button states immediately
    if (window.validateInputs) window.validateInputs();
}

async function handleGeneration(backend) {
    // If called without arguments, default to 'cpp' (the only backend now)
    backend = backend || 'cpp';

    if (isGenerating) { stopGeneration(); return; }

    // Clear log at start of new generation
    if (consoleDiv) consoleDiv.innerHTML = '';

    const prompt = promptBox?.value?.trim();
    if (!prompt && window.currentMode !== 'i2t') {
        alert('Please enter a prompt');
        return;
    }

    // --- I2T Interactive path (always active for I2T) ---
    // Uses the persistent /api/i2t/query endpoint via llama-server
    if (window.currentMode === 'i2t') {
        const allSlots = (typeof getImageSlots === 'function') ? getImageSlots() : [];
        const validSlots = allSlots.filter(s => s.path && s.role !== 'none');
        if (validSlots.length === 0) {
            alert('Please upload at least one image first.');
            stopGeneration();
            return;
        }
        const i2tModelSelect = document.getElementById('i2t-model-select');
        const modelDir = i2tModelSelect?.value || null;
        const imageSpecs = validSlots.map(s => ({ path: s.path, role: s.role }));
        const firstImagePath = validSlots[0]?.path || null;
        await runI2TInteractive(firstImagePath, prompt, modelDir, null, imageSpecs);
        return;
    }

    isGenerating = true;
    
    // Stop any active tagging to free VRAM for generation
    if (typeof stopAutoTag === "function") {
        stopAutoTag();
    }
    setGenerationState(true);
    startTimer();

    const payload = buildPayload(backend);
    // Save IA data for cache update after generation
    const iaDataForCache = payload.image_analysis || null;
    // Clear pending IA export after consuming it
    window.pendingIAExport = null;

    // Update cache immediately so the IA button activates without re-fetching JSON
    if (iaDataForCache) {
        window.cachedImageAnalysis = iaDataForCache;
    }

    abortController = new AbortController();

    try {
        // All generation modes now route through /api/generate (sdcpp backend)
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: abortController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: IMG|')) {
                    const imgPath = line.replace('data: IMG|', '').trim();
                    displayResult(imgPath);
                    updateStatusGen('Done!', 'success');
                } else if (line.startsWith('data: TXT|')) {
                    const textContent = line.replace('data: TXT|', '').trim().replace(/\\n/g, '\n');
                    displayTextResult(textContent);
                    updateStatusGen('Analysis Complete!', 'success');
                } else if (line.startsWith('data: ERROR|')) {
                    const errorJson = line.replace('data: ERROR|', '').trim();
                    try {
                        const errorData = JSON.parse(errorJson);
                        const errorMsg = errorData.error || 'Unknown error';
                        const errorDetails = errorData.details || '';
                        const errorCode = errorData.code || '';
                        alert(`❌ Error: ${errorMsg}\n\nCode: ${errorCode}\n\nDetails: ${errorDetails}`);
                    } catch (e) {
                        alert('❌ Error: ' + errorJson);
                    }
                    updateStatusGen('Error!', 'error');
                } else if (line.startsWith('data: LOG|')) {
                    const logMsg = line.replace('data: LOG|', '').trim();
                    appendLog(logMsg);
                } else if (line.startsWith('data: ERR|')) {
                    const errMsg = line.replace('data: ERR|', '').trim();
                    alert('❌ Error: ' + errMsg);
                    updateStatusGen('Error!', 'error');
                } else if (line.startsWith('data: TAGGING_RESUME|')) {
                    // T2I/I2I generation preempted an auto-tag batch.
                    // The backend has restored the tag server and saved
                    // a checkpoint; trigger the resume automatically
                    // so the user doesn't have to click anything.
                    const resumeFrom = (line.split('from=')[1] || '0').trim();
                    console.log('[Generate] Auto-resuming tagging from index', resumeFrom);
                    if (typeof window.resumeTagging === 'function') {
                        // Small delay so the popup is fully closed by
                        // the [Stop Tagging] call earlier in the flow.
                        setTimeout(() => window.resumeTagging(), 250);
                    }
                }
            }
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            console.log('Generation aborted by user.');
        } else {
            console.error('Generation error:', e);
            updateStatusGen('Error: ' + e.message, 'error');
        }
    }

    stopGeneration();
}

/**
 * Normalize an image path for the backend: strip /outputs/ or /uploads/ URL prefix
 * so the backend can resolve it via OUTPUT_DIR.
 * @param {string|null} path
 * @returns {string|null}
 */
function normalizeImagePath(path) {
    if (!path) return null;
    if (path.startsWith('/outputs/')) {
        return path.replace('/outputs/', '');
    }
    if (path.startsWith('/uploads/')) {
        return path.replace('/uploads/', '');
    }
    return path;
}

function buildPayload(backend) {
    // Mode map: all modes now use sdcpp backends directly
    const modeMap = { 't2i': 't2i_cpp', 'i2i': 'i2i_cpp' };
    let mode = modeMap[window.currentMode] || 't2i_cpp';


    // === Phase 5: N-Slot Multi-Image Role-Based Mode Mapping ===
    const allImageSlots = (typeof getImageSlots === 'function') ? getImageSlots() : [];
    const faceSlot = allImageSlots.find(s => s.role === 'face');

    if ((mode === 'i2i_cpp') && faceSlot) {
        mode = 'i2i_faceswap';
        console.log('[buildPayload] Found face role slot -> i2i_faceswap mode');
    }

    // Model path selection
    let modelPath = modelSelect?.value;

    // === Qwen-Image-Edit auto-routing ===
    // The qwen-image-edit GGUF is a diffusion-only *edit* model: the generic
    // i2i_cpp (img2img) worker cannot open it as a checkpoint (-m) and sd.cpp
    // dies with "get sd version from file failed". Any I2I request with this
    // model selected must use the dedicated i2i_edit worker (semantic edit:
    // prompt + reference image + Qwen2.5-VL text encoder).
    const qwenEditSelected = !!modelPath && (window.isQwenImageEditModel ? window.isQwenImageEditModel(modelPath) : /qwen-image-edit|qwen-rapid/i.test(modelPath.split('/').pop() || modelPath));
    if (qwenEditSelected && mode === 'i2i_cpp') {
        mode = 'i2i_edit';
        console.log('[buildPayload] Qwen-Image-Edit model selected -> i2i_edit mode');
    } else if (qwenEditSelected && window.currentMode === 't2i') {
        console.warn('[buildPayload] Qwen-Image-Edit is an edit model: switch to I2I with an input image.');
    }
    // Explicit 'edit' backend request (legacy Edit-Image button)
    if (backend === 'edit') {
        mode = 'i2i_edit';
        console.log('[buildPayload] Explicit edit backend -> i2i_edit mode');
    }
    const i2tModelSelect = document.getElementById('i2t-model-select');


    // Slider values
    let topK = parseFloat(sliderTopK?.value || 1.0);
    let temperature = parseFloat(sliderTemp?.value || 0.6);
    let strength = parseFloat(sliderStrength?.value || 0.75);
    let mixRatio = 0.5;
    let guidance = parseFloat(sliderCfg?.value || 1.5);

    
    // === Phase 5: Compute images array from imageSlots ===
    const imageSlotsPayload = (typeof getImageSlots === 'function') ? getImageSlots() : [];
    let imagesArray;
    if (imageSlotsPayload.length > 0) {
        imagesArray = imageSlotsPayload.map(s => ({ path: s.path, role: s.role }));
    } else {
        imagesArray = [];
        const upi1 = window.uploadedPathInput || document.getElementById('uploaded-path');
        const upi2 = window.uploadedPathInput2 || document.getElementById('uploaded-path-2');
        if (upi1?.value) {
            imagesArray.push({ path: upi1.value, role: (typeof getSlotRole === 'function') ? getSlotRole(1) : 'init' });
        }
        if (upi2?.value) {
            imagesArray.push({ path: upi2.value, role: (typeof getSlotRole === 'function') ? getSlotRole(2) : 'none' });
        }
    }
    imagesArray = imagesArray.filter(img => img.role && img.role !== 'none' && img.path);
    // Normalize all image paths to relative format for the backend
    imagesArray = imagesArray.map(img => ({
        ...img,
        path: normalizeImagePath(img.path) || img.path
    }));

    // Backward compat: extract single-image fields
    const initImgSlot = imagesArray.find(s => s.role === 'init');
    const refImgSlots = imagesArray.filter(s => s.role === 'ref');
    const faceImgSlot = imagesArray.find(s => s.role === 'face');
    const maskImgSlot = imagesArray.find(s => s.role === 'mask');
    const controlImgSlot = imagesArray.find(s => s.role === 'control');
    const blendImgSlot = imagesArray.find(s => s.role === 'blend');

    return {
        mode: mode,
        prompt: promptBox?.value || '',
        negative_prompt: (negativePromptBox?.value?.trim() && negativePromptBox?.style.display !== 'none') ? negativePromptBox?.value?.trim() : '',
        width: parseInt(sliderW?.value || 1024),
        height: parseInt(sliderH?.value || 1024),
        steps: parseInt(sliderSteps?.value || 15),
        guidance: guidance,
        use_slg: document.getElementById('use-slg')?.checked || false,
        use_apg: document.getElementById('use-apg')?.checked || false,
        seed: parseInt(seedInput?.value) || -1,
        randomize: randomizeCheckbox?.checked || false,
        loras: getLoraConfigs(),
        top_k: topK,
        temperature: temperature,
        strength: strength,
        mix_ratio: mixRatio,
        model_path: modelPath,
        vae_path: vaeSelect?.value || null,
        scheduler: schedulerSelect?.value || 'euler',
        scheduler_type: schedulerTypeSelect?.value || 'auto',
        images: imagesArray,
        init_image: normalizeImagePath(initImgSlot ? initImgSlot.path : ((window.uploadedPathInput || document.getElementById('uploaded-path'))?.value || null)),
        init_image_2: normalizeImagePath((refImgSlots[0] || faceImgSlot || maskImgSlot || controlImgSlot || blendImgSlot)?.path || null),
        slot1_role: (typeof getSlotRole === 'function') ? getSlotRole(1) : 'init',
        slot2_role: (typeof getSlotRole === 'function') ? getSlotRole(2) : 'none',
        ref_image: normalizeImagePath(refImgSlots[0]?.path || null),
        face_image: normalizeImagePath(faceImgSlot?.path || null),
        mask_image: normalizeImagePath(maskImgSlot?.path || null),
        control_image: normalizeImagePath(controlImgSlot?.path || null),
        control_strength: 0.8,
        pm_style_strength: 20.0,
        face_enhance: faceEnhanceCheckbox?.checked || false,
        scale: parseFloat(document.getElementById('scale-factor')?.value || 4),
        noise_level: parseInt(document.getElementById('noise-level')?.value || 20),
        text_encoder: llmSelect?.value === "default" ? null : llmSelect?.value,
        ...(mode === 'analyze_llama' && { i2t_model_path: modelPath }),
        image_analysis: window.pendingIAExport || exportIAState(),
        // Max image dimension for VRAM-saving resize (I2I input images)
        max_img_size: parseInt(document.getElementById('maximg-i2i')?.value || 1024)
    };
}

function displayResult(path) {
    // For I2T mode, hide the image result panel - only text results should be shown
    if (window.currentMode === 'i2t') {
        if (resultImg) resultImg.style.display = 'none';
        if (placeholder) placeholder.style.display = 'none';
        // Keep the text result area visible for I2T
        return;
    }

    if (resultImg) {
        resultImg.src = path;
        resultImg.style.display = 'block';
    }
    if (placeholder) placeholder.style.display = 'none';
    window.currentResultPath = path;

    // Refresh history panel if it's open
    const historyPanel = document.getElementById('history-panel');
    if (historyPanel && !historyPanel.classList.contains('collapsed')) {
        if (window.loadHistory) {
            window.loadHistory();
        }
    }

    // Update preferences statistics after successful generation
    if (window.Preferences && window.Preferences.updateStatsAfterGeneration) {
        const payload = buildPayload('cpp');
        window.Preferences.updateStatsAfterGeneration(
            payload.mode,
            payload.model_path,
            payload
        );
    }

    // Check image JSON to enable/disable toolbar buttons based on metadata
    checkImageMetadataAndUpdateButtons(path);
}

/**
 * Fetch the JSON metadata for the given image path and update
 * the image toolbar buttons accordingly:
 *  - 🧠 IA button: enabled only if JSON has "image_analysis" key
 *  - ➜ I2I button: enabled only if JSON has source_images data
 * Also caches the image_analysis data in window.cachedImageAnalysis for sendToIA().
 * @param {string} imagePath - Path to the result image
 */
function checkImageMetadataAndUpdateButtons(imagePath) {
    const btnIA = document.getElementById('btn-ia-toolbar');
    const btnI2I = document.getElementById('btn-i2i-toolbar');

    if (!imagePath) return;

    // ARF is enabled whenever there is a real result image (no metadata needed)
    if (window.setArfToolbar) window.setArfToolbar(true);

    const jsonPath = imagePath.replace(/\.(png|jpg|jpeg)$/i, '.json');

    fetch(jsonPath)
        .then(res => {
            if (!res.ok) throw new Error('JSON not found');
            return res.json();
        })
        .then(data => {
            const newIAData = data.image_analysis || null;

            // If the new image has IA data, always update the cache.
            // If it doesn't, keep the existing cache (don't discard).
            if (newIAData !== null) {
                window.cachedImageAnalysis = newIAData;
            }

            // Update buttons based on cache state
            const hasIA = window.cachedImageAnalysis !== null;
            if (btnIA && hasIA) {
                btnIA.disabled = false;
                btnIA.style.opacity = '1';
                btnIA.style.cursor = 'pointer';
                btnIA.title = 'Send to Image Analysis (IA data found)';
            } else if (btnIA) {
                btnIA.disabled = true;
                btnIA.style.opacity = '0.4';
                btnIA.style.cursor = 'not-allowed';
                btnIA.title = 'No IA metadata available for this image';
            }

            // Check for source_images
            const sourceImages = data?.inputs?.source_images || data?.image_analysis?.source_images;
            const hasSourceImages = sourceImages && Array.isArray(sourceImages) && sourceImages.length > 0 && sourceImages[0];
            if (btnI2I && hasSourceImages) {
                btnI2I.disabled = false;
                btnI2I.style.opacity = '1';
                btnI2I.style.cursor = 'pointer';
                btnI2I.title = 'Send to Image-to-Image (source image found)';
            } else if (btnI2I) {
                btnI2I.disabled = true;
                btnI2I.style.opacity = '0.4';
                btnI2I.style.cursor = 'not-allowed';
                btnI2I.title = 'No source image available for this result';
            }
        })
        .catch(() => {
            // JSON fetch failed — keep existing cache state
            // Update buttons based on existing cache
            const hasIA = window.cachedImageAnalysis !== null;
            if (btnIA) {
                if (hasIA) {
                    btnIA.disabled = false;
                    btnIA.style.opacity = '1';
                    btnIA.style.cursor = 'pointer';
                    btnIA.title = 'Send to Image Analysis (cached IA data)';
                } else {
                    btnIA.disabled = true;
                    btnIA.style.opacity = '0.4';
                    btnIA.style.cursor = 'not-allowed';
                    btnIA.title = 'No metadata JSON found';
                }
            }
            if (btnI2I) {
                btnI2I.disabled = true;
                btnI2I.style.opacity = '0.4';
                btnI2I.style.cursor = 'not-allowed';
                btnI2I.title = 'No metadata JSON found';
            }
        });
}

function displayTextResult(text) {
    // Hide the result image in I2T mode
    if (resultImg) resultImg.style.display = 'none';
    if (placeholder) placeholder.style.display = 'none';

    const textResultArea = document.getElementById('text-result-area');
    const positivePromptBox = document.getElementById('i2t-positive-prompt');
    const negativePromptBox = document.getElementById('i2t-negative-prompt');
    const thinkingBox = document.getElementById('thinking-box');

    // Make sure text result area is visible
    if (textResultArea) textResultArea.classList.remove('hidden');

    // Parse the three sections from the response
    let positivePrompt = '';
    let negativePrompt = '';
    let reasoning = '';

    // Try JSON format first
    if (text.trim().startsWith('{')) {
        try {
            const json = JSON.parse(text);
            positivePrompt = json.positive_prompt || json.positive || json.prompt || '';
            negativePrompt = json.negative_prompt || json.negative || '';
            reasoning = json.reasoning || json.thinking || json.ragionamento || '';
        } catch (e) {
            // Fall through to marker parsing
        }
    }

    // Try marker format: ###POSITIVE### ... ###NEGATIVE### ... ###REASONING###
    if (!positivePrompt && !reasoning) {
        const posMatch = text.match(/###POSITIVE###\n?([\s\S]*?)(?=###NEGATIVE###|###REASONING###|$)/i);
        const negMatch = text.match(/###NEGATIVE###\n?([\s\S]*?)(?=###POSITIVE###|###REASONING###|$)/i);
        const reasonMatch = text.match(/###REASONING###\n?([\s\S]*?)(?=###POSITIVE###|###NEGATIVE###|$)/i);

        if (posMatch) positivePrompt = posMatch[1].trim();
        if (negMatch) negativePrompt = negMatch[1].trim();
        if (reasonMatch) reasoning = reasonMatch[1].trim();
    }

    // Fallback: old format with ###THINKING### and ###ANSWER###
    if (!positivePrompt && text.includes('###THINKING###')) {
        const parts = text.split(/(?:###THINKING###|###ANSWER###)/);
        if (parts.length >= 2) {
            reasoning = parts[1]?.trim() || '';
            // The answer part becomes the positive prompt
            positivePrompt = parts[2]?.trim() || parts[1]?.trim() || '';
        }
    }

    // Fallback: if we only have one block, treat as positive prompt
    if (!positivePrompt && text.trim()) {
        positivePrompt = text.trim();
    }

    // Sanitize all text to remove problematic characters before display
    positivePrompt = sanitizeForJSON(positivePrompt);
    negativePrompt = sanitizeForJSON(negativePrompt);
    reasoning = sanitizeForJSON(reasoning);

    // Populate the three boxes
    if (positivePromptBox) positivePromptBox.textContent = positivePrompt;
    if (negativePromptBox) negativePromptBox.textContent = negativePrompt;
    if (thinkingBox) thinkingBox.textContent = reasoning;

    // Expand thinking box by default for I2T results
    if (thinkingBox && thinkingBox.classList.contains('collapsed')) {
        thinkingBox.classList.remove('collapsed');
        const icon = document.getElementById('thinking-toggle-icon');
        if (icon) icon.innerText = '▼';
    }

    // Refresh history panel if it's open
    const historyPanel = document.getElementById('history-panel');
    if (historyPanel && !historyPanel.classList.contains('collapsed')) {
        if (window.loadHistory) {
            window.loadHistory();
        }
    }

    // Update preferences statistics
    if (window.Preferences && window.Preferences.updateStatsAfterGeneration) {
        const payload = buildPayload('cpp');
        window.Preferences.updateStatsAfterGeneration('i2t', payload.model_path, payload);
    }
}

function appendLog(msg) {
    if (consoleDiv) {
        consoleDiv.innerHTML += '<div>' + msg + '</div>';
        consoleDiv.scrollTop = consoleDiv.scrollHeight;
    }
}

function startTimer() {
    console.log("Starting timer...");
    window.timerID = setInterval(function () {
        var now = Date.now();
        var diff = now - window.startTime;
        var secs = (diff / 1000).toFixed(1);
        document.getElementById("timer").innerText = secs + "s";
    }, 100);
    window.startTime = Date.now();
}

function stopTimer() {
    if (window.timerID) { clearInterval(window.timerID); window.timerID = null; }
}

function stopGeneration() {
    // Abort I2T query if in progress (Image Analysis mode)
    if (window.i2tAbortController) {
        try {
            window.i2tAbortController.abort();
            console.log('[Stop] I2T query aborted');
        } catch (e) {
            console.error('[Stop] Error aborting I2T:', e);
        }
        window.i2tAbortController = null;
    }
    
    // Clean up live I2T streaming element if present
    const liveEl = document.getElementById('i2t-live-response');
    if (liveEl) liveEl.remove();
    
    isGenerating = false;
    window.isProcessing = false;
    stopTimer();
    setGenerationState(false);
    updateStatusGen('Ready', 'info');
}

function updateStatusGen(text, type) {
    if (statusText) { statusText.innerText = text; statusText.className = 'status-text status-' + (type || 'info'); }
}

// ===================================================================
// IA Bidirectional Data Flow
// ===================================================================

/**
 * Sanitize a string to ensure it only contains valid UTF-8 characters.
 * Replaces problematic characters with their closest ASCII equivalent or '?'.
 * Handles:
 *   - Typographic quotes/dashes → ASCII (prevents mojibake in C++ binary)
 *   - Control characters (except \n, \r, \t)
 *   - Non-BMP surrogate pairs
 *   - Invalid UTF-8 byte sequences
 * @param {string} text - The text to sanitize
 * @returns {string} Sanitized text safe for UTF-8 transmission
 */
function sanitizeForJSON(text) {
    if (!text) return '';
    let cleaned = text;
    
    // Step 1: DOUBLE-ENCODING RECOVERY (NEW)
    // Detect C1 control chars (0x80-0x9F) which identify UTF-8 bytes
    // misinterpreted as Latin-1. E.g. â\x80\x94 (3 Latin-1 chars) are
    // actually the 3 bytes E2 80 94 which decode as '—' (U+2014).
    // Builds a byte array (charCode → byte) then re-decodes as UTF-8.
    if (/[\x80-\x9F]/.test(cleaned)) {
        try {
            const bytes = new Uint8Array(cleaned.length);
            for (let i = 0; i < cleaned.length; i++) {
                bytes[i] = cleaned.charCodeAt(i) & 0xFF;
            }
            const decoder = new TextDecoder('utf-8', { fatal: false });
            const recovered = decoder.decode(bytes);
            if (recovered !== cleaned) {
                cleaned = recovered;
            }
        } catch(e) {
            // Recovery failed silently, continue with original text
        }
    }
    
    // Step 2: Replace typographic Unicode characters with ASCII equivalents
    const typographicMap = {
        '\u2018': "'", '\u2019': "'",  // smart single quotes
        '\u201c': '"', '\u201d': '"',  // smart double quotes
        '\u201e': '"',                  // double low-9 quote
        '\u2013': '-', '\u2014': '--', // en dash, em dash
        '\u2026': '...',               // ellipsis
        '\u00ab': '"', '\u00bb': '"',  // angle quotes
        '\u2032': "'", '\u2033': '"',  // prime, double prime
        '\u00b7': '*', '\u2022': '*',  // middle dot, bullet
        '\ufeff': '',                   // BOM
        '\u00a0': ' ',                  // no-break space
    };
    for (const [char, replacement] of Object.entries(typographicMap)) {
        cleaned = cleaned.replaceAll(char, replacement);
    }
    
    // Step 3: Remove ALL control characters including C1 range 0x80-0x9F
    // (previously only removed 0x00-0x1F range, missing the corrupted bytes)
    cleaned = cleaned.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, '');
    
    // Step 4: Replace non-BMP characters (surrogate pairs that might cause issues)
    cleaned = cleaned.replace(/[\uD800-\uDFFF]/g, '?');
    
    // Step 5: Verify UTF-8 validity via encodeURIComponent
    try {
        encodeURIComponent(cleaned);
        return cleaned;
    } catch (e) {
        // If encoding fails, aggressively clean to ASCII range only
        return cleaned.replace(/[^\x20-\x7E\x0A\x0D\x09]/g, '?');
    }
}

/**
 * Recursively sanitize all string values in an object for safe JSON transmission.
 * @param {Object} obj - The object to sanitize
 * @returns {Object} Sanitized object
 */
function sanitizeObjectForJSON(obj) {
    if (typeof obj === 'string') {
        return sanitizeForJSON(obj);
    } else if (Array.isArray(obj)) {
        return obj.map(item => sanitizeObjectForJSON(item));
    } else if (obj && typeof obj === 'object') {
        const sanitized = {};
        for (const key in obj) {
            if (Object.prototype.hasOwnProperty.call(obj, key)) {
                sanitized[key] = sanitizeObjectForJSON(obj[key]);
            }
        }
        return sanitized;
    }
    return obj;
}

/**
 * Export the current IA (Image Analysis) state as a JSON object.
 * Collects all IA parameters, prompts, results, and chat history.
 * All string fields are sanitized for safe UTF-8 transmission.
 * @returns {Object|null} IA state object, or null if no IA data available
 */
function exportIAState() {
    // ONLY export when we are actually in Image Analysis (i2t) mode
    // This prevents polluting the 'prompt' field when called from T2I/I2I tabs
    // where the prompt box has been overwritten with positive_prompt_result
    if (window.currentMode !== 'i2t') return null;
    
    const positivePrompt = document.getElementById('i2t-positive-prompt')?.textContent || '';
    const reasoning = document.getElementById('thinking-box')?.textContent || '';
    
    // No IA data to export
    if (!positivePrompt && !reasoning) return null;
    
    const iaData = {
        "source_images": [
            document.getElementById('uploaded-path')?.value || '',
            document.getElementById('uploaded-path-2')?.value || ''
        ],
        "vision_model": document.getElementById('i2t-model-select')?.value || '',
        "top_k": parseFloat(document.getElementById('topk-i2t')?.value || 1.0),
        "temperature": parseFloat(document.getElementById('temp-i2t')?.value || 0.6),
        "denoise_strength": parseFloat(document.getElementById('strength-i2t')?.value || 0.75),
        "mix_ratio": parseFloat(document.getElementById('mix-ratio-i2t')?.value || 0.5),
        "context_size": parseInt(document.getElementById('ctx-i2t')?.value || 16384),
        "max_img_size": parseInt(document.getElementById('maximg-i2t')?.value || 1024),
        "prompt": document.getElementById('prompt')?.value || '',
        "negative_prompt": document.getElementById('negative-prompt')?.value || '',
        "chat_with_ai": window.i2tConversationArray || [],
        "positive_prompt_result": positivePrompt,
        "negative_prompt_result": document.getElementById('i2t-negative-prompt')?.textContent || '',
        "reasoning": reasoning
    };
    
    // Sanitize all string values for safe UTF-8 transmission
    return sanitizeObjectForJSON(iaData);
}

/**
 * Helper: set a slider value and its display span if the elements exist.
 * @param {string} sliderId - ID of the input[type=range] element
 * @param {string} valId - ID of the span showing the value
 * @param {number} value - The value to set
 */
function setSliderIfExists(sliderId, valId, value) {
    if (value === undefined || value === null) return;
    const slider = document.getElementById(sliderId);
    const valDisplay = document.getElementById(valId);
    if (slider) {
        slider.value = value;
        if (valDisplay) valDisplay.innerText = value;
    }
}

/**
 * Restore the IA tab state from a saved IA data object.
 * Loads source images, model selector, slider values, prompts,
 * analysis results, and chat conversation.
 * @param {Object} iaData - The IA data object from exportIAState or image JSON
 */
function restoreIAState(iaData) {
    if (!iaData) return;

    // Restore source images
    if (iaData.source_images) {
        if (iaData.source_images[0]) {
            document.getElementById('uploaded-path').value = iaData.source_images[0];
            if (window.showPreviewFromPath) {
                window.showPreviewFromPath(iaData.source_images[0], 1);
            }
        }
        if (iaData.source_images[1]) {
            document.getElementById('uploaded-path-2').value = iaData.source_images[1];
            if (window.showPreviewFromPath) {
                window.showPreviewFromPath(iaData.source_images[1], 2);
            }
        }
    }

    // Restore Vision Model selector
    const modelSelect = document.getElementById('i2t-model-select');
    if (iaData.vision_model && modelSelect) {
        for (let i = 0; i < modelSelect.options.length; i++) {
            const opt = modelSelect.options[i];
            if (opt.value === iaData.vision_model ||
                opt.value.endsWith('/' + iaData.vision_model) ||
                opt.value.endsWith(iaData.vision_model)) {
                modelSelect.selectedIndex = i;
                break;
            }
        }
    }

    // Restore I2T sliders
    setSliderIfExists('topk-i2t', 'val-topk-i2t', iaData.top_k);
    setSliderIfExists('temp-i2t', 'val-temp-i2t', iaData.temperature);
    setSliderIfExists('strength-i2t', 'val-strength-i2t', iaData.denoise_strength);
    setSliderIfExists('mix-ratio-i2t', 'val-mix-i2t', iaData.mix_ratio);
    setSliderIfExists('ctx-i2t', 'val-ctx-i2t', iaData.context_size);
    setSliderIfExists('maximg-i2t', 'val-maximg-i2t', iaData.max_img_size);
    // Add 'px' suffix to the display value
    const maxImgVal = document.getElementById('val-maximg-i2t');
    if (maxImgVal && iaData.max_img_size) {
        maxImgVal.innerText = iaData.max_img_size + 'px';
    }
    // Restore I2I max image size (for re-execution from history)
    setSliderIfExists('maximg-i2i', 'val-maximg-i2i', iaData.max_img_size);
    const maxImgI2iVal = document.getElementById('val-maximg-i2i');
    if (maxImgI2iVal && iaData.max_img_size) {
        maxImgI2iVal.innerText = iaData.max_img_size + 'px';
    }

    // Restore prompts (user input fields)
    const promptBox = document.getElementById('prompt');
    if (iaData.prompt && promptBox) promptBox.value = iaData.prompt;

    const negPromptBox = document.getElementById('negative-prompt');
    if (iaData.negative_prompt && negPromptBox) {
        negPromptBox.value = iaData.negative_prompt;
        negPromptBox.style.display = 'block';
        const toggle = document.getElementById('negative-prompt-toggle');
        if (toggle) toggle.textContent = '▼ Hide';
    }

    // Restore result boxes (IA output: positive, negative, reasoning)
    const posBox = document.getElementById('i2t-positive-prompt');
    if (iaData.positive_prompt_result && posBox) {
        posBox.textContent = iaData.positive_prompt_result;
    }

    const negResultBox = document.getElementById('i2t-negative-prompt');
    if (iaData.negative_prompt_result && negResultBox) {
        negResultBox.textContent = iaData.negative_prompt_result;
    }

    const thinkBox = document.getElementById('thinking-box');
    if (iaData.reasoning && thinkBox) {
        thinkBox.textContent = iaData.reasoning;
        thinkBox.classList.remove('collapsed');
        const icon = document.getElementById('thinking-toggle-icon');
        if (icon) icon.innerText = '▼';
    }

    // Show text result area if we have results
    const textResultArea = document.getElementById('text-result-area');
    if (textResultArea && (iaData.positive_prompt_result || iaData.reasoning)) {
        textResultArea.classList.remove('hidden');
    }

    // Restore chat conversation
    if (iaData.chat_with_ai && Array.isArray(iaData.chat_with_ai)) {
        window.i2tConversationArray = iaData.chat_with_ai;
        // Rebuild conversation display
        if (window.updateConversationDisplay) {
            window.updateConversationDisplay(null, null);
        }
    }

    updateStatusGen('IA state restored from image metadata', 'success');
}

/**
 * Send current result image to IA (Image Analysis) tab.
 * Reads the image's JSON metadata, extracts IA data if present,
 * and restores the full IA configuration.
 */
function sendToIA() {
    const currentPath = window.currentResultPath;
    
    if (!currentPath) {
        // Fallback: try reading from result-img src
        const resultImg = document.getElementById('result-img');
        const imgSrc = resultImg?.src || '';
        if (imgSrc && imgSrc !== window.location.href && !imgSrc.endsWith('/outputs/')) {
            // Extract clean path from the src (remove timestamp query params)
            const cleanSrc = imgSrc.split('?')[0];
            window.currentResultPath = cleanSrc;
            // Retry with the recovered path
            return sendToIA();
        }
        alert('No image to send to IA. Generate an image first.');
        return;
    }

    // Switch to IA tab first (so the UI panels are ready)
    switchTab('i2t');

    // Use cached IA data if available (loaded by checkImageMetadataAndUpdateButtons)
    if (window.cachedImageAnalysis) {
        setTimeout(() => {
            restoreIAState(window.cachedImageAnalysis);
        }, 200);
        return;
    }

    // Fallback: fetch the JSON and extract IA data
    const jsonPath = currentPath.replace(/\.(png|jpg|jpeg)$/i, '.json');

    fetch(jsonPath)
        .then(res => {
            if (!res.ok) throw new Error('JSON not found at: ' + jsonPath);
            return res.json();
        })
        .then(data => {
            // Extract IA data from the JSON's "image_analysis" tag
            const iaData = data.image_analysis || null;
            // Update cache so the toolbar button reflects the data
            window.cachedImageAnalysis = iaData;

            // Wait for tab switch DOM updates
            setTimeout(() => {
                if (iaData) {
                    // Full IA state restoration
                    restoreIAState(iaData);
                } else {
                    // No IA data, just load the result image as source
                    const sourceImage = data.outputs?.files?.[0] || data.meta?.filename || '';
                    if (sourceImage) {
                        const imgUrl = '/outputs/' + sourceImage;
                        document.getElementById('uploaded-path').value = imgUrl;
                        if (window.showPreviewFromPath) {
                            window.showPreviewFromPath(imgUrl, 1);
                        }
                    }
                    updateStatusGen('Image loaded (no IA data found)', 'info');
                }
            }, 200);
        })
        .catch((err) => {
            // JSON not found - just load the current result image as source for IA
            setTimeout(() => {
                document.getElementById('uploaded-path').value = currentPath;
                if (window.showPreviewFromPath) {
                    window.showPreviewFromPath(currentPath, 1);
                }
                updateStatusGen('Image loaded (no metadata JSON)', 'info');
            }, 200);
        });
}

/**
 * Send current result image to I2I (Image-to-Image) tab.
 * Reads the image's JSON metadata, extracts IA data if present,
 * and pre-fills the positive/negative prompts from the IA analysis.
 */
function sendToI2I() {
    const currentPath = window.currentResultPath;
    if (!currentPath) {
        alert('No image to send');
        return;
    }

    // Switch to I2I tab first
    switchTab('i2i');

    setTimeout(() => {
        // Load the result image into input slot 1
        if (window.loadToInput) {
            window.loadToInput(currentPath, 1);
        }

        // Attempt to load IA data from the image's JSON
        const jsonPath = currentPath.replace(/\.(png|jpg|jpeg)$/i, '.json');
        fetch(jsonPath)
            .then(res => {
                if (!res.ok) throw new Error('JSON not found');
                return res.json();
            })
            .then(data => {
                const iaData = data.image_analysis;
                if (iaData) {
                    // Set the prompt from IA's positive_prompt_result (analysis output)
                    const promptBox = document.getElementById('prompt');
                    if (iaData.positive_prompt_result && promptBox) {
                        promptBox.value = iaData.positive_prompt_result;
                    }
                    // Set negative prompt from IA's negative_prompt_result (analysis output)
                    const negBox = document.getElementById('negative-prompt');
                    if (iaData.negative_prompt_result && negBox) {
                        negBox.value = iaData.negative_prompt_result;
                        negBox.style.display = 'block';
                        const toggle = document.getElementById('negative-prompt-toggle');
                        if (toggle) toggle.textContent = '▼ Hide';
                    }
                    updateStatusGen('Prompts pre-filled from IA analysis', 'success');
                }
            })
            .catch(() => { /* no IA data, that's ok */ });
    }, 200);
}

/**
 * Send current result image to I2T (Analysis) tab by
 * loading the image into upload slot 1.
 * @deprecated Use sendToIA() which extracts full IA state.
 */
function sendToI2T() {
    const currentPath = window.currentResultPath;
    if (!currentPath) {
        alert('No image to send');
        return;
    }
    switchTab('i2t');
    setTimeout(() => {
        document.getElementById('uploaded-path').value = currentPath;
        if (window.showPreviewFromPath) {
            window.showPreviewFromPath(currentPath, 1);
        }
        updateStatusGen('Image loaded for analysis', 'info');
    }, 200);
}

/**
 * Send complete IA data (all prompts + params) to T2I tab.
 * Exports the entire IA state via exportIAState() and stores it in memory
 * so that when the user generates in T2I, buildPayload() includes image_analysis.
 */
function sendFullIAtoT2I() {
    const iaData = exportIAState();
    if (!iaData) {
        alert('No IA analysis data to send. Run an analysis first.');
        return;
    }

    // Switch to T2I tab first (this restores the prompt buffer)
    switchTab('t2i');

    setTimeout(() => {
        // Set the main prompt from IA's positive_prompt_result (analysis output)
        const promptBox = document.getElementById('prompt');
        if (iaData.positive_prompt_result && promptBox) {
            promptBox.value = iaData.positive_prompt_result;
        }

        // Set negative prompt from IA's negative_prompt_result (analysis output)
        const negBox = document.getElementById('negative-prompt');
        if (iaData.negative_prompt_result && negBox) {
            negBox.value = iaData.negative_prompt_result;
            negBox.style.display = 'block';
            const toggle = document.getElementById('negative-prompt-toggle');
            if (toggle) toggle.textContent = '▼ Hide';
        }

        // NOW store IA data — after prompts are set and tab switch is complete.
        // This prevents the input-event watcher from firing during tab switch.
        window.pendingIAExport = iaData;
        window._iaPromptChanged = false;

        updateStatusGen('IA data loaded into prompts. Will be saved to image JSON unless you modify the prompts.', 'success');
    }, 200);
}

/**
 * Send complete IA data (all prompts + params + source image) to I2I tab.
 * Exports the entire IA state via exportIAState() and stores it in memory
 * so that when the user generates in I2I, buildPayload() includes image_analysis.
 */
function sendFullIAtoI2I() {
    const iaData = exportIAState();
    if (!iaData) {
        alert('No IA analysis data to send. Run an analysis first.');
        return;
    }

    // Switch to I2I tab first (this restores the prompt buffer)
    switchTab('i2i');

    setTimeout(() => {
        // Load the first source image into I2I input slot
        if (iaData.source_images && iaData.source_images[0]) {
            if (window.loadToInput) {
                window.loadToInput(iaData.source_images[0], 1);
            }
        }

        // Set the main prompt from IA's positive_prompt_result (analysis output)
        const promptBox = document.getElementById('prompt');
        if (iaData.positive_prompt_result && promptBox) {
            promptBox.value = iaData.positive_prompt_result;
        }

        // Set negative prompt from IA's negative_prompt_result (analysis output)
        const negBox = document.getElementById('negative-prompt');
        if (iaData.negative_prompt_result && negBox) {
            negBox.value = iaData.negative_prompt_result;
            negBox.style.display = 'block';
            const toggle = document.getElementById('negative-prompt-toggle');
            if (toggle) toggle.textContent = '▼ Hide';
        }

        // NOW store IA data — after prompts are set and tab switch is complete.
        // This prevents the input-event watcher from firing during tab switch.
        window.pendingIAExport = iaData;
        window._iaPromptChanged = false;

        updateStatusGen('IA data loaded into prompts. Will be saved to image JSON unless you modify the prompts.', 'success');
    }, 200);
}

/**
 * Watch for user modifications to the prompt or negative-prompt fields.
 * If the user edits either field while window.pendingIAExport is set,
 * prompt them to decide whether to keep or discard the IA metadata.
 */
function setupPendingIAWatch() {
    const promptBox = document.getElementById('prompt');
    const negBox = document.getElementById('negative-prompt');

    const handleInput = () => {
        // Only trigger if IA data is pending and we haven't already asked
        if (!window.pendingIAExport || window._iaPromptChanged) return;

        // Mark as changed to avoid repeated dialogs
        window._iaPromptChanged = true;

        // Ask user what to do
        const keepIa = confirm(
            'You modified the prompt text.\n\n' +
            'The generated image still has IA metadata loaded from your analysis.\n' +
            'Do you want to KEEP it attached to the next generated image?\n\n' +
            '  • OK     → IA metadata will be saved into the next image JSON,\n' +
            '              preserving the prompt generator state.\n' +
            '  • Cancel → IA metadata will be discarded.\n\n' +
            '(You can still generate with your modified prompt either way.)'
        );
        if (!keepIa) {
            window.pendingIAExport = null;
            updateStatusGen('IA metadata discarded (prompts modified by user)', 'info');
        } else {
            updateStatusGen('IA metadata preserved despite prompt changes', 'success');
        }
    };

    if (promptBox) {
        promptBox.addEventListener('input', handleInput);
    }
    if (negBox) {
        negBox.addEventListener('input', handleInput);
    }
}

/**
 * Copy text from an element to clipboard.
 * @param {string} elementId - ID of the element to copy from
 * @param {Event} event - Optional event object (to stop propagation)
 */
function copyToClipboard(elementId, event) {
    if (event) event.stopPropagation();
    const el = document.getElementById(elementId);
    if (!el) {
        alert('Element not found');
        return;
    }
    const text = el.textContent || el.innerText || '';
    if (!text.trim()) {
        alert('Nothing to copy');
        return;
    }
    navigator.clipboard.writeText(text).then(() => {
        // Brief visual feedback
        if (event && event.target) {
            const btn = event.target;
            const originalText = btn.textContent;
            btn.textContent = '✓ Copied';
            setTimeout(() => { btn.textContent = originalText; }, 1500);
        }
    }).catch(err => {
        console.error('Copy failed:', err);
        alert('Copy failed: ' + err);
    });
}

/**
 * Send prompt text to T2I or I2I tab.
 * @param {string} targetMode - 't2i' or 'i2i'
 * @param {string} sourceElementId - Optional ID of element to get text from (default: 'i2t-positive-prompt')
 */
function sendPromptTo(targetMode, sourceElementId) {
    const sourceId = sourceElementId || 'i2t-positive-prompt';
    const sourceEl = document.getElementById(sourceId);
    const text = sourceEl?.textContent?.trim() || sourceEl?.innerText?.trim() || '';

    if (!text) {
        alert('No prompt to send');
        return;
    }

    // Capture the source image path BEFORE switching tabs,
    // while imageSlots still belongs to the I2T context.
    let sourceImagePath = null;
    let sourceImageUrl = null;
    if (targetMode === 'i2i') {
        // Try to get the first image from imageSlots (N-Slot API)
        if (typeof getImageSlots === 'function') {
            const slots = getImageSlots();
            const initSlot = slots.find(s => s.role === 'init') || slots[0];
            if (initSlot && initSlot.path) {
                sourceImagePath = initSlot.path;
            }
        }
        // Fallback to legacy hidden input
        if (!sourceImagePath) {
            const legacyInput = document.getElementById('uploaded-path');
            if (legacyInput && legacyInput.value) {
                sourceImagePath = legacyInput.value;
            }
        }

        // Convert absolute URL path (/outputs/...) to relative server path
        // so the backend can resolve it correctly via OUTPUT_DIR.
        if (sourceImagePath) {
            if (sourceImagePath.startsWith('/outputs/')) {
                sourceImageUrl = sourceImagePath;
                sourceImagePath = sourceImagePath.replace('/outputs/', '');
            } else if (sourceImagePath.startsWith('/uploads/')) {
                sourceImageUrl = sourceImagePath;
                sourceImagePath = sourceImagePath.replace('/uploads/', '');
            }
        }
    }

    // Switch to target tab and set the prompt
    if (targetMode === 't2i') {
        switchTab('t2i');
    } else if (targetMode === 'i2i') {
        switchTab('i2i');
    }

    // Wait a moment for tab switch, then set prompt and load image
    setTimeout(() => {
        const promptBox = document.getElementById('prompt');
        if (promptBox) {
            promptBox.value = text;
            // Scroll to prompt box
            promptBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        // Also load the source image into I2I input slot
        if (targetMode === 'i2i' && sourceImagePath) {
            // Set the path directly in imageSlots (synchronous data operation)
            if (typeof imageSlots !== 'undefined' && imageSlots.length > 0 && imageSlots[0]) {
                imageSlots[0].path = sourceImagePath;
                if (sourceImageUrl) {
                    imageSlots[0].previewDataUrl = sourceImageUrl;
                }
                imageSlots[0].filename = sourceImagePath.split('/').pop();
                // Sync legacy hidden input
                const legacyInput = document.getElementById('uploaded-path');
                if (legacyInput) legacyInput.value = sourceImagePath;
                // Re-render slots to show preview (this also calls syncLegacyInputs)
                if (typeof renderSlots === 'function') {
                    renderSlots();
                } else if (window.renderSlots) {
                    window.renderSlots();
                }
                // NOTE: Do NOT call showPreviewFromPath here, because it would
                // overwrite imageSlots[0].path with the absolute URL path.
                // renderSlots() already updates the DOM preview correctly.
            } else {
                // Fallback: directly set legacy inputs
                const legacyInput = document.getElementById('uploaded-path');
                if (legacyInput) legacyInput.value = sourceImagePath;
                // Use showPreviewFromPath only as last resort, but pass the
                // relative path (sourceImagePath) NOT the URL (sourceImageUrl),
                // because showPreviewFromPath sets targetSlot.path = serverPath.
                if (window.showPreviewFromPath) {
                    window.showPreviewFromPath(sourceImagePath, 1);
                }
            }
        }
    }, 100);
}

/**
 * Send the given prompt text to the 🪄 Agent Refine GOAL and load the current
 * INIT image as the Agent reference (single action from Image Analysis).
 * @param {string} sourceElementId - ID of the element holding the prompt text
 */
function sendPromptToARF(sourceElementId) {
    const el = document.getElementById(sourceElementId || 'i2t-positive-prompt');
    const text = (el && (el.textContent || el.innerText || '') || '').trim();
    if (!text) { alert('No prompt to send'); return; }

    // Capture the INIT image BEFORE switching tabs (shared I2T/I2I slot, role init)
    let refUrl = null;
    if (typeof getImageSlots === 'function') {
        const slots = getImageSlots();
        const init = slots.find(s => s.role === 'init') || slots[0];
        const slot = (init && typeof getSlot === 'function') ? getSlot(init.id) : null;
        if (slot) {
            // Prefer the SERVER path (normalized -> keeps subfolders, e.g.
            // agent_refine_*/iter.png); fall back to a usable preview URL.
            if (slot.path && window.normalizeSourcePath) {
                const n = window.normalizeSourcePath(slot.path);
                if (n && n.url) refUrl = n.url;
            }
            if (!refUrl && slot.previewDataUrl &&
                /^(data:|blob:|https?:|\/outputs\/|\/uploads\/)/.test(slot.previewDataUrl)) {
                refUrl = slot.previewDataUrl;
            }
        }
    }
    if (!refUrl) {
        const legacy = document.getElementById('uploaded-path');
        if (legacy && legacy.value && window.normalizeSourcePath) {
            const n = window.normalizeSourcePath(legacy.value);
            if (n && n.url) refUrl = n.url;
        }
    }

    // Switch to the Agent Refine tab (nav index 3)
    const nav = document.querySelectorAll('.nav-btn')[3];
    if (window.switchTab) window.switchTab('agent', nav);

    setTimeout(() => {
        const g = document.getElementById('agent-goal');
        if (g) {
            g.value = text;
            g.dispatchEvent(new Event('input', { bubbles: true }));
            g.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        if (refUrl && window.agentUseAsRef) window.agentUseAsRef(refUrl);
        if (window.updateStatusGen) {
            window.updateStatusGen('🪄 Prompt' + (refUrl ? ' + init image' : '') + ' inviati ad Agent Refine', 'success');
        }
        if (window.log) window.log('🪄 I2T → Agent Refine: goal impostato' + (refUrl ? ' + immagine di riferimento' : ''), false, true);
    }, 250);
}
window.sendPromptToARF = sendPromptToARF;

/**
 * Send negative prompt text to T2I or I2I tab, setting the negative-prompt textarea.
 * @param {string} targetMode - 't2i' or 'i2i'
 * @param {string} sourceElementId - ID of element to get text from
 */
function sendNegativePromptTo(targetMode, sourceElementId) {
    const sourceEl = document.getElementById(sourceElementId);
    const text = sourceEl?.textContent?.trim() || sourceEl?.innerText?.trim() || '';

    if (!text) {
        alert('No negative prompt to send');
        return;
    }

    // Capture the source image path BEFORE switching tabs
    let sourceImagePath = null;
    let sourceImageUrl = null;
    if (targetMode === 'i2i') {
        if (typeof getImageSlots === 'function') {
            const slots = getImageSlots();
            const initSlot = slots.find(s => s.role === 'init') || slots[0];
            if (initSlot && initSlot.path) {
                sourceImagePath = initSlot.path;
            }
        }
        if (!sourceImagePath) {
            const legacyInput = document.getElementById('uploaded-path');
            if (legacyInput && legacyInput.value) {
                sourceImagePath = legacyInput.value;
            }
        }

        // Convert absolute URL path (/outputs/...) to relative server path
        if (sourceImagePath) {
            if (sourceImagePath.startsWith('/outputs/')) {
                sourceImageUrl = sourceImagePath;
                sourceImagePath = sourceImagePath.replace('/outputs/', '');
            } else if (sourceImagePath.startsWith('/uploads/')) {
                sourceImageUrl = sourceImagePath;
                sourceImagePath = sourceImagePath.replace('/uploads/', '');
            }
        }
    }

    // Switch to target tab
    if (targetMode === 't2i') {
        switchTab('t2i');
    } else if (targetMode === 'i2i') {
        switchTab('i2i');
    }

    // Wait a moment for tab switch, then set negative prompt and load image
    setTimeout(() => {
        const negPromptBox = document.getElementById('negative-prompt');
        if (negPromptBox) {
            negPromptBox.value = text;
            // Also make the negative prompt section visible
            if (negPromptBox.style.display === 'none') {
                toggleNegativePrompt();
            }
            negPromptBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        // Also load the source image into I2I input slot
        if (targetMode === 'i2i' && sourceImagePath) {
            // Set the path directly in imageSlots (synchronous data operation)
            if (typeof imageSlots !== 'undefined' && imageSlots.length > 0 && imageSlots[0]) {
                imageSlots[0].path = sourceImagePath;
                if (sourceImageUrl) {
                    imageSlots[0].previewDataUrl = sourceImageUrl;
                }
                imageSlots[0].filename = sourceImagePath.split('/').pop();
                // Sync legacy hidden input
                const legacyInput = document.getElementById('uploaded-path');
                if (legacyInput) legacyInput.value = sourceImagePath;
                // Re-render slots to show preview (this also calls syncLegacyInputs)
                if (typeof renderSlots === 'function') {
                    renderSlots();
                } else if (window.renderSlots) {
                    window.renderSlots();
                }
                // NOTE: Do NOT call showPreviewFromPath here, because it would
                // overwrite imageSlots[0].path with the absolute URL path.
                // renderSlots() already updates the DOM preview correctly.
            } else {
                // Fallback: directly set legacy inputs
                const legacyInput = document.getElementById('uploaded-path');
                if (legacyInput) legacyInput.value = sourceImagePath;
                // Use showPreviewFromPath only as last resort, but pass the
                // relative path (sourceImagePath) NOT the URL (sourceImageUrl),
                // because showPreviewFromPath sets targetSlot.path = serverPath.
                if (window.showPreviewFromPath) {
                    window.showPreviewFromPath(sourceImagePath, 1);
                }
            }
        }
    }, 100);
}

/**
 * Send thinking text to T2I or I2I tab.
 */
function sendThinkingTo(targetMode) {
    const thinkingBox = document.getElementById('thinking-box');
    const text = thinkingBox?.textContent?.trim() || '';
    sendPromptTo(targetMode, 'thinking-box');
}

/**
 * Auto-select qwen_image_vae in the VAE dropdown when Edit Image button is clicked.
 * This VAE is required for Qwen Image Edit mode to work properly.
 */
function autoSelectQwenImageVAE() {
    const vaeSelect = document.getElementById('vae-select');
    if (!vaeSelect || vaeSelect.options.length === 0) {
        console.log('autoSelectQwenImageVAE: VAE selector not available');
        return;
    }
    
    // Look for qwen_image_vae in the options
    const qwenVAEKeywords = ['qwen_image_vae', 'qwen-image-vae', 'vae_qwen_image'];
    
    for (let i = 0; i < vaeSelect.options.length; i++) {
        const opt = vaeSelect.options[i];
        const optText = opt.innerText.toLowerCase();
        const optValue = opt.value.toLowerCase();
        
        for (const keyword of qwenVAEKeywords) {
            if (optText.includes(keyword) || optValue.includes(keyword)) {
                vaeSelect.selectedIndex = i;
                console.log('autoSelectQwenImageVAE: Selected VAE option', i, '-', opt.innerText);
                return;
            }
        }
    }
    
    console.log('autoSelectQwenImageVAE: qwen_image_vae not found in VAE selector');
}

window.initGenerate = initGenerate;
window.handleGeneration = handleGeneration;
window.stopGeneration = stopGeneration;
window.sendToI2I = sendToI2I;
window.sendToI2T = sendToI2T;
window.setGenerationState = setGenerationState;
window.autoSelectQwenImageVAE = autoSelectQwenImageVAE;
window.checkImageMetadataAndUpdateButtons = checkImageMetadataAndUpdateButtons;

// ===================================================================
// Interactive I2T (persistent llama-server)
// ===================================================================

/**
 * Send a query to the persistent llama-server via /api/i2t/query.
 * Uses SSE streaming, same as handleGeneration().
 *
 * @param {string} imagePath - Path to the uploaded image
 * @param {string} prompt    - The prompt text
 * @param {string} modelDir  - Model directory (optional)
 */
async function runI2TInteractive(imagePath, prompt, modelDir, userInput = null, imageSpecs = null) {
    // Phase 6: N-Image I2T support
    // imagePath: str (single, backward compat)
    // imageSpecs: Array<{path, role}> (N-Image mode, takes priority if provided)
    // Clear log at start
    const consoleDiv = document.getElementById('console');
    if (consoleDiv) consoleDiv.innerHTML = '';

    isGenerating = true;
    
    // Stop any active tagging to free VRAM for generation
    if (typeof stopAutoTag === "function") {
        stopAutoTag();
    }
    setGenerationState(true);
    startTimer();

    // Build N-image specs if not provided
    let actualImageSpecs = imageSpecs;
    if (!actualImageSpecs && imagePath && imagePath.trim() !== '') {
        // Backward compat: single image, role=init
        actualImageSpecs = [{ path: imagePath, role: 'init' }];
    }
    if (!actualImageSpecs || actualImageSpecs.length === 0) {
        // No images - virtual T2T mode
        actualImageSpecs = null;
    }
    const isVirtualMode = !actualImageSpecs;
    const numImages = actualImageSpecs ? actualImageSpecs.length : 0;

    if (isVirtualMode) {
        window.i2tVirtualMode = true;
        // Show text-result-area immediately for virtual T2T
        const textResultArea = document.getElementById('text-result-area');
        if (textResultArea) textResultArea.classList.remove('hidden');
    }

    // Different format instructions for virtual T2T vs real I2T vs N-Image I2T (Phase 7)
    let formatInstruction;
    if (isVirtualMode) {
        // Virtual T2T - simpler format, just need the answer
        formatInstruction = `Provide a clear and detailed response to the user's question.`;
    } else if (numImages > 1) {
        // Phase 7: N-Image I2T - the model receives N labeled images and must
        // synthesize them into ONE English prompt for I2I generation. The user
        // can reference images by number (Image 1, Image 2, ...) in their request.
        formatInstruction = `[FORMAT - N-IMAGE SYNTHESIS]
You are analyzing ${numImages} images numbered Image 1, Image 2, ..., Image ${numImages}.
The user's request below may reference these images by number (e.g. "face from image 2",
"style of image 3", "background from image 1"). Extract the relevant visual details from
the referenced images and synthesize them into ONE coherent English prompt for image-to-image
generation. The final prompt must describe the target image directly - do NOT mention
"image 1", "image 2" or any image numbers in the final prompt.

Divide your response into exactly three sections using these EXACT markers (no other text outside markers):

###POSITIVE###
[Detailed English prompt for image generation that incorporates elements from the referenced images. 50-150 words.]
###NEGATIVE###
[Things to avoid in the generated image. 10-30 words.]
###REASONING###
[Explain which images you referenced and what specific elements you extracted from each. List the visual details that come from each image number.]
[/FORMAT]`;
    } else {
        // Real I2T - single image with POSITIVE/NEGATIVE/REASONING sections
        formatInstruction = `[FORMAT]
Divide your response into exactly three sections using these EXACT markers (no other text outside markers):

###POSITIVE###
[Describe what to include in the image - positive prompt for image generation]
###NEGATIVE###
[Describe what to exclude in the image - negative prompt]
###REASONING###
[Explain your analysis and reasoning process]
[/FORMAT]`;
    }

    const fullPrompt = `${formatInstruction}\n\nUser request: ${prompt}`;

    // Build payload - include conversation for virtual T2T mode
    // enable_thinking defaults to false: the chat template prefills the
    // empty reasoning channel for Gemma 4, and Qwen 3.x just ignores
    // the kwarg. The per-model params.json can still override this.
    const payload = {
        prompt: fullPrompt,
        model_dir: modelDir || null,
        temperature: parseFloat(document.getElementById('temp-i2t')?.value || 0.8),  // Slightly higher default
        max_tokens: 1024,
        enable_thinking: false,
    };
    
    // Add ctx_size from slider if available
    const ctxSlider = document.getElementById('ctx-i2t');
    if (ctxSlider) {
        const ctxValue = parseInt(ctxSlider.value);
        if (ctxValue && ctxValue > 0) {
            payload.ctx_size = ctxValue;
        }
    }

    // Add max image dimension for resizing (saves VRAM)
    const maxImgSlider = document.getElementById('maximg-i2t');
    const maxImgSize = maxImgSlider ? parseInt(maxImgSlider.value) : 1024;
    if (maxImgSize && maxImgSize > 0) {
        payload.max_img_size = maxImgSize;
    }

    // Phase 6: N-Image I2T - send image_specs array
    if (actualImageSpecs && actualImageSpecs.length > 0) {
        if (actualImageSpecs.length === 1) {
            // Single image: use backward-compat image_path field
            payload.image_path = actualImageSpecs[0].path;
        } else {
            // N images: send image_specs array with paths and roles
            payload.image_specs = actualImageSpecs.map(s => ({
                path: s.path,
                role: s.role || 'ref'
            }));
        }
    }
    
    // Include conversation history for virtual T2T context
    if (isVirtualMode && window.i2tConversationHistory) {
        payload.conversation = window.i2tConversationHistory;
    }
    
    console.log("I2T Interactive payload:", payload);

    // Store abortController globally so STOP button can abort I2T processing
    window.i2tAbortController = new AbortController();
    const abortController = window.i2tAbortController;
    
    // Store userInput for conversation display
    const currentUserInput = userInput;

    try {
        const response = await fetch('/api/i2t/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: abortController.signal,
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: TXT_CHUNK|')) {
                    // Phase 8B: Real-time streaming chunk
                    const chunkContent = line.replace('data: TXT_CHUNK|', '').replace(/\\n/g, '\n');
                    fullResponse = (fullResponse || '') + chunkContent;
                    // Stream to the text-result-area live (no chat pollution - Phase 8A)
                    if (!currentUserInput) {
                        // For initial analysis: incrementally update the result boxes
                        // We need to incrementally fill the boxes, but the parser needs the
                        // FULL text to extract POSITIVE/NEGATIVE/REASONING. So we just
                        // show the text in a "live" status area during streaming.
                        updateStatusGen(`Streaming... (${fullResponse.length} chars)`, 'info');
                    } else {
                        // For chat follow-up: append to chat display in real-time
                        appendToChatDisplayLive(fullResponse);
                    }
                } else if (line.startsWith('data: TXT|')) {
                    // Final complete text payload (sent at the end of streaming)
                    const textContent = line.replace('data: TXT|', '').trim().replace(/\\n/g, '\n');
                    fullResponse = textContent;

                    // Chat follow-up (userInput provided): only update conversation display
                    // Initial analysis (no userInput): fill positive/negative/reasoning boxes
                    // Phase 8B: Clean up the live streaming element if it exists
                    const liveEl = document.getElementById('i2t-live-response');
                    if (liveEl) liveEl.remove();

                    if (currentUserInput) {
                        // Chat mode - skip displayTextResult to avoid polluting positive/negative boxes
                        updateStatusGen('Response received!', 'success');
                    } else {
                        // Initial analysis - fill the three result boxes with parsed result
                        displayTextResult(textContent);
                        updateStatusGen('Analysis Complete!', 'success');
                    }
                    
                    // Initial analysis (no chat follow-up): store prompt for "Send Full IA" but
                    // do NOT push to chat history (Phase 8A - chat is a separate channel for
                    // direct conversation with the AI, not for image analysis results).
                    if (!currentUserInput && !isVirtualMode && fullResponse && fullResponse.trim().length > 0) {
                        // Store the produced prompt so user can send it to I2I generation
                        window.i2tLastProducedPrompt = fullResponse;
                        window.i2tVirtualMode = true;
                        // NOTE: chat history is intentionally NOT updated here. The IA result
                        // is shown in the dedicated text-result-area, not in the chat display.
                    }
                    
                    // Show the interactive bar for continued conversation (for both initial and follow-up)
                    showI2TInteractiveBar(true);
                } else if (line.startsWith('data: LOG|')) {
                    const logMsg = line.replace('data: LOG|', '').trim();
                    appendLog(logMsg);
                } else if (line.startsWith('data: DONE|')) {
                    // completed
                } else if (line.startsWith('data: ERR|')) {
                    const errMsg = line.replace('data: ERR|', '').trim();
                    alert('❌ Error: ' + errMsg);
                    updateStatusGen('Error!', 'error');
                    showI2TInteractiveBar(false);
                } else if (line.startsWith('data: ERROR|')) {
                    const errorJson = line.replace('data: ERROR|', '').trim();
                    try {
                        const errorData = JSON.parse(errorJson);
                        alert(`❌ Error: ${errorData.error || errorJson}`);
                    } catch (e) {
                        alert('❌ Error: ' + errorJson);
                    }
                    updateStatusGen('Error!', 'error');
                    showI2TInteractiveBar(false);
                } else if (line.startsWith('data: TAGGING_RESUME|')) {
                    // T2I/I2I generation preempted an auto-tag batch.
                    // The backend has restored the tag server and saved
                    // a checkpoint; trigger the resume automatically
                    // so the user doesn't have to click anything.
                    const resumeFrom = (line.split('from=')[1] || '0').trim();
                    console.log('[Generate] Auto-resuming tagging from index', resumeFrom);
                    if (typeof window.resumeTagging === 'function') {
                        // Small delay so the popup is fully closed by
                        // the [Stop Tagging] call earlier in the flow.
                        setTimeout(() => window.resumeTagging(), 250);
                    }
                }
            }
        }

        // Update conversation display if we have user input and non-empty response
        if (currentUserInput && fullResponse && fullResponse.trim().length > 0) {
            // Extract just the answer part (after ###ANSWER###) for cleaner display
            let cleanResponse = fullResponse;
            if (fullResponse.includes('###ANSWER###')) {
                const parts = fullResponse.split('###ANSWER###');
                cleanResponse = parts[parts.length - 1].trim();
            }
            updateConversationDisplay(currentUserInput, cleanResponse);
        } else if (currentUserInput && (!fullResponse || fullResponse.trim().length === 0)) {
            // Empty response - log warning but don't add to conversation
            console.warn('I2T: Empty response received, not adding to conversation');
            updateStatusGen('Empty response - try again', 'warning');
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            console.log('I2T query aborted by user.');
        } else {
            console.error('I2T Interactive error:', e);
            updateStatusGen('Error: ' + e.message, 'error');
        }
    }

    stopGeneration();
}

/**
 * Shutdown the persistent llama-server to free VRAM.
 */
async function shutdownI2TServer() {
    try {
        const res = await fetch('/api/i2t/shutdown', { method: 'POST' });
        const data = await res.json();
        console.log('I2T server shutdown:', data);
        showI2TInteractiveBar(false);
        updateStatusGen('Model unloaded', 'info');
    } catch (e) {
        console.error('Failed to shutdown I2T server:', e);
    }
}

/**
 * Check if the persistent llama-server is running and update UI.
 */
async function checkI2TServerStatus() {
    const statusEl = document.getElementById('i2t-chat-status');
    if (!statusEl) return;
    try {
        const res = await fetch('/api/i2t/status');
        const data = await res.json();
        if (data.running) {
            statusEl.innerHTML = '● Model loaded in VRAM';
            statusEl.style.color = '#4ade80';
        } else {
            statusEl.innerHTML = '○ No model loaded';
            statusEl.style.color = '#888';
        }
    } catch (e) {
        statusEl.innerHTML = '○ Status unknown';
        statusEl.style.color = '#888';
    }
}

/**
 * Show/hide the interactive I2T action bar.
 */
function showI2TInteractiveBar(visible) {
    const chatSection = document.getElementById('i2t-chat-section');
    if (chatSection) {
        if (visible) {
            chatSection.classList.remove('hidden');
        } else {
            chatSection.classList.add('hidden');
        }
    }
    if (visible) checkI2TServerStatus();
}

/**
 * Called when the user presses Enter on the chat input.
 *
 * Builds a conversational prompt by prepending the previous answer to the user's comment.
 * This allows the user to have a back-and-forth dialogue with the model about the image.
 *
 * Format sent to server:
 *   [Previous Answer]
 *   ---
 *   [User's new comment from i2t-chat-input]
 */
async function retryI2T() {
    if (window.isProcessing) {
        if (confirm('Abort current process?')) {
            if (abortController) abortController.abort();
            stopGeneration();
        }
        return;
    }

    // Phase 6: N-Image I2T - gather all slots with images
    const allSlots = (typeof getImageSlots === 'function') ? getImageSlots() : [];
    const validSlots = allSlots.filter(s => s.path && s.role !== 'none');
    const imagePath = validSlots[0]?.path || null;
    const imageSpecs = validSlots.length > 0 ? validSlots.map(s => ({ path: s.path, role: s.role })) : null;

    const i2tModelSelect = document.getElementById('i2t-model-select');
    const modelDir = i2tModelSelect?.value || null;

    // Get the previous answer from conversation history
    let previousAnswer = '';
    if (window.i2tConversationArray && window.i2tConversationArray.length > 0) {
        const lastExchange = window.i2tConversationArray[window.i2tConversationArray.length - 1];
        previousAnswer = lastExchange?.response || '';
    }

    // Get the user's new input from the dedicated chat input
    const commentBox = document.getElementById('i2t-chat-input');
    const userInput = commentBox?.value?.trim() || '';

    if (!userInput.trim()) {
        alert('Please type a comment or question in the text box below.');
        return;
    }

    // Check if we're in virtual T2T mode (no images at all)
    const isVirtualMode = !imageSpecs || imageSpecs.length === 0;

    if (isVirtualMode) {
        // Virtual T2T mode - accumulate conversation
        // Build conversational prompt with accumulated history
        let conversationalPrompt;
        if (previousAnswer) {
            conversationalPrompt = `Previous answer:\n${previousAnswer}\n---\nUser comment:\n${userInput}`;
        } else {
            // First query - just use the prompt as-is
            conversationalPrompt = userInput;
        }

        // Clear the comment box after reading
        if (commentBox) commentBox.value = '';

        await runI2TInteractive(null, conversationalPrompt, modelDir, userInput, null);
    } else {
        // Real I2T mode with image(s) - prepend conversation if available
        let conversationalPrompt;
        if (previousAnswer) {
            conversationalPrompt = `Previous answer:\n${previousAnswer}\n---\nUser comment:\n${userInput}`;
        } else {
            conversationalPrompt = userInput;
        }

        // Clear the comment box after reading
        if (commentBox) commentBox.value = '';

        // Phase 6: pass imageSpecs (N-image) instead of single imagePath
        await runI2TInteractive(imagePath, conversationalPrompt, modelDir, userInput, imageSpecs);
    }
}

/**
 * Update the conversation display area with the full conversation history.
 * User questions are shown in italic, model answers in normal text.
 *
 * @param {string} userInput - The user's question/comment
 * @param {string} modelResponse - The model's response
 */
/**
 * Phase 8B: Append text live to the chat display as it streams in.
 * Used for chat follow-up in I2T mode to show real-time token-by-token output.
 */
function appendToChatDisplayLive(text) {
    const convDisplay = document.getElementById('i2t-chat-display');
    if (!convDisplay) return;
    // Use a temporary "live" div that gets replaced by the final conversation
    let liveEl = document.getElementById('i2t-live-response');
    if (!liveEl) {
        liveEl = document.createElement('div');
        liveEl.id = 'i2t-live-response';
        liveEl.style.cssText = 'color: #4ade80; margin-bottom: 12px; font-style: italic;';
        liveEl.textContent = '';
        convDisplay.appendChild(liveEl);
    }
    // Strip ### markers for cleaner live display
    let displayText = text.replace(/###POSITIVE###\n?/g, '').replace(/###NEGATIVE###\n?/g, '').replace(/###REASONING###\n?/g, '').replace(/###THINKING###\n?/g, '').replace(/###ANSWER###\n?/g, '');
    // Sanitize text to remove problematic characters before display
    displayText = sanitizeForJSON(displayText);
    liveEl.textContent = displayText;
    convDisplay.scrollTop = convDisplay.scrollHeight;
}

window.appendToChatDisplayLive = appendToChatDisplayLive;

function updateConversationDisplay(userInput, modelResponse) {
    const convDisplay = document.getElementById('i2t-chat-display');
    if (!convDisplay) return;
    
    // Initialize conversation history array if not exists
    if (!window.i2tConversationArray) {
        window.i2tConversationArray = [];
    }
    
    // Add new exchange to history (only if userInput and modelResponse are non-empty)
    if (userInput && userInput.trim() && modelResponse && modelResponse.trim()) {
        window.i2tConversationArray.push({
            user: userInput,
            response: modelResponse
        });
    }
    
    // Build HTML for full conversation
    let html = '';
    if (window.i2tConversationArray.length === 0) {
        // Show placeholder when no conversation history
        html = '<div style="color: #666; font-style: italic;">Start a conversation by typing below...</div>';
    } else {
        for (const exchange of window.i2tConversationArray) {
            html += `<div style="margin-bottom: 12px;">
                <div style="color: #8ab4f8; font-style: italic;">❓ ${escapeHtml(exchange.user)}</div>
                <div style="color: #4ade80; margin-top: 4px;">✅ ${escapeHtml(exchange.response)}</div>
            </div>`;
        }
    }
    
    convDisplay.innerHTML = html;
    if (html && window.i2tConversationArray.length > 0) {
        // Scroll to bottom
        convDisplay.scrollTop = convDisplay.scrollHeight;
    }
}

/**
 * Escape HTML special characters to prevent XSS.
 * Also sanitizes typographic characters (smart quotes, dashes, etc.)
 * to prevent mojibake in the C++ sd binary when text is copied to prompt.
 */
function escapeHtml(text) {
    if (!text) return '';
    // Sanitize typographic characters FIRST (before HTML escaping)
    // This ensures the text displayed in the chat is already clean
    const sanitized = sanitizeForJSON(text);
    const div = document.createElement('div');
    div.textContent = sanitized;
    return div.innerHTML;
}

/**
 * Enable/disable the To Prompt button based on text selection in chat display.
 */
function updateToPromptButton() {
    const convDisplay = document.getElementById('i2t-chat-display');
    const toPromptBtn = document.getElementById('btn-i2t-to-prompt');
    if (!toPromptBtn || !convDisplay) return;

    const selection = window.getSelection();
    const hasSelection = selection && selection.toString().trim().length > 0 &&
                         convDisplay.contains(selection.anchorNode);

    if (hasSelection) {
        toPromptBtn.disabled = false;
        toPromptBtn.style.opacity = '1';
        toPromptBtn.style.cursor = 'pointer';
        toPromptBtn.title = 'Send selected text to main Prompt';
    } else {
        toPromptBtn.disabled = true;
        toPromptBtn.style.opacity = '0.4';
        toPromptBtn.style.cursor = 'default';
        toPromptBtn.title = 'Select text in the chat above to enable, then click to send to main Prompt';
    }
}

/**
 * Convert the selected text from chat display to the main prompt.
 * Called when user clicks "To Prompt" button.
 * Only works when text is selected in the chat display.
 */
function convertConversationToPrompt() {
    const convDisplay = document.getElementById('i2t-chat-display');
    const toPromptBtn = document.getElementById('btn-i2t-to-prompt');

    if (!convDisplay) {
        alert('Chat display not found.');
        return;
    }

    // Get the selected text
    const selection = window.getSelection();
    const selectedText = selection ? selection.toString().trim() : '';

    if (!selectedText) {
        alert('Please select some text in the chat above first.');
        toPromptBtn.disabled = true;
        toPromptBtn.style.opacity = '0.4';
        return;
    }

    // Sanitize selected text to remove problematic characters before copying
    const sanitizedText = sanitizeForJSON(selectedText);
    
    // Copy to main prompt box
    const promptBox = document.getElementById('prompt');
    if (promptBox) {
        promptBox.value = sanitizedText;
        promptBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    updateStatusGen('Selected text sent to prompt box', 'success');
}

/**
 * Clear the entire chat conversation.
 */
function clearChat() {
    if (!window.i2tConversationArray || window.i2tConversationArray.length === 0) {
        updateStatusGen('Chat already empty', 'info');
        return;
    }

    // Reset conversation data
    window.i2tConversationArray = [];
    window.i2tConversationHistory = null;

    // Clear the display
    const convDisplay = document.getElementById('i2t-chat-display');
    if (convDisplay) {
        convDisplay.innerHTML = '<div style="color: #6688aa; font-style: italic;">Start a conversation by typing below...</div>';
    }

    // Clear the input
    const chatInput = document.getElementById('i2t-chat-input');
    if (chatInput) {
        chatInput.value = '';
    }

    // Disable the To Prompt button
    const toPromptBtn = document.getElementById('btn-i2t-to-prompt');
    if (toPromptBtn) {
        toPromptBtn.disabled = true;
        toPromptBtn.style.opacity = '0.4';
    }

    updateStatusGen('Chat cleared', 'info');
}

// Wire up the I2T chat input and action buttons
document.addEventListener('DOMContentLoaded', function () {
    const chatInput = document.getElementById('i2t-chat-input');
    if (chatInput) {
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                retryI2T();
            }
        });
    }

    // To Prompt button - sends selected text
    const toPromptBtn = document.getElementById('btn-i2t-to-prompt');
    if (toPromptBtn) {
        toPromptBtn.onclick = convertConversationToPrompt;
        // Start disabled
        toPromptBtn.disabled = true;
        toPromptBtn.style.opacity = '0.4';
    }

    // Clear Chat button
    const clearChatBtn = document.getElementById('btn-i2t-clear-chat');
    if (clearChatBtn) {
        clearChatBtn.onclick = clearChat;
    }

    // Track text selection in chat display to enable/disable To Prompt button
    const convDisplay = document.getElementById('i2t-chat-display');
    if (convDisplay) {
        document.addEventListener('mouseup', updateToPromptButton);
        document.addEventListener('keyup', updateToPromptButton);
    }

    // Periodic status check
    setInterval(checkI2TServerStatus, 5000);
});

// Make functions globally accessible
window.runI2TInteractive = runI2TInteractive;
window.shutdownI2TServer = shutdownI2TServer;
window.checkI2TServerStatus = checkI2TServerStatus;
window.showI2TInteractiveBar = showI2TInteractiveBar;
window.retryI2T = retryI2T;
window.copyToClipboard = copyToClipboard;
window.sendPromptTo = sendPromptTo;
window.sendNegativePromptTo = sendNegativePromptTo;
window.sendThinkingTo = sendThinkingTo;
window.updateConversationDisplay = updateConversationDisplay;
window.convertConversationToPrompt = convertConversationToPrompt;
window.clearChat = clearChat;
window.updateToPromptButton = updateToPromptButton;

// IA Bidirectional Data Flow exports
window.exportIAState = exportIAState;
window.restoreIAState = restoreIAState;
window.sendToIA = sendToIA;
window.sendToI2I = sendToI2I;
window.sendToI2T = sendToI2T;
window.sendFullIAtoT2I = sendFullIAtoT2I;
window.sendFullIAtoI2I = sendFullIAtoI2I;

console.log('Generate loaded');
