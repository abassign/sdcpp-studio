/**
 * ui.js - User Interface Logic
 * Handles tabs, sliders, and visual updates.
 */

// References - These are also set as window.* in config.js
// We declare them with 'const' to provide local bindings without conflicting with the
// existing window.* properties. The browser allows 'const X' even when window.X already exists,
// because 'const' creates a script-level binding (not a property on the global object).
const sliderW = document.getElementById('width');
const sliderH = document.getElementById('height');
const valW = document.getElementById('val-w');
const valH = document.getElementById('val-h');
const imgArea = document.getElementById('image-area');
const placeholder = document.querySelector('.placeholder');
const resultImg = document.getElementById('result-img');
const thinkingBox = document.getElementById('thinking-box');
// window.promptBox and window.uploadedPathInput are already declared in config.js as window.* properties.
// Don't redeclare them here - some browsers throw "Identifier already declared" if you do.
// All references in this file use window.promptBox / window.uploadedPathInput directly.

window.currentResultPath = "";


// ============================================================================
// Phase 5: Slot Visibility Helper (N-Slot Dynamic)
// ============================================================================
function updateSlotVisibilityForMode(mode) {
    // Modes that need 1 image only (init)
    const singleImageModes = [];
    // Modes that need 1+ images (init + optional refs)
    const multiImageModes = ['i2i', 'i2t'];
    // t2i doesn't need any image
    const noImageModes = ['t2i'];

    const addSlotBtn = document.getElementById('btn-add-slot');
    const slotsContainer = document.getElementById('image-slots-container');
    const slot2Container = document.getElementById('slot-2-container');
    const swapContainer = document.getElementById('swap-container');

    if (noImageModes.includes(mode)) {
        // Hide entire upload panel
        if (slotsContainer) slotsContainer.classList.add('hidden');
        if (addSlotBtn) addSlotBtn.classList.add('hidden');
        if (slot2Container) slot2Container.classList.add('hidden');
        if (swapContainer) swapContainer.classList.add('hidden');
        return;
    }

    if (slotsContainer) slotsContainer.classList.remove('hidden');

    // In single-image modes, hide slot 2+ (keep only init slot)
    const slots = document.querySelectorAll('#image-slots-container .upload-slot');
    if (singleImageModes.includes(mode)) {
        // Hide all slots after slot 1
        slots.forEach((slot, idx) => {
            if (idx >= 1) slot.classList.add('hidden');
        });
        if (addSlotBtn) addSlotBtn.classList.add('hidden');
    } else {
        // Multi-image modes: show all slots
        slots.forEach(slot => slot.classList.remove('hidden'));
        if (addSlotBtn && typeof imageSlots !== 'undefined' && imageSlots.length < 8) {
            addSlotBtn.classList.remove('hidden');
        } else if (addSlotBtn) {
            addSlotBtn.classList.add('hidden');
        }
    }

    if (slot2Container) {
        if (singleImageModes.includes(mode)) {
            slot2Container.classList.add('hidden');
        } else {
            slot2Container.classList.remove('hidden');
        }
    }
}

window.updateSlotVisibilityForMode = updateSlotVisibilityForMode;

window.switchTab = function (mode, btn) {
    try {
        // --- Leaving 'agent' mode -> hide agent panel, restore standard workspace ---
        if (mode !== 'agent') {
            // move the standard History panel back to the main layout first
            if (window.agentExitWorkspace) window.agentExitWorkspace();
            const agentPanel = document.getElementById('agent-panel');
            if (agentPanel) agentPanel.classList.add('hidden');
            const pl = document.querySelector('.panel-left');
            if (pl) pl.style.display = '';
            const pc = document.querySelector('.panel-center-container');
            if (pc) pc.style.display = '';
            // restore the standard columns container (hidden only in agent mode)
            const panelsC = document.querySelector('.panels-container');
            if (panelsC) panelsC.style.display = '';
        }

        // --- Auto-shutdown persistent llama-server when switching away from I2T ---
        // Free GPU VRAM when the user switches to a different module.
        if (window.currentMode === 'i2t' && mode !== 'i2t') {
            console.log('[switchTab] Leaving I2T → shutting down persistent llama-server');
            if (window.shutdownI2TServer) {
                window.shutdownI2TServer();
            }
            // Hide the interactive bar
            if (window.showI2TInteractiveBar) {
                window.showI2TInteractiveBar(false);
            }
        }

        // Save current prompt to buffer
        if (window.currentMode && window.promptBuffers.hasOwnProperty(window.currentMode) && window.promptBox) {
            window.promptBuffers[window.currentMode] = window.promptBox.value;
        }

        window.currentMode = mode;

        // Phase 1E: Restore saved slot roles for this mode
        if (typeof restoreSlotRoles === 'function') {
            try { restoreSlotRoles(mode); } catch (e) { /* ignore */ }
        }

        // Restore new prompt from buffer
        if (window.promptBox) window.promptBox.value = window.promptBuffers[mode] || "";

        // Show image toolbar for all non-IA modes
        const imageToolbar = document.getElementById('image-toolbar');
        if (mode !== 'i2t') {
            if (imageToolbar) imageToolbar.classList.remove('hidden');
            const savedImg = window.imageBuffers[mode];
            if (savedImg) {
                window.currentResultPath = savedImg;
                if (resultImg) {
                    resultImg.src = savedImg + "?t=" + Date.now();
                    resultImg.style.display = 'block';
                }
                if (placeholder) placeholder.style.display = 'none';
                if (imgArea) imgArea.style.display = 'flex';
            } else {
                // No image for this mode -> Show Placeholder
                window.currentResultPath = "";
                if (resultImg) {
                    resultImg.style.display = 'none';
                    resultImg.src = "";
                }
                if (placeholder) placeholder.style.display = 'block';
                if (imgArea) imgArea.style.display = 'flex';
            }
        } else {
            // I2T mode: hide result image, image toolbar, show placeholder
            window.currentResultPath = "";
            if (resultImg) {
                resultImg.style.display = 'none';
                resultImg.src = "";
            }
            if (imageToolbar) imageToolbar.classList.add('hidden');
            if (placeholder) placeholder.style.display = 'block';
            if (imgArea) imgArea.style.display = 'flex';
        }

        // Active Button State
        if (!btn) {
            // Fallback inference if btn not passed
            const btns = document.querySelectorAll('.nav-btn');
            if (mode === 'i2t' && btns[0]) btn = btns[0];
            else if (mode === 't2i' && btns[1]) btn = btns[1];
            else if (mode === 'i2i' && btns[2]) btn = btns[2];
            else if (mode === 'agent' && btns[3]) btn = btns[3];

        }
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        if (btn) btn.classList.add('active');

        // Panel Visibility Reset
        const panels = [
            'upload-panel', 'params-panel', 'i2i-specific-params', 'i2t-params-panel',
            'text-result-area', 'i2t-model-container'
        ];
        panels.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.add('hidden');
        });

        // Specific Panel Logic
        const btnGenerate = document.getElementById('btn-generate');
        const btnOrig = document.querySelector('.btn-orig');
        const slot2Container = document.getElementById('slot-2-container');
        const swapContainer = document.getElementById('swap-container');
        const loraSec = document.querySelector('.lora-section');
        const colMix = document.getElementById('col-mix');
        const historyBtns = ['btn-hist-top', 'btn-hist-i2i', 'btn-hist-i2t'];

        // Hide History Buttons initially
        historyBtns.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.add('hidden');
        });

        // Is LoRA visible? Default yes
        if (loraSec) loraSec.classList.remove('hidden');

        if (mode === 't2i') {
            document.getElementById('params-panel').classList.remove('hidden');
            const imageArea = document.getElementById('image-area');
            if (imageArea) {
                imageArea.classList.remove('hidden');
            }
            if (document.getElementById('btn-hist-top')) document.getElementById('btn-hist-top').classList.remove('hidden');

            // Show Text Encoder in T2I mode
            const textEncoderGroupT2i = document.getElementById('llm-select-container');
            if (textEncoderGroupT2i) textEncoderGroupT2i.classList.remove('hidden');

            // Show standard generate button
            if (btnGenerate) btnGenerate.classList.remove('hidden');
            if (btnGenerate) btnGenerate.innerText = "🚀 GENERATE";
            if (window.promptBox) window.promptBox.placeholder = "Describe your imagination here...";
            if (btnOrig) { btnOrig.style.opacity = "0.3"; btnOrig.style.cursor = "not-allowed"; }

        } else if (mode === 'i2i') {
            document.getElementById('upload-panel').classList.remove('hidden');
            document.getElementById('params-panel').classList.remove('hidden');
            document.getElementById('i2i-specific-params').classList.remove('hidden');
            const imageArea = document.getElementById('image-area');
            if (imageArea) {
                imageArea.classList.remove('hidden');
            }
            if (document.getElementById('btn-hist-i2i')) document.getElementById('btn-hist-i2i').classList.remove('hidden');

            // Show Text Encoder in I2I mode
            const textEncoderGroupI2i = document.getElementById('llm-select-container');
            if (textEncoderGroupI2i) textEncoderGroupI2i.classList.remove('hidden');

            // Slots
            if (slot2Container) slot2Container.classList.remove('hidden');
            if (swapContainer) swapContainer.classList.remove('hidden');
            updateSlotVisibilityForMode(mode);

            if (btnGenerate) btnGenerate.classList.remove('hidden');
            if (btnGenerate) btnGenerate.innerText = "🎨 RESTYLE IMAGE";
            if (window.promptBox) window.promptBox.placeholder = "Describe how to modify the image...";
            if (btnOrig) { btnOrig.style.opacity = "1"; btnOrig.style.cursor = "pointer"; }

        } else if (mode === 'i2t') {
            // Hide LoRA in I2T mode - must be FIRST
            if (loraSec) loraSec.classList.add('hidden');

            document.getElementById('upload-panel').classList.remove('hidden');

            // I2T: show both slots (user wants to be able to analyze 2 images)
            // Slot 1 is the main image, Slot 2 is optional comparison
                        updateSlotVisibilityForMode(mode);

            // Hide main params-panel, show I2T-specific params panel instead
            document.getElementById('params-panel').classList.add('hidden');
            document.getElementById('i2t-params-panel').classList.remove('hidden');
            
            // Show I2T model selector
            const i2tModelContainer = document.getElementById('i2t-model-container');
            if (i2tModelContainer) i2tModelContainer.classList.remove('hidden');

            // I2T: Hide image-area (toolbar hidden in general switchTab logic), show text result area
            const imageArea = document.getElementById('image-area');
            if (imageArea) {
                imageArea.classList.add('hidden');
            }

            const textResultArea = document.getElementById('text-result-area');
            if (textResultArea) {
                textResultArea.classList.remove('hidden');
            }

            // Check if there's an uploaded image and conversation history
            // (window.uploadedPathInput is already declared at module level, line 16)
            const hasImage = window.uploadedPathInput?.value?.trim();
            const hasConversation = window.i2tConversationArray && window.i2tConversationArray.length > 0;
            
            // Virtual T2T mode persists if:
            // - No image uploaded yet, OR
            // - Image IS uploaded but there's accumulated conversation history
            // This allows user to keep refining the prompt even after uploading an image
            if (!hasImage || hasConversation) {
                // Virtual T2T mode - show interactive bar
                // User can continue refining prompt even with image present
                window.i2tVirtualMode = true;
                if (window.showI2TInteractiveBar) {
                    window.showI2TInteractiveBar(true);
                }
            } else {
                // Real I2T mode - image is uploaded and no conversation yet
                window.i2tVirtualMode = false;
            }

            // Hide I2I-specific params (Mix/Strength) since I2T has its own in i2t-params-panel
            document.getElementById('i2i-specific-params').classList.add('hidden');
            if (document.getElementById('btn-hist-i2t')) document.getElementById('btn-hist-i2t').classList.remove('hidden');

            // Show Text Encoder in I2T mode
            const textEncoderGroup = document.getElementById('llm-select-container');
            if (textEncoderGroup) textEncoderGroup.classList.remove('hidden');

            if (slot2Container) slot2Container.classList.remove('hidden');
            if (swapContainer) swapContainer.classList.remove('hidden');

            if (btnGenerate) btnGenerate.classList.remove('hidden');
            if (btnGenerate) btnGenerate.innerText = "🧠 ANALYZE IMAGE(S)";
            if (window.promptBox) window.promptBox.placeholder = "Ask a question about the image(s)...";
            if (btnOrig) { btnOrig.style.opacity = "1"; btnOrig.style.cursor = "pointer"; }

        } else if (mode === 'agent') {
            // === Dedicated Agentic Refine workspace (M2) ===
            // Hide the standard left/right columns and show the agent panel.
            if (loraSec) loraSec.classList.add('hidden');
            if (btnGenerate) btnGenerate.classList.add('hidden');
            ['upload-panel', 'params-panel', 'i2t-params-panel', 'i2i-specific-params']
                .forEach(id => { const el = document.getElementById(id); if (el) el.classList.add('hidden'); });
            const imageToolbar2 = document.getElementById('image-toolbar');
            if (imageToolbar2) imageToolbar2.classList.add('hidden');
            const pl = document.querySelector('.panel-left');
            if (pl) pl.style.display = 'none';
            const pc = document.querySelector('.panel-center-container');
            if (pc) pc.style.display = 'none';
            // hide the whole standard column container: agent panel lives right
            // below it in main-layout and fills the freed space entirely.
            const panelsC = document.querySelector('.panels-container');
            if (panelsC) panelsC.style.display = 'none';
            const agentPanel = document.getElementById('agent-panel');
            if (agentPanel) agentPanel.classList.remove('hidden');
            // agent panel is a direct flex child of main-layout: make sure it gets
            // the full remaining height in the column layout
            if (agentPanel) agentPanel.style.flex = '1';
            // dock the standard History panel at the right of the workspace
            if (window.agentEnterWorkspace) window.agentEnterWorkspace();
            ['btn-hist-top', 'btn-hist-i2i', 'btn-hist-i2t'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.add('hidden');
            });
        }

        // Trigger updates
        if (window.updatePreviews) window.updatePreviews();
        if (window.updateParamVisibility) window.updateParamVisibility();
        if (window.validateInputs) window.validateInputs();

        // Refresh History View
        if (window.refreshHistoryView) {
            window.currentPage = 0;
            window.refreshHistoryView();
        }

        // Notify Preferences system of mode change
        if (window.Preferences && window.Preferences.onModeChanged) {
            window.Preferences.onModeChanged(mode);
        }

    } catch (e) {
        console.error("Error in switchTab:", e);
        if (window.log) window.log("UI Error: " + e.message, true);
    }
};

window.toggleSystemLog = function () {
    const drawer = document.getElementById('system-log-drawer');
    const btn = document.getElementById('btn-log-toggle');
    if (!drawer || !btn) return;

    drawer.classList.toggle('collapsed');
    const isCollapsed = drawer.classList.contains('collapsed');
    btn.innerText = isCollapsed ? '▼' : '▲';
};

window.toggleThinking = function () {
    if (!thinkingBox) return;
    const isCollapsed = thinkingBox.classList.toggle('collapsed');
    const icon = document.getElementById('thinking-toggle-icon');
    if (icon) icon.innerText = isCollapsed ? "▶" : "▼";
};

window.setRatio = function (wRatio, hRatio) {
    if (!sliderW || !sliderH) return;
    const currentW = parseInt(sliderW.value);
    const newH = Math.round(currentW * (hRatio / wRatio));
    // Clamp
    const clampedH = Math.max(512, Math.min(2048, newH));
    sliderH.value = clampedH;
    valH.innerText = clampedH;
};

window.setOriginalRatio = function () {
    // Delegate to the shared implementation (main.js) which reads the INIT
    // image dimensions from the I2T/I2I slots. Falls back to the old cache.
    if (typeof window.setOriginalRatioImpl === 'function') return window.setOriginalRatioImpl();
    if (window.currentMode === 't2i') return;
    if (window.uploadedImageDims && window.uploadedImageDims.w > 0) {
        sliderW.value = window.uploadedImageDims.w;
        valW.innerText = window.uploadedImageDims.w; // Assumes valW exists
        sliderH.value = window.uploadedImageDims.h;
        valH.innerText = window.uploadedImageDims.h;
        window.log(`Resized to Original: ${window.uploadedImageDims.w}x${window.uploadedImageDims.h}`);
    } else {
        window.log("No image loaded for Original Ratio", true);
    }
};

window.showImage = function (url) {
    if (placeholder) placeholder.style.display = 'none';
    if (imgArea) imgArea.style.display = 'flex';

    window.currentResultPath = url;

    // Save to buffer
    if (window.currentMode && window.imageBuffers.hasOwnProperty(window.currentMode)) {
        window.imageBuffers[window.currentMode] = url;
    }

    if (resultImg) {
        resultImg.src = url + "?t=" + Date.now();
        resultImg.style.display = 'block';
    }
    if (window.setArfToolbar) window.setArfToolbar(true);
};

window.updateParamVisibility = function () {
    const isI2T = (window.currentMode === 'i2t');

    // Disable generation params during Analysis (I2T) if they don't apply
    // Note: Some models might use them, but typically I2T is different.
    const ids = ['width', 'height', 'steps', 'guidance', 'seed', 'randomize'];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.disabled = isI2T;
            el.style.opacity = isI2T ? "0.5" : "1";
            el.style.cursor = isI2T ? "not-allowed" : "pointer";
        }
    });

    // Validation
    if (window.validateInputs) window.validateInputs();
};

window.validateInputs = function () {
    // If we are currently processing, don't disable buttons (one might be a STOP button)
    if (window.isProcessing) return;

    const btnGenerate = document.getElementById('btn-generate');

    let isValid = false;

    if (window.currentMode === 't2i') {
        isValid = window.promptBox && window.promptBox.value.trim().length > 0;
    } else {
        // I2I, I2T need image
        isValid = window.uploadedPathInput && !!window.uploadedPathInput.value;
    }

    // Main Generate Button (sdcpp backend - handles all model types)
    if (btnGenerate) {
        btnGenerate.disabled = !isValid;
        btnGenerate.style.opacity = isValid ? "1" : "0.5";
        btnGenerate.style.cursor = isValid ? "pointer" : "not-allowed";
        btnGenerate.title = isValid ? "" : "Please check inputs";
    }
    
    // Edit Image Button (disabled - feature removed)
    const btnEditImage = document.getElementById('btn-edit-image');
    if (btnEditImage) {
        btnEditImage.disabled = true;
        btnEditImage.style.opacity = "0.5";
        btnEditImage.style.cursor = "not-allowed";
        btnEditImage.title = "Edit Image feature removed";
    }
};

console.log("UI module loaded.");
