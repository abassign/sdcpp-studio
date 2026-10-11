// main.js - Entry point
// ====================
// Aspect Ratio Logic
var lastActiveDim = 'w'; // Default to width

function setRatio(ratioW, ratioH) {
    var wInput = document.getElementById('width');
    var hInput = document.getElementById('height');
    var valW = document.getElementById('val-w');
    var valH = document.getElementById('val-h');

    var w = parseInt(wInput.value);
    var h = parseInt(hInput.value);

    if (lastActiveDim === 'w') {
        // Width is master, calculate Height
        // Aspect Ratio = ratioW / ratioH = NewW / NewH => NewH = NewW * (ratioH / ratioW)
        var newH = Math.round(w * (ratioH / ratioW));

        // Snap to step
        var step = parseInt(hInput.step) || 32;
        newH = Math.round(newH / step) * step;

        // Clamp (manual checks as changing value via JS doesn't always respect min/max)
        var min = parseInt(hInput.min) || 512;
        var max = parseInt(hInput.max) || 2048;
        newH = Math.max(min, Math.min(newH, max));

        hInput.value = newH;
        if (valH) valH.innerText = newH;

    } else {
        // Height is master, calculate Width
        // Aspect Ratio = ratioW / ratioH = NewW / NewH => NewW = NewH * (ratioW / ratioH)
        var newW = Math.round(h * (ratioW / ratioH));

        // Snap to step
        var step = parseInt(wInput.step) || 32;
        newW = Math.round(newW / step) * step;

        // Clamp
        var min = parseInt(wInput.min) || 512;
        var max = parseInt(wInput.max) || 2048;
        newW = Math.max(min, Math.min(newW, max));

        wInput.value = newW;
        if (valW) valW.innerText = newW;
    }
}

function initRatioLogic() {
    var wInput = document.getElementById('width');
    var hInput = document.getElementById('height');

    if (wInput) {
        wInput.addEventListener('input', function () {
            lastActiveDim = 'w';
        });
    }
    if (hInput) {
        hInput.addEventListener('input', function () {
            lastActiveDim = 'h';
        });
    }
}

function setOriginalRatio() {
    // "Orig": match the dimensions of the INIT/reference image (the shared slot
    // with role 'init'), NOT the last generated result. In T2I there is no
    // init image, so the button is disabled/unused.
    if (window.currentMode === 't2i') return;

    const applyDims = (w, h) => {
        const wInput = document.getElementById('width');
        const hInput = document.getElementById('height');
        const valW = document.getElementById('val-w');
        const valH = document.getElementById('val-h');
        if (!wInput || !hInput || !w || !h) return;

        // Expand limits if the reference is outside the current min/max
        if (w > parseInt(wInput.max)) wInput.max = w;
        if (w < parseInt(wInput.min)) wInput.min = w;
        if (h > parseInt(hInput.max)) hInput.max = h;
        if (h < parseInt(hInput.min)) hInput.min = h;

        wInput.value = w;
        hInput.value = h;
        if (valW) valW.innerText = w;
        if (valH) valH.innerText = h;

        lastActiveDim = 'w';
        // keep the shared dims cache in sync (legacy compat)
        try { window.uploadedImageDims = { w: w, h: h }; } catch (e) { /* ignore */ }
    };

    // Resolve the init image URL from the shared I2T/I2I slots.
    let refUrl = null;
    if (typeof window.getImageSlots === 'function') {
        const refs = window.getImageSlots();
        const init = refs.find(s => s.role === 'init') || refs[0];
        const slot = (init && typeof window.getSlot === 'function') ? window.getSlot(init.id) : null;
        if (slot) {
            refUrl = slot.previewDataUrl ||
                (slot.path ? '/outputs/' + String(slot.path).split('/').pop() : null);
        }
    }

    if (!refUrl) {
        if (window.uploadedImageDims && window.uploadedImageDims.w > 0) {
            applyDims(window.uploadedImageDims.w, window.uploadedImageDims.h);
        } else {
            alert("Load an Init image first to use 🔒 Orig.");
        }
        return;
    }

    const img = new Image();
    img.onload = () => applyDims(img.naturalWidth, img.naturalHeight);
    img.onerror = () => {
        if (window.uploadedImageDims && window.uploadedImageDims.w > 0) {
            applyDims(window.uploadedImageDims.w, window.uploadedImageDims.h);
        } else {
            alert("Could not read the Init image dimensions.");
        }
    };
    img.src = refUrl;
}
window.setOriginalRatioImpl = setOriginalRatio;

function exitButtonClicked() {
    alert("Thanks for using sdcpp-studio! Server shutting down...");
    fetch("/api/exit", { method: "POST" })
        .then(function () {
            document.body.innerHTML = "<h1 style='color:white;text-align:center;margin-top:20%'>System Shutdown.</h1>";
        })
        .catch(function () {
            document.body.innerHTML = "<h1 style='color:white;text-align:center;margin-top:20%'>System Shutdown.</h1>";
        });
}

function toggleSystemLog() {
    var drawer = document.getElementById("system-log-drawer");
    if (drawer) drawer.classList.toggle("collapsed");
}

function toggleNegativePrompt() {
    const textarea = document.getElementById('negative-prompt');
    const btn = document.getElementById('negative-prompt-toggle');
    if (textarea && btn) {
        if (textarea.style.display === 'none') {
            textarea.style.display = 'block';
            btn.textContent = '▼ Hide';
        } else {
            textarea.style.display = 'none';
            btn.textContent = '▼ Show';
        }
    }
}

function toggleI2TChat() {
    const content = document.getElementById('i2t-chat-content');
    const btn = document.getElementById('i2t-chat-toggle');
    if (content && btn) {
        if (content.style.display === 'none') {
            content.style.display = 'block';
            btn.textContent = '▼ Hide';
        } else {
            content.style.display = 'none';
            btn.textContent = '▼ Show';
        }
    }
}

// Assign to window AFTER functions are defined
window.toggleHistory = toggleHistory;
window.toggleHistoryFilter = toggleHistoryFilter;
window.toggleHistorySize = toggleHistorySize;
window.changePage = changePage;
window.viewHistoryItem = viewHistoryItem;
window.toggleAddModel = toggleAddModel;
window.inspectModel = inspectModel;
window.closeInspectModal = closeInspectModal;
window.downloadSelectedFiles = downloadSelectedFiles;
window.downloadModel = downloadModel;
window.deleteImage = deleteImage;
window.swapUploadedImages = swapUploadedImages;
window.handleGeneration = handleGeneration;
window.stopGeneration = stopGeneration;
window.sendToI2I = sendToI2I;
window.sendToI2T = sendToI2T;
window.sendToIA = sendToIA;
window.sendFullIAtoT2I = sendFullIAtoT2I;
window.sendFullIAtoI2I = sendFullIAtoI2I;
window.exportIAState = exportIAState;
window.restoreIAState = restoreIAState;
window.handleI2T = handleI2T;
window.toggleThinking = toggleThinking;
window.scanLoras = scanLoras;
window.addLoraSlot = addLoraSlot;
window.setRatio = setRatio;
window.setOriginalRatio = setOriginalRatio;
window.toggleSystemLog = toggleSystemLog;
window.toggleNegativePrompt = toggleNegativePrompt;
window.toggleI2TChat = toggleI2TChat;
window.exitButtonClicked = exitButtonClicked;

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', async function () {
    console.log('sdcpp-studio initializing...');

    initUpload();
    initLora();
    initGenerate();
    initHistory();
    initRatioLogic(); // Initialize ratio listeners
    await initApp();

    // Initialize I2T slider value displays
    const sliderTopKI2t = document.getElementById('topk-i2t');
    const valTopKI2t = document.getElementById('val-topk-i2t');
    const sliderTempI2t = document.getElementById('temp-i2t');
    const valTempI2t = document.getElementById('val-temp-i2t');
    const sliderStrengthI2t = document.getElementById('strength-i2t');
    const valStrengthI2t = document.getElementById('val-strength-i2t');
    const sliderMixI2t = document.getElementById('mix-ratio-i2t');
    const valMixI2t = document.getElementById('val-mix-i2t');
    const sliderCtxI2t = document.getElementById('ctx-i2t');
    const valCtxI2t = document.getElementById('val-ctx-i2t');
    
    if (sliderTopKI2t && valTopKI2t) {
        sliderTopKI2t.addEventListener('input', () => valTopKI2t.innerText = sliderTopKI2t.value);
    }
    if (sliderTempI2t && valTempI2t) {
        sliderTempI2t.addEventListener('input', () => valTempI2t.innerText = sliderTempI2t.value);
    }
    if (sliderStrengthI2t && valStrengthI2t) {
        sliderStrengthI2t.addEventListener('input', () => valStrengthI2t.innerText = sliderStrengthI2t.value);
    }
    if (sliderMixI2t && valMixI2t) {
        sliderMixI2t.addEventListener('input', () => valMixI2t.innerText = sliderMixI2t.value);
    }
    if (sliderCtxI2t && valCtxI2t) {
        sliderCtxI2t.addEventListener('input', () => valCtxI2t.innerText = sliderCtxI2t.value);
    }
    const sliderMaxImgI2t = document.getElementById('maximg-i2t');
    const valMaxImgI2t = document.getElementById('val-maximg-i2t');
    if (sliderMaxImgI2t && valMaxImgI2t) {
        sliderMaxImgI2t.addEventListener('input', () => valMaxImgI2t.innerText = sliderMaxImgI2t.value + 'px');
    }
    const sliderMaxImgI2i = document.getElementById('maximg-i2i');
    const valMaxImgI2i = document.getElementById('val-maximg-i2i');
    if (sliderMaxImgI2i && valMaxImgI2i) {
        sliderMaxImgI2i.addEventListener('input', () => valMaxImgI2i.innerText = sliderMaxImgI2i.value + 'px');
    }

    console.log('sdcpp-studio ready!');
});
