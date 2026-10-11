// mask_editor.js - inpainting mask painter (contextual to the I2I "Mask" slot).
// =============================================================================
// Opens a full-screen editor with the Init image (slot 1) as background; the user
// paints in red the areas to MODIFY. On save it produces a black/white PNG
// (white = modify, black = preserve) and assigns it to the slot with role 🎭 Mask.
// sd.cpp consumes it with --mask for inpaint-capable models (FLUX.1 / FLUX.2 / FLUX-FILL).
// =============================================================================
(function () {
    'use strict';

    const OVERLAY = '#ff4040';          // SOLID color: transparency comes from the canvas CSS
    const ALPHA_THRESHOLD = 20;

    let overlayEl = null;
    let bgCanvas = null, maskCanvas = null, bgCtx = null, maskCtx = null;
    let scale = 1, naturalW = 0, naturalH = 0;
    let targetSlotId = null;
    let drawing = false, lastX = null, lastY = null;
    let brushSize = 40;
    let erase = false;
    let overlayOpacity = 0.45;
    let dilatePx = 10;      // dilation (growth) around the object
    let featherPx = 6;      // edge feathering
    let previewCanvas = null;
    let lassoCanvas = null, lassoCtx = null;
    let lassoPoints = [];
    let tool = 'brush';     // brush | eraser | lasso
    let _endHandler = null, _escHandler = null;

    function injectStyles() {
        if (document.getElementById('mask-editor-styles')) return;
        const st = document.createElement('style');
        st.id = 'mask-editor-styles';
        st.textContent = `
#mask-editor-overlay{position:fixed;inset:0;z-index:10050;background:rgba(0,0,0,.85);
  display:flex;align-items:center;justify-content:center;padding:14px;font-family:inherit;}
#mask-editor-overlay .me-box{background:#1b1b1b;border:1px solid #444;border-radius:10px;
  padding:12px;display:flex;flex-direction:column;gap:10px;max-width:96vw;max-height:96vh;}
#mask-editor-overlay .me-title{font-weight:bold;color:#eee;font-size:14px;}
#mask-editor-overlay .me-toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
#mask-editor-overlay .me-toolbar button{border:1px solid #555;background:#2a2a2a;color:#ddd;
  border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer;}
#mask-editor-overlay .me-toolbar button.active{background:#3a5f8f;border-color:#4a90d9;color:#fff;}
#mask-editor-overlay .me-toolbar input[type=range]{width:130px;vertical-align:middle;}
#mask-editor-overlay .me-stage{position:relative;overflow:auto;border:1px solid #444;background:#000;
  border-radius:6px;touch-action:none;}
#mask-editor-overlay .me-canvaswrap{position:relative;}
#mask-editor-overlay canvas{position:absolute;top:0;left:0;display:block;}
#mask-editor-overlay canvas.me-bg{position:relative;}
#mask-editor-overlay .me-hint{font-size:11px;color:#999;}
#mask-editor-overlay .me-actions{display:flex;justify-content:space-between;gap:10px;}
#mask-editor-overlay .me-actions button{border:none;border-radius:6px;padding:8px 14px;
  font-size:13px;font-weight:600;cursor:pointer;}
#mask-editor-overlay .me-save{background:#2e7d32;color:#fff;}
#mask-editor-overlay .me-cancel{background:#444;color:#eee;}
`;
        document.head.appendChild(st);
    }

    function closeEditor() {
        if (_endHandler) { window.removeEventListener('mouseup', _endHandler); _endHandler = null; }
        if (_escHandler) { document.removeEventListener('keydown', _escHandler); _escHandler = null; }
        if (overlayEl && overlayEl.parentNode) overlayEl.parentNode.removeChild(overlayEl);
        overlayEl = null; bgCanvas = maskCanvas = bgCtx = maskCtx = null;
    }

    function toUrl(path, preview) {
        if (preview) return preview;
        const s = String(path || '');
        if (s.startsWith('/outputs/') || s.startsWith('/uploads/') || s.startsWith('http') || s.startsWith('data:')) return s;
        return '/outputs/' + s.split('/').pop();
    }

    function pos(evt) {
        const r = maskCanvas.getBoundingClientRect();
        const cx = evt.touches ? evt.touches[0].clientX : evt.clientX;
        const cy = evt.touches ? evt.touches[0].clientY : evt.clientY;
        return {
            x: (cx - r.left) * (maskCanvas.width / r.width),
            y: (cy - r.top) * (maskCanvas.height / r.height)
        };
    }

    function stroke(x0, y0, x1, y1) {
        maskCtx.save();
        maskCtx.lineCap = 'round';
        maskCtx.lineJoin = 'round';
        maskCtx.lineWidth = brushSize;
        if (erase) {
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.strokeStyle = 'rgba(0,0,0,1)';
            maskCtx.fillStyle = 'rgba(0,0,0,1)';
        } else {
            maskCtx.globalCompositeOperation = 'source-over';
            maskCtx.strokeStyle = OVERLAY;
            maskCtx.fillStyle = OVERLAY;
        }
        maskCtx.beginPath();
        maskCtx.arc(x1, y1, brushSize / 2, 0, Math.PI * 2);
        maskCtx.fill();
        if (x0 != null) {
            maskCtx.beginPath();
            maskCtx.moveTo(x0, y0);
            maskCtx.lineTo(x1, y1);
            maskCtx.stroke();
        }
        maskCtx.restore();
    }

    function clearMask() {
        maskCtx.clearRect(0, 0, naturalW, naturalH);
    }

    function invertMask() {
        const img = maskCtx.getImageData(0, 0, naturalW, naturalH);
        const d = img.data;
        for (let i = 0; i < d.length; i += 4) {
            if (d[i + 3] > ALPHA_THRESHOLD) {           // was painted -> clear
                d[i] = 0; d[i + 1] = 0; d[i + 2] = 0; d[i + 3] = 0;
            } else {                                    // was empty -> paint
                d[i] = 255; d[i + 1] = 64; d[i + 2] = 64; d[i + 3] = 255;
            }
        }
        maskCtx.putImageData(img, 0, 0);
    }

    // Live lasso outline (yellow dashed line)
    function drawLassoPreview() {
        if (!lassoCtx) return;
        lassoCtx.clearRect(0, 0, naturalW, naturalH);
        if (lassoPoints.length < 2) return;
        lassoCtx.save();
        lassoCtx.strokeStyle = '#ffd54a';
        lassoCtx.lineWidth = Math.max(2, brushSize / 6);
        lassoCtx.setLineDash([8, 6]);
        lassoCtx.beginPath();
        lassoCtx.moveTo(lassoPoints[0][0], lassoPoints[0][1]);
        for (let i = 1; i < lassoPoints.length; i++) lassoCtx.lineTo(lassoPoints[i][0], lassoPoints[i][1]);
        lassoCtx.closePath();
        lassoCtx.stroke();
        lassoCtx.restore();
    }

    // Fill the lasso polygon into the mask
    function fillLasso(pts) {
        if (!pts || pts.length < 3) return;
        maskCtx.save();
        maskCtx.globalCompositeOperation = 'source-over';
        maskCtx.fillStyle = OVERLAY;
        maskCtx.beginPath();
        maskCtx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) maskCtx.lineTo(pts[i][0], pts[i][1]);
        maskCtx.closePath();
        maskCtx.fill();
        maskCtx.restore();
    }

    // Mask shape: opaque white on a TRANSPARENT background, already
    // dilated (grown) and feathered (soft edges).
    function buildMaskShape() {
        // 1. hard mask: white where painted, transparent elsewhere
        const hard = document.createElement('canvas');
        hard.width = naturalW; hard.height = naturalH;
        const hctx = hard.getContext('2d');
        const src = maskCtx.getImageData(0, 0, naturalW, naturalH).data;
        const hd = hctx.createImageData(naturalW, naturalH);
        for (let i = 0; i < src.length; i += 4) {
            const on = src[i + 3] > ALPHA_THRESHOLD;
            hd.data[i] = 255; hd.data[i + 1] = 255; hd.data[i + 2] = 255;
            hd.data[i + 3] = on ? 255 : 0;
        }
        hctx.putImageData(hd, 0, 0);

        // 2. dilation: union of offset copies in a ring -> grows ~dilatePx
        const grown = document.createElement('canvas');
        grown.width = naturalW; grown.height = naturalH;
        const gctx = grown.getContext('2d');
        if (dilatePx > 0) {
            const steps = 36;
            for (let s = 0; s < steps; s++) {
                const a = (s / steps) * Math.PI * 2;
                gctx.drawImage(hard, Math.round(Math.cos(a) * dilatePx), Math.round(Math.sin(a) * dilatePx));
            }
        }
        gctx.drawImage(hard, 0, 0);

        // 3. edge feathering
        const shape = document.createElement('canvas');
        shape.width = naturalW; shape.height = naturalH;
        const sctx = shape.getContext('2d');
        if (featherPx > 0) sctx.filter = 'blur(' + featherPx + 'px)';
        sctx.drawImage(grown, 0, 0);
        sctx.filter = 'none';
        return shape;
    }

    // Final mask saved: white/gray on black background.
    function buildFinalMaskCanvas() {
        const shape = buildMaskShape();
        const out = document.createElement('canvas');
        out.width = naturalW; out.height = naturalH;
        const octx = out.getContext('2d');
        octx.fillStyle = '#000';
        octx.fillRect(0, 0, naturalW, naturalH);
        octx.drawImage(shape, 0, 0);
        return out;
    }

    // Live preview of the dilated/feathered area (gold) over the paint (red).
    function updatePreview() {
        if (!previewCanvas) return;
        const pctx = previewCanvas.getContext('2d');
        pctx.clearRect(0, 0, naturalW, naturalH);
        if (dilatePx <= 0 && featherPx <= 0) { previewCanvas.style.opacity = '0'; return; }
        pctx.drawImage(buildMaskShape(), 0, 0);
        pctx.globalCompositeOperation = 'source-in';
        pctx.fillStyle = 'rgba(255,196,0,0.9)';
        pctx.fillRect(0, 0, naturalW, naturalH);
        pctx.globalCompositeOperation = 'source-over';
        previewCanvas.style.opacity = '0.55';
    }

    function saveMask() {
        const out = buildFinalMaskCanvas();

        const fname = 'mask_' + (targetSlotId || 0) + '_' + Date.now() + '.png';
        const previewData = out.toDataURL('image/png');
        out.toBlob(async (blob) => {
            if (!blob) { alert('Could not generate the mask.'); return; }
            const file = new File([blob], fname, { type: 'image/png' });
            const fd = new FormData();
            fd.append('file', file);
            try {
                const res = await fetch('/api/models/upload', { method: 'POST', body: fd });
                const data = await res.json();
                if (!data.path) throw new Error(data.error || 'upload failed');
                const slot = window.getSlot ? window.getSlot(targetSlotId) : null;
                if (slot) {
                    slot.path = data.path;
                    slot.file = file;
                    slot.filename = fname;
                    slot.previewDataUrl = data.url || previewData;
                    if (typeof window.renderSlots === 'function') window.renderSlots();
                }
                closeEditor();
            } catch (e) {
                alert('Mask save failed: ' + e.message);
            }
        }, 'image/png');
    }

    function buildUI(img) {
        injectStyles();
        naturalW = img.naturalWidth || img.width;
        naturalH = img.naturalHeight || img.height;
        if (!naturalW || !naturalH) { alert('Invalid Init image.'); return; }

        const maxW = Math.min(window.innerWidth * 0.9, 1100);
        const maxH = window.innerHeight * 0.68;
        scale = Math.min(maxW / naturalW, maxH / naturalH, 1);
        const dispW = Math.round(naturalW * scale);
        const dispH = Math.round(naturalH * scale);

        overlayEl = document.createElement('div');
        overlayEl.id = 'mask-editor-overlay';
        overlayEl.innerHTML = `
          <div class="me-box">
            <div class="me-title">✏️ Mask editor — paint the areas to MODIFY</div>
            <div class="me-toolbar">
              <button id="me-brush" class="active" type="button">🖌 Brush</button>
              <button id="me-eraser" type="button">🩹 Eraser</button>
              <button id="me-lasso" type="button">🪢 Lasso (fill)</button>
              <label class="me-hint">Size <input id="me-size" type="range" min="4" max="200" value="40"></label>
              <label class="me-hint">Opacity <input id="me-opacity" type="range" min="10" max="90" value="45"></label>
              <label class="me-hint">Dilation <input id="me-dilate" type="range" min="0" max="80" value="10"></label>
              <label class="me-hint">Feather <input id="me-feather" type="range" min="0" max="60" value="6"></label>
              <button id="me-invert" type="button">◐ Invert</button>
              <button id="me-clear" type="button">✕ Clear</button>
            </div>
            <div class="me-stage">
              <div class="me-canvaswrap" style="width:${dispW}px;height:${dispH}px;">
                <canvas class="me-bg" width="${naturalW}" height="${naturalH}" style="width:${dispW}px;height:${dispH}px;"></canvas>
                <canvas class="me-mask" width="${naturalW}" height="${naturalH}" style="width:${dispW}px;height:${dispH}px;"></canvas>
                <canvas class="me-preview" width="${naturalW}" height="${naturalH}" style="width:${dispW}px;height:${dispH}px;pointer-events:none;opacity:0;"></canvas>
                <canvas class="me-lasso" width="${naturalW}" height="${naturalH}" style="width:${dispW}px;height:${dispH}px;pointer-events:none;"></canvas>
              </div>
            </div>
            <div class="me-hint">Red = area to modify · <b style="color:#ffc400">gold = dilation</b> (growth around the object) · black = area to preserve.<br>🖌 Brush/Eraser draw · <b>🪢 Lasso</b>: draw an outline and release to fill it. Mask ${naturalW}×${naturalH} px. In the prompt describe the <b>result</b> for the masked area (e.g. "bright red leather pants"), not the mask. For FLUX.1-dev: Denoise Strength ~0.7-0.85 and Guidance ~3.5.</div>
            <div class="me-actions">
              <button class="me-cancel" id="me-cancel" type="button">Cancel</button>
              <button class="me-save" id="me-save" type="button">💾 Save mask</button>
            </div>
          </div>`;
        document.body.appendChild(overlayEl);

        bgCanvas = overlayEl.querySelector('canvas.me-bg');
        maskCanvas = overlayEl.querySelector('canvas.me-mask');
        previewCanvas = overlayEl.querySelector('canvas.me-preview');
        lassoCanvas = overlayEl.querySelector('canvas.me-lasso');
        bgCtx = bgCanvas.getContext('2d');
        maskCtx = maskCanvas.getContext('2d');
        lassoCtx = lassoCanvas.getContext('2d');
        maskCanvas.style.opacity = overlayOpacity;
        bgCtx.drawImage(img, 0, 0, naturalW, naturalH);

        const setTool = (t) => {
            tool = t;
            erase = (t === 'eraser');
            overlayEl.querySelector('#me-brush').classList.toggle('active', t === 'brush');
            overlayEl.querySelector('#me-eraser').classList.toggle('active', t === 'eraser');
            overlayEl.querySelector('#me-lasso').classList.toggle('active', t === 'lasso');
            if (lassoCtx && t !== 'lasso') { lassoCtx.clearRect(0, 0, naturalW, naturalH); lassoPoints = []; }
        };
        overlayEl.querySelector('#me-brush').onclick = () => setTool('brush');
        overlayEl.querySelector('#me-eraser').onclick = () => setTool('eraser');
        overlayEl.querySelector('#me-lasso').onclick = () => setTool('lasso');
        overlayEl.querySelector('#me-size').oninput = (e) => { brushSize = parseInt(e.target.value, 10) || 40; };
        overlayEl.querySelector('#me-opacity').oninput = (e) => {
            overlayOpacity = Math.max(0.05, Math.min(0.95, (parseInt(e.target.value, 10) || 45) / 100));
            maskCanvas.style.opacity = overlayOpacity;
        };
        overlayEl.querySelector('#me-dilate').oninput = (e) => {
            dilatePx = parseInt(e.target.value, 10) || 0;
            updatePreview();
        };
        overlayEl.querySelector('#me-feather').oninput = (e) => {
            featherPx = parseInt(e.target.value, 10) || 0;
            updatePreview();
        };
        overlayEl.querySelector('#me-invert').onclick = () => { invertMask(); updatePreview(); };
        overlayEl.querySelector('#me-clear').onclick = () => {
            clearMask();
            if (previewCanvas) previewCanvas.getContext('2d').clearRect(0, 0, naturalW, naturalH);
            if (previewCanvas) previewCanvas.style.opacity = '0';
        };
        overlayEl.querySelector('#me-cancel').onclick = closeEditor;
        overlayEl.querySelector('#me-save').onclick = saveMask;

        const start = (e) => {
            e.preventDefault();
            drawing = true;
            const p = pos(e);
            if (tool === 'lasso') {
                lassoPoints = [[p.x, p.y]];
                drawLassoPreview();
                return;
            }
            stroke(null, null, p.x, p.y);
            lastX = p.x; lastY = p.y;
        };
        const move = (e) => {
            if (!drawing) return;
            e.preventDefault();
            const p = pos(e);
            if (tool === 'lasso') {
                lassoPoints.push([p.x, p.y]);
                drawLassoPreview();
                return;
            }
            stroke(lastX, lastY, p.x, p.y);
            lastX = p.x; lastY = p.y;
        };
        _endHandler = () => {
            if (!drawing) return;
            if (tool === 'lasso') {
                fillLasso(lassoPoints);
                lassoPoints = [];
                if (lassoCtx) lassoCtx.clearRect(0, 0, naturalW, naturalH);
            }
            drawing = false; lastX = lastY = null;
            updatePreview();
        };
        _escHandler = (e) => { if (e.key === 'Escape') closeEditor(); };

        maskCanvas.addEventListener('mousedown', start);
        maskCanvas.addEventListener('mousemove', move);
        window.addEventListener('mouseup', _endHandler);
        maskCanvas.addEventListener('touchstart', start, { passive: false });
        maskCanvas.addEventListener('touchmove', move, { passive: false });
        maskCanvas.addEventListener('touchend', _endHandler);
        document.addEventListener('keydown', _escHandler);
    }

    window.openMaskEditor = function (slotId) {
        targetSlotId = slotId;
        const refs = (typeof window.getImageSlots === 'function') ? window.getImageSlots() : [];
        const initRef = refs.find(s => s.role === 'init') || refs[0];
        const initSlot = initRef && window.getSlot ? window.getSlot(initRef.id) : null;
        if (!initSlot || !initSlot.path) {
            alert("Load an Init image (slot 1) first to draw the mask.");
            return;
        }
        const url = toUrl(initSlot.path, initSlot.previewDataUrl);
        const img = new Image();
        img.onload = () => buildUI(img);
        img.onerror = () => alert("Could not load the Init image.");
        img.src = url;
    };
})();
