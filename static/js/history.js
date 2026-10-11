// history.js - History management
// ============================

let historyPage = 0;
let historyFilter = 'all';
let historyIconSize = 128; // 64 or 128
let itemsPerPage = 10; // Default, will be calculated
let resizeTimeout;
let autoCloseTimer = null;
let historyRefreshState = 'idle'; // 'idle', 'loading', 'loaded'
let historyTotalCount = 0;

// Batch selection state: Map filename -> full item object
let batchSelectedMap = new Map();
let batchSelectionMode = false;

function enterBatchSelectionMode() {
    batchSelectionMode = true;
    batchSelectedMap.clear();
    updateBatchToolbar();

    const toolbar = document.getElementById('history-toolbar');
    const batchToolbar = document.getElementById('history-batch-toolbar');
    if (toolbar) toolbar.classList.add('hidden');
    if (batchToolbar) batchToolbar.classList.remove('hidden');

    // Update all existing history items
    document.querySelectorAll('.history-item').forEach(item => {
        item.classList.remove('batch-selected');
        const saveBtn = item.querySelector('[data-batch-btn]');
        if (saveBtn) {
            saveBtn.innerText = '⬇️';
            saveBtn.title = 'Select for batch download';
        }
    });
}

function cancelBatchSelection() {
    batchSelectionMode = false;
    batchSelectedMap.clear();

    const toolbar = document.getElementById('history-toolbar');
    const batchToolbar = document.getElementById('history-batch-toolbar');
    if (toolbar) toolbar.classList.remove('hidden');
    if (batchToolbar) batchToolbar.classList.add('hidden');

    document.querySelectorAll('.history-item').forEach(item => {
        item.classList.remove('batch-selected');
    });
}

function toggleItemSelection(item, div, btn) {
    const filename = item.filename || (item.image ? item.image.split('/').pop() : '');
    if (!filename) return;

    if (batchSelectedMap.has(filename)) {
        batchSelectedMap.delete(filename);
        div.classList.remove('batch-selected');
        if (btn) { btn.innerText = '⬇️'; btn.title = 'Select for batch download'; }
    } else {
        batchSelectedMap.set(filename, item);
        div.classList.add('batch-selected');
        if (btn) { btn.innerText = '☑️'; btn.title = 'Selected (click to deselect)'; }
    }
    updateBatchToolbar();
}

function updateBatchToolbar() {
    const countEl = document.getElementById('batch-count');
    if (countEl) {
        countEl.innerText = String(batchSelectedMap.size);
    }
}

function showBatchConfirmDialog(items, onConfirm) {
    // Remove existing dialog
    const old = document.getElementById('batch-confirm-dialog');
    if (old) old.remove();

    const overlay = document.createElement('div');
    overlay.id = 'batch-confirm-dialog';
    overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); z-index:9999; display:flex; align-items:center; justify-content:center;';

    const box = document.createElement('div');
    box.style.cssText = 'background:#1a1a2e; border:1px solid #444; border-radius:8px; padding:20px; max-width:360px; width:90%; box-shadow:0 4px 20px rgba(0,0,0,0.5);';

    const count = items.length;
    const title = document.createElement('div');
    title.innerText = 'Download ' + count + ' item' + (count > 1 ? 's' : '') + '?';
    title.style.cssText = 'font-size:16px; font-weight:bold; color:#fff; margin-bottom:12px;';

    const msg = document.createElement('div');
    msg.innerText = 'Each item includes: image + JSON metadata + source images (if any).';
    msg.style.cssText = 'font-size:12px; color:#aaa; margin-bottom:16px; line-height:1.4;';

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex; gap:8px; flex-direction:column;';

    // Download only
    const btnDl = document.createElement('button');
    btnDl.innerText = '💾 Download only';
    btnDl.style.cssText = 'background:#1a5a1a; color:#4ade80; border:none; border-radius:6px; padding:10px; cursor:pointer; font-size:13px; font-weight:bold;';
    btnDl.onclick = function() {
        overlay.remove();
        onConfirm('download');
    };

    // Download + Delete
    const btnDlDel = document.createElement('button');
    btnDlDel.innerText = '💾🗑️ Download & Delete from server';
    btnDlDel.style.cssText = 'background:#5a3a1a; color:#fbbf24; border:none; border-radius:6px; padding:10px; cursor:pointer; font-size:13px; font-weight:bold;';
    btnDlDel.onclick = function() {
        overlay.remove();
        onConfirm('download-delete');
    };

    // Cancel
    const btnCancel = document.createElement('button');
    btnCancel.innerText = '❌ Cancel';
    btnCancel.style.cssText = 'background:#3a1a1a; color:#ff8a8a; border:none; border-radius:6px; padding:10px; cursor:pointer; font-size:13px;';
    btnCancel.onclick = function() {
        overlay.remove();
        onConfirm('cancel');
    };

    btnRow.appendChild(btnDl);
    btnRow.appendChild(btnDlDel);
    btnRow.appendChild(btnCancel);

    box.appendChild(title);
    box.appendChild(msg);
    box.appendChild(btnRow);
    overlay.appendChild(box);
    document.body.appendChild(overlay);

    // Close on click outside
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) {
            overlay.remove();
            onConfirm('cancel');
        }
    });
}

async function batchDownloadSelected() {
    if (batchSelectedMap.size === 0) {
        alert('No items selected. Click ⬇️ on images to select them first.');
        return;
    }

    const items = Array.from(batchSelectedMap.values());

    showBatchConfirmDialog(items, async function(action) {
        if (action === 'cancel') return;

        const deleteAfter = (action === 'download-delete');

        // Show progress
        const countEl = document.getElementById('batch-count');
        let completed = 0;
        if (countEl) countEl.innerText = '⏳' + completed + '/' + items.length;

        for (const item of items) {
            try {
                await window.saveHistoryItemToLocal(item, null);
                completed++;
                if (countEl) countEl.innerText = '⏳' + completed + '/' + items.length;

                // Delete from server if requested
                if (deleteAfter) {
                    const fn = item.filename || (item.image ? item.image.split('/').pop() : '');
                    if (fn) {
                        try {
                            await fetch('/api/history/delete', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ filename: fn })
                            });
                            console.log('[Batch] Deleted from server:', fn);
                        } catch (delErr) {
                            console.warn('[Batch] Delete failed for', fn, delErr);
                        }
                    }
                }
            } catch (e) {
                console.error('[Batch] Failed for', item.filename || item.image, e);
            }
        }

        if (countEl) countEl.innerText = '✅' + completed + '/' + items.length;

        // Refresh history if we deleted items
        if (deleteAfter && window.loadHistory) {
            await window.loadHistory();
        }

        setTimeout(() => {
            cancelBatchSelection();
        }, 2000);
    });
}

function calculateItemsPerPage() {
    const listContainer = document.getElementById('history-list');
    if (!listContainer) return 10;

    // Available height: Window height - Header (50) - Status Bar (36) - System Log (if open? assumes collapsed or overlay) - Padding
    // Better: Measure the container's parent height.
    const panel = document.getElementById('history-panel');
    if (!panel) return 10;

    // The history list takes remaining space. 
    // Header height ~ 45px (history-header)
    // Footer controls ~ 40px (controlsDiv)
    // We can just measure history-list's clientHeight if it's set to flex-grow

    // However, if we populate it, it might grow.
    // Let's use the window height or the panel parent height.
    // .panel-center-container height is 100% of main-layout.

    // Approx calculation strategy:
    const footerHeight = 40; // Approx height for footer controls
    const availableHeight = panel.clientHeight - 50 - footerHeight; // - Header - Footer
    const itemHeight = historyIconSize + 12; // Size + Gap/Border (approx 10-12px)

    const count = Math.floor(availableHeight / itemHeight);
    // Reduce by 1 to ensure the last item is never cut off
    return Math.max(1, count - 1);
}

function updatePagination() {
    const newCount = calculateItemsPerPage();
    if (newCount !== itemsPerPage) {
        itemsPerPage = newCount;
        historyPage = 0; // Reset to page 0 on resize to avoid out of bounds
        loadHistory();
    }
}

window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(updatePagination, 200);
});

// ============================================================================
// TAGGING SYSTEM - Model selector + Progress Popup
// Reuses loadI2TModels() from models.js for consistent model selection
// ============================================================================

let taggingDialog = null;
let taggingProgressPopup = null;
let taggingAbortController = null;
let taggingActive = false;
let taggingSelectedModel = '';  // empty = auto/default

async function initHistory() {
    // I2T models are loaded by models.js via loadI2TModels()
    // We just ensure the i2t-model-select is available for our dialog
}

/**
 * Show the Tagging dialog with model selector (reuses IA tab's model list)
 */
function showTaggingDialog() {
    // Remove existing dialog
    if (taggingDialog) {
        taggingDialog.remove();
        taggingDialog = null;
    }

    taggingDialog = document.createElement('div');
    taggingDialog.id = 'tagging-dialog';
    taggingDialog.className = 'tagging-dialog-overlay';

    taggingDialog.innerHTML = `
        <div class="tagging-dialog-box" id="tagging-dialog-box">
            <div class="tagging-dialog-header">
                <span style="font-size:14px; font-weight:bold;">🏷️ Tagging Images</span>
                <button class="tagging-dialog-close" onclick="hideTaggingDialog()">&times;</button>
            </div>
            <div class="tagging-dialog-body">
                <div style="margin-bottom:12px;">
                    <label style="font-size:11px; color:#aaa; display:block; margin-bottom:4px;">I2T model (same selector as the IA tab):</label>
                    <div style="display:flex; gap:6px; align-items:center;">
                        <select id="tagging-model-select" class="input-text" style="flex:1; font-size:11px;">
                            <option value="">Auto (use default model)</option>
                        </select>
                        <button class="btn-xs" onclick="refreshTaggingModels()" title="Refresh model list">🔄</button>
                    </div>
                    <div id="tagging-model-status" style="font-size:10px; color:#888; margin-top:3px;"></div>
                </div>
                <div style="background:#1a1a2a; border-radius:6px; padding:10px; margin-bottom:12px;">
                    <div style="font-size:11px; color:#888; margin-bottom:4px;">ℹ️ Info</div>
                    <div style="font-size:10px; color:#666; line-height:1.5;">
                        All images in the outputs folder that do not yet have semantic tags are analyzed.
                        The selected model is used to generate automatic descriptions.
                    </div>
                </div>
            </div>
            <div class="tagging-dialog-footer">
                <button class="btn-xs" onclick="hideTaggingDialog()" style="background:#555;">Cancel</button>
                <button class="btn-xs" id="tagging-start-btn" onclick="startTagging()" style="background:#1a5a1a; font-weight:bold;">▶️ Start Tagging</button>
            </div>
        </div>
    `;

    document.body.appendChild(taggingDialog);
    makeDraggable(document.getElementById('tagging-dialog-box'));

    // Load models into the select
    populateTaggingModelSelect();

    // Click outside to close
    taggingDialog.addEventListener('click', (e) => {
        if (e.target === taggingDialog) hideTaggingDialog();
    });
}

function hideTaggingDialog() {
    if (taggingDialog) {
        taggingDialog.remove();
        taggingDialog = null;
    }
}

/**
 * Populate the tagging model select using the same data source as the IA tab.
 * Reuses the i2t-model-select from the IA panel if already loaded, or fetches fresh.
 */
function populateTaggingModelSelect() {
    const select = document.getElementById('tagging-model-select');
    if (!select) return;

    // Try to reuse the IA tab's model select options
    const i2tSelect = document.getElementById('i2t-model-select');

    if (i2tSelect && i2tSelect.options.length > 1) {
        // IA models already loaded - copy them
        select.innerHTML = '';
        for (let i = 0; i < i2tSelect.options.length; i++) {
            const opt = document.getElementById('i2t-model-select').options[i];
            const newOpt = document.createElement('option');
            newOpt.value = opt.value;
            newOpt.textContent = opt.textContent;
            select.appendChild(newOpt);
        }
        const status = document.getElementById('tagging-model-status');
        if (status) status.textContent = `${select.options.length - 1} models available`;
    } else {
        // Fetch from API (same endpoint as IA tab)
        fetch('/api/config/i2t-models')
            .then(res => res.json())
            .then(data => {
                select.innerHTML = '';
                const defaultOpt = document.createElement('option');
                defaultOpt.value = '';
                defaultOpt.textContent = 'Auto (use default model)';
                select.appendChild(defaultOpt);

                if (data.models && data.models.length > 0) {
                    data.models.forEach(model => {
                        const opt = document.createElement('option');
                        opt.value = model.path;
                        opt.textContent = model.name;
                        select.appendChild(opt);
                    });
                }
                const status = document.getElementById('tagging-model-status');
                if (status) status.textContent = `${data.models ? data.models.length : 0} models available`;
            })
            .catch(err => {
                console.error('[Tagging] Failed to load models:', err);
                const status = document.getElementById('tagging-model-status');
                if (status) {
                    status.textContent = 'Error loading models';
                    status.style.color = '#ef4444';
                }
            });
    }
}

function refreshTaggingModels() {
    // Call the global loadI2TModels from models.js, then repopulate
    if (typeof loadI2TModels === 'function') {
        loadI2TModels().then(() => {
            populateTaggingModelSelect();
        });
    } else {
        // Fallback: fetch directly
        fetch('/api/config/i2t-models')
            .then(res => res.json())
            .then(data => {
                const select = document.getElementById('tagging-model-select');
                if (!select) return;
                const currentVal = select.value;
                select.innerHTML = '';
                const defaultOpt = document.createElement('option');
                defaultOpt.value = '';
                defaultOpt.textContent = 'Auto (use default model)';
                select.appendChild(defaultOpt);
                if (data.models) {
                    data.models.forEach(model => {
                        const opt = document.createElement('option');
                        opt.value = model.path;
                        opt.textContent = model.name;
                        select.appendChild(opt);
                    });
                }
                select.value = currentVal;
                const status = document.getElementById('tagging-model-status');
                if (status) status.textContent = `${data.models ? data.models.length : 0} models available (updated)`;
            });
    }
}

/**
 * Start the tagging process and show the progress popup
 */
function startTagging() {
    const select = document.getElementById('tagging-model-select');
    taggingSelectedModel = select ? select.value : '';

    // Close the dialog
    hideTaggingDialog();

    // Show the progress popup
    showTaggingProgressPopup();

    // Start the tagging process
    runTaggingProcess(taggingSelectedModel);
}

/**
 * Show the tagging progress popup (draggable, closable)
 */
function showTaggingProgressPopup() {
    if (taggingProgressPopup) {
        taggingProgressPopup.remove();
    }

    taggingProgressPopup = document.createElement('div');
    taggingProgressPopup.id = 'tagging-progress-popup';
    taggingProgressPopup.className = 'tagging-progress-popup';

    taggingProgressPopup.innerHTML = `
        <div class="tagging-progress-box" id="tagging-progress-box">
            <div class="tagging-progress-header">
                <span style="font-size:12px; font-weight:bold;">🏷️ Tagging in progress...</span>
                <button class="tagging-progress-close" onclick="closeTaggingProgressPopup()">&times;</button>
            </div>
            <div class="tagging-progress-body">
                <div class="tagging-progress-stats">
                    <div class="tagging-stat">
                        <span class="tagging-stat-label">Images:</span>
                        <span class="tagging-stat-value" id="tagging-total">--</span>
                    </div>
                    <div class="tagging-stat">
                        <span class="tagging-stat-label">Completed:</span>
                        <span class="tagging-stat-value" id="tagging-completed">0</span>
                    </div>
                    <div class="tagging-stat">
                        <span class="tagging-stat-label">Errors:</span>
                        <span class="tagging-stat-value" id="tagging-errors" style="color:#ff6666;">0</span>
                    </div>
                    <div class="tagging-stat">
                        <span class="tagging-stat-label">Avg time:</span>
                        <span class="tagging-stat-value" id="tagging-avg-time">--</span>
                    </div>
                </div>
                <div class="tagging-progress-bar-container">
                    <div class="tagging-progress-bar" id="tagging-progress-bar"></div>
                </div>
                <div class="tagging-current-image" id="tagging-current-image">
                    <span style="color:#666;">Waiting...</span>
                </div>
                <div class="tagging-current-summary" id="tagging-current-summary"></div>
                <div class="tagging-current-tags" id="tagging-current-tags"></div>
                <div class="tagging-elapsed" id="tagging-elapsed"></div>
            </div>
            <div class="tagging-progress-footer">
                <button class="btn-xs" id="tagging-stop-btn" onclick="stopAutoTag()" style="background:#5a1a1a; color:#ff6666; font-weight:bold;">⏹️ Stop Tagging</button>
            </div>
        </div>
    `;

    document.body.appendChild(taggingProgressPopup);
    makeDraggable(document.getElementById('tagging-progress-box'));
}

/**
 * Close the progress popup (tagging continues in background)
 */
function closeTaggingProgressPopup() {
    if (taggingProgressPopup) {
        taggingProgressPopup.remove();
        taggingProgressPopup = null;
    }
}


/**
 * Update the tagging progress popup with SSE data
 * Used by both runTaggingProcess and resumeTagging
 */
function updateTaggingProgress(data) {
    const progressBar = document.getElementById('tagging-progress-bar');
    const currentImage = document.getElementById('tagging-current-image');
    const currentSummary = document.getElementById('tagging-current-summary');
    const currentTags = document.getElementById('tagging-current-tags');
    const elapsedDiv = document.getElementById('tagging-elapsed');
    const totalSpan = document.getElementById('tagging-total');
    const completedSpan = document.getElementById('tagging-completed');
    const errorsSpan = document.getElementById('tagging-errors');
    const avgTimeSpan = document.getElementById('tagging-avg-time');
    const btnAuto = document.getElementById('btn-auto-tag');
    const btnStopSemantic = document.getElementById('btn-stop-tag');
    
    if (data.total !== undefined && data.total !== null) {
        if (totalSpan) totalSpan.textContent = data.total;
    }
    
    if (data.status === 'starting_server') {
        if (currentImage) currentImage.innerHTML = '<span style="color:#aaa;">⏳ ' + (data.message || 'Starting server...') + '</span>';
    } else if (data.status === 'server_ready') {
        if (currentImage) currentImage.innerHTML = '<span style="color:#4a4;">✅ ' + (data.message || 'Server ready') + '</span>';
    } else if (data.status === 'processing') {
        if (completedSpan) completedSpan.textContent = data.current;
        if (progressBar) {
            const pct = data.total > 0 ? (data.current / data.total * 100) : 0;
            progressBar.style.width = pct + '%';
        }
        if (currentImage) currentImage.innerHTML = '<span style="color:#ccc;">⏳ ' + data.current + '/' + data.total + '</span> <span style="color:#888;">' + (data.image || '') + '</span>';
        if (currentSummary) currentSummary.innerHTML = '<span style="color:#666;">Processing...</span>';
        if (currentTags) currentTags.innerHTML = '';
        if (elapsedDiv) elapsedDiv.innerHTML = '';
    } else if (data.status === 'progress') {
        if (data.last_status === 'success') {
            if (completedSpan) completedSpan.textContent = data.current;
        } else {
            if (errorsSpan) errorsSpan.textContent = (parseInt(errorsSpan.textContent) || 0) + 1;
        }
        if (progressBar) {
            const pct = data.total > 0 ? (data.current / data.total * 100) : 0;
            progressBar.style.width = pct + '%';
        }
        const icon = data.last_status === 'success' ? '✅' : '❌';
        if (currentImage) currentImage.innerHTML = '<span style="color:#ccc;">' + icon + ' ' + data.current + '/' + data.total + '</span> <span style="color:#aaa; font-size:11px;">' + (data.image || '') + '</span>';
        if (data.summary) {
            if (currentSummary) currentSummary.innerHTML = '<div style="font-size:11px; color:#bbb; margin-top:4px; line-height:1.4; max-height:60px; overflow-y:auto;">' + data.summary + '</div>';
        } else {
            if (currentSummary) currentSummary.innerHTML = '';
        }
        if (data.tags && data.tags.length > 0) {
            if (currentTags) currentTags.innerHTML = data.tags.map(t => '<span class="tag-chip">' + t + '</span>').join('');
        } else {
            if (currentTags) currentTags.innerHTML = '';
        }
        if (data.elapsed_time) {
            if (elapsedDiv) elapsedDiv.innerHTML = '<span style="color:#888; font-size:10px;">⏱️ ' + data.elapsed_time + 's per image</span>';
        }
    } else if (data.status === 'completed') {
        if (progressBar) progressBar.style.width = '100%';
        if (currentImage) currentImage.innerHTML = '<span style="color:#4f4;">✅ Tagging completed!</span>';
        if (currentSummary) currentSummary.innerHTML = '<div style="font-size:12px; color:#4f4; margin-top:4px;">Images tagged: ' + data.success + '/' + data.total + ' | Errors: ' + data.errors + '</div>';
        if (currentTags) currentTags.innerHTML = '';
        taggingActive = false;
        if (btnAuto) btnAuto.style.display = 'inline-block';
        if (btnStopSemantic) btnStopSemantic.style.display = 'none';
    } else if (data.status === 'stopped') {
        if (currentImage) currentImage.innerHTML = '<span style="color:#fa0;">⏹️ Tagging stopped</span>';
        if (currentSummary) currentSummary.innerHTML = '<div style="font-size:11px; color:#fa0; margin-top:4px;">Completed: ' + data.current + '/' + data.total + '</div>';
        if (currentSummary) currentSummary.innerHTML += '<div style="margin-top:6px;"><button class="btn-xs" onclick="resumeTagging()" style="background:#1a5a1a; font-weight:bold;">▶️ Resume Tagging</button></div>';
        taggingActive = false;
        if (btnAuto) btnAuto.style.display = 'inline-block';
        if (btnStopSemantic) btnStopSemantic.style.display = 'none';
    } else if (data.status === 'suspended') {
        if (currentImage) currentImage.innerHTML = '<span style="color:#fa0;">⏸️ Tagging suspended (IA server active)</span>';
        if (currentSummary) {
            currentSummary.innerHTML = '<div style="font-size:11px; color:#fa0; margin-top:4px;">' + (data.message || 'IA server active, tagging waiting') + '</div>';
            currentSummary.innerHTML += '<div style="margin-top:6px;"><button class="btn-xs" onclick="resumeTagging()" style="background:#1a5a1a; font-weight:bold;">▶️ Retry / Resume</button></div>';
        }
    } else if (data.status === 'error') {
        if (currentImage) currentImage.innerHTML = '<span style="color:#f44;">❌ Error</span>';
        if (currentSummary) currentSummary.innerHTML = '<div style="font-size:11px; color:#f44; margin-top:4px;">' + (data.message || 'Unknown error') + '</div>';
        taggingActive = false;
        if (btnAuto) btnAuto.style.display = 'inline-block';
        if (btnStopSemantic) btnStopSemantic.style.display = 'none';
    }
}

/**
 * Run the tagging process via SSE
 */
function runTaggingProcess(modelPath) {
    taggingActive = true;
    taggingAbortController = new AbortController();

    const model = modelPath || 'server';
    const progressBar = document.getElementById('tagging-progress-bar');
    const currentImage = document.getElementById('tagging-current-image');
    const currentSummary = document.getElementById('tagging-current-summary');
    const currentTags = document.getElementById('tagging-current-tags');
    const elapsedDiv = document.getElementById('tagging-elapsed');
    const totalSpan = document.getElementById('tagging-total');
    const completedSpan = document.getElementById('tagging-completed');
    const errorsSpan = document.getElementById('tagging-errors');
    const avgTimeSpan = document.getElementById('tagging-avg-time');
    const btnStop = document.getElementById('tagging-stop-btn');
    const btnAuto = document.getElementById('btn-auto-tag');
    const btnStopSemantic = document.getElementById('btn-stop-tag');

    let totalImages = 0;
    let completedCount = 0;
    let errorCount = 0;
    let totalElapsed = 0;

    if (btnAuto) btnAuto.style.display = 'none';
    if (btnStopSemantic) btnStopSemantic.style.display = 'block';

    fetch(`/api/tags/auto-tag-all?model=${encodeURIComponent(model)}`, {
        method: 'POST',
        signal: taggingAbortController.signal
    })
    .then(res => {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function read() {
            return reader.read().then(({done, value}) => {
                if (done) {
                    if (buffer) processLine(buffer);
                    taggingActive = false;
                    if (btnStop) btnStop.disabled = true;
                    return;
                }

                buffer += decoder.decode(value, {stream: true});
                const lines = buffer.split('\n');
                buffer = lines.pop();

                lines.forEach(processLine);
                return read();
            });
        }

        function processLine(line) {
            if (!line.startsWith('data: ')) return;
            try {
                const data = JSON.parse(line.substring(6));
                console.log('[Tagging]', data);

                if (data.total !== undefined && data.total !== null) totalImages = data.total;
                if (totalSpan) totalSpan.textContent = totalImages;

                if (data.status === 'starting_server') {
                    currentImage.innerHTML = `<span style="color:#aaa;">⏳ ${data.message || 'Starting server...'}</span>`;
                } else if (data.status === 'server_ready') {
                    currentImage.innerHTML = `<span style="color:#4a4;">✅ ${data.message || 'Server ready'}</span>`;
                } else if (data.status === 'processing') {
                    completedCount = data.current;
                    if (completedSpan) completedSpan.textContent = completedCount;
                    if (progressBar) {
                        const pct = totalImages > 0 ? (data.current / totalImages * 100) : 0;
                        progressBar.style.width = pct + '%';
                    }
                    currentImage.innerHTML = `<span style="color:#ccc;">⏳ ${data.current}/${totalImages}</span> <span style="color:#888;">${data.image || ''}</span>`;
                    currentSummary.innerHTML = '<span style="color:#666;">Processing...</span>';
                    currentTags.innerHTML = '';
                    elapsedDiv.innerHTML = '';
                } else if (data.status === 'progress') {
                    completedCount = data.current;
                    if (data.last_status === 'success') {
                        if (completedSpan) completedSpan.textContent = completedCount;
                    } else {
                        errorCount++;
                        if (errorsSpan) errorsSpan.textContent = errorCount;
                    }

                    if (progressBar) {
                        const pct = totalImages > 0 ? (data.current / totalImages * 100) : 0;
                        progressBar.style.width = pct + '%';
                    }

                    const icon = data.last_status === 'success' ? '✅' : '❌';
                    currentImage.innerHTML = `<span style="color:#ccc;">${icon} ${data.current}/${totalImages}</span> <span style="color:#aaa; font-size:11px;">${data.image || ''}</span>`;

                    // Show summary text
                    if (data.summary) {
                        currentSummary.innerHTML = `<div style="font-size:11px; color:#bbb; margin-top:4px; line-height:1.4; max-height:60px; overflow-y:auto;">${data.summary}</div>`;
                    } else {
                        currentSummary.innerHTML = '';
                    }

                    // Show tags
                    if (data.tags && data.tags.length > 0) {
                        currentTags.innerHTML = data.tags.map(t =>
                            `<span class="tag-chip">${t}</span>`
                        ).join('');
                    } else {
                        currentTags.innerHTML = '';
                    }

                    // Show elapsed time
                    if (data.elapsed_time) {
                        totalElapsed += data.elapsed_time;
                        const avg = completedCount > 0 ? (totalElapsed / completedCount).toFixed(1) : '--';
                        if (avgTimeSpan) avgTimeSpan.textContent = avg + 's';
                        elapsedDiv.innerHTML = `<span style="color:#888; font-size:10px;">⏱️ ${data.elapsed_time}s per image</span>`;
                    }
                } else if (data.status === 'completed') {
                    if (progressBar) progressBar.style.width = '100%';
                    if (completedSpan) completedSpan.textContent = data.success || completedCount;
                    currentImage.innerHTML = `<span style="color:#4f4;">✅ Tagging completed!</span>`;
                    currentSummary.innerHTML = `<div style="font-size:12px; color:#4f4; margin-top:4px;">Images tagged: ${data.success}/${data.total} | Errors: ${data.errors}</div>`;
                    currentTags.innerHTML = '';
                    taggingActive = false;
                    if (btnAuto) btnAuto.style.display = 'inline-block';
                    if (btnStopSemantic) btnStopSemantic.style.display = 'none';
                } else if (data.status === 'stopped') {
                    currentImage.innerHTML = `<span style="color:#fa0;">⏹️ Tagging stopped</span>`;
                    currentSummary.innerHTML = `<div style="font-size:11px; color:#fa0; margin-top:4px;">Completed: ${data.current}/${data.total}</div>`;
                    currentSummary.innerHTML += `<div style="margin-top:6px;"><button class="btn-xs" onclick="resumeTagging()" style="background:#1a5a1a; font-weight:bold;">▶️ Resume Tagging</button></div>`;
                    taggingActive = false;
                    if (btnAuto) btnAuto.style.display = 'inline-block';
                    if (btnStopSemantic) btnStopSemantic.style.display = 'none';
                } else if (data.status === 'suspended') {
                    currentImage.innerHTML = `<span style="color:#fa0;">⏸️ Tagging suspended (IA server active)</span>`;
                    currentSummary.innerHTML = `<div style="font-size:11px; color:#fa0; margin-top:4px;">${data.message || 'IA server active, tagging waiting'}</div>`;
                    currentSummary.innerHTML += `<div style="margin-top:6px;"><button class="btn-xs" onclick="resumeTagging()" style="background:#1a5a1a; font-weight:bold;">▶️ Retry / Resume</button></div>`;
                } else if (data.status === 'error') {
                    currentImage.innerHTML = `<span style="color:#f44;">❌ Error</span>`;
                    currentSummary.innerHTML = `<div style="font-size:11px; color:#f44; margin-top:4px;">${data.message || 'Unknown error'}</div>`;
                    taggingActive = false;
                    if (btnAuto) btnAuto.style.display = 'inline-block';
                    if (btnStopSemantic) btnStopSemantic.style.display = 'none';
                }
            } catch (e) {
                console.error('[Tagging] Parse error:', e);
            }
        }

        return read();
    })
    .catch(err => {
        if (err.name === 'AbortError') {
            console.log('[Tagging] Aborted by user');
        } else {
            console.error('[Tagging] Error:', err);
            if (currentImage) currentImage.innerHTML = `<span style="color:#f44;">❌ Error: ${err.message}</span>`;
        }
        taggingActive = false;
        if (btnAuto) btnAuto.style.display = 'inline-block';
        if (btnStopSemantic) btnStopSemantic.style.display = 'none';
    });
}

// Stop auto-tagging process
function stopAutoTag() {
    console.log('[AutoTag] Stopping...');
    
    // Abort the local fetch connection first (this stops the SSE stream immediately)
    if (taggingAbortController) {
        taggingAbortController.abort();
        taggingAbortController = null;
    }
    
    // Also notify the backend to stop the tagging loop
    fetch('/api/tags/auto-tag-stop', {
        method: 'POST'
    })
    .then(res => res.json())
    .then(data => {
        console.log('[AutoTag] Stop response:', data);
    })
    .catch(err => {
        console.error('[AutoTag] Stop error:', err);
    });
    
    // Update UI immediately. The popup stays open with a 'Resume
    // Tagging' button so the user can resume the batch from the
    // checkpoint (preserved in VramManager.checkpoint).
    taggingActive = false;
    const btnAuto = document.getElementById('btn-auto-tag');
    const btnStopSemantic = document.getElementById('btn-stop-tag');
    const currentImage = document.getElementById('tagging-current-image');
    const currentSummary = document.getElementById('tagging-current-summary');
    if (btnAuto) btnAuto.style.display = 'inline-block';
    if (btnStopSemantic) btnStopSemantic.style.display = 'none';
    if (currentImage) currentImage.innerHTML = '<span style="color:#fa0;">⏹️ Tagging stopped</span>';
    if (currentSummary) {
        currentSummary.innerHTML = '<div style="font-size:11px; color:#fa0; margin-top:4px;">⏹️ Stopped manually. You can resume from the checkpoint.</div>';
        currentSummary.innerHTML += '<div style="margin-top:6px;"><button class="btn-xs" onclick="resumeTagging()" style="background:#1a5a1a; font-weight:bold;">▶️ Resume Tagging</button></div>';
    }
}

// Draggable helper
function makeDraggable(element) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    
    element.onmousedown = dragMouseDown;
    
    function dragMouseDown(e) {
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.classList.contains('tag-model-item')) {
            return;
        }
        e.preventDefault();
        pos3 = e.clientX;
        pos4 = e.clientY;
        document.onmouseup = closeDragElement;
        document.onmousemove = elementDrag;
    }
    
    function elementDrag(e) {
        e.preventDefault();
        pos1 = pos3 - e.clientX;
        pos2 = pos4 - e.clientY;
        pos3 = e.clientX;
        pos4 = e.clientY;
        element.style.top = (element.offsetTop - pos2) + "px";
        element.style.left = (element.offsetLeft - pos1) + "px";
    }
    
    function closeDragElement() {
        document.onmouseup = null;
        document.onmousemove = null;
    }
}

async function toggleHistory() {
    const panel = document.getElementById('history-panel');
    const btn = document.getElementById('history-toggle-btn');
    if (panel.classList.contains('collapsed')) {
        panel.classList.remove('collapsed');
        if (btn) btn.classList.add('hidden');
        await loadHistory();
    } else {
        // Clear any pending auto-close timer if user manually closes
        if (autoCloseTimer) { clearTimeout(autoCloseTimer); autoCloseTimer = null; }
        panel.classList.add('collapsed');
        if (btn) btn.classList.remove('hidden');
    }
}


function createHistoryItemDOM(item, badgeHtml) {
    const div = document.createElement('div');
    div.className = 'history-item size-' + historyIconSize;

    const imageUrl = item.image || (item.filename ? '/outputs/' + item.filename : null);
    const filename = item.filename || (item.image ? item.image.split('/').pop() : 'unknown');
    
    // --- I2T Text Card ---
    // For I2T items (is_text_result), show a text preview card instead of an image thumbnail
    let thumbContent = '';
    if (item.is_text_result) {
        const textPreview = item.text_preview || 'No text content';
        // Escape HTML in preview to avoid XSS
        const escapedPreview = textPreview
            .replace(/&/g, '&')
            .replace(/</g, '<')
            .replace(/>/g, '>')
            .replace(/"/g, '"');
        
        thumbContent =
            '<div style="display:flex; flex-direction:column; width:100%; height:100%; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); overflow:hidden; box-sizing:border-box;">' +
                '<div style="display:flex; align-items:center; gap:4px; padding:4px 6px; background:rgba(0,0,0,0.3); border-bottom:1px solid #333; flex-shrink:0;">' +
                    '<span style="font-size:14px;">📝</span>' +
                    '<span style="font-size:9px; color:#8af; font-weight:bold; text-transform:uppercase;">Analysis</span>' +
                '</div>' +
                '<div style="flex:1; padding:6px; font-size:10px; color:#ccc; line-height:1.4; overflow:hidden; display:-webkit-box; -webkit-line-clamp:5; -webkit-box-orient:vertical;">' +
                    escapedPreview +
                '</div>' +
            '</div>';
    } else {
        const sourceImages = item.params?.source_images;
        const sourceImageName = (sourceImages && Array.isArray(sourceImages) && sourceImages.length > 0)
            ? sourceImages[0]
            : '';
        
        if (imageUrl && imageUrl !== '/outputs/undefined') {
            thumbContent = '<img src="' + imageUrl + '" loading="lazy" alt="' + filename + '">';
        } else if (sourceImageName) {
            thumbContent = '<div style="display:flex; align-items:center; justify-content:center; width:100%; height:100%; text-align:center; font-size:9px; color:#aaa; word-break:break-all; padding:2px; box-sizing:border-box;">' + sourceImageName + '</div>';
        } else {
            thumbContent = '<div style="display:flex; align-items:center; justify-content:center; width:100%; height:100%; text-align:center; font-size:9px; color:#666;">No Image</div>';
        }
    }
    
    const itemMode = (item.mode || item.params?.mode || "IMG").toUpperCase();
    const rawTs = item.timestamp || item.params?.timestamp || 0;
    // Ensure timestamp is a valid number
    const timestamp = (typeof rawTs === 'number' && !isNaN(rawTs)) ? rawTs : 0;
    
    if (!badgeHtml) {
        badgeHtml = '<span class="mode-badge" style="background:#444; padding:2px 5px; border-radius:3px; font-size:9px;">' + itemMode + '</span>';
    } else {
        badgeHtml = badgeHtml.replace('{mode}', itemMode);
    }
    // ARF run: it is a multi-image folder -> show count badge
    if (item.arf && item.arf_steps) {
        badgeHtml += '<span class="arf-badge" style="background:#4a3a12; color:#e6c84a; padding:1px 5px; border-radius:3px; font-size:9px; font-weight:bold; margin-left:3px; border:1px solid #6a5a1a;" title="Agent Refine run with ' + item.arf_steps.length + ' images. Click to choose the step.">ARF ×' + item.arf_steps.length + '</span>';
    }
    
    // Append IA badge if this image has image_analysis data
    if (item.has_image_analysis) {
        badgeHtml += '<span class="ia-badge" style="background:#2a4a5a; color:#8cf; padding:1px 5px; border-radius:3px; font-size:9px; font-weight:bold; margin-left:3px; border:1px solid #3a6a7a;" title="Has Image Analysis data">IA</span>';
    }
    
    const tags = item.tags || [];
    const tagLabels = tags.slice(0, 3).map(t => t.label || t.text || '').filter(Boolean);
    const tagsDisplay = tagLabels.length > 0 ? tagLabels.join(', ') : '';
    
    div.innerHTML =
        '<div class="history-thumb">' + thumbContent + '</div>' +
        '<div class="history-info" style="position:relative;">' +
        badgeHtml +
        '<span style="display:block; font-size:10px; color:#fff; font-weight:bold; margin-top:5px;">' + formatDateYYMMDD(timestamp) + '</span>' +
        (tagsDisplay ? '<div style="font-size:9px; color:#8a8; margin-top:3px; overflow:hidden; max-height:30px; line-height:1.2;">' + tagsDisplay + '</div>' : '') +
        '</div>';

    // Full dimension caption below the image (W × H px), same style as the
    // Progression rail (.ag-tile-dim).
    const histThumb = div.querySelector('.history-thumb');
    const histImg = div.querySelector('.history-thumb img');
    if (histThumb && histImg) {
        const dim = document.createElement('div');
        dim.className = 'history-dim loading';
        dim.textContent = '…';
        histThumb.appendChild(dim);
        const updateDim = () => {
            if (histImg.naturalWidth && histImg.naturalHeight) {
                dim.textContent = histImg.naturalWidth + ' × ' + histImg.naturalHeight + ' px';
                dim.classList.remove('loading');
            }
        };
        if (histImg.complete && histImg.naturalWidth) updateDim();
        else histImg.addEventListener('load', updateDim);
    }

    const actionsDiv = document.createElement('div');
    actionsDiv.style.cssText = "display: flex; flex-direction: column; gap: 6px; margin-top: 6px;";

    // Row 1: JSON + tags (only for large icon size)
    if (historyIconSize === 128) {
        const metaRow = document.createElement('div');
        metaRow.style.cssText = "display: flex; gap: 4px;";

        const btnTags = document.createElement('button');
        btnTags.innerText = "tags";
        btnTags.title = "View Semantic Tags";
        btnTags.style.cssText = "background: #2a4a2a; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 5px; font-size: 10px; line-height:1; flex:1;";
        btnTags.onclick = async function (e) {
            e.stopPropagation();
            try {
                const res = await fetch('/api/tags/image/' + encodeURIComponent(filename));
                if (res.ok) {
                    const data = await res.json();
                    if (data.status === 'success') {
                        viewJsonInModal(data.data, "Semantic Tags");
                    } else {
                        alert('Error loading tags: ' + (data.detail || 'Unknown error'));
                    }
                } else {
                    alert('No semantic tags found for ' + filename);
                }
            } catch (err) {
                alert('Error loading tags: ' + err.message);
            }
        };
        metaRow.appendChild(btnTags);

        const btnJson = document.createElement('button');
        btnJson.innerText = "JSON";
        btnJson.title = "View Metadata";
        btnJson.style.cssText = "background: #335; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 5px; font-size: 10px; line-height:1; flex:1;";
        btnJson.onclick = function (e) {
            e.stopPropagation();
            viewJsonInModal(item, "Item JSON");
        };
        metaRow.appendChild(btnJson);

        actionsDiv.appendChild(metaRow);
    }

    // Row 2: Save to local + Delete
    const delRow = document.createElement('div');
    delRow.style.cssText = "display: flex; gap: 4px;";

    const btnSave = document.createElement('button');
    btnSave.setAttribute('data-batch-btn', 'true');
    btnSave.innerText = "⬇️";
    btnSave.title = "Save to local directory (image + JSON + sources)";
    btnSave.style.cssText = "background: #1a5a8a; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 5px; font-size: 12px; line-height:1; flex:1;";
    btnSave.onclick = function (e) {
        e.stopPropagation();
        // Batch selection mode: toggle selection for this item
        if (!batchSelectionMode) {
            enterBatchSelectionMode();
        }
        toggleItemSelection(item, div, btnSave);
    };
    delRow.appendChild(btnSave);

    const btnDel = document.createElement('button');
    btnDel.innerText = "X";
    btnDel.title = "Delete this image";
    btnDel.style.cssText = "background: #500; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 5px; font-size: 12px; line-height:1; flex:1;";
    btnDel.onclick = function (e) {
        e.stopPropagation();
        deleteHistoryItem(filename);
    };
    delRow.appendChild(btnDel);

    actionsDiv.appendChild(delRow);

    actionsDiv.appendChild(delRow);

    if (historyIconSize === 128 && window.currentMode && window.currentMode !== 't2i') {
        const moveRow = document.createElement('div');
        moveRow.style.cssText = "display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 4px;";

        const btn1 = document.createElement('button');
        btn1.innerText = ">1";
        btn1.title = "Load into Input Image 1";
        btn1.style.cssText = "background: #242; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 6px; font-size: 10px; font-weight:bold;";
        btn1.onclick = function (e) {
            e.stopPropagation();
            loadToInput(imageUrl, 1);
        };
        moveRow.appendChild(btn1);

        const btn2 = document.createElement('button');
        btn2.innerText = ">2";
        btn2.title = "Load into Input Image 2";
        btn2.style.cssText = "background: #242; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 6px; font-size: 10px; font-weight:bold;";
        btn2.onclick = function (e) {
            e.stopPropagation();
            loadToInput(imageUrl, 2);
        };
        moveRow.appendChild(btn2);

        // For I2T text result items, add a "📤 Prompt" button
        // to send the analysis text to the current prompt box
        if (item.is_text_result) {
            const textPreview = item.text_preview || '';
            const btnPrompt = document.createElement('button');
            btnPrompt.innerText = "📤 Prompt";
            btnPrompt.title = "Send analysis text to prompt box";
            btnPrompt.style.cssText = "background: #1a3a5a; border: none; border-radius: 3px; cursor: pointer; color: #8cf; padding: 2px 6px; font-size: 10px; font-weight:bold; flex:1;";
            btnPrompt.onclick = function (e) {
                e.stopPropagation();
                if (window.promptBox) {
                    window.promptBox.value = textPreview;
                    window.promptBox.focus();
                }
            };
            moveRow.appendChild(btnPrompt);
        }

        actionsDiv.insertBefore(moveRow, actionsDiv.firstChild);
    }

    div.querySelector('.history-info').appendChild(actionsDiv);

    div.onclick = function () { restoreHistoryState(item); };
    return div;
}

// ============================================================================
// ARF multi-image picker (History integration)
// ----------------------------------------------------------------------------
// An Agent Refine run is a folder with MANY step images. History lists it as a
// single image item (the last generated step, visible default). Clicking it
// opens this picker: the user sees all step thumbnails + a large preview of the
// image that will actually be used. Default = last generated step + its data.
// The chosen image keeps its DIRECT path inside agent_refine_<ts>/ (no copies),
// and a JSON describing the selected step is exposed as window.agentStepJson.
// ============================================================================
let arfPickItem = null;
let arfPickStep = null;
// Preview zoom mode: 'fit' = enlarge to fill the available area (upscaling
// allowed), 'limit' = never exceed the image's real pixel size (still shrinks
// to fit when the image is larger than the area).
let arfPickZoom = 'fit';
// The picker DOM normally lives in the main document, but it can be DETACHED
// into an independent always-on-top window (Document Picture-in-Picture, with a
// normal popup as fallback). arfDoc always points to the document currently
// holding the picker, so every picker-internal DOM lookup goes through it.
let arfDoc = document;
let arfPipWin = null;

function $arf(id) { return arfDoc.getElementById(id); }
function arfQueryAll(sel) { return arfDoc.querySelectorAll(sel); }

// Apply the current zoom mode to the large preview. Called on every selection
// and on image load (naturalWidth is needed for the 'limit' cap).
function arfApplyZoom() {
    const pv = $arf('arf-picker-preview');
    const btn = $arf('arf-picker-zoom');
    if (!pv) return;
    if (arfPickZoom === 'limit' && pv.naturalWidth) {
        // cap at the real pixel size, but keep the min() so a big image still
        // shrinks down to the available area instead of overflowing.
        pv.style.width = 'auto';
        pv.style.height = 'auto';
        pv.style.maxWidth = 'min(100%, ' + pv.naturalWidth + 'px)';
        pv.style.maxHeight = 'min(100%, ' + pv.naturalHeight + 'px)';
        if (btn) { btn.innerText = '⤢ Fill area'; btn.title = 'Enlarge the image to fill the available area (allow upscaling)'; }
    } else {
        // Fill: force the image to grow/shrink to the available area, keeping
        // its aspect ratio (object-fit: contain). Without width/height:100% a
        // small image would stay at its natural size (max-* only limits).
        pv.style.width = '100%';
        pv.style.height = '100%';
        pv.style.maxWidth = 'none';
        pv.style.maxHeight = 'none';
        if (btn) { btn.innerText = '1:1 max'; btn.title = 'Limit the preview to the actual pixel size of the image'; }
    }
}
window.arfApplyZoom = arfApplyZoom;

function arfToggleZoom() {
    arfPickZoom = (arfPickZoom === 'fit') ? 'limit' : 'fit';
    arfApplyZoom();
}
window.arfToggleZoom = arfToggleZoom;

// ---- Detach / re-attach (independent always-on-top window) -----------------
function arfUpdateDetachBtn() {
    const db = $arf('arf-picker-detach');
    if (!db) return;
    if (arfPipWin) {
        db.innerText = '⇲';
        db.title = 'Re-attach the picker to the main window';
        db.classList.add('on');
    } else {
        db.innerText = '⧉';
        db.title = 'Detach the picker into an independent always-on-top window (stays open while a job runs)';
        db.classList.remove('on');
    }
}
window.arfUpdateDetachBtn = arfUpdateDetachBtn;

// Copy stylesheets into the detached document and keep relative URLs
// (/outputs/..., css/...) resolving against the main page.
function arfCopyStyles(targetDoc) {
    const base = targetDoc.createElement('base');
    base.href = document.baseURI;
    targetDoc.head.appendChild(base);
    document.querySelectorAll('link[rel="stylesheet"], style').forEach(node => {
        try { targetDoc.head.appendChild(node.cloneNode(true)); } catch (e) {}
    });
}

// Expose the inline-onclick handlers in the detached window's scope.
function arfSyncHandlers(win) {
    ['arfSelect', 'arfToggleZoom', 'arfApplyZoom', 'arfClosePicker', 'arfDeleteStep',
     'arfApplyStep', 'arfOpenInAgent', 'arfDetach', 'arfUpdateDetachBtn',
     'copyAgentPrompt'].forEach(name => {
        if (typeof window[name] === 'function') {
            try { win[name] = window[name]; } catch (e) {}
        }
    });
}

async function arfDetach() {
    if (arfPipWin) { arfAttach(); return; }
    const modal = document.getElementById('arf-picker-modal');
    if (!modal || modal.classList.contains('hidden')) return;
    let win = null;
    // 1) Document Picture-in-Picture: a real always-on-top, independent window.
    try {
        if (window.documentPictureInPicture && !window.documentPictureInPicture.window) {
            win = await window.documentPictureInPicture.requestWindow({ width: 940, height: 720 });
        }
    } catch (e) {
        console.warn('Document PiP unavailable, falling back to a popup window:', e);
        win = null;
    }
    // 2) Fallback: a normal browser popup window (still fully independent).
    if (!win) {
        win = window.open('', 'arfPickerWindow', 'popup=yes,width=940,height=720');
        if (!win) {
            alert('Popup bloccato dal browser: consenti i popup per staccare la finestra.');
            return;
        }
    }
    // Build the detached document and move the picker node into it.
    try {
        arfCopyStyles(win.document);
        win.document.title = 'ARF — step picker';
        win.document.body.style.margin = '0';
        win.document.body.style.background = '#111';
        win.document.body.appendChild(modal);
        arfDoc = win.document;
        arfPipWin = win;
        arfSyncHandlers(win);
        modal.classList.remove('hidden');
        arfUpdateDetachBtn();
    } catch (e) {
        console.error('Detach failed:', e);
        arfDoc = document;
        arfPipWin = null;
        try { win.close(); } catch (e2) {}
        return;
    }
    // If the user closes the detached window, bring the picker back / close it.
    win.addEventListener('pagehide', function () {
        if (arfPipWin !== win) return;
        try {
            const m = win.document.getElementById('arf-picker-modal');
            if (m) { m.classList.add('hidden'); document.body.appendChild(m); }
        } catch (e2) {}
        arfPipWin = null;
        arfDoc = document;
        arfPickItem = null;
        arfPickStep = null;
        arfUpdateDetachBtn();
    });
    try { win.focus(); } catch (e) {}
    // re-run the selection so the preview/zoom lays out in the new window
    if (arfPickStep) arfSelect(arfPickStep);
}
window.arfDetach = arfDetach;

// Bring the picker DOM back into the main document and close the detached win.
function arfAttach() {
    const win = arfPipWin;
    if (!win) return;
    // Move the node back BEFORE closing the window.
    try {
        const m = win.document.getElementById('arf-picker-modal');
        if (m && m.parentNode !== document.body) document.body.appendChild(m);
    } catch (e) {}
    arfPipWin = null;
    arfDoc = document;
    arfUpdateDetachBtn();
    if (win && !win.closed) { try { win.close(); } catch (e) {} }
}
window.arfAttach = arfAttach;

function arfClosePicker() {
    // If detached, re-attach first so the DOM survives and the window closes.
    if (arfPipWin) arfAttach();
    const modal = document.getElementById('arf-picker-modal');
    if (modal) modal.classList.add('hidden');
    arfPickItem = null;
    arfPickStep = null;
}
window.arfClosePicker = arfClosePicker;

function arfSelect(step) {
    if (!step) return;
    arfPickStep = step;
    // highlight the chosen thumbnail in the list
    arfQueryAll('#arf-picker-thumbs .arf-thumb').forEach(t => {
        t.classList.toggle('sel', t.dataset.url === step.image);
    });
    const pv = $arf('arf-picker-preview');
    const pb = $arf('arf-picker-prompt');
    const metaEl = $arf('arf-picker-meta');
    const dimEl = $arf('arf-picker-dims');
    if (pv) {
        pv.src = step.image + '?t=' + Date.now();
        pv.onload = () => {
            if (dimEl && pv.naturalWidth) dimEl.innerText = pv.naturalWidth + ' × ' + pv.naturalHeight + ' px';
            arfApplyZoom();
        };
        // apply immediately too (cached images may not fire onload again)
        arfApplyZoom();
    }
    const db = $arf('arf-picker-delete');
    if (db) db.disabled = !step.filename;
    if (pb) pb.innerText = step.prompt || '(no prompt for this step)';
    if (metaEl) {
        const m = [];
        if (step.resolution) m.push('resolution ' + step.resolution);
        if (step.goal_reached !== undefined && step.goal_reached !== null) m.push('GOAL_REACHED: ' + (step.goal_reached ? 'yes' : 'no'));
        if (step.combined !== undefined && step.combined !== null) m.push('combined ' + step.combined + '/10');
        else if (step.adherence !== undefined && step.adherence !== null) m.push('adherence ' + step.adherence + '/10');
        if (step.fidelity !== undefined && step.fidelity !== null) m.push('fidelity ' + step.fidelity + '/10');
        if (step.strategy) m.push('[' + (step.mode === 'reground' ? 'REGROUND' : 'EXPLORE') + ':' + step.strategy + ']');
        let loraNames = [];
        if (Array.isArray(step.loras)) {
            loraNames = step.loras.map(l => String(l.path || '').split('/').pop()).filter(Boolean);
        } else if (step.loras && Array.isArray(step.loras.files)) {
            loraNames = step.loras.files.map(f => f[0]).filter(Boolean);
        }
        if (loraNames.length) m.push('lora: ' + loraNames.join(', '));
        metaEl.innerText = m.join(' · ') || '';
    }
    const ub = $arf('arf-picker-use');
    if (ub) ub.innerText = '✔ Use this step (#' + step.iter + ')';
}
window.arfSelect = arfSelect;

function openArfPicker(item) {
    if (!item || !item.arf_steps || !item.arf_steps.length) return;
    // If the picker is detached, keep the floating window and just point it to
    // the newly selected ARF run (clicking another ARF icon in History must not
    // re-attach/close the floating window).
    arfDoc = arfPipWin ? arfPipWin.document : document;
    let modal = $arf('arf-picker-modal');
    if (!modal && arfPipWin) {
        // Safety net: the detached node is gone -> fall back to the main window.
        arfAttach();
        arfDoc = document;
        modal = $arf('arf-picker-modal');
    }
    if (!modal) return;
    arfPickItem = item;
    // In agent mode "Use this step" already loads the run into the Agent
    // panel, so the explicit "Apri in Agent Refine" button is redundant there.
    const agentBtn = $arf('arf-picker-agent');
    if (agentBtn) agentBtn.classList.toggle('hidden', window.currentMode === 'agent');
    // header shows the run folder + goal
    const h = $arf('arf-picker-title');
    if (h) {
        const goal = (item.inputs && item.inputs.goal) || (item.params && item.params.goal) || '';
        h.innerText = '🪄 ARF — ' + item.arf_dir + (goal ? ('  ·  “' + goal.slice(0, 60) + (goal.length > 60 ? '…' : '') + '”') : '');
    }
    arfRenderThumbs();
    const db = $arf('arf-picker-delete');
    if (db) db.disabled = false;
    arfUpdateDetachBtn();
    modal.classList.remove('hidden');
    // default: the LAST generated step (item.image) = the visible default
    const last = item.arf_steps[item.arf_steps.length - 1];
    if (last) arfSelect(last);
    // bring the floating window to the front when it follows a new selection
    if (arfPipWin) { try { arfPipWin.focus(); } catch (e) {} }
}
window.openArfPicker = openArfPicker;

// (Re)build the step thumbnails from the current in-memory run (arfPickItem).
function arfRenderThumbs() {
    const thumbs = $arf('arf-picker-thumbs');
    if (!thumbs || !arfPickItem) return;
    thumbs.innerHTML = '';
    arfPickItem.arf_steps.forEach(st => {
        const t = arfDoc.createElement('div');
        t.className = 'arf-thumb';
        t.dataset.url = st.image;
        const img = arfDoc.createElement('img');
        img.src = st.image;
        img.loading = 'lazy';
        const cap = arfDoc.createElement('div');
        cap.className = 'arf-thumb-cap';
        cap.innerText = '#' + st.iter + (st.resolution ? ' · ' + st.resolution : '');
        t.appendChild(img);
        t.appendChild(cap);
        t.addEventListener('click', () => arfSelect(st));
        thumbs.appendChild(t);
    });
}
window.arfRenderThumbs = arfRenderThumbs;

// Delete ONLY the selected step image from the ARF run folder (image file,
// its critique/prompt notes and the manifest entry). The run is preserved.
function arfDeleteStep() {
    const st = arfPickStep;
    const item = arfPickItem;
    if (!st || !item || !st.filename) return;
    const label = '#' + st.iter + ' (' + st.filename + ')';
    if (!confirm('Delete step ' + label + ' from ARF run "' + item.arf_dir + '"?\n\n' +
                 'The image file and its critique/prompt notes will be removed from the run folder.')) return;
    const btn = $arf('arf-picker-delete');
    if (btn) btn.disabled = true;
    fetch('/api/history/delete_arf_step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ arf_dir: item.arf_dir, filename: st.filename })
    })
        .then(r => r.json())
        .then(data => {
            if (!data || data.status !== 'deleted') {
                alert('Delete failed: ' + ((data && (data.detail || data.error)) || 'unknown error'));
                if (btn) btn.disabled = false;
                return;
            }
            // Drop the step from the in-memory run, then refresh the picker.
            const idx = item.arf_steps.indexOf(st);
            item.arf_steps = item.arf_steps.filter(s => s !== st);
            if (item.arf_steps.length) {
                const next = item.arf_steps[Math.min(Math.max(idx, 0), item.arf_steps.length - 1)];
                item.image = item.arf_steps[item.arf_steps.length - 1].image;
                arfRenderThumbs();
                arfSelect(next);
                if (window.updateStatusGen) window.updateStatusGen('🗑 ARF step removed — ' + item.arf_steps.length + ' left', 'success');
            } else {
                arfClosePicker();
                if (window.updateStatusGen) window.updateStatusGen('🗑 ARF step removed — run folder has no more images', 'success');
            }
            // Keep the History list/badge in sync with the updated run.
            if (window.loadHistory) window.loadHistory();
        })
        .catch(e => {
            console.error(e);
            alert('Delete failed: ' + e);
            if (btn) btn.disabled = false;
        });
}
window.arfDeleteStep = arfDeleteStep;

function arfApplyStep() {
    const st = arfPickStep;
    const item = arfPickItem;
    if (!st || !item) { arfClosePicker(); return; }
    const inAgentMode = window.currentMode === 'agent';
    if (inAgentMode) {
        // Inside Agent Refine the picker loads the whole run into the Agent
        // panel (reference + goal + params + steps + active run folder) instead
        // of using the standard result panel, which is hidden in agent mode.
        if (window.agentLoadArfRun) {
            window.agentLoadArfRun(item, st);
        } else if (window.agentUseAsRef) {
            window.agentUseAsRef(st.image);
            if (window.updateStatusGen) window.updateStatusGen('🪄 ARF step #' + st.iter + ' impostato come riferimento in Agent Refine', 'success');
            if (window.log) window.log('🪄 ARF ' + item.arf_dir + ' → step #' + st.iter + ' usato come riferimento Agent Refine', false, true);
        }
    } else {
        // put the step prompt in the main prompt box
        const pb = document.getElementById('prompt');
        if (pb && st.prompt) pb.value = st.prompt;
        // make this image the visible one in the result panel (direct path in folder)
        if (window.showImage) window.showImage(st.image);
        if (window.updateStatusGen) window.updateStatusGen('🪄 ARF step #' + st.iter + ' selected: visible image updated (' + st.image + ')', 'success');
        if (window.log) window.log('🪄 ARF ' + item.arf_dir + ' → step #' + st.iter + ' · coupled JSON ready (window.agentStepJson)', false, true);
    }
    // coupled JSON for the SELECTED image + its data (always)
    window.agentStepJson = {
        mode: 'arf',
        arf_dir: item.arf_dir,
        image: st.image,
        filename: st.filename || '',
        iter: st.iter,
        prompt: st.prompt || '',
        resolution: st.resolution || '',
        width: st.width || 0,
        height: st.height || 0,
        // LoRAs used for this image (same shape as T2I/I2I JSON)
        loras: st.loras || (item.params && item.params.loras) || { path: '', files: [] },
        // extra I2T/I2I reference images used by the run (run-level)
        ref_images: (item.params && item.params.ref_images) || (item.inputs && item.inputs.ref_images) || [],
        goal_reached: st.goal_reached ?? null,
        goal: (item.inputs && item.inputs.goal) || '',
        model_path: (item.params && item.params.model_path) || '',
        seed: (item.params && item.params.seed) || 0,
        steps: (item.params && item.params.steps) || 0,
        guidance: (item.params && item.params.guidance) || 0,
        scheduler: (item.params && item.params.scheduler) || '',
        original_image: item.original_image || '',
    };
    arfClosePicker();
}
window.arfApplyStep = arfApplyStep;

// Open the whole ARF run inside the 🪄 Agent Refine panel (works from any tab):
// switch to agent mode, then load goal/params/steps/reference of the chosen step.
function arfOpenInAgent() {
    const st = arfPickStep;
    const item = arfPickItem;
    if (!st || !item) { arfClosePicker(); return; }
    const alreadyAgent = window.currentMode === 'agent';
    if (!alreadyAgent && window.switchTab) {
        // switch to the Agent tab first so the panel DOM is visible
        const nav = document.querySelectorAll('.nav-btn')[3];
        window.switchTab('agent', nav);
    }
    const doLoad = () => {
        if (window.agentLoadArfRun) window.agentLoadArfRun(item, st);
        arfClosePicker();
    };
    if (alreadyAgent) doLoad();
    else setTimeout(doLoad, 250);  // wait for the agent panel to be shown
}
window.arfOpenInAgent = arfOpenInAgent;


// Update refresh button visual state
function updateRefreshButtonState(state, message) {
    const btn = document.getElementById('btn-history-refresh');
    const countSpan = document.getElementById('history-count');
    
    if (!btn) return;
    
    historyRefreshState = state;
    
    switch(state) {
        case 'loading':
            btn.style.background = '#4a4a1a';
            btn.style.color = '#ff0';
            btn.innerHTML = '⏳';
            btn.disabled = true;
            if (countSpan) {
                countSpan.textContent = '...';
                countSpan.style.background = '#4a4a1a';
                countSpan.style.color = '#ff0';
            }
            break;
        case 'loaded':
            btn.style.background = '#1a4a1a';
            btn.style.color = '#4ade80';
            btn.innerHTML = '🔄';
            btn.disabled = false;
            if (countSpan && historyTotalCount > 0) {
                countSpan.textContent = historyTotalCount;
                countSpan.style.background = '#1a4a1a';
                countSpan.style.color = '#4ade80';
            }
            // Reset to idle after 2 seconds
            setTimeout(() => updateRefreshButtonState('idle'), 2000);
            break;
        case 'error':
            btn.style.background = '#5a1a1a';
            btn.style.color = '#ff4444';
            btn.innerHTML = '❌';
            btn.disabled = false;
            if (countSpan) {
                countSpan.textContent = 'err';
                countSpan.style.background = '#5a1a1a';
                countSpan.style.color = '#ff4444';
            }
            setTimeout(() => updateRefreshButtonState('idle'), 3000);
            break;
        case 'idle':
        default:
            btn.style.background = '';
            btn.style.color = '';
            btn.innerHTML = '🔄';
            btn.disabled = false;
            if (countSpan) {
                if (historyTotalCount > 0) {
                    countSpan.textContent = historyTotalCount;
                }
                countSpan.style.background = '#333';
                countSpan.style.color = '#888';
            }
            break;
    }
}

// Refresh history with visual feedback
async function refreshHistory() {
    if (historyRefreshState === 'loading') return; // Prevent double-click
    
    console.log('[History] Refreshing...');
    updateRefreshButtonState('loading');
    
    try {
        // Reset to page 0 on refresh
        historyPage = 0;
        await loadHistory();
        updateRefreshButtonState('loaded');
        console.log('[History] Refresh complete. Total:', historyTotalCount);
    } catch (err) {
        console.error('[History] Refresh error:', err);
        updateRefreshButtonState('error');
    }
}

async function loadHistory() {
    if (activeViews.length > 0) {
        // Load active view instead if any is active
        await executeActiveViews();
        return;
    }
    try {
        if (itemsPerPage === 10) itemsPerPage = calculateItemsPerPage(); // Init if needed
        const res = await fetch('/api/history/?page=' + historyPage + '&filter=' + historyFilter + '&limit=' + itemsPerPage);
        const data = await res.json();
        const list = document.getElementById('history-list');
        const items = data.history || [];
        
        // Update total count
        historyTotalCount = data.total || 0;
        const countSpan = document.getElementById('history-count');
        if (countSpan) {
            countSpan.textContent = historyTotalCount;
        }

        if (list) {
            if (items.length > 0) {
                list.innerHTML = '';
                items.forEach(item => {
                    const div = createHistoryItemDOM(item, null);
                    list.appendChild(div);
                });

                // Page controls
                const totalPages = data.pages || Math.ceil((data.total || 0) / itemsPerPage) || 1;

                const footer = document.getElementById('history-footer');
                if (footer) {
                    footer.innerHTML = ''; // Clear previous controls

                    const controlsDiv = document.createElement('div');
                    controlsDiv.style.width = "100%";
                    controlsDiv.style.display = "flex";
                    controlsDiv.style.justifyContent = "space-between";
                    controlsDiv.style.alignItems = "center";
                    controlsDiv.style.padding = "0 5px"; // Small padding

                    const btnPrev = document.createElement('button');
                    btnPrev.className = 'btn-xs';
                    btnPrev.innerText = '◀';
                    btnPrev.onclick = () => changePage(-1);
                    btnPrev.disabled = historyPage <= 0;

                    const btnNext = document.createElement('button');
                    btnNext.className = 'btn-xs';
                    btnNext.innerText = '▶';
                    btnNext.onclick = () => changePage(1);
                    btnNext.disabled = historyPage >= totalPages - 1;

                    const pageInfo = document.createElement('span');
                    pageInfo.style.fontSize = '11px';
                    pageInfo.style.color = '#888';
                    pageInfo.innerText = 'Page ' + (historyPage + 1) + ' / ' + totalPages;

                    controlsDiv.appendChild(btnPrev);
                    controlsDiv.appendChild(pageInfo);
                    controlsDiv.appendChild(btnNext);

                    footer.appendChild(controlsDiv);
                }
            } else {
                list.innerHTML = '<div style="padding:20px;text-align:center;color:#888">No history yet</div>';
                const footer = document.getElementById('history-footer');
                if (footer) footer.innerHTML = '';
            }
        }
    } catch (e) { console.error('History load error:', e); }
}

function formatDateYYMMDD(timestamp) {
    // Handle invalid timestamps
    if (timestamp === null || timestamp === undefined || isNaN(timestamp)) {
        return '------';
    }
    const d = new Date(timestamp * 1000);
    if (isNaN(d.getTime())) {
        return '------';
    }
    return d.toISOString().slice(2, 10).replace(/-/g, '');
}

function toggleHistoryFilter() {
    const btn = document.getElementById('btn-history-filter');
    const filters = ['all', 't2i', 'i2i', 'i2t', 'arf', 'image_analysis'];
    const idx = filters.indexOf(historyFilter);
    historyFilter = filters[(idx + 1) % filters.length];
    if (btn) {
        if (historyFilter === 'image_analysis') {
            btn.textContent = 'IA';
        } else {
            btn.textContent = historyFilter.toUpperCase();
        }
    }
    historyPage = 0;
    loadHistory();
}

function toggleHistorySize() {
    historyIconSize = historyIconSize === 64 ? 128 : 64;
    itemsPerPage = calculateItemsPerPage(); // Recalc for new size
    historyPage = 0; // Reset page
    loadHistory();
}

function changePage(delta) {
    historyPage = Math.max(0, historyPage + delta);
    loadHistory();
}

function deleteHistoryItem(filename) {
    if (!confirm('Are you sure you want to delete ' + filename + '?')) return;

    fetch('/api/history/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: filename })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.status === 'deleted') {
                loadHistory();
            } else {
                alert('Error deleting: ' + (data.error || 'Unknown'));
            }
        })
        .catch(function (e) { console.error(e); });
}

function viewJsonInModal(item, title) {
    // Use the new JSONViewer
    if (window.JSONViewer) {
        // If item has history metadata (params, meta, inputs, outputs), show only that
        // Otherwise show the full item (which may be just semantic tags)
        if (item.params || item.meta || item.inputs || item.outputs) {
            JSONViewer.openFromData({
                filename: item.filename,
                image: item.image,
                timestamp: item.timestamp,
                mode: item.mode,
                params: item.params,
                inputs: item.inputs,
                outputs: item.outputs,
                meta: item.meta,
                tags: item.tags
            });
        } else {
            JSONViewer.openFromData(item);
        }
    } else {
        alert("JSONViewer not loaded");
    }
}
function restoreHistoryState(item) {
    // ARF runs contain MULTIPLE images: open the step picker instead of a
    // plain restore. The picker shows which image is the effective visible one
    // (default = the last generated step) and lets the user choose another.
    if (item && item.arf && window.openArfPicker) {
        window.openArfPicker(item);
        return;
    }
    const p = item.params;
    console.log("=== Restoring history state ===");
    console.log("Full history item:", item);
    console.log("Parameters (p):", p);

    // --- TASK 1: Auto-switch to the correct tab based on image mode ---
    const historyMode = (item.mode || p?.mode || '').toLowerCase();
    const validModes = ['t2i', 'i2i', 'i2t'];
    console.log("[restoreHistoryState] Detected mode:", historyMode, "| currentMode:", window.currentMode);
    
    if (historyMode && validModes.includes(historyMode) && historyMode !== window.currentMode) {
        console.log("[restoreHistoryState] Switching tab to:", historyMode);
        // Switch tab BEFORE restoring parameters so the correct UI panels are shown
        window.switchTab(historyMode);
    }
    
    // Ensure selectors are available - try to get them from DOM if not cached
    if (!vaeSelect) {
        window.vaeSelect = document.getElementById('vae-select');
        console.log('Retrieved vaeSelect from DOM:', !!window.vaeSelect);
    }
    if (!llmSelect) {
        window.llmSelect = document.getElementById('llm-select');
        console.log('Retrieved llmSelect from DOM:', !!window.llmSelect);
    }
    if (!schedulerTypeSelect) {
        window.schedulerTypeSelect = document.getElementById('scheduler-type-select');
        console.log('Retrieved schedulerTypeSelect from DOM:', !!window.schedulerTypeSelect);
    }
    if (!schedulerSelect) {
        window.schedulerSelect = document.getElementById('scheduler-select');
        console.log('Retrieved schedulerSelect from DOM:', !!window.schedulerSelect);
    }

    console.log("Available selectors - vaeSelect:", !!vaeSelect, "llmSelect:", !!llmSelect, "schedulerTypeSelect:", !!schedulerTypeSelect, "schedulerSelect:", !!schedulerSelect);
    if (vaeSelect) console.log("VAE select options count:", vaeSelect.options.length);
    if (llmSelect) console.log("LLM select options count:", llmSelect.options.length);

    // Update JSON Viewer if open
    if (window.JSONViewer && JSONViewer.isOpen && JSONViewer.isOpen()) {
        JSONViewer.openFromData(item);
    }

    if (!p) {
        console.warn("No parameters found");
        if (statusText) statusText.innerText = "No parameters in history file";
        return;
    }

    if (statusText) statusText.innerText = "Restoring parameters...";

    // Restore Prompt
    if (p.prompt && promptBox) promptBox.value = p.prompt;
    
    // Restore Negative Prompt
    if (p.negative_prompt && negativePromptBox) {
        negativePromptBox.value = p.negative_prompt;
        // Make sure it's visible when restoring from history
        negativePromptBox.style.display = 'block';
        const btn = document.getElementById('negative-prompt-toggle');
        if (btn) btn.textContent = '▼ Hide';
    }

    // Helper function to restore both VAE and Text Encoder after model is loaded
    // Defined BEFORE it's called to avoid ReferenceError
    const restoreVAEAndTextEncoder = async () => {
        // Restore VAE
        console.log('VAE restoration - vaeSelect exists:', !!vaeSelect, 'vaeSelect.options.length:', vaeSelect?.options?.length, 'vae:', p.vae);
        const vaeToRestore = (p.vae && p.vae.trim() !== '') ? p.vae : null;
        console.log('VAE to restore:', vaeToRestore);
        if (vaeToRestore && vaeSelect) {
            // Wait for VAE selector to be populated (max 10 seconds)
            let vaeWaitTime = 0;
            while (vaeSelect.options.length <= 0 && vaeWaitTime < 10000) {
                console.log('Waiting for VAE selector to be populated...');
                await new Promise(resolve => setTimeout(resolve, 100));
                vaeWaitTime += 100;
            }
            if (vaeSelect.options.length > 0) {
                restoreVAE();
            } else {
                console.warn('VAE selector not populated after waiting');
            }
        } else {
            console.log('VAE restoration skipped - no vae');
        }

        // Restore Text Encoder
        console.log('Text encoder restoration - llmSelect exists:', !!llmSelect, 'llmSelect.options.length:', llmSelect?.options?.length, 'text_encoder:', p.text_encoder);
        const textEncoderToRestore = (p.text_encoder && p.text_encoder.trim() !== '') ? p.text_encoder : null;
        console.log('Text encoder to restore:', textEncoderToRestore);
        if (textEncoderToRestore && llmSelect) {
            // Wait for text encoder selector to be populated (max 10 seconds)
            let encoderWaitTime = 0;
            while (llmSelect.options.length <= 0 && encoderWaitTime < 10000) {
                console.log('Waiting for text encoder selector to be populated...');
                await new Promise(resolve => setTimeout(resolve, 100));
                encoderWaitTime += 100;
            }
            if (llmSelect.options.length > 0) {
                restoreTextEncoder();
            } else {
                console.warn('Text encoder selector not populated after waiting');
            }
        } else {
            console.log('Text encoder restoration skipped - no text_encoder');
        }
    };

    // Restore Model - cerca il modello per nome e poi ripristina VAE e text_encoder
    if (p.model && window.selectModelFromHistory) {
        window.selectModelFromHistory(p.model).then(found => {
            if (found) {
                console.log('Model restored from history:', p.model);
                // After model is restored, restore VAE and Text Encoder
                // They need the model to be selected first to populate the selectors
                restoreVAEAndTextEncoder();
            } else {
                console.warn('Model not found in selector:', p.model);
            }
        }).catch(err => {
            console.error('Error restoring model:', err);
        });
    } else {
        // If no model to restore, still try to restore VAE and Text Encoder
        restoreVAEAndTextEncoder();
    }

    // Restore Sliders
    if (p.width && sliderW) { sliderW.value = p.width; if (valW) valW.innerText = p.width; }
    if (p.height && sliderH) { sliderH.value = p.height; if (valH) valH.innerText = p.height; }
    if (p.steps && sliderSteps) { sliderSteps.value = p.steps; if (valSteps) valSteps.innerText = p.steps; }
    if (p.guidance !== undefined && p.guidance !== null && sliderCfg) { sliderCfg.value = p.guidance; if (valCfg) valCfg.innerText = p.guidance; }
    if (p.top_k && sliderTopK) { sliderTopK.value = p.top_k; if (valTopK) valTopK.innerText = p.top_k; }
    if (p.temperature && sliderTemp) { sliderTemp.value = p.temperature; if (valTemp) valTemp.innerText = p.temperature; }
    if (p.strength && sliderStrength) { sliderStrength.value = p.strength; if (valStrength) valStrength.innerText = p.strength; }
    if (p.clip_skip && sliderClip) { sliderClip.value = p.clip_skip; if (valClip) valClip.innerText = p.clip_skip; }

    // Helper function to restore VAE
    const restoreVAE = () => {
        // Get the VAE from JSON - only use vae field (vae_path is deprecated)
        let vaeFilename = p.vae ? p.vae.split('/').pop() : null;
        
        if (!vaeFilename || !vaeSelect) {
            console.log('restoreVAE: skipped - vaeFilename:', vaeFilename, 'vaeSelect exists:', !!vaeSelect);
            return;
        }
        
        // Clean the filename - remove extension for comparison
        const cleanVaeFilename = vaeFilename.replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
        console.log('Looking for VAE with filename:', vaeFilename, '(cleaned:', cleanVaeFilename + ')');
        
        let vaeFound = false;
        for (let i = 0; i < vaeSelect.options.length; i++) {
            const opt = vaeSelect.options[i];
            const optText = opt.innerText;
            console.log('VAE option[' + i + ']: innerText="' + optText + '"');
            
            // Extract filename from option text (format: "displayName (filename)")
            const match = optText.match(/\(([^)]+)\)/);
            if (match) {
                const optFilename = match[1];
                const cleanOptFilename = optFilename.replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
                console.log('  -> Option filename:', optFilename, '(cleaned:', cleanOptFilename + ')');
                
                // Compare cleaned filenames
                if (cleanOptFilename === cleanVaeFilename || optFilename === vaeFilename) {
                    console.log('  -> MATCH FOUND! Setting selectedIndex to', i);
                    vaeSelect.selectedIndex = i;
                    vaeFound = true;
                    console.log('VAE restored from history:', vaeFilename);
                    break;
                }
            }
        }
        if (!vaeFound) {
            console.warn('VAE not found in selector:', vaeFilename, '- using auto');
            // Set to default (Auto)
            vaeSelect.selectedIndex = 0;
        }
    };

    // Helper function to restore Text Encoder
    const restoreTextEncoder = () => {
        // Get the text encoder from JSON - only use text_encoder field (llm is deprecated)
        let textEncoderFilename = p.text_encoder ? p.text_encoder.split('/').pop() : null;
        
        if (!textEncoderFilename || !llmSelect) {
            console.log('restoreTextEncoder: skipped - text_encoder:', textEncoderFilename, 'llmSelect exists:', !!llmSelect);
            return;
        }
        
        // Clean the filename - remove extension for comparison
        const cleanTextEncoderFilename = textEncoderFilename.replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
        console.log('Looking for text encoder with filename:', textEncoderFilename, '(cleaned:', cleanTextEncoderFilename + ')');
        
        let textEncoderFound = false;
        for (let i = 0; i < llmSelect.options.length; i++) {
            const opt = llmSelect.options[i];
            const optText = opt.innerText || opt.textContent || '';
            console.log('Text encoder option[' + i + ']: innerText="' + optText + '"');
            
            // Extract filename from option text (format: "displayName (filename) - source - type")
            const match = optText.match(/\(([^)]+)\)/);
            if (match) {
                const optFilename = match[1];
                const cleanOptFilename = optFilename.replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
                console.log('  -> Option filename:', optFilename, '(cleaned:', cleanOptFilename + ')');
                
                // Compare cleaned filenames
                if (cleanOptFilename === cleanTextEncoderFilename || optFilename === textEncoderFilename) {
                    console.log('  -> MATCH FOUND! Setting selectedIndex to', i);
                    llmSelect.selectedIndex = i;
                    textEncoderFound = true;
                    console.log('Text encoder restored from history:', textEncoderFilename);
                    break;
                }
            }
        }
        if (!textEncoderFound) {
            console.warn('Text encoder not found in selector:', textEncoderFilename, '- using default');
            // Set to default (first option)
            llmSelect.selectedIndex = 0;
        }
    };


    // Restore Scheduler Type
    if (p.scheduler_type && schedulerTypeSelect) {
        for (let i = 0; i < schedulerTypeSelect.options.length; i++) {
            const opt = schedulerTypeSelect.options[i];
            if (opt.value === p.scheduler_type) {
                schedulerTypeSelect.selectedIndex = i;
                console.log('Scheduler type restored from history:', p.scheduler_type);
                break;
            }
        }
    }

    // Restore Sampler
    if (p.sampler && schedulerSelect) {
        for (let i = 0; i < schedulerSelect.options.length; i++) {
            const opt = schedulerSelect.options[i];
            if (opt.value === p.sampler) {
                schedulerSelect.selectedIndex = i;
                console.log('Sampler restored from history:', p.sampler);
                break;
            }
        }
    }

    // Restore LoRAs
    // Handle both old format: [{path, weight}, ...] and new format: {path: "...", files: [[filename, weight, active], ...]}
    let lorasToRestore = [];
    if (p.loras) {
        if (Array.isArray(p.loras)) {
            // Old format: array of {path, weight}
            lorasToRestore = p.loras;
        } else if (p.loras.files && Array.isArray(p.loras.files)) {
            // New format: {path: "...", files: [[filename, weight, active], ...]}
            const basePath = p.loras.path || "";
            lorasToRestore = p.loras.files.map(file => ({
                path: basePath ? basePath + '/' + file[0] : file[0],
                weight: file[1],
                active: file[2]
            }));
        }
    }
    
    // Clear existing LoRAs first
    const loraContainer = document.getElementById('lora-container');
    if (loraContainer) loraContainer.innerHTML = '';
    
    if (lorasToRestore.length > 0) {
        console.log('Restoring LoRAs:', lorasToRestore);
        
        // Add slots for each LoRA
        lorasToRestore.forEach((lora, index) => {
            console.log('Adding LoRA slot:', index, lora);
            // Trigger addLoraSlot to create the slot UI
            if (window.addLoraSlot) {
                window.addLoraSlot();
            }
        });
        
        // Wait for DOM to update, then set values
        setTimeout(() => {
            const slots = document.querySelectorAll('.lora-slot');
            lorasToRestore.forEach((lora, index) => {
                if (slots[index]) {
                    const select = slots[index].querySelector('select');
                    const weightInput = slots[index].querySelector('input[type="number"]');
                    
                    // Find and select the LoRA by path
                    if (select) {
                        for (let i = 0; i < select.options.length; i++) {
                            const opt = select.options[i];
                            // Match by path or by filename
                            if (opt.value === lora.path || opt.value.endsWith('/' + lora.path) || opt.value.endsWith(lora.path)) {
                                select.selectedIndex = i;
                                break;
                            }
                        }
                    }
                    
                    // Set weight
                    if (weightInput && lora.weight !== undefined) {
                        weightInput.value = lora.weight;
                    }
                }
            });
            console.log('LoRAs restored successfully');
        }, 100);
    }

    // Restore Source Images into upload inputs (Browse 1 / Browse 2 slots)
    // showPreviewFromPath handles showing "image not available {filename}" on error
    if (p.source_image_1 && (window.uploadedPathInput || document.getElementById("uploaded-path"))) {
        (window.uploadedPathInput || document.getElementById("uploaded-path")).value = p.source_image_1;
        if (window.showPreviewFromPath) {
            window.showPreviewFromPath(p.source_image_1, 1);
        }
    }
    if (p.source_image_2 && (window.uploadedPathInput2 || document.getElementById("uploaded-path-2"))) {
        (window.uploadedPathInput2 || document.getElementById("uploaded-path-2")).value = p.source_image_2;
        if (window.showPreviewFromPath) {
            window.showPreviewFromPath(p.source_image_2, 2);
        }
    }

    // Show result image (only for non-I2T modes; I2T shows text results)
    const isTextResult = item.is_text_result || (historyMode === 'i2t');
    if (!isTextResult && item.image && resultImg) {
        resultImg.src = item.image;
        resultImg.style.display = 'block';
        console.log("[restoreHistoryState] item.image:", item.image, "has_image_analysis:", item.has_image_analysis);
        // Update toolbar buttons based on the image's JSON metadata
        if (window.checkImageMetadataAndUpdateButtons) {
            window.checkImageMetadataAndUpdateButtons(item.image);
        }
    } else if (resultImg) {
        resultImg.style.display = 'none';
    }
    if (placeholder) placeholder.style.display = 'none';
    
    // Restore I2T text result into the text result area
    if (isTextResult && item.output && item.output.text_content) {
        const tc = item.output.text_content;
        const positivePromptBox = document.getElementById('i2t-positive-prompt');
        const negativePromptBox = document.getElementById('i2t-negative-prompt');
        const thinkingBoxLocal = document.getElementById('thinking-box');
        const textResultArea = document.getElementById('text-result-area');
        
        // Populate boxes
        if (positivePromptBox) {
            positivePromptBox.textContent = tc.final_answer || tc.raw_full_response || '';
        }
        if (negativePromptBox) {
            negativePromptBox.textContent = '';
        }
        if (thinkingBoxLocal) {
            thinkingBoxLocal.textContent = tc.thinking_process || '';
            // Expand thinking box if content exists
            if (tc.thinking_process && thinkingBoxLocal.classList.contains('collapsed')) {
                thinkingBoxLocal.classList.remove('collapsed');
                const icon = document.getElementById('thinking-toggle-icon');
                if (icon) icon.innerText = '▼';
            }
        }
        // Show text result area
        if (textResultArea) {
            textResultArea.classList.remove('hidden');
        }
    }

    // Clear previous timer if exists (reset delay)
    if (autoCloseTimer) clearTimeout(autoCloseTimer);

    // Set new timer for 5 seconds
    autoCloseTimer = setTimeout(() => {
        const panel = document.getElementById('history-panel');
        if (panel && !panel.classList.contains('collapsed')) {
            toggleHistory();
        }
        autoCloseTimer = null;
    }, 5000);

    if (statusText) statusText.innerText = "Parameters restored (Auto-close in 5s)";
}

async function handleI2T() {
    if (!(window.uploadedPathInput?.value || document.getElementById("uploaded-path")?.value)) { alert('Please upload an image first'); return; }
    updateStatusHist('Analyzing image...');

    const formData = new FormData();
    formData.append('image', uploadedPathInput.value);
    formData.append('prompt', promptBox?.value || 'Describe this image in detail');
    formData.append('temperature', parseFloat(sliderTemp?.value || 0.6));

    try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (answerBox) answerBox.innerText = data.answer || '';
        if (thinkingBox) thinkingBox.innerText = data.thinking || '';
        if (textResultArea) textResultArea.classList.remove('hidden');
        updateStatusHist('Analysis complete', 'success');
    } catch (e) { updateStatusHist('Error: ' + e.message, 'error'); }
}

function toggleThinking() {
    const box = thinkingBox;
    const icon = document.getElementById('thinking-toggle-icon');
    if (box) box.classList.toggle('collapsed');
    if (icon) icon.textContent = box?.classList.contains('collapsed') ? '>' : 'v';
}

function updateStatusHist(text, type) {
    if (statusText) { statusText.innerText = text; statusText.className = 'status-text status-' + (type || 'info'); }
}

// ==========================================
// SEMANTIC TAGGING & VIRTUAL GROUPING
// ==========================================

let semanticMode = false;
let activeViews = [];
let tempViewResults = null;
let tempViewQuery = '';
let tagsCache = null;
let currentViewMode = 'normal'; // 'normal' or specific view type
let viewResultsCache = [];
let viewPage = 0;
let viewItemsPerPage = 20;

// Toggle semantic panel visibility
function toggleSemanticMode() {
    semanticMode = !semanticMode;
    const panel = document.getElementById('semantic-panel');
    const btn = document.getElementById('btn-semantic-toggle');
    
    if (panel) {
        panel.style.display = semanticMode ? 'block' : 'none';
    }
    if (btn) {
        btn.style.background = semanticMode ? '#1a4a1a' : '';
        btn.style.color = semanticMode ? '#4ade80' : '';
    }
    
    if (semanticMode) {
        loadSemanticTagsStatus();
    }
}

// Load tags status for info
async function loadSemanticTagsStatus() {
    try {
        const res = await fetch('/api/tags/status');
        const data = await res.json();
        if (data.status === 'success') {
            console.log('[Semantic] Tags status:', data.data);
        }
    } catch (err) {
        console.error('[Semantic] Failed to load status:', err);
    }
}

// Execute semantic search from input - shows results in main history list
async function executeSemanticSearch() {
    const input = document.getElementById('semantic-search-input');
    const query = input ? input.value.trim() : '';
    
    if (!query) {
        alert('Inserisci un termine di ricerca');
        return;
    }
    
    // Show loading in semantic panel
    const resultsDiv = document.getElementById('semantic-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = '<div style="padding:10px; color:#888;">⏳ Searching...</div>';
    }
    
    // Set view mode to search
    currentViewMode = 'search';
    tempViewQuery = query;
    viewPage = 0;
    
    const btnCancel = document.getElementById('btn-cancel-view');
    if (btnCancel) btnCancel.classList.remove('hidden');
    
    try {
        const res = await fetch('/api/tags/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: {
                    logic: 'OR',
                    conditions: [
                        {field: 'summary', operator: 'contains', value: query},
                        {field: 'tags', operator: 'contains', value: query}
                    ]
                }
            })
        });
        
        const data = await res.json();
        
        if (data.count === 0) {
            if (resultsDiv) {
                resultsDiv.innerHTML = '<div style="padding:10px; color:#888;">No results for "' + query + '"</div>';
            }
            // Clear history list and show message
            const list = document.getElementById('history-list');
            if (list) {
                list.innerHTML = '<div style="padding:20px;text-align:center;color:#888">No results for "' + query + '"</div>';
            }
            const footer = document.getElementById('history-footer');
            if (footer) footer.innerHTML = '';
            return;
        }
        
        // Store results and render in history list with pagination
        viewResultsCache = data.images || [];
        tempViewResults = viewResultsCache;
        historyTotalCount = data.count;
        
        // Update count display
        const countSpan = document.getElementById('history-count');
        if (countSpan) {
            countSpan.textContent = data.count + ' results';
        }
        
        // Render first page in history list
        renderFilteredResults();
        
        // Update semantic panel with summary
        if (resultsDiv) {
            resultsDiv.innerHTML = '<div style="padding:6px; font-size:11px; color:#4ade80;">✅ ' + data.count + ' results found - shown in the main list</div>';
        }
        
    } catch (err) {
        if (resultsDiv) {
            resultsDiv.innerHTML = '<div style="padding:10px; color:#c66;">Error: ' + err.message + '</div>';
        }
    }
}

// Cache for full image data from search results
let searchResultsCache = {};

// Render filtered results in history list with pagination
function renderFilteredResults() {
    const list = document.getElementById('history-list');
    if (!list) return;
    
    const items = tempViewResults || [];
    const total = items.length;
    
    if (total === 0) {
        list.innerHTML = '<div style="padding:20px;text-align:center;color:#888">No results</div>';
        const footer = document.getElementById('history-footer');
        if (footer) footer.innerHTML = '';
        return;
    }
    
    // Calculate pagination
    const startIdx = viewPage * itemsPerPage;
    const endIdx = Math.min(startIdx + itemsPerPage, total);
    const pageItems = items.slice(startIdx, endIdx);
    const totalPages = Math.ceil(total / itemsPerPage);
    
    // Render items using standard history item DOM
    list.innerHTML = '';
    pageItems.forEach(img => {
        const filename = (img.image || '').split('/').pop() || 'unknown';
        
        // Store full image data in cache for click handler
        searchResultsCache[filename] = img;
        
        // Create item with click handler that fetches full data
        const item = {
            image: img.image || '',
            filename: filename,
            timestamp: img.timestamp || 0,
            mode: 'IMG',
            params: img.params || {},
            tags: img.tags || [],
            has_image_analysis: img.has_image_analysis || false,
            _searchResult: true, // Flag to indicate this is from search
            _filename: filename  // Store filename for retrieval
        };
        
        const div = createHistoryItemDOM(item, null);
        
        // Override click handler to fetch full data
        div.onclick = function() {
            handleSearchResultClick(filename);
        };
        
        list.appendChild(div);
    });
    
    // Render pagination controls
    const footer = document.getElementById('history-footer');
    if (footer) {
        footer.innerHTML = '';
        
        const controlsDiv = document.createElement('div');
        controlsDiv.style.width = "100%";
        controlsDiv.style.display = "flex";
        controlsDiv.style.justifyContent = "space-between";
        controlsDiv.style.alignItems = "center";
        controlsDiv.style.padding = "0 5px";
        
        const btnPrev = document.createElement('button');
        btnPrev.className = 'btn-xs';
        btnPrev.innerText = '◀';
        btnPrev.onclick = () => { viewPage--; renderFilteredResults(); };
        btnPrev.disabled = viewPage <= 0;
        
        const btnNext = document.createElement('button');
        btnNext.className = 'btn-xs';
        btnNext.innerText = '▶';
        btnNext.onclick = () => { viewPage++; renderFilteredResults(); };
        btnNext.disabled = viewPage >= totalPages - 1;
        
        const pageInfo = document.createElement('span');
        pageInfo.style.fontSize = '11px';
        pageInfo.style.color = '#888';
        pageInfo.innerText = 'Page ' + (viewPage + 1) + ' / ' + totalPages + ' (' + total + ' results)';
        
        controlsDiv.appendChild(btnPrev);
        controlsDiv.appendChild(pageInfo);
        controlsDiv.appendChild(btnNext);
        
        footer.appendChild(controlsDiv);
    }
}

// Handle click on search result - fetch full data from history API
async function handleSearchResultClick(filename) {
    console.log('[Search] Clicked on:', filename);
    
    // Show loading status
    if (statusText) statusText.innerText = "Loading image data...";
    
    try {
        // Use the new /api/history/item endpoint
        const res = await fetch('/api/history/item?filename=' + encodeURIComponent(filename));
        const data = await res.json();
        
        if (data.status === 'success' && data.data) {
            restoreHistoryState(data.data);
        } else {
            // Fallback: try to find in cached search results
            const cachedData = searchResultsCache[filename];
            if (cachedData) {
                restoreHistoryState({
                    filename: filename,
                    image: cachedData.image || '/outputs/' + filename,
                    params: {},
                    timestamp: cachedData.timestamp || 0
                });
            } else {
                alert('Could not load image data: ' + (data.detail || 'Unknown error'));
            }
        }
    } catch (err) {
        console.error('[Search] Error loading image:', err);
        if (statusText) statusText.innerText = "Error loading image: " + err.message;
    }
}

// Execute view by type (tag, mood, style)
async function executeView(viewType) {
    const resultsDiv = document.getElementById('semantic-results');
    if (!resultsDiv) return;
    
    viewPage = 0;
    currentViewMode = viewType;
    
    const btnCancel = document.getElementById('btn-cancel-view');
    if (btnCancel) btnCancel.classList.remove('hidden');
    
    resultsDiv.innerHTML = '<div style="padding:10px; color:#888;">⏳ Loading ' + viewType + '...</div>';
    
    try {
        // Get all tags and group by category
        const res = await fetch('/api/tags/all');
        const data = await res.json();
        
        if (!data.tags) {
            resultsDiv.innerHTML = '<div style="padding:10px; color:#c66;">No tags found</div>';
            return;
        }
        
        // Count tags by category
        const counts = {};
        Object.values(data.tags).forEach(imgData => {
            if (imgData.tags && Array.isArray(imgData.tags)) {
                imgData.tags.forEach(tag => {
                    if (tag.category === viewType || (viewType === 'tag' && tag.category)) {
                        const key = tag.category + ':' + tag.value;
                        if (!counts[key]) {
                            counts[key] = {category: tag.category, value: tag.value, count: 0, sample: imgData.image};
                        }
                        counts[key].count++;
                    }
                });
            }
        });
        
        // Sort by count
        const sorted = Object.values(counts).sort((a, b) => b.count - a.count);
        
        if (sorted.length === 0) {
            resultsDiv.innerHTML = '<div style="padding:10px; color:#888;">No results for ' + viewType + '</div>';
            return;
        }
        
        let html = '<div style="padding:6px; font-size:11px; color:#aaa;">' + sorted.length + ' gruppi per ' + viewType + '</div>';
        html += '<div style="display:flex; flex-direction:column; gap:3px; padding:4px; max-height:250px; overflow-y:auto;">';
        
        sorted.forEach(group => {
            const sampleImg = group.sample || '';
            html += '<div onclick="filterByTagValue(\'' + group.category + '\', \'' + group.value.replace(/'/g, "\\'") + '\')" style="display:flex; align-items:center; gap:8px; padding:6px; background:#2a2a2a; border-radius:4px; cursor:pointer;">';
            if (sampleImg) {
                html += '<img src="' + sampleImg + '" style="width:32px; height:32px; object-fit:cover; border-radius:3px;" loading="lazy">';
            }
            html += '<span style="font-size:11px; color:#ddd; flex:1;">' + group.value + '</span>';
            html += '<span style="font-size:10px; color:#888; background:#3a3a3a; padding:2px 6px; border-radius:3px;">' + group.count + '</span>';
            html += '</div>';
        });
        
        html += '</div>';
        resultsDiv.innerHTML = html;
        
    } catch (err) {
        resultsDiv.innerHTML = '<div style="padding:10px; color:#c66;">Error: ' + err.message + '</div>';
    }
}

// Filter by specific tag value - shows results in main history list
async function filterByTagValue(category, value) {
    const resultsDiv = document.getElementById('semantic-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = '<div style="padding:10px; color:#888;">⏳ Filtro ' + category + ':' + value + '...</div>';
    }
    
    // Set view mode
    currentViewMode = 'filter';
    viewPage = 0;
    
    const btnCancel = document.getElementById('btn-cancel-view');
    if (btnCancel) btnCancel.classList.remove('hidden');
    
    try {
        const res = await fetch('/api/tags/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: {
                    logic: 'AND',
                    conditions: [
                        {field: 'tags', category: category, operator: 'contains', value: value}
                    ]
                }
            })
        });
        
        const data = await res.json();
        
        if (data.count === 0) {
            if (resultsDiv) {
                resultsDiv.innerHTML = '<div style="padding:10px; color:#888;">No results for ' + value + '</div>';
            }
            const list = document.getElementById('history-list');
            if (list) {
                list.innerHTML = '<div style="padding:20px;text-align:center;color:#888">No results for ' + value + '</div>';
            }
            const footer = document.getElementById('history-footer');
            if (footer) footer.innerHTML = '';
            return;
        }
        
        // Store results and render in history list
        viewResultsCache = data.images || [];
        tempViewResults = viewResultsCache;
        historyTotalCount = data.count;
        
        // Update count display
        const countSpan = document.getElementById('history-count');
        if (countSpan) {
            countSpan.textContent = data.count + ' ' + value;
        }
        
        // Render first page
        renderFilteredResults();
        
        // Update semantic panel with summary
        if (resultsDiv) {
            resultsDiv.innerHTML = '<div style="padding:6px; font-size:11px; color:#4ade80;">✅ ' + data.count + ' images tagged "' + value + '" - shown in the main list</div>';
        }
        
    } catch (err) {
        if (resultsDiv) {
            resultsDiv.innerHTML = '<div style="padding:10px; color:#c66;">Error: ' + err.message + '</div>';
        }
    }
}

function cancelView() {
    currentViewMode = 'normal';
    viewResultsCache = [];
    tempViewResults = null;
    viewPage = 0;
    
    const resultsDiv = document.getElementById('semantic-results');
    if (resultsDiv) resultsDiv.innerHTML = '';
    
    const btnCancel = document.getElementById('btn-cancel-view');
    if (btnCancel) btnCancel.classList.add('hidden');
    
    // Reset count display
    const countSpan = document.getElementById('history-count');
    if (countSpan) {
        countSpan.textContent = historyTotalCount || '0';
    }
    
    // Reload normal history
    loadHistory();
}

async function executeActiveViews() {
    if (currentViewMode === 'search' || currentViewMode === 'filter') {
        // Re-render filtered results
        renderFilteredResults();
    } else if (currentViewMode !== 'normal') {
        await executeView(currentViewMode);
    }
}

// --------------------------------------------------------------------------
// Source-image path normalizer (shared by History/upload/Agent flows)
// Accepts: /outputs/x/y.png, /uploads/name.png, /app/outputs/x/y.png,
// http(s)://host/outputs/x/y.png, or a bare relative path (with or without a
// leading 'outputs/'). Returns {server, url} with server RELATIVE to OUTPUT_DIR
// (e.g. "agent_refine_1788.../iter_002_img.png") and url a browser URL
// ("/outputs/agent_refine_1788.../iter_002_img.png"). null if unusable.
// --------------------------------------------------------------------------
window.normalizeSourcePath = function (u) {
    if (!u || u === 'undefined') return null;
    let s = String(u).trim();
    if (!s) return null;
    // full http(s) URL -> strip scheme+host and keep path after /outputs|/uploads
    let m = s.match(/^https?:\/\/[^/]+\/(outputs|uploads)\/(.+)$/i);
    if (m) s = m[2];
    // absolute web path
    if (s.startsWith('/outputs/')) s = s.slice('/outputs/'.length);
    else if (s.startsWith('/uploads/')) s = s.slice('/uploads/'.length);
    // absolute container disk path
    else if (s.startsWith('/app/')) {
        s = s.slice('/app/'.length);
        if (s.startsWith('outputs/')) s = s.slice('outputs/'.length);
    }
    // leftover relative 'outputs/...' prefix (from older normalizations)
    if (s.startsWith('outputs/')) s = s.slice('outputs/'.length);
    // data: URLs / any other scheme -> not usable as a disk path
    if (/^[a-z][a-z0-9+.-]*:/i.test(s) && !/^outputs\//i.test(s)) return null;
    s = s.replace(/^\/+/, '');
    if (!s) return null;
    return { server: s, url: '/outputs/' + s };
};

function loadToInput(imageUrl, slot) {
    if (!imageUrl || imageUrl === 'undefined') return;

    // Normalize to a server path RELATIVE to OUTPUT_DIR + a browser URL.
    // Handles images living in subfolders (e.g. agent_refine_*/iter_0.png),
    // absolute disk paths, and absolute http(s) URLs.
    const norm = window.normalizeSourcePath ? window.normalizeSourcePath(imageUrl) : null;
    if (!norm) return;
    const serverPath = norm.server;   // e.g. "agent_refine_1788.../iter_002_img.png"
    const browserUrl = norm.url;      // e.g. "/outputs/agent_refine_1788.../iter_002_img.png"

    // Phase 9: Use the new N-Slot API (imageSlots array) when available
    // Falls back to legacy uploadedPathInput/uploadedPathInput2 if imageSlots is not initialized
    if (slot === 1) {
        // Try to load into the first slot of imageSlots
        if (typeof imageSlots !== 'undefined' && imageSlots.length > 0 && imageSlots[0]) {
            const slot0 = imageSlots[0];
            slot0.path = serverPath;
            slot0.previewDataUrl = browserUrl;
            slot0.filename = serverPath.split('/').pop();
            if (typeof renderSlots === 'function') {
                renderSlots();
            } else if (window.renderSlots) {
                window.renderSlots();
            }
            // Legacy compatibility: also update hidden input
            const legacyInput = document.getElementById('uploaded-path');
            if (legacyInput) legacyInput.value = serverPath;
            if (window.showPreviewFromPath) {
                window.showPreviewFromPath(browserUrl, 1);
            }
        } else {
            // Fallback to legacy path
            const legacyInput = document.getElementById('uploaded-path');
            if (legacyInput) legacyInput.value = serverPath;
            if (window.showPreviewFromPath) {
                window.showPreviewFromPath(browserUrl, 1);
            }
        }
    } else if (slot === 2) {
        // Slot 2 - similar logic
        if (typeof imageSlots !== 'undefined' && imageSlots.length > 1 && imageSlots[1]) {
            const slot1 = imageSlots[1];
            slot1.path = serverPath;
            slot1.previewDataUrl = browserUrl;
            slot1.filename = serverPath.split('/').pop();
            if (typeof renderSlots === 'function') {
                renderSlots();
            } else if (window.renderSlots) {
                window.renderSlots();
            }
            const legacyInput2 = document.getElementById('uploaded-path-2');
            if (legacyInput2) legacyInput2.value = serverPath;
            if (window.showPreviewFromPath) {
                window.showPreviewFromPath(browserUrl, 2);
            }
        } else {
            const legacyInput2 = document.getElementById('uploaded-path-2');
            if (legacyInput2) legacyInput2.value = serverPath;
            if (window.showPreviewFromPath) {
                window.showPreviewFromPath(browserUrl, 2);
            }
        }
    }
}

function viewHistoryItem(item) {
    // Normalize item and restore
    if (typeof item === 'string') {
        try { item = JSON.parse(item); } catch (e) { return; }
    }
    restoreHistoryState(item);
}

function openZoom() {
    const img = document.getElementById('result-img');
    if (img && img.src && img.style.display !== 'none') {
        // Open image in a new tab for zoom
        window.open(img.src, '_blank');
    } else {
        alert('No image to zoom. Generate or load an image first.');
    }
}

// Export functions to window for HTML onclick handlers
window.toggleHistory = toggleHistory;
window.toggleHistoryFilter = toggleHistoryFilter;
window.toggleHistorySize = toggleHistorySize;
window.changePage = changePage;
window.viewHistoryItem = viewHistoryItem;
window.toggleSemanticMode = toggleSemanticMode;
window.executeSemanticSearch = executeSemanticSearch;
window.executeView = executeView;
window.cancelView = cancelView;
window.filterByTagValue = filterByTagValue;
window.loadToInput = loadToInput;
window.handleI2T = handleI2T;
window.toggleThinking = toggleThinking;
window.initHistory = initHistory;
window.openZoom = openZoom;
window.loadHistory = loadHistory;
window.refreshHistory = refreshHistory;
window.stopAutoTag = stopAutoTag;
// Tagging system exports
window.showTaggingDialog = showTaggingDialog;
window.hideTaggingDialog = hideTaggingDialog;
window.populateTaggingModelSelect = populateTaggingModelSelect;
window.refreshTaggingModels = refreshTaggingModels;
window.startTagging = startTagging;
window.closeTaggingProgressPopup = closeTaggingProgressPopup;

// Resume tagging that was interrupted by generation
async function resumeTagging() {
    console.log('[Tagging] Resuming interrupted tagging...');
    
    // Show the progress popup
    showTaggingProgressPopup();
    
    // Start a new SSE stream for the resumed tagging
    const btnAuto = document.getElementById('btn-auto-tag');
    const btnStopSemantic = document.getElementById('btn-stop-tag');
    if (btnAuto) btnAuto.style.display = 'none';
    if (btnStopSemantic) btnStopSemantic.style.display = 'block';
    
    try {
        const res = await fetch('/api/tags/resume', { method: 'POST' });
        
        // The resume endpoint returns a StreamingResponse, not JSON
        if (res.body) {
            // It's a streaming response - process it like runTaggingProcess
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            taggingAbortController = new AbortController();
            taggingActive = true;
            
            // Process the SSE stream
            function processLine(line) {
                if (!line.startsWith('data: ')) return;
                try {
                    const data = JSON.parse(line.substring(6));
                    console.log('[Tagging Resume]', data);
                    // Reuse the same processLine logic from runTaggingProcess
                    updateTaggingProgress(data);
                } catch (e) {
                    console.error('[Tagging Resume] Parse error:', e);
                }
            }
            
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();
                    lines.forEach(processLine);
                }
            } catch (err) {
                if (err.name !== 'AbortError') {
                    console.error('[Tagging Resume] Stream error:', err);
                }
            }
            
            taggingActive = false;
            if (btnAuto) btnAuto.style.display = 'inline-block';
            if (btnStopSemantic) btnStopSemantic.style.display = 'none';
        } else {
            // It's a JSON response (error or no_images)
            const data = await res.json();
            console.log('[Tagging Resume] Response:', data);
            if (data.status === 'error') {
                alert('Error: ' + (data.message || 'Unknown error'));
            }
            taggingActive = false;
            if (btnAuto) btnAuto.style.display = 'inline-block';
            if (btnStopSemantic) btnStopSemantic.style.display = 'none';
        }
    } catch (err) {
        console.error('[Tagging Resume] Error:', err);
        alert('Error while resuming tagging');
        taggingActive = false;
        if (btnAuto) btnAuto.style.display = 'inline-block';
        if (btnStopSemantic) btnStopSemantic.style.display = 'none';
    }
}
window.resumeTagging = resumeTagging;
window.batchDownloadSelected = batchDownloadSelected;
window.cancelBatchSelection = cancelBatchSelection;
