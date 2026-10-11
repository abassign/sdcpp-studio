// upload.js - File upload + Dynamic N-Slots (Phase 5)
// =============================================================================
// ARCHITECTURE: imageSlots[] is the dynamic array managing N images.
// Each slot is: {id, role, path, file, previewDataUrl, filename}
// The first slot is always 'init'. Later slots have a configurable role
// (default: 'ref' if not specified).
// BACKWARD COMPAT: legacy ids file-input/uploaded-path + file-input-2/
// uploaded-path-2 are kept in sync with imageSlots[0] and imageSlots[1] for
// compatibility with existing code that reads those fields.
// =============================================================================

const MAX_SLOTS = 8;          // Max number of slots
let imageSlots = [];          // Dynamic slot array
let _nextSlotId = 1;          // Unique id counter

// Elementi DOM legacy (Phase 1G compatibility)
let fileInput, fileInput2, uploadedPathInput, uploadedPathInput2;
let preview1, preview2, btnDel1, btnDel2;
let uploadZone1, uploadZone2;
let swapContainer, slotsContainer, addSlotBtn;

// =============================================================================
// Inizializzazione
// =============================================================================

function initUpload() {
    // Cache legacy elements
    fileInput        = document.getElementById('file-input');
    fileInput2       = document.getElementById('file-input-2');
    uploadedPathInput  = document.getElementById('uploaded-path');
    uploadedPathInput2 = document.getElementById('uploaded-path-2');
    preview1 = document.getElementById('preview-1');
    preview2 = document.getElementById('preview-2');
    btnDel1  = document.getElementById('btn-del-1');
    btnDel2  = document.getElementById('btn-del-2');
    uploadZone1 = document.getElementById('upload-zone-1');
    uploadZone2 = document.getElementById('upload-zone-2');
    swapContainer = document.getElementById('swap-container');
    slotsContainer = document.getElementById('image-slots-container');
    addSlotBtn = document.getElementById('btn-add-slot');

    // File input listeners
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const s = imageSlots[0];
            if (e.target.files[0] && s) handleFileUpload(e.target.files[0], s.id);
        });
    }
    if (fileInput2) {
        fileInput2.addEventListener('change', (e) => {
            const s = imageSlots[1];
            if (e.target.files[0] && s) handleFileUpload(e.target.files[0], s.id);
        });
    }

    // Initial slots: 1 init + 1 ref
    if (slotsContainer && imageSlots.length === 0) {
        addImageSlot('init', { skipRender: true });
        addImageSlot('ref',  { skipRender: true });
        renderSlots();
    }

    // Show add-slot button on first user interaction (call to onUploadChange from any other init)
    if (typeof onUploadChange === 'function') onUploadChange();

    // Collapse any expanded preview when the click lands outside its image
    // ("the image loses focus" -> shrink back to the 128px icon).
    document.addEventListener('click', (e) => {
        document.querySelectorAll('.upload-slot.expanded').forEach(el => {
            const im = el.querySelector('img.uploaded-preview');
            if (im && e.target === im) return;
            el.classList.remove('expanded');
        });
    });
}

// =============================================================================
// Slot Management (Phase 5: N-Slot Dinamici)
// =============================================================================

/**
 * Adds a new slot to the list. If skipRender is true, it does not re-render the DOM
 * (utile per setup iniziale batch).
 */
function addImageSlot(role = 'ref', opts = {}) {
    if (imageSlots.length >= MAX_SLOTS) {
        console.warn(`[upload] Max ${MAX_SLOTS} slots reached`);
        return null;
    }
    const id = _nextSlotId++;
    // First slot is forced to init role
    if (imageSlots.length === 0) role = 'init';
    const slot = {
        id,
        role,
        path: '',
        file: null,
        previewDataUrl: '',
        filename: ''
    };
    imageSlots.push(slot);
    if (!opts.skipRender) renderSlots();
    return id;
}

/**
 * Rimuove uno slot per ID. Non rimuove il primo slot (init).
 */
function removeImageSlot(id) {
    const idx = imageSlots.findIndex(s => s.id === id);
    if (idx === -1) return;
    if (idx === 0) {
        console.warn('[upload] Cannot remove init slot (slot 1)');
        return;
    }
    imageSlots.splice(idx, 1);
    renderSlots();
    if (typeof onUploadChange === 'function') onUploadChange();
}

/**
 * Moves a slot up or down by one position.
 */
function moveSlot(id, direction) {
    const idx = imageSlots.findIndex(s => s.id === id);
    if (idx === -1) return;
    const newIdx = direction === 'up' ? idx - 1 : idx + 1;
    // Free reordering: any slot can move, including above the init (index 0).
    if (newIdx < 0 || newIdx >= imageSlots.length) return;
    [imageSlots[idx], imageSlots[newIdx]] = [imageSlots[newIdx], imageSlots[idx]];
    renderSlots();
    if (typeof onUploadChange === 'function') onUploadChange();
}

/**
 * Restituisce la lista di slot attivi (con immagine caricata e ruolo != 'none')
 * nel formato atteso dal backend: [{path, role}, ...]
 */
function getImageSlots() {
    return imageSlots
        .filter(s => s.path && s.role && s.role !== 'none')
        .map(s => ({ path: s.path, role: s.role, id: s.id }));
}

/**
 * Restituisce uno slot per ID.
 */
function getSlot(id) {
    return imageSlots.find(s => s.id === id) || null;
}

/**
 * Renderizza tutti gli slot nel container DOM.
 * IMPORTANTE: ricostruisce il DOM mantenendo lo state (path, preview, ruolo).
 */
function renderSlots() {
    if (!slotsContainer) return;
    slotsContainer.innerHTML = '';
    imageSlots.forEach((slot, idx) => {
        const slotEl = buildSlotElement(slot, idx);
        slotsContainer.appendChild(slotEl);
    });
    // Show add-slot button if under limit
    if (addSlotBtn) {
        if (imageSlots.length < MAX_SLOTS) {
            addSlotBtn.classList.remove('hidden');
        } else {
            addSlotBtn.classList.add('hidden');
        }
    }
    // Sync legacy hidden inputs (Phase 1G compatibility)
    syncLegacyInputs();
    // Trigger change handler
    if (typeof onUploadChange === 'function') onUploadChange();
    // Restore roles from localStorage for first 2 slots
    const mode = window.currentMode || 'default';
    for (let i = 0; i < Math.min(2, imageSlots.length); i++) {
        const savedRole = localStorage.getItem(`slot${i+1}_role_${mode}`);
        if (savedRole && imageSlots[i].role !== savedRole) {
            imageSlots[i].role = savedRole;
        }
    }
}

/**
 * Costruisce l'elemento DOM per uno slot.
 */
function buildSlotElement(slot, idx) {
    const div = document.createElement('div');
    div.className = 'upload-slot';
    div.id = `upload-slot-${slot.id}`;
    div.dataset.slotId = slot.id;
    div.dataset.role = slot.role;
    const info = getRoleInfo(slot.role);
    const isFirst = (idx === 0);

    div.innerHTML = `
        <div class="slot-header">
            <span class="slot-number">${idx + 1}</span>
            <span class="slot-role-badge" data-role="${slot.role}">${info.badge}</span>
            <div class="slot-header-actions">
                <button class="btn btn-icon slot-move-up" data-slot-id="${slot.id}"
                    ${idx === 0 ? 'disabled' : ''} title="Move up (reading priority)">▲</button>
                <button class="btn btn-icon slot-move-down" data-slot-id="${slot.id}"
                    ${idx === imageSlots.length - 1 ? 'disabled' : ''} title="Move down (reading priority)">▼</button>
            </div>
        </div>
        <div class="upload-zone-custom" data-slot-id="${slot.id}"
             id="upload-zone-${slot.id}">
            <div class="upload-text" style="pointer-events:none; color:#666;">Drop Image Here</div>
            <img class="uploaded-preview hidden" alt="preview" />
        </div>
        <div class="slot-footer">
            <select class="slot-role-select" data-slot-id="${slot.id}">
                ${buildRoleOptions(slot, isFirst)}
            </select>
            <button class="btn btn-sm slot-browse-btn" data-slot-id="${slot.id}">📂</button>
            ${slot.role === 'mask' ? `<button class="btn btn-sm slot-mask-btn" data-slot-id="${slot.id}" title="Disegna la maschera (inpainting)">✏️</button>` : ''}
            <button class="btn btn-sm btn-browse-delete slot-delete-btn hidden" data-slot-id="${slot.id}">🗑️</button>
        </div>
        <div class="slot-hint">${info.hint}</div>
    `;

    // Wire up events
    const zone = div.querySelector('.upload-zone-custom');
    const textEl = div.querySelector('.upload-text');
    const img = div.querySelector('img.uploaded-preview');
    const select = div.querySelector('select.slot-role-select');
    const browseBtn = div.querySelector('.slot-browse-btn');
    const delBtn = div.querySelector('.slot-delete-btn');
    const moveUp = div.querySelector('.slot-move-up');
    const moveDown = div.querySelector('.slot-move-down');
    const maskBtn = div.querySelector('.slot-mask-btn');

    // Restore preview if any
    if (slot.previewDataUrl) {
        img.src = slot.previewDataUrl;
        img.classList.remove('hidden');
        if (textEl) textEl.classList.add('hidden');
        delBtn.classList.remove('hidden');
    }

    // Click on zone -> trigger file picker (use legacy input for first 2 slots)
    const openFilePicker = () => {
        if (idx === 0 && fileInput) fileInput.click();
        else if (idx === 1 && fileInput2) fileInput2.click();
        else {
            // For slot >=3, create a transient file input
            const tmpInput = document.createElement('input');
            tmpInput.type = 'file';
            tmpInput.accept = 'image/*';
            tmpInput.onchange = (e) => {
                if (e.target.files[0]) handleFileUpload(e.target.files[0], slot.id);
            };
            tmpInput.click();
        }
    };
    zone.addEventListener('click', openFilePicker);
    // Single click on the image toggles the expanded (column-width) view and
    // must NOT open the file picker. Clicking elsewhere collapses it.
    img.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        div.classList.toggle('expanded');
    });
    if (browseBtn) browseBtn.addEventListener('click', (e) => { e.stopPropagation(); openFilePicker(); });
    if (maskBtn) maskBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (typeof window.openMaskEditor === 'function') window.openMaskEditor(slot.id);
    });
    if (delBtn) delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        // Trash CLOSES the slot window (removes it). The init slot (first) is
        // special: it cannot be removed, so it is only cleared.
        const idx = imageSlots.findIndex(s => s.id === slot.id);
        if (idx > 0) removeImageSlot(slot.id);
        else deleteImage(slot.id);
    });
    if (moveUp) moveUp.addEventListener('click', (e) => { e.stopPropagation(); moveSlot(slot.id, 'up'); });
    if (moveDown) moveDown.addEventListener('click', (e) => { e.stopPropagation(); moveSlot(slot.id, 'down'); });

    // Drag and drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.style.borderColor = 'var(--accent)';
    });
    zone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        zone.style.borderColor = '#444';
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.style.borderColor = '#444';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) handleFileUpload(file, slot.id);
    });

    // Role change
    select.addEventListener('change', (e) => {
        onRoleChange(slot.id);
    });

    return div;
}

/**
 * Costruisce le <option> per il selettore di ruolo, disabilitando 'init' per
 * slot non-primari e abilitando 'init' solo per il primo slot.
 */
function buildRoleOptions(slot, isFirst) {
    const allRoles = [
        { value: 'init',    label: '🎯 Init Image' },
        { value: 'ref',     label: '🎨 Style Reference' },
        { value: 'face',    label: '👤 Face Source' },
        { value: 'mask',    label: '🎭 Mask' },
        { value: 'control', label: '📐 Control' },
        { value: 'blend',   label: '🔄 Blend' },
    ];
    if (!isFirst) {
        allRoles.unshift({ value: 'none', label: '🚫 Disabled' });
    } else {
        allRoles[0] = { value: 'init', label: '🎯 Init Image' };
    }
    return allRoles.map(r =>
        `<option value="${r.value}" ${slot.role === r.value ? 'selected' : ''}>${r.label}</option>`
    ).join('');
}

/**
 * Sincronizza gli input hidden legacy con i primi due slot.
 * BACKWARD COMPAT: Phase 1G codice legge uploaded-path / uploaded-path-2.
 */
function syncLegacyInputs() {
    // Slot 1 → uploaded-path
    if (imageSlots[0]) {
        if (uploadedPathInput) uploadedPathInput.value = imageSlots[0].path || '';
        if (preview1 && imageSlots[0].previewDataUrl) {
            preview1.src = imageSlots[0].previewDataUrl;
            preview1.classList.remove('hidden');
        } else if (preview1) {
            preview1.src = '';
            preview1.classList.add('hidden');
        }
    } else {
        if (uploadedPathInput) uploadedPathInput.value = '';
    }
    // Slot 2 → uploaded-path-2
    if (imageSlots[1]) {
        if (uploadedPathInput2) uploadedPathInput2.value = imageSlots[1].path || '';
        if (preview2 && imageSlots[1].previewDataUrl) {
            preview2.src = imageSlots[1].previewDataUrl;
            preview2.classList.remove('hidden');
        } else if (preview2) {
            preview2.src = '';
            preview2.classList.add('hidden');
        }
    } else {
        if (uploadedPathInput2) uploadedPathInput2.value = '';
    }
}

// =============================================================================
// File Upload / Preview
// =============================================================================

async function handleFileUpload(file, slotId) {
    const slot = getSlot(slotId);
    if (!slot) {
        console.error(`[upload] No slot found for id ${slotId}`);
        return;
    }
    const formData = new FormData();
    formData.append('file', file);
    try {
        const res = await fetch('/api/models/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.path) {
            slot.path = data.path;
            slot.file = file;
            slot.filename = file.name;
            // Read preview
            const reader = new FileReader();
            reader.onload = (e) => {
                slot.previewDataUrl = e.target.result;
                renderSlots();
            };
            reader.readAsDataURL(file);
            // Save role for this mode
            try {
                const mode = window.currentMode || 'default';
                const slotIdx = imageSlots.indexOf(slot);
                localStorage.setItem(`slot${slotIdx+1}_role_${mode}`, slot.role);
            } catch (e) { /* localStorage may be disabled */ }
            if (typeof onUploadChange === 'function') onUploadChange();
        }
    } catch (e) { console.error('Upload error:', e); }
}

function showPreview(file, slot) {
    // Backward compat: for legacy callers
    const reader = new FileReader();
    reader.onload = (e) => {
        // Find or create slot
        let targetSlot = imageSlots[slot - 1];
        if (!targetSlot) {
            // If slot doesn't exist, add it
            addImageSlot(slot === 1 ? 'init' : 'ref');
            targetSlot = imageSlots[imageSlots.length - 1];
        }
        targetSlot.previewDataUrl = e.target.result;
        targetSlot.file = file;
        renderSlots();
    };
    reader.readAsDataURL(file);
}

function deleteImage(slot) {
    // The slot-trash button passes the slot ID; legacy callers pass a 1-based
    // index (1 or 2). Resolve by ID FIRST: IDs are unique and, after any
    // add/remove (or an Agent Refine run restore), they no longer equal
    // index+1. Treating a numeric ID as an index deleted the wrong slot and
    // left an un-removable "ghost" image behind.
    let targetSlot = null;
    if (typeof slot === 'number') {
        targetSlot = getSlot(slot) || imageSlots[slot - 1] || null;
    } else {
        targetSlot = getSlot(slot);
    }
    if (!targetSlot) return;
    targetSlot.path = '';
    targetSlot.file = null;
    targetSlot.previewDataUrl = '';
    targetSlot.filename = '';
    renderSlots();
    if (typeof onUploadChange === 'function') onUploadChange();
}

function swapUploadedImages() {
    // Backward compat: swap slot 1 and slot 2
    if (imageSlots.length < 2) return;
    [imageSlots[0], imageSlots[1]] = [imageSlots[1], imageSlots[0]];
    renderSlots();
}

/**
 * Clear the images from ALL slots and remove the extra slots, returning to the
 * default state: slot 1 (init) + slot 2 (ref), both empty. The roles of the
 * base slots are kept. Useful when a run/history auto-restored reference images
 * and you want to reset them in one go.
 */
function clearAllImageSlots() {
    // Remove slots beyond the first two via the id API (keeps _nextSlotId)
    while (imageSlots.length > 2) {
        const last = imageSlots[imageSlots.length - 1];
        if (typeof window.removeImageSlot === 'function') window.removeImageSlot(last.id);
        else imageSlots.pop();
    }
    if (!imageSlots.length) addImageSlot('init');
    // Clear the images, keeping the roles
    imageSlots.forEach(s => {
        s.path = '';
        s.file = null;
        s.previewDataUrl = '';
        s.filename = '';
    });
    renderSlots();
    if (typeof onUploadChange === 'function') onUploadChange();
}

function showPreviewFromPath(serverPath, slot) {
    // Backward compat: set preview for a legacy slot
    if (!serverPath) return;
    // Robust normalization (handles subfolders like agent_refine_*/img.png and
    // absolute /app or http URLs). Falls back to inline handling if the shared
    // helper is not loaded yet.
    let rel, browserUrl;
    if (window.normalizeSourcePath) {
        const norm = window.normalizeSourcePath(serverPath);
        if (!norm) return;
        rel = norm.server;
        browserUrl = norm.url;
    } else {
        let s = serverPath;
        if (s.startsWith('/app/')) s = s.substring(4);
        if (s.startsWith('/outputs/')) s = s.substring('/outputs/'.length);
        else if (s.startsWith('/uploads/')) s = s.substring('/uploads/'.length);
        if (s.startsWith('outputs/')) s = s.substring('outputs/'.length);
        rel = s.replace(/^\/+/, '');
        browserUrl = '/outputs/' + rel;
    }
    let targetSlot;
    if (typeof slot === 'number' && slot < 100) {
        targetSlot = imageSlots[slot - 1];
        if (!targetSlot) {
            addImageSlot(slot === 1 ? 'init' : 'ref');
            targetSlot = imageSlots[imageSlots.length - 1];
        }
    } else {
        targetSlot = getSlot(slot);
    }
    if (!targetSlot) return;
    targetSlot.path = rel;
    targetSlot.previewDataUrl = browserUrl;
    targetSlot.filename = rel.split('/').pop();
    renderSlots();
}

// =============================================================================
// Multi-Image Role Management (Phase 1D, kept for backward compat)
// =============================================================================

const ROLE_LABELS = {
    init:    { badge: '🎯 Init Image',   hint: "L'immagine di partenza da modificare" },
    ref:     { badge: '🎨 Reference',    hint: "Riferimento stilistico per guidare la generazione" },
    face:    { badge: '👤 Face Source',  hint: "Foto da cui estrarre il volto (face swap con PhotoMaker)" },
    mask:    { badge: '🎭 Mask',         hint: "Maschera: bianco = area da modificare, nero = preservare" },
    control: { badge: '📐 Control',      hint: "Guida strutturale (Canny edge, depth, pose)" },
    blend:   { badge: '🔄 Blend',        hint: "Fusa con init image tramite Mix Ratio (pixel blend)" },
    none:    { badge: '🚫 Disabled',     hint: "Slot non usato. Seleziona un ruolo per attivarlo." }
};

function getRoleInfo(role) {
    return ROLE_LABELS[role] || ROLE_LABELS.none;
}

function onRoleChange(slotId) {
    const select = document.querySelector(`select.slot-role-select[data-slot-id="${slotId}"]`);
    if (!select) return;
    const slot = getSlot(slotId);
    if (!slot) return;
    setSlotRole(slotId, select.value);
    renderSlots();
}

function setSlotRole(slotId, role) {
    const slot = getSlot(slotId);
    if (!slot) return;
    slot.role = role;
    // Update DOM if exists
    const select = document.querySelector(`select.slot-role-select[data-slot-id="${slotId}"]`);
    if (select && select.value !== role) select.value = role;
    // Update badge
    const slotEl = document.getElementById(`upload-slot-${slotId}`);
    if (slotEl) {
        const badge = slotEl.querySelector('.slot-role-badge');
        if (badge) {
            const info = getRoleInfo(role);
            badge.textContent = info.badge;
            badge.dataset.role = role;
        }
        const hint = slotEl.querySelector('.slot-hint');
        if (hint) hint.textContent = getRoleInfo(role).hint;
        slotEl.dataset.role = role;
    }
    // Persist per mode (use 1-based index for first 2 slots to maintain localStorage keys)
    try {
        const mode = window.currentMode || 'default';
        const idx = imageSlots.indexOf(slot);
        if (idx >= 0 && idx < 2) {
            localStorage.setItem(`slot${idx+1}_role_${mode}`, role);
        }
    } catch (e) { /* localStorage may be disabled */ }
}

/**
 * Backward compat: get role for slot N (1-based, max 2)
 */
function getSlotRole(slot) {
    const targetSlot = imageSlots[slot - 1];
    return targetSlot ? (targetSlot.role || 'none') : 'none';
}

function restoreSlotRoles(mode) {
    if (!mode) return;
    try {
        const s1 = localStorage.getItem(`slot1_role_${mode}`);
        const s2 = localStorage.getItem(`slot2_role_${mode}`);
        if (s1 && imageSlots[0]) setSlotRole(imageSlots[0].id, s1);
        if (s2 && imageSlots[1]) setSlotRole(imageSlots[1].id, s2);
    } catch (e) { /* ignore */ }
}

function autoAssignRoles() {
    // Phase 1D/5: Analyze prompt to suggest the best role for slot 2+
    const promptEl = document.getElementById('prompt');
    const prompt = (promptEl?.value || '').toLowerCase();

    const faceKeywords    = ['volto', 'face', 'viso', 'ritratto', 'identità', 'persona', 'sostituisci persona', 'chi è'];
    const maskKeywords    = ['maschera', 'mask', 'solo lo', 'solo il', 'solo la', 'rimuovi sfondo', 'replace'];
    const controlKeywords = ['canny', 'edge', 'depth', 'pose', 'scheletro', 'skeleton', 'contorno'];
    const blendKeywords   = ['fuso', 'blend', 'mix', 'doppia esposizione', 'composizione'];
    const refKeywords     = ['stile', 'style', 'riferimento', 'reference', 'ispirato', 'come'];

    const hasFace    = faceKeywords.some(k => prompt.includes(k));
    const hasMask    = maskKeywords.some(k => prompt.includes(k));
    const hasControl = controlKeywords.some(k => prompt.includes(k));
    const hasBlend   = blendKeywords.some(k => prompt.includes(k));
    const hasRef     = refKeywords.some(k => prompt.includes(k));

    let suggested = 'ref'; // default
    if (hasFace) suggested = 'face';
    else if (hasMask) suggested = 'mask';
    else if (hasControl) suggested = 'control';
    else if (hasBlend) suggested = 'blend';

    // Apply to slot 2 (or create it if doesn't exist)
    if (imageSlots.length < 2) {
        addImageSlot(suggested);
    } else {
        setSlotRole(imageSlots[1].id, suggested);
    }
    renderSlots();
    console.log(`[autoAssignRoles] Suggested slot 2 role: ${suggested} (hasRef=${hasRef})`);
}

function onUploadChange() {
    // Triggered when images or roles change
    const hasImage1 = !!(imageSlots[0] && imageSlots[0].path);
    const hasImage2 = !!(imageSlots[1] && imageSlots[1].path);
    const hasAnyRef = imageSlots.slice(1).some(s => s.path && s.role && s.role !== 'none');
    // Show swap only with 2 same-role slots and both have images
    if (swapContainer) {
        const sameRole = imageSlots[0] && imageSlots[1] && imageSlots[0].role === imageSlots[1].role;
        const canSwap = hasImage1 && hasImage2 && sameRole &&
                        imageSlots[0].role !== 'face' &&
                        imageSlots[0].role !== 'mask' &&
                        imageSlots[0].role !== 'control';
        if (canSwap) swapContainer.classList.remove('hidden');
        else swapContainer.classList.add('hidden');
    }
    // Show auto-role button when at least 1 image is loaded
    const autoBtn = document.getElementById('btn-auto-role');
    if (autoBtn) {
        if (hasImage1 || hasAnyRef) {
            autoBtn.classList.remove('hidden');
        } else {
            autoBtn.classList.add('hidden');
        }
    }
    // Show "clear all slots" only when there is something to clear
    const clearBtn = document.getElementById('btn-clear-slots');
    if (clearBtn) {
        const anyImage = imageSlots.some(s => s.path || s.previewDataUrl);
        if (anyImage) clearBtn.classList.remove('hidden');
        else clearBtn.classList.add('hidden');
    }
}

// =============================================================================
// Exports
// =============================================================================

window.initUpload = initUpload;
window.handleFileUpload = handleFileUpload;
window.showPreview = showPreview;
window.deleteImage = deleteImage;
window.swapUploadedImages = swapUploadedImages;
window.showPreviewFromPath = showPreviewFromPath;
window.addImageSlot = addImageSlot;
window.removeImageSlot = removeImageSlot;
window.moveSlot = moveSlot;
window.getImageSlots = getImageSlots;
window.getSlot = getSlot;
window.clearAllImageSlots = clearAllImageSlots;
window.renderSlots = renderSlots;
window.onRoleChange = onRoleChange;
window.setSlotRole = setSlotRole;
window.getSlotRole = getSlotRole;
window.restoreSlotRoles = restoreSlotRoles;
window.autoAssignRoles = autoAssignRoles;
window.onUploadChange = onUploadChange;

console.log('Upload (N-Slot) loaded');
