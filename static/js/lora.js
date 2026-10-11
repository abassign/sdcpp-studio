// lora.js - Gestione LoRA
// ========================
// Custom LoRA picker per slot + per-LoRA metadata JSON editor + compatibility
// filter vs the currently selected checkpoint model.
//
// Data model of a LoRA file entry (from GET /api/models/loras):
//   { name, path, size, meta: { display_name, trigger_words[],
//                                compatible_models[], base_arch, has_meta } }
// The metadata lives in a sidecar JSON "<lora-file>.json" managed through
// POST /api/models/loras/meta/read | /meta/save

const LORA_EXT_RE = /\.(safetensors|ckpt|pt|pth)$/i;

// shared mutable state (kept on window for debugging / other modules)
window.loraFiles = window.loraFiles || [];
window.loraFilterOn = true; // false = "All", true = "only compatible with selected model"
let _openPicker = null;     // currently open .lora-picker element
let _metaEditorBusy = false;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function loraStem(name) {
    return String(name || '').replace(LORA_EXT_RE, '');
}
function loraDisplayName(f) {
    const m = f && f.meta;
    if (m && m.display_name) return m.display_name;
    return loraStem(f && f.name);
}
function currentModelPath() {
    return (window.modelSelect && window.modelSelect.value) || '';
}
function loraCompatList(f) {
    const m = f && f.meta;
    return (m && Array.isArray(m.compatible_models)) ? m.compatible_models : [];
}
function isLoraCompatible(f, modelPath) {
    if (!modelPath || !f) return false;
    return loraCompatList(f).indexOf(modelPath) !== -1;
}
/** 'ok' compatible · 'no' tagged but not this model · 'unknown' not tagged · 'none' no model selected */
function loraStatus(f, modelPath) {
    if (!modelPath) return 'none';
    if (isLoraCompatible(f, modelPath)) return 'ok';
    return (f.meta && f.meta.has_meta) ? 'no' : 'unknown';
}
function dotClass(status) {
    return { ok: 'dot-ok', no: 'dot-no', unknown: 'dot-unknown', none: 'dot-none' }[status] || 'dot-none';
}
function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function slotSelect(slot) { return slot.querySelector('select.lora-native-select'); }
function slotWeight(slot) { return slot.querySelector('input[type="number"].lora-weight'); }

function countCompatibles() {
    const mp = currentModelPath();
    if (!mp) return 0;
    return (window.loraFiles || []).filter(f => isLoraCompatible(f, mp)).length;
}

function updateFilterUI() {
    const btn = document.getElementById('btn-lora-filter');
    const statusEl = document.getElementById('lora-filter-status');
    const total = (window.loraFiles || []).length;
    const on = window.loraFilterOn;
    if (btn) {
        btn.textContent = on ? '🎯 Filter' : '📋 All';
        btn.classList.toggle('filter-active', on);
        btn.classList.toggle('filter-off', !on);
        btn.title = on
            ? 'Filter is ON: the lists only show LoRAs compatible with the selected model. Click to show ALL LoRAs.'
            : 'Showing ALL LoRAs. Click to filter by the selected model compatibility.';
    }
    if (statusEl) {
        if (!total) {
            statusEl.textContent = '';
            statusEl.classList.remove('active');
        } else if (on) {
            const ok = countCompatibles();
            statusEl.textContent = `✓ ${ok}/${total} match`;
            statusEl.classList.toggle('active', ok > 0);
            statusEl.title = currentModelPath()
                ? 'LoRAs compatible with the currently selected model / total'
                : 'Select a checkpoint model first';
        } else {
            statusEl.textContent = `all ${total}`;
            statusEl.classList.remove('active');
            statusEl.title = 'Showing all LoRAs';
        }
    }
}

// ---------------------------------------------------------------------------
// scanning
// ---------------------------------------------------------------------------

async function scanLoras() {
    const folder = (window.loraFolderInput && window.loraFolderInput.value)
        || window.modelsDir + '/loras' || '/app/models/loras';
    try {
        const res = await fetch('/api/models/loras?folder=' + encodeURIComponent(folder));
        const data = await res.json();
        window.loraFiles = data.files || [];
        updateFilterUI();
        refreshAllSlots();
    } catch (e) { console.error('LoRA scan error:', e); }
}

// ---------------------------------------------------------------------------
// slots / picker
// ---------------------------------------------------------------------------

function addLoraSlot() {
    const container = window.loraContainer;
    if (!container) return null;

    const slot = document.createElement('div');
    slot.className = 'lora-slot';

    const picker = document.createElement('div');
    picker.className = 'lora-picker';
    picker.innerHTML =
        '<button type="button" class="lora-picker-btn input-text" title="Choose a LoRA">' +
            '<span class="lp-placeholder">-- Select LoRA --</span><span class="lp-caret">▼</span></button>' +
        '<select class="lora-native-select" tabindex="-1" aria-hidden="true"></select>' +
        '<div class="lora-picker-popup hidden">' +
            '<input type="text" class="lora-picker-search" placeholder="🔍 search name / trigger word...">' +
            '<div class="lora-picker-info"></div>' +
            '<div class="lora-picker-list"></div>' +
        '</div>';
    slot.appendChild(picker);

    const btn = picker.querySelector('.lora-picker-btn');
    const select = picker.querySelector('select');
    const popup = picker.querySelector('.lora-picker-popup');
    const search = popup.querySelector('.lora-picker-search');
    const info = popup.querySelector('.lora-picker-info');
    const list = popup.querySelector('.lora-picker-list');

    // --- selection state lives in the (hidden) native select ---
    select.addEventListener('change', () => syncPickerLabel(picker));

    // open/close popup
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (popup.classList.contains('hidden')) openPicker(picker);
        else closePicker();
    });
    search.addEventListener('input', () => renderPickerList(picker));
    search.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); }
    });

    // weight input
    const weightInput = document.createElement('input');
    weightInput.type = 'number';
    weightInput.className = 'input-text lora-weight';
    weightInput.placeholder = 'Weight';
    weightInput.value = '1.0';
    weightInput.step = '0.1';
    weightInput.min = '-10.0';
    weightInput.max = '10.0';

    // remove button
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.innerText = '×';
    removeBtn.className = 'btn';
    removeBtn.title = 'Remove LoRA slot';
    removeBtn.style.cssText = 'background:#500; padding:2px 8px;';
    removeBtn.onclick = () => { if (picker === _openPicker) _openPicker = null; slot.remove(); };

    slot.appendChild(weightInput);
    slot.appendChild(removeBtn);

    container.appendChild(slot);
    refreshSlot(slot);
    return slot;
}

function openPicker(picker) {
    closePicker();
    _openPicker = picker;
    const popup = picker.querySelector('.lora-picker-popup');
    popup.classList.remove('hidden');
    const search = popup.querySelector('.lora-picker-search');
    search.value = '';

    // place the popup (position: fixed) under the trigger, flipped up if needed
    const btn = picker.querySelector('.lora-picker-btn');
    const rect = btn.getBoundingClientRect();
    const popW = Math.max(rect.width, 330);
    const popH = 420;
    let left = rect.left;
    if (left + popW > window.innerWidth - 8) left = Math.max(8, window.innerWidth - 8 - popW);
    let top = rect.bottom + 4;
    if (top + popH > window.innerHeight - 8) top = Math.max(8, rect.top - 4 - popH);
    popup.style.left = left + 'px';
    popup.style.top = top + 'px';
    popup.style.width = popW + 'px';
    popup.style.maxHeight = Math.min(popH, window.innerHeight - 16) + 'px';

    renderPickerList(picker);
    search.focus();
}

function closePicker() {
    if (_openPicker) {
        const popup = _openPicker.querySelector('.lora-picker-popup');
        if (popup) popup.classList.add('hidden');
        _openPicker = null;
    }
}

document.addEventListener('click', (e) => {
    if (_openPicker && !e.target.closest('.lora-picker')) closePicker();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closePicker(); closeLoraMetaEditor(); }
});

/** Which loras are choosable right now (filter-aware). */
function visibleLoras() {
    const mp = currentModelPath();
    const all = window.loraFiles || [];
    if (!window.loraFilterOn || !mp) {
        return all.map(f => ({ f, status: loraStatus(f, mp), chosen: false }));
    }
    return all
        .filter(f => isLoraCompatible(f, mp))
        .map(f => ({ f, status: 'ok', chosen: false }));
}

function renderPickerList(picker) {
    const popup = picker.querySelector('.lora-picker-popup');
    const search = popup.querySelector('.lora-picker-search');
    const info = popup.querySelector('.lora-picker-info');
    const list = popup.querySelector('.lora-picker-list');
    const select = picker.querySelector('select');
    const chosen = select.value || '';
    const chosenFile = (window.loraFiles || []).find(f => f.path === chosen);

    let items = visibleLoras();

    // pin currently selected LoRA even when filter would hide it (don't lose state)
    if (chosen && chosenFile && !items.some(it => it.f.path === chosen)) {
        items.unshift({
            f: chosenFile,
            status: loraStatus(chosenFile, currentModelPath()),
            chosen: true,
            pinned: true
        });
    }

    // search filter
    const q = search.value.trim().toLowerCase();
    if (q) {
        items = items.filter(({ f }) => {
            const hay = (loraDisplayName(f) + ' ' + (f.name || '') + ' ' + loraStem(f.name) + ' '
                + ((f.meta && f.meta.base_arch) || '') + ' '
                + ((f.meta && f.meta.trigger_words) || []).join(' ')).toLowerCase();
            return hay.indexOf(q) !== -1;
        });
    }

    // info line
    const mp = currentModelPath();
    let infoTxt = '';
    if (window.loraFilterOn) {
        infoTxt = mp
            ? 'Only LoRAs compatible with <b>' + esc(mp.split('/').pop()) + '</b> are listed.'
            : 'Select a model first — showing everything.';
    } else {
        infoTxt = mp
            ? '<span style="color:#86efac">● compatible</span> &nbsp;'
              + '<span style="color:#f59e0b">● no metadata</span> &nbsp;'
              + '<span style="color:#ef4444">● not compatible</span>'
            : 'No model selected.';
    }
    info.innerHTML = infoTxt;

    list.innerHTML = '';
    if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'lora-picker-empty';
        if (q) {
            empty.innerHTML = 'No LoRA matches <b>' + esc(q) + '</b>.';
        } else if (window.loraFilterOn && mp) {
            empty.innerHTML =
                'No LoRA is tagged as compatible with the selected model yet.<br>' +
                'Click <b>⚙</b> on a LoRA (in <b>📋 All</b> view) to create its metadata<br>' +
                'and check the checkpoints it works with.';
            const goAll = document.createElement('button');
            goAll.className = 'mini-btn';
            goAll.textContent = '📋 Show all LoRAs';
            goAll.onclick = (e) => { e.stopPropagation(); window.loraFilterOn = false; updateFilterUI(); renderPickerList(picker); };
            empty.appendChild(document.createElement('br'));
            empty.appendChild(goAll);
        } else if (!(window.loraFiles || []).length) {
            empty.innerHTML = 'No LoRA files found.<br>Check the folder above and press 🔄 Scan.';
        } else {
            empty.innerHTML = 'Nothing to show.';
        }
        list.appendChild(empty);
        return;
    }

    items.forEach(({ f, status, pinned }) => {
        const row = document.createElement('div');
        row.className = 'lora-row' + (f.path === chosen ? ' selected' : '');
        row.dataset.path = f.path;

        const m = f.meta || {};
        const hasMeta = !!m.has_meta;

        // dot + main
        const dot = document.createElement('span');
        dot.className = 'dot ' + dotClass(status);
        dot.title = status === 'ok' ? 'Compatible with the selected model'
            : status === 'no' ? 'Not compatible with the selected model'
            : status === 'unknown' ? 'No metadata yet — create it with ⚙'
            : '';

        const main = document.createElement('div');
        main.className = 'lora-row-main';

        const name = document.createElement('div');
        name.className = 'lora-row-name';
        const nm = document.createElement('span');
        nm.textContent = loraDisplayName(f);
        nm.title = f.path;
        name.appendChild(nm);
        if (pinned) {
            const b = document.createElement('span');
            b.className = 'badge warn';
            b.textContent = 'not compatible here';
            name.appendChild(b);
        }
        if (!hasMeta) {
            const b = document.createElement('span');
            b.className = 'badge warn';
            b.textContent = 'no info';
            name.appendChild(b);
        } else if (m.base_arch) {
            const b = document.createElement('span');
            b.className = 'badge';
            b.textContent = m.base_arch;
            name.appendChild(b);
        }

        const sub = document.createElement('div');
        const tw = (m.trigger_words || []).slice(0, 2);
        if (tw.length) {
            sub.className = 'lora-row-sub tw';
            sub.textContent = 'trigger: ' + tw.join(' · ');
        } else {
            sub.className = 'lora-row-sub';
            sub.textContent = f.name;
        }
        main.appendChild(name);
        main.appendChild(sub);

        // edit metadata button
        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'lora-row-edit' + (hasMeta ? ' has-meta' : '');
        editBtn.textContent = '⚙';
        editBtn.title = hasMeta ? 'Edit metadata (JSON)' : 'Create metadata (JSON) for this LoRA';
        editBtn.onclick = (e) => {
            e.stopPropagation();
            closePicker();
            openLoraMetaEditor(f.path);
        };

        row.appendChild(dot);
        row.appendChild(main);
        row.appendChild(editBtn);

        // selection
        row.addEventListener('click', () => {
            selectLora(picker, f.path);
            closePicker();
        });

        list.appendChild(row);
    });
}

function selectLora(picker, path) {
    const select = picker.querySelector('select');
    // guarantee the option exists in the native select too
    let found = false;
    for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].value === path) { select.selectedIndex = i; found = true; break; }
    }
    if (!found) {
        const opt = document.createElement('option');
        opt.value = path;
        opt.textContent = path;
        select.appendChild(opt);
        select.value = path;
    }
    select.dispatchEvent(new Event('change'));
}

function syncPickerLabel(picker) {
    const select = picker.querySelector('select');
    const btn = picker.querySelector('.lora-picker-btn');
    const val = select.value;
    if (!val) {
        btn.dataset.val = '';
        if (!btn.querySelector('.lp-placeholder')) {
            btn.innerHTML = '<span class="lp-placeholder">-- Select LoRA --</span><span class="lp-caret">▼</span>';
        }
        btn.title = 'Choose a LoRA';
        return;
    }
    const file = (window.loraFiles || []).find(f => f.path === val);
    const label = esc(file ? loraDisplayName(file) : (val.split('/').pop() || val));
    btn.dataset.val = val;
    btn.innerHTML = '<span class="lp-label">' + label + '</span><span class="lp-caret">▼</span>';
    btn.title = val;
}

// --- refresh a single slot: the (hidden) native select keeps ALL scanned
// LoRAs so legacy code (history restore, getLoraConfigs) works even when the
// popup filter is ON. Filtering only affects the rows rendered in the popup. ---
function refreshSlot(slot) {
    const select = slotSelect(slot);
    if (!select) return;
    const old = select.value;

    select.innerHTML = '';
    const files = window.loraFiles || [];
    if (old && !files.some(f => f.path === old)) {
        const opt = document.createElement('option');
        opt.value = old;
        opt.textContent = old;
        select.appendChild(opt);
    }
    files.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f.path;
        opt.textContent = f.path;
        select.appendChild(opt);
    });
    select.value = files.some(f => f.path === old) ? old : '';

    syncPickerLabel(slot.querySelector('.lora-picker'));
}

function refreshAllSlots() {
    document.querySelectorAll('.lora-slot').forEach(refreshSlot);
    // keep an open popup in sync
    if (_openPicker) renderPickerList(_openPicker);
}

// legacy name
function updateLoraDropdowns() { refreshAllSlots(); }


// ---------------------------------------------------------------------------
// global toggles
// ---------------------------------------------------------------------------

function toggleLoraFilter() {
    window.loraFilterOn = !window.loraFilterOn;
    updateFilterUI();
    refreshAllSlots();
}

// model change => recompute compatibility
if (window.modelSelect) {
    window.modelSelect.addEventListener('change', () => {
        updateFilterUI();
        refreshAllSlots();
    });
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

function initLora() {
    if (window.btnScan) window.btnScan.addEventListener('click', scanLoras);
    if (window.btnAddLora) window.btnAddLora.addEventListener('click', () => addLoraSlot());
    updateFilterUI();
}

// ---------------------------------------------------------------------------
// metadata editor (JSON) modal
// ---------------------------------------------------------------------------

function openLoraMetaEditor(loraPath) {
    if (_metaEditorBusy) return;
    _metaEditorBusy = true;

    let root = document.getElementById('lora-meta-root');
    if (!root) {
        root = document.createElement('div');
        root.id = 'lora-meta-root';
        document.body.appendChild(root);
    }
    root.innerHTML =
        '<div class="modal-overlay" id="lora-meta-overlay">' +
            '<div class="modal-content">' +
                '<div class="modal-header">' +
                    '<span>⚙ LoRA Metadata</span>' +
                    '<button class="close-btn" onclick="window.closeLoraMetaEditor()">×</button>' +
                '</div>' +
                '<div class="modal-body" id="lora-meta-body">Loading…</div>' +
                '<div class="modal-footer">' +
                    '<button class="btn btn-sm" onclick="window.closeLoraMetaEditor()">Cancel</button>' +
                    '<button class="btn btn-sm" style="background:var(--accent)" id="btn-meta-save">💾 Save (JSON)</button>' +
                '</div>' +
            '</div>' +
        '</div>';

    const overlay = root.querySelector('#lora-meta-overlay');
    overlay.addEventListener('click', (e) => { if (e.target === overlay) closeLoraMetaEditor(); });

    const body = root.querySelector('#lora-meta-body');
    body.innerHTML = '<div style="text-align:center;padding:30px;">Loading metadata…</div>';

    fetch('/api/models/loras/meta/read', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lora_path: loraPath })
    })
        .then(r => r.json())
        .then(data => {
            if (data.detail) throw new Error(data.detail);
            renderMetaEditor(loraPath, data);
        })
        .catch(err => {
            body.innerHTML = '<div class="lora-meta-error">Failed to load metadata: ' + esc(err.message) + '</div>';
        })
        .finally(() => { _metaEditorBusy = false; });
}

function renderMetaEditor(loraPath, data) {
    const body = document.getElementById('lora-meta-body');
    if (!body) return;

    const meta = data.meta || {};
    const checkpoints = data.checkpoints || [];
    const suggested = new Set(data.suggested_paths || []);
    const existingPaths = new Set(meta.compatible_models || []);
    const arch = meta.base_arch || (data.detected && data.detected.arch) || '';
    const archSources = (data.detected && data.detected.arch_sources) || [];

    // build the checkpoint pick state: start from existing tags when the meta
    // file already exists, otherwise from the auto-detected suggestions
    const checked = new Set(existingPaths.size ? existingPaths : suggested);

    const prettyFolder = (f) => {
        if (!f) return '(root)';
        if (f === 'checkpoints') return '(checkpoints root)';
        return f;
    };

    // checkpoint model rows, grouped by folder
    const rows = checkpoints.map(cp => ({ ...cp, missing: false }));
    (meta.compatible_models || []).forEach(p => {
        if (!checkpoints.some(c => c.path === p)) {
            rows.push({ path: p, name: p.split('/').pop(), folder: '⚠ Missing', family: '', missing: true });
        }
    });
    const folders = [...new Set(rows.map(r => r.folder || ''))].sort((a, b) => {
        if (a === '⚠ Missing') return 1;
        if (b === '⚠ Missing') return -1;
        return a.localeCompare(b);
    });

    body.innerHTML =
        '<div style="font-size:13px; font-weight:600;">' + esc(data.file.name) + '</div>' +
        '<div class="lora-meta-file">' + esc(data.file.path) + '</div>' +

        '<div class="lora-meta-section">' +
            '<div class="lora-meta-title">📛 Name in the list' +
                '<span class="lora-meta-arch">base <span class="val">' + esc(arch || '?') + '</span></span>' +
            '</div>' +
            '<label class="lora-meta-label">Display name (default = file name)</label>' +
            '<input type="text" class="lora-meta-input" id="lora-meta-display" value="' + esc(meta.display_name || '') + '">' +
            (archSources.length
                ? '<div style="font-size:10px;color:#666;margin-top:3px;">detected from: ' + esc(archSources.join(' · ')) + '</div>'
                : '') +
        '</div>' +

        '<div class="lora-meta-section">' +
            '<div class="lora-meta-title">💬 Trigger words</div>' +
            '<label class="lora-meta-label">One per line (optional)</label>' +
            '<textarea class="lora-meta-textarea" id="lora-meta-tw" placeholder="e.g.&#10;masterpiece&#10;character name">' +
                esc((meta.trigger_words || []).join('\n')) +
            '</textarea>' +
            '<div class="lora-tw-preview" id="lora-tw-preview"></div>' +
        '</div>' +

        '<div class="lora-meta-section">' +
            '<div class="lora-meta-title">🧩 Compatible models' +
                '<span class="lora-meta-arch" id="lora-cp-count"></span></div>' +
            '<label class="lora-meta-label">Check the checkpoints this LoRA can work with (done once per LoRA).</label>' +
            '<div class="lora-cp-toolbar">' +
                '<input type="text" id="lora-cp-search" placeholder="🔍 filter checkpoints…">' +
                '<button class="btn-sm" id="btn-cp-auto" title="Re-apply the auto-detected suggestions">✨ Auto</button>' +
                '<button class="btn-sm" id="btn-cp-clear">Clear</button>' +
            '</div>' +
            '<div id="lora-cp-list"></div>' +
        '</div>' +
        (data.exists
            ? '<div class="lora-meta-footer-hint">💾 Sidecar file <code>' + esc(loraPath) + '.json</code> exists — saving updates it.</div>'
            : '<div class="lora-meta-footer-hint">ℹ️ No metadata file yet. Save creates <code>' + esc(loraPath) + '.json</code> next to the LoRA.</div>');

    // --- trigger words live preview ---
    const twInput = body.querySelector('#lora-meta-tw');
    const preview = body.querySelector('#lora-tw-preview');
    const renderTw = () => {
        preview.innerHTML = '';
        parseTriggerWords(twInput.value).slice(0, 30).forEach(w => {
            const chip = document.createElement('span');
            chip.className = 'chip';
            chip.textContent = w;
            preview.appendChild(chip);
        });
    };
    twInput.addEventListener('input', renderTw);
    renderTw();

    // --- compatible models checklist ---
    const cpList = body.querySelector('#lora-cp-list');
    const cpSearch = body.querySelector('#lora-cp-search');
    const counter = body.querySelector('#lora-cp-count');
    const rowEls = [];

    const updateCounter = () => {
        const n = body.querySelectorAll('#lora-cp-list input[type=checkbox]:checked').length;
        counter.textContent = n + ' tagged';
    };

    const buildGroups = () => {
        cpList.innerHTML = '';
        folders.forEach(folder => {
            const groupRows = rows.filter(r => (r.folder || '') === folder);
            if (!groupRows.length) return;

            const group = document.createElement('div');
            group.className = 'lora-cp-group';
            group.dataset.folder = folder;

            const header = document.createElement('div');
            header.className = 'lora-cp-group-header';
            header.innerHTML = '<span>' + esc(prettyFolder(folder)) + '</span>' +
                '<span class="gcount">' + groupRows.length + '</span>' +
                '<span class="glink" data-act="all">select all</span>' +
                '<span class="glink" data-act="none">none</span>';
            group.appendChild(header);

            groupRows.forEach(cp => {
                const rowEl = document.createElement('div');
                rowEl.className = 'lora-cp-row';
                rowEl.dataset.path = cp.path;
                rowEl.dataset.name = (cp.name || cp.path).toLowerCase();
                rowEl.dataset.folder = (cp.folder || '').toLowerCase();
                rowEl.dataset.family = (cp.family || '').toLowerCase();

                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = checked.has(cp.path);
                cb.dataset.path = cp.path;

                const name = document.createElement('span');
                name.className = 'cp-name';
                name.textContent = cp.name || cp.path;
                name.title = cp.path;

                const fam = document.createElement('span');
                fam.className = 'cp-fam';
                fam.textContent = cp.family_label || cp.family || '?';

                rowEl.appendChild(cb);
                rowEl.appendChild(name);
                if (cp.family) rowEl.appendChild(fam);
                if (cp.missing) {
                    const miss = document.createElement('span');
                    miss.className = 'cp-missing';
                    miss.textContent = 'no longer on disk';
                    rowEl.appendChild(miss);
                }

                rowEl.addEventListener('click', (e) => {
                    if (e.target !== cb && !e.target.closest('input')) cb.checked = !cb.checked;
                    updateCounter();
                });
                cb.addEventListener('change', updateCounter);

                group.appendChild(rowEl);
                rowEls.push(rowEl);
            });

            // group quick actions
            const headerAll = group.querySelector('[data-act=all]');
            const headerNone = group.querySelector('[data-act=none]');
            headerAll.addEventListener('click', (e) => {
                e.stopPropagation();
                group.querySelectorAll('input[type=checkbox]').forEach(c => { c.checked = true; });
                updateCounter();
            });
            headerNone.addEventListener('click', (e) => {
                e.stopPropagation();
                group.querySelectorAll('input[type=checkbox]').forEach(c => { c.checked = false; });
                updateCounter();
            });

            cpList.appendChild(group);
        });
        updateCounter();
    };
    buildGroups();

    cpSearch.addEventListener('input', () => {
        const q = cpSearch.value.trim().toLowerCase();
        rowEls.forEach(el => {
            const show = !q || el.dataset.name.indexOf(q) !== -1
                || el.dataset.folder.indexOf(q) !== -1
                || el.dataset.family.indexOf(q) !== -1;
            el.style.display = show ? '' : 'none';
        });
    });

    body.querySelector('#btn-cp-auto').addEventListener('click', () => {
        rowEls.forEach(el => { el.querySelector('input').checked = suggested.has(el.dataset.path); });
        updateCounter();
    });
    body.querySelector('#btn-cp-clear').addEventListener('click', () => {
        rowEls.forEach(el => { el.querySelector('input').checked = false; });
        updateCounter();
    });

    // --- save ---
    const saveBtn = document.getElementById('btn-meta-save');
    saveBtn.disabled = false;
    saveBtn.addEventListener('click', () => saveLoraMeta(loraPath, body));
}

function parseTriggerWords(value) {
    const out = [];
    String(value || '').split(/[\n,;]+/).forEach(tok => {
        const s = tok.trim().replace(/\s+/g, ' ');
        if (s && out.indexOf(s) === -1) out.push(s);
    });
    return out;
}

async function saveLoraMeta(loraPath, body) {
    const saveBtn = document.getElementById('btn-meta-save');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';

    const compat = [];
    body.querySelectorAll('#lora-cp-list input[type=checkbox]:checked').forEach(cb => {
        if (cb.dataset.path) compat.push(cb.dataset.path);
    });

    const payload = {
        lora_path: loraPath,
        meta: {
            display_name: (body.querySelector('#lora-meta-display') || {}).value || '',
            trigger_words: parseTriggerWords((body.querySelector('#lora-meta-tw') || {}).value),
            compatible_models: compat
        }
    };

    try {
        const res = await fetch('/api/models/loras/meta/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok || data.status !== 'success') throw new Error(data.detail || 'Save failed');

        // update cached entry in place (no full rescan needed)
        const files = window.loraFiles || [];
        const idx = files.findIndex(f => f.path === loraPath);
        if (idx !== -1) files[idx] = { ...files[idx], meta: data.meta };

        closeLoraMetaEditor();
        updateFilterUI();
        refreshAllSlots();
        if (window.updateStatus) window.updateStatus('LoRA metadata saved ✓', 'success');
    } catch (err) {
        console.error('save meta error', err);
        if (window.updateStatus) window.updateStatus('Failed to save metadata', 'error');
        let errBox = body.querySelector('.lora-meta-error');
        if (!errBox) {
            errBox = document.createElement('div');
            errBox.className = 'lora-meta-error';
            body.appendChild(errBox);
        }
        errBox.textContent = 'Error: ' + err.message;
        saveBtn.disabled = false;
        saveBtn.textContent = '💾 Save (JSON)';
    }
}

function closeLoraMetaEditor() {
    const root = document.getElementById('lora-meta-root');
    if (root) root.innerHTML = '';
    _metaEditorBusy = false;
}

// ---------------------------------------------------------------------------
// reading selected LoRAs (legacy interface — used by generate.js)
// ---------------------------------------------------------------------------

function getLoraConfigs() {
    const configs = [];
    const slots = document.querySelectorAll('.lora-slot');
    slots.forEach(slot => {
        const select = slotSelect(slot);
        const weight = slotWeight(slot);
        if (select && select.value) configs.push({ path: select.value, weight: parseFloat(weight && weight.value) || 1.0 });
    });
    return configs;
}

// keep legacy name working (defined above as well, harmless)

window.initLora = initLora;
window.scanLoras = scanLoras;
window.addLoraSlot = addLoraSlot;
window.getLoraConfigs = getLoraConfigs;
window.updateLoraDropdowns = updateLoraDropdowns;
window.toggleLoraFilter = toggleLoraFilter;
window.openLoraMetaEditor = openLoraMetaEditor;
window.closeLoraMetaEditor = closeLoraMetaEditor;
window.loraDisplayName = loraDisplayName;
console.log('LoRA loaded');
