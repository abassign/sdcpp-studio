// models.js - Gestione modelli
// =============================

function formatDate(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * Parser per raggruppare modelli per cartella + ruolo
 * @param {Array} models - Lista modelli dall'API
 * @returns {Object} - Oggetto raggruppato { folder: { role: [models] } }
 */
function parseModelsByFolderAndRole(models) {
    const grouped = {};

    models.forEach(model => {
        // Estrai cartella genitore dal path
        const pathParts = model.path.replace(/\\/g, '/').split('/');
        // The parent folder is the second-to-last element (e.g. checkpoints, i2t, loras)
        let folder = pathParts[pathParts.length - 2];
        if (!folder || folder === model.path) {
            folder = 'root';
        }

        // Ruolo del modello (text_encoder, lora, checkpoint)
        const role = model.role || model.category || 'checkpoint';

        if (!grouped[folder]) grouped[folder] = {};
        if (!grouped[folder][role]) grouped[folder][role] = [];

        grouped[folder][role].push(model);
    });

    return grouped;
}

/**
 * Ottiene l'etichetta leggibile per il ruolo
 * @param {string} role - Ruolo del modello
 * @returns {string} - Etichetta leggibile
 */
function getRoleLabel(role) {
    const labels = {
        'text_encoder': 'Text Encoder',
        'lora': 'LoRA',
        'checkpoint': 'Checkpoint'
    };
    return labels[role] || role;
}

async function initModelSelector() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        modelSelect.innerHTML = "";

        if (data.models && data.models.length > 0) {
            // Filter models by mode
            let filteredModels;
            if (window.currentMode === 'i2t') {
                // Per i2t mostra solo modelli i2t
                filteredModels = data.models.filter(m => m.category === 'i2t');
            } else {
                // Per t2i, i2i - mostra solo checkpoint
                filteredModels = data.models.filter(m => m.category === 'checkpoint' || !m.category);
            }

            if (filteredModels.length === 0) {
                const opt = document.createElement('option');
                opt.innerText = `No ${window.currentMode === 'i2t' ? 'I2T' : 'checkpoint'} models found`;
                modelSelect.appendChild(opt);
                return;
            }

            // Raggruppa i modelli per cartella (sottocartella di checkpoints)
            const grouped = {};
            filteredModels.forEach(m => {
                const pathParts = m.path.replace(/\\/g, '/').split('/');
                // Trova la posizione di 'checkpoints' nel path
                let folder = 'root';
                if (pathParts.includes('checkpoints')) {
                    const idx = pathParts.indexOf('checkpoints');
                    if (idx + 1 < pathParts.length) {
                        folder = pathParts[idx + 1];
                    }
                }

                // Estrai solo il nome del modello
                let modelName = m.name || m.id;
                if (modelName.includes('/')) {
                    modelName = modelName.split('/').pop();
                }
                modelName = modelName.replace(/\.(safetensors|gguf|bin)$/i, '');

                if (!grouped[folder]) grouped[folder] = [];
                grouped[folder].push({
                    path: m.path,
                    name: modelName
                });
            });

            // Ordina le cartelle
            const sortedFolders = Object.keys(grouped).sort();

            sortedFolders.forEach(folder => {
                const models = grouped[folder];
                // Ordina i modelli nella cartella
                models.sort((a, b) => a.name.localeCompare(b.name));

                if (models.length === 1) {
                    // Un solo modello: mostra solo la cartella come opzione
                    const opt = document.createElement('option');
                    opt.value = models[0].path;
                    opt.innerText = folder;
                    modelSelect.appendChild(opt);
                } else {
                    // Multiple models: create an optgroup per folder
                    const folderGroup = document.createElement('optgroup');
                    folderGroup.label = folder;

                    models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m.path;
                        opt.innerText = m.name;
                        folderGroup.appendChild(opt);
                    });

                    modelSelect.appendChild(folderGroup);
                }
            });
        } else {
            const opt = document.createElement('option');
            opt.innerText = "No models found";
            modelSelect.appendChild(opt);
        }

        setTimeout(() => { checkModelCapabilities(modelSelect.value); }, 500);

    } catch (e) { console.error("Failed to list models", e); }
}

async function checkModelCapabilities(modelPath) {
    if (llmSelectContainer) {
        // First check by name convention for immediate UI response
        const isZImage = modelPath && modelPath.toLowerCase().includes('z_image');
        llmSelectContainer.classList.toggle('hidden', !isZImage);

        // Then query backend for actual encoders to be robust
        if (modelPath) {
            try {
                // Wait for text encoders to load
                await loadTextEncoders(modelPath);

                // If encoders were found and loaded into the select (more than just the default option), show the container
                if (llmSelect && llmSelect.options && llmSelect.options.length > 1) {
                    llmSelectContainer.classList.remove('hidden');
                }
            } catch (e) {
                console.error("Error checking model capabilities for text encoders:", e);
            }
        }
    }
}

async function loadTextEncoders(modelPath) {
    try {
        const res = await fetch('/api/models/text_encoders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: modelPath })
        });
        const data = await res.json();

        if (llmSelect && data.encoders && data.encoders.length > 0) {
            llmSelect.innerHTML = "";

            // Add default option
            const defaultOpt = document.createElement('option');
            defaultOpt.value = "default";
            defaultOpt.innerText = "Model default text encoder";
            llmSelect.appendChild(defaultOpt);

            // Add available encoders
            data.encoders.forEach(encoder => {
                const opt = document.createElement('option');
                opt.value = encoder.path;
                // Get the filename for matching with JSON
                const fileName = encoder.path.split('/').pop();
                // Show both display name and filename for easier matching
                opt.innerText = `${encoder.name} (${fileName}) - ${encoder.source} - ${encoder.type.toUpperCase()}`;
                llmSelect.appendChild(opt);
            });

            // Load selected encoder (returns true if at least one was added besides default)
            await loadSelectedEncoder(modelPath);
            return true;
        }
        return false;
    } catch (e) {
        console.error("Failed to load text encoders", e);
        return false;
    }
}

async function initVaeSelector() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();

        if (!vaeSelect) return;
        vaeSelect.innerHTML = "";

        // Add default option
        const defaultOpt = document.createElement('option');
        defaultOpt.value = "";
        defaultOpt.innerText = "Auto (Default)";
        vaeSelect.appendChild(defaultOpt);

        if (data.models && data.models.length > 0) {
            // Filter only VAE models
            const vaeModels = data.models.filter(m => m.category === 'vae');

            vaeModels.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.path;

                // Extract clean name
                let modelName = m.name || m.id;
                if (modelName.includes('/')) {
                    modelName = modelName.split('/').pop();
                }
                modelName = modelName.replace(/\.(safetensors|gguf|bin|pt|ckpt)$/i, '');

                // Get the filename for matching with JSON
                const fileName = m.path.split('/').pop();
                
                // Show both display name and filename for easier matching
                opt.innerText = `${modelName} (${fileName})`;
                vaeSelect.appendChild(opt);
            });
        }
    } catch (e) {
        console.error("Failed to load VAE models", e);
    }
    // Apply the model-compatibility filters (if a model is already selected).
    try { if (window.applyAllVaeFilters) window.applyAllVaeFilters(); } catch (e) { /* ignore */ }
    try { if (window.applyAllSchedFilters) window.applyAllSchedFilters(); } catch (e) { /* ignore */ }
}

// ===========================================================================
// VAE compatibility filter
// ---------------------------------------------------------------------------
// Show only the VAE files that can actually be used with the currently selected
// checkpoint, mirroring the LoRA picker filter. It is a UI convenience:
//   * "Auto" is always kept;
//   * an unknown model or an unknown VAE file is NEVER hidden (we only hide a
//     VAE when we are sure it belongs to a DIFFERENT architecture family);
//   * an incompatible selected VAE falls back to Auto (the workers auto-detect
//     the right one, e.g. wan_2.1 vs wan2.2 for Krea).
// It is applied to every VAE selector: T2I and I2I share #vae-select, and the
// Agent Refine panel has #agent-vae -> all three panels are covered.
// ===========================================================================
window.vaeFilterOn = true;

function _vaeBaseName(p) {
    return String(p || '').replace(/\\/g, '/').split('/').pop().toLowerCase();
}

// Checkpoint -> required VAE family (null = unknown -> show everything).
function vaeFamilyForModel(modelPath) {
    const n = _vaeBaseName(modelPath);
    if (!n) return null;
    if (/flux2|flux[._-]?2|ideogram/.test(n)) return 'flux2';
    if (/flux/.test(n)) return 'flux';                        // FLUX.1 + FLUX.1 Krea
    if (/wan[._-]?2\.2|wan22|krea[._-]?2\.2/.test(n)) return 'wan2.2';
    if (/wan[._-]?2\.1|wan21|krea/.test(n)) return 'wan2.1';  // Krea 2 / Krea Realtime
    if (/qwen.*2[._-]?1/.test(n)) return 'qwen_image_2_1';
    if (/qwen/.test(n)) return 'qwen_image';
    if (/z[._-]?image/.test(n)) return 'zimage';
    if (/pony|sdxl|sd[._-]?xl|sd[._-]?3|sd[._-]?1[._-]?5|sd15/.test(n)) return 'sd';
    return null;
}

// VAE file -> family (null = unknown -> treated as compatible / kept visible).
function vaeFamilyForFile(vaePath) {
    const n = _vaeBaseName(vaePath);
    if (!n) return null;
    if (/wan[._-]?2\.2|wan22/.test(n)) return 'wan2.2';
    if (/wan[._-]?2\.1|wan21/.test(n)) return 'wan2.1';
    if (/flux2/.test(n)) return 'flux2';
    if (/^(ae|ae\.safetensors|ae\.sft|taef1)$/.test(n) || /flux[._-]?(ae|vae)/.test(n)) return 'flux';
    if (/qwen.*2[._-]?1/.test(n)) return 'qwen_image_2_1';
    if (/qwen/.test(n)) return 'qwen_image';
    if (/z[._-]?image/.test(n)) return 'zimage';
    if (/vae[._-]?ft|sdxl|sd[._-]?xl|kl[._-]?f8|taesd|sd15|sd[._-]?1[._-]?5/.test(n)) return 'sd';
    return null;
}

function vaeCompatible(vaePath, modelPath) {
    const mf = vaeFamilyForModel(modelPath);
    if (!mf) return true;            // unknown model -> never hide anything
    const vf = vaeFamilyForFile(vaePath);
    if (!vf) return true;            // unknown VAE -> keep it visible
    return vf === mf;
}

// Hide incompatible options in one VAE <select>. Returns stats.
function applyVaeFilter(selectEl, modelPath) {
    if (!selectEl) return null;
    const on = !!window.vaeFilterOn;
    const mf = on ? vaeFamilyForModel(modelPath) : null;
    let total = 0, shown = 0, reset = false, selectedHidden = false;
    Array.from(selectEl.options).forEach(o => {
        if (o.value === '' || o.value === 'auto' || o.value === 'default') {
            o.hidden = false; o.disabled = false; return;
        }
        total++;
        const ok = !mf || vaeCompatible(o.value, modelPath);
        o.hidden = !ok;
        o.disabled = !ok;
        if (ok) shown++;
        if (!ok && o.selected) selectedHidden = true;
    });
    // If the current VAE is incompatible, fall back to Auto for safety.
    if (selectedHidden) {
        const auto = Array.from(selectEl.options).find(o => o.value === '' || o.value === 'auto' || o.value === 'default');
        if (auto) { auto.selected = true; reset = true; }
        try { selectEl.dispatchEvent(new Event('change')); } catch (e) {}
    }
    return { on, active: !!mf, total, shown, reset };
}

function updateVaeFilterButtons() {
    const on = !!window.vaeFilterOn;
    ['btn-vae-filter', 'btn-agent-vae-filter'].forEach(id => {
        const b = document.getElementById(id);
        if (!b) return;
        b.textContent = on ? '🎯' : '📋';
        b.classList.toggle('filter-on', on);
        b.title = on
            ? 'Filter ON: each VAE list shows only files compatible with the selected model. Click to show ALL.'
            : 'Showing ALL VAE files. Click to filter by the selected model compatibility.';
    });
}

// Apply to every VAE selector (T2I, I2I and Agent Refine).
function applyAllVaeFilters() {
    const t2iSel = document.getElementById('vae-select');
    const t2iModel = document.getElementById('model-select');
    applyVaeFilter(t2iSel, t2iModel && t2iModel.value);

    const aSel = document.getElementById('agent-vae');
    const aModel = document.getElementById('agent-diffusion');
    applyVaeFilter(aSel, aModel && aModel.value);

    updateVaeFilterButtons();
    if (window.refreshAgentModulesSummary) {
        try { window.refreshAgentModulesSummary(); } catch (e) { /* ignore */ }
    }
}
window.applyVaeFilter = applyVaeFilter;
window.applyAllVaeFilters = applyAllVaeFilters;
window.vaeFamilyForModel = vaeFamilyForModel;
window.vaeFamilyForFile = vaeFamilyForFile;
window.vaeCompatible = vaeCompatible;

function toggleVaeFilter() {
    window.vaeFilterOn = !window.vaeFilterOn;
    applyAllVaeFilters();
}
window.toggleVaeFilter = toggleVaeFilter;

// ===========================================================================
// Scheduler-type compatibility filter
// ---------------------------------------------------------------------------
// The scheduler type (the sigma schedule) is model dependent: some types only
// make sense for one architecture (flux / flux2 / logit_normal / ltx2 /
// llada_image). The generic schedules (discrete, karras, exponential, ...) are
// always listed. An unknown model hides nothing. Mirrors the VAE filter and is
// applied to T2I, I2I (shared #scheduler-type-select) and Agent Refine
// (#agent-scheduler-type).
// ===========================================================================
window.schedFilterOn = true;

// Model -> "scheduler family" (null = unknown -> show everything).
function schedFamilyForModel(modelPath) {
    const n = _vaeBaseName(modelPath);
    if (!n) return null;
    if (/ltx/.test(n)) return 'ltx';
    if (/llada/.test(n)) return 'llada';
    if (/ideogram/.test(n)) return 'ideogram';
    if (/flux2|flux[._-]?2/.test(n)) return 'flux2';
    if (/qwen.*2[._-]?1/.test(n)) return 'flux';     // Qwen-Image-2.1 native = flux
    if (/flux/.test(n)) return 'flux';                // FLUX.1 + FLUX.1 Krea
    if (/wan[._-]?2\.2|wan22|krea[._-]?2\.2/.test(n)) return 'wan2.2';
    if (/wan[._-]?2\.1|wan21|krea/.test(n)) return 'wan2.1';
    if (/qwen/.test(n)) return 'qwen_image';
    if (/z[._-]?image/.test(n)) return 'zimage';
    if (/pony|sdxl|sd[._-]?xl|sd[._-]?3|sd[._-]?1[._-]?5|sd15/.test(n)) return 'sd';
    return null;
}

// scheduler type -> the ONLY family it belongs to (null = generic, always shown).
const SCHED_EXCLUSIVE_FAMILY = {
    flux:         'flux',
    flux2:        'flux2',
    logit_normal: 'ideogram',
    ltx2:         'ltx',
    llada_image:  'llada',
};

function schedTypeCompatible(schedValue, modelPath) {
    const fam = schedFamilyForModel(modelPath);
    if (!fam) return true;                            // unknown model -> keep all
    const owner = SCHED_EXCLUSIVE_FAMILY[schedValue];
    if (!owner) return true;                          // generic schedule -> keep
    return owner === fam;
}

// Hide incompatible options in one scheduler-type <select>. Returns stats.
function applySchedFilter(selectEl, modelPath) {
    if (!selectEl) return null;
    const on = !!window.schedFilterOn;
    const fam = on ? schedFamilyForModel(modelPath) : null;
    let total = 0, shown = 0, selectedHidden = false;
    Array.from(selectEl.options).forEach(o => {
        if (o.value === '' || o.value === 'auto' || o.value === 'default') {
            o.hidden = false; o.disabled = false; return;
        }
        total++;
        const ok = !fam || schedTypeCompatible(o.value, modelPath);
        o.hidden = !ok;
        o.disabled = !ok;
        if (ok) shown++;
        if (!ok && o.selected) selectedHidden = true;
    });
    let reset = false;
    // If the current scheduler is incompatible, fall back to the first option
    // (Discrete for T2I/I2I, Auto for Agent).
    if (selectedHidden) {
        const first = selectEl.options[0];
        if (first) { first.selected = true; reset = true; }
        try { selectEl.dispatchEvent(new Event('change')); } catch (e) {}
    }
    return { on, active: !!fam, total, shown, reset };
}

function updateSchedFilterButtons() {
    const on = !!window.schedFilterOn;
    ['btn-sched-filter', 'btn-agent-sched-filter'].forEach(id => {
        const b = document.getElementById(id);
        if (!b) return;
        b.textContent = on ? '🎯' : '📋';
        b.classList.toggle('filter-on', on);
        b.title = on
            ? 'Filter ON: only scheduler types compatible with the selected model are listed. Click to show ALL.'
            : 'Showing ALL scheduler types. Click to filter by model compatibility.';
    });
}

// Apply to every scheduler-type selector (T2I, I2I and Agent Refine).
function applyAllSchedFilters() {
    const sel = document.getElementById('scheduler-type-select');
    const model = document.getElementById('model-select');
    applySchedFilter(sel, model && model.value);

    const aSel = document.getElementById('agent-scheduler-type');
    const aModel = document.getElementById('agent-diffusion');
    applySchedFilter(aSel, aModel && aModel.value);

    updateSchedFilterButtons();
    if (window.refreshAgentModulesSummary) {
        try { window.refreshAgentModulesSummary(); } catch (e) { /* ignore */ }
    }
}
window.applySchedFilter = applySchedFilter;
window.applyAllSchedFilters = applyAllSchedFilters;
window.schedFamilyForModel = schedFamilyForModel;
window.schedTypeCompatible = schedTypeCompatible;

function toggleSchedFilter() {
    window.schedFilterOn = !window.schedFilterOn;
    applyAllSchedFilters();
}
window.toggleSchedFilter = toggleSchedFilter;

async function loadSelectedEncoder(modelPath) {
    try {
        const res = await fetch(`/api/models/text_encoders/selected?model_path=${encodeURIComponent(modelPath)}`);
        const data = await res.json();

        if (data.encoder_path) {
            llmSelect.value = data.encoder_path;
        } else {
            llmSelect.value = "default";
        }
    } catch (e) {
        console.error("Failed to load selected encoder", e);
    }
}

async function selectTextEncoder() {
    const modelPath = modelSelect.value;
    const encoderPath = llmSelect.value === "default" ? null : llmSelect.value;

    try {
        const res = await fetch('/api/models/text_encoders/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: modelPath, encoder_path: encoderPath })
        });
        const data = await res.json();

        if (data.status === 'success') {
            updateStatus('Text encoder selected', 'success');
        }
    } catch (e) {
        console.error("Failed to select text encoder", e);
        updateStatus('Failed to select text encoder', 'error');
    }
}

function toggleAddModel() {
    if (modelAddRow) modelAddRow.classList.toggle('hidden');
}

llmSelect.addEventListener('change', selectTextEncoder);

// Add event listener to model selector to trigger UI validation (e.g. disable standard generate for GGUF)
if (modelSelect) {
    modelSelect.addEventListener('change', async () => {
        await checkModelCapabilities(modelSelect.value);
        // Re-filter the VAE / scheduler-type lists for the newly selected model
        if (window.applyAllVaeFilters) window.applyAllVaeFilters();
        if (window.applyAllSchedFilters) window.applyAllSchedFilters();
        // === Qwen-Image-Edit hint ===
        // Diffusion-only edit model: needs I2I mode + input image. Routing is
        // handled automatically in buildPayload() -> i2i_edit worker.
        const mName = (modelSelect.value || '').split('/').pop() || '';
        const isQwenEdit = window.isQwenImageEditModel ? window.isQwenImageEditModel(modelSelect.value) : /qwen-image-edit|qwen-rapid/i.test(mName);
        if (isQwenEdit) {
            if (window.currentMode === 't2i') {
                window.log('⚠️ Qwen-Image-Edit is an EDIT model: switch to the I2I tab and load an image.', true);
            } else if (window.currentMode === 'i2i') {
                window.log('✏️ Qwen-Image-Edit active: semantic edit via prompt + reference image. The Denoise Strength parameter is ignored and the output keeps the input resolution.', false, true);
            }
        }
        if (window.validateInputs) window.validateInputs();
    });
}

function updateStatus(text, type) {
    if (statusText) { statusText.innerText = text; statusText.className = 'status-text status-' + (type || 'info'); }
}

/**
 * Seleziona automaticamente il modello in base al nome salvato nella history
 * @param {string} modelName - Nome del modello (es: GLM-Image)
 * @returns {boolean} - True if the model was selected, false otherwise
 */
async function selectModelFromHistory(modelName) {
    if (!modelName || !modelSelect) {
        console.warn('selectModelFromHistory: modelName o modelSelect non disponibile');
        return false;
    }

    console.log('selectModelFromHistory: cercando modello', modelName);

    // Pulisci il nome del modello da eventuali estensioni
    const cleanModelName = modelName.replace(/\.(safetensors|gguf|bin|pth|pt)$/i, '').trim();

    let found = false;

    // Cerca per nome (solo l'ultima parte del percorso)
    for (let i = 0; i < modelSelect.options.length; i++) {
        const option = modelSelect.options[i];
        const optionPath = option.value.replace(/\\/g, '/');
        const optionName = optionPath.split('/').pop();

        // Confronta solo il nome, senza estensione
        const cleanOptionName = optionName.replace(/\.(safetensors|gguf|bin|pth|pt)$/i, '');

        if (cleanOptionName === cleanModelName) {
            modelSelect.selectedIndex = i;
            found = true;
            console.log('selectModelFromHistory: trovato modello', cleanModelName, '->', option.value);
            break;
        }
    }

    if (found) {
        // Chiama checkModelCapabilities per aggiornare la UI
        await checkModelCapabilities(modelSelect.value);
        return true;
    } else {
        console.warn('selectModelFromHistory: modello non trovato', modelName);
        return false;
    }
}

// Qwen Image Edit Model Selector
// ===============================

// I2T Model Selector
// ==================

let i2tModelSelect = null;
let i2tModelContainer = null;
let i2tModelStatus = null;

async function loadI2TModels() {
    try {
        const res = await fetch('/api/config/i2t-models');
        const data = await res.json();
        
        if (!i2tModelSelect) {
            i2tModelSelect = document.getElementById('i2t-model-select');
            i2tModelContainer = document.getElementById('i2t-model-container');
            i2tModelStatus = document.getElementById('i2t-model-status');
        }
        
        if (!i2tModelSelect) return;
        
        i2tModelSelect.innerHTML = "";
        
        // Add default option
        const defaultOpt = document.createElement('option');
        defaultOpt.value = "";
        defaultOpt.innerText = "Auto (Use default from settings)";
        i2tModelSelect.appendChild(defaultOpt);
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const opt = document.createElement('option');
                opt.value = model.path;
                opt.innerText = model.name;
                i2tModelSelect.appendChild(opt);
            });
            
            // Load currently selected model
            await loadSelectedI2TModel();
            
            // Show the container
            if (i2tModelContainer) {
                i2tModelContainer.classList.remove('hidden');
            }
        } else {
            const opt = document.createElement('option');
            opt.value = "";
            opt.innerText = "No I2T models found in i2t folder";
            i2tModelSelect.appendChild(opt);
            
            if (i2tModelStatus) {
                i2tModelStatus.innerText = "No models found. Download models to /models/i2t";
                i2tModelStatus.style.color = "#fbbf24";
            }
        }
    } catch (e) {
        console.error("Failed to load I2T models:", e);
        if (i2tModelStatus) {
            i2tModelStatus.innerText = "Error loading models";
            i2tModelStatus.style.color = "#ef4444";
        }
    }
}

async function loadSelectedI2TModel() {
    try {
        const res = await fetch('/api/config/settings');
        const data = await res.json();
        const selectedPath = data.i2t_model_path || "";
        
        if (i2tModelSelect) {
            // Try to find the exact path
            let found = false;
            for (let i = 0; i < i2tModelSelect.options.length; i++) {
                if (i2tModelSelect.options[i].value === selectedPath) {
                    i2tModelSelect.selectedIndex = i;
                    found = true;
                    break;
                }
            }
            
            // If not found but a path exists, try to match by filename only
            if (!found && selectedPath) {
                const filename = selectedPath.split('/').pop();
                for (let i = 0; i < i2tModelSelect.options.length; i++) {
                    const optionPath = i2tModelSelect.options[i].value;
                    if (optionPath && optionPath.endsWith(filename)) {
                        i2tModelSelect.selectedIndex = i;
                        found = true;
                        break;
                    }
                }
            }
        }
    } catch (e) {
        console.error("Failed to load selected I2T model:", e);
    }
}

async function saveI2TModel(modelPath) {
    try {
        const res = await fetch('/api/config/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                i2t_model_path: modelPath || null
            })
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            if (i2tModelStatus) {
                i2tModelStatus.innerText = "✓ Saved";
                i2tModelStatus.style.color = "#4ade80";
                setTimeout(() => {
                    if (i2tModelStatus) {
                        i2tModelStatus.innerText = "";
                    }
                }, 2000);
            }
            return true;
        } else {
            throw new Error(data.detail || "Failed to save");
        }
    } catch (e) {
        console.error("Failed to save I2T model:", e);
        if (i2tModelStatus) {
            i2tModelStatus.innerText = "✗ Error saving";
            i2tModelStatus.style.color = "#ef4444";
        }
        return false;
    }
}

// Initialize I2T selector when the page loads
function initI2TSelector() {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadI2TModels);
    } else {
        loadI2TModels();
    }
}

// Event listener for i2t-model-select
document.addEventListener('DOMContentLoaded', function() {
    const i2tModelSelectEl = document.getElementById('i2t-model-select');
    if (i2tModelSelectEl) {
        i2tModelSelectEl.addEventListener('change', async function() {
            const selectedPath = this.value;
            await saveI2TModel(selectedPath || null);
        });
    }
    
    // Initialize the I2T selector
    initI2TSelector();
});

window.loadI2TModels = loadI2TModels;
window.loadSelectedI2TModel = loadSelectedI2TModel;
window.saveI2TModel = saveI2TModel;

console.log('Qwen Image Edit selector loaded');

window.initModelSelector = initModelSelector;
window.checkModelCapabilities = checkModelCapabilities;
window.toggleAddModel = toggleAddModel;
window.loadTextEncoders = loadTextEncoders;
window.loadSelectedEncoder = loadSelectedEncoder;
window.selectTextEncoder = selectTextEncoder;
window.updateStatus = updateStatus;
window.selectModelFromHistory = selectModelFromHistory;
console.log('Models loaded');
