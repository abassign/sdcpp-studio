// model_downloader.js - Advanced Model Inspection & Downloading
// =============================================================

// Helper: Format Bytes
function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

function formatDate(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

// Helper: Parse Repo ID or URL
function parseRepoId(input) {
    let repo = input.trim();

    // If it's a URL (Civitai), return as is
    if (repo.startsWith("http")) return repo;

    // Remove protocol and domain
    repo = repo.replace(/^https?:\/\//, '').replace(/^huggingface\.co\//, '');
    // Remove /tree/main, /blob/main etc.
    repo = repo.split('/tree/')[0].split('/blob/')[0];

    // Ensure we take first two parts (User/Repo)
    const parts = repo.split('/').filter(p => p.length > 0);
    if (parts.length >= 2) {
        return `${parts[0]}/${parts[1]}`;
    }
    return repo;
}

let currentRepoId = "";
let inspectModal = null;
let inspectList = null;
let isDragging = false;
let dragOffset = { x: 0, y: 0 };

// Make modal draggable
function initDraggableModal() {
    const modal = document.getElementById('model-inspect-modal');
    if (!modal) return;
    
    const content = modal.querySelector('.modal-content');
    if (!content) return;
    
    // Enable dragging via header
    const header = modal.querySelector('.modal-header');
    if (!header) return;
    
    header.style.cursor = 'move';
    
    header.addEventListener('mousedown', (e) => {
        isDragging = true;
        const rect = content.getBoundingClientRect();
        dragOffset.x = e.clientX - rect.left;
        dragOffset.y = e.clientY - rect.top;
        content.style.transition = 'none';
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const x = e.clientX - dragOffset.x;
        const y = e.clientY - dragOffset.y;
        content.style.left = x + 'px';
        content.style.top = y + 'px';
        content.style.right = 'auto';
        content.style.bottom = 'auto';
    });
    
    document.addEventListener('mouseup', () => {
        isDragging = false;
        content.style.transition = '';
    });
}

// Ensure DOM elements are found (called when initiating action)
function ensureInspectElements() {
    inspectModal = document.getElementById('model-inspect-modal');
    inspectList = document.getElementById('model-inspect-list');
    // Initialize draggable when modal elements are ready
    initDraggableModal();
}

function setDefaultDownloadPath() {
    const subdirInput = document.getElementById('dl-target-subdir');
    if (!subdirInput) return;
    
    // Set default based on current mode
    if (window.currentMode === 'i2t') {
        subdirInput.value = 'GLM-4.1V-9B-Thinking';  // Common default for I2T
    } else if (window.currentMode === 't2i' || window.currentMode === 'i2i') {
        subdirInput.value = '';  // Checkpoints go to root of checkpoints/
    } else {
        subdirInput.value = '';
    }
}

async function inspectModel() {
    ensureInspectElements();
    setDefaultDownloadPath();
    const inputEl = document.getElementById('hf-repo-input');

    if (!inputEl) {
        alert("System Error: Input field 'hf-repo-input' not found.");
        return;
    }

    let raw = inputEl.value;
    if (!raw || raw.trim() === "") return alert("Please enter a Hugging Face Repo ID");

    const repo = parseRepoId(raw);
    currentRepoId = repo;

    if (repo.startsWith("http")) {
        alert("Inspection is only available for Hugging Face repositories. For Civitai, use 'Get Model' directly.");
        return;
    }

    if (inspectList) inspectList.innerHTML = '<div style="text-align:center; padding:20px;">Fetching file list...</div>';
    if (inspectModal) inspectModal.classList.remove('hidden');

    try {
        const res = await fetch('/api/models/inspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repo })
        });
        const data = await res.json();

        if (data.error) {
            if (inspectList) inspectList.innerHTML = `<div style="color:red; padding:10px;">Error: ${data.error}</div>`;
            return;
        }

        if (inspectList) inspectList.innerHTML = "";

        // Handle both old (string[]) and new ({name, size, created_at?}[]) formats
        let validFiles = [];
        if (data.files.length > 0 && typeof data.files[0] === 'string') {
            validFiles = data.files.map(f => ({ name: f, size: 0, created_at: null }));
        } else {
            validFiles = data.files.map(f => ({
                name: f.name,
                size: f.size || 0,
                created_at: f.created_at || null
            }));
        }

        // Filter relevant files (case insensitive)
        const relevant = validFiles.filter(obj => {
            const ext = obj.name.split('.').pop().toLowerCase();
            return ["safetensors", "bin", "json", "yaml", "gguf", "pt", "ckpt"].includes(ext);
        });

        if (relevant.length === 0) {
            if (inspectList) inspectList.innerHTML = '<div style="padding:10px;">No relevant model files found.</div>';
            return;
        }

        relevant.forEach(fileObj => {
            const file = fileObj.name;
            const sizeStr = fileObj.size > 0 ? formatBytes(fileObj.size) : "";
            const dateStr = fileObj.created_at ? formatDate(fileObj.created_at) : "";

            const div = document.createElement('div');
            div.className = "file-list-item";
            // Simple style for the items if css class missing
            div.style.padding = "5px";
            div.style.borderBottom = "1px solid #333";
            div.style.display = "flex";
            div.style.alignItems = "center";

            const label = document.createElement('label');
            label.style.flex = "1";
            label.style.display = "flex";
            label.style.alignItems = "center";
            label.style.cursor = "pointer";

            const checkbox = document.createElement('input');
            checkbox.type = "checkbox";
            checkbox.value = file;
            checkbox.style.marginRight = "10px";

            // Auto-select likely candidates
            if (file.endsWith(".json")) checkbox.checked = true;
            if (file.toLowerCase().includes("fp16") && file.endsWith(".safetensors")) checkbox.checked = true;
            // For GGUF, maybe don't auto-select as there are many quants

            const nameSpan = document.createElement('span');
            nameSpan.innerText = file;

            const sizeSpan = document.createElement('span');
            sizeSpan.className = "size";
            sizeSpan.style.marginLeft = "auto";
            sizeSpan.style.color = "#888";
            sizeSpan.style.fontSize = "0.8em";
            sizeSpan.innerText = sizeStr;

            const dateSpan = document.createElement('span');
            dateSpan.className = "date";
            dateSpan.style.marginLeft = "15px";
            dateSpan.style.color = "#aaa";
            dateSpan.style.fontSize = "0.8em";
            dateSpan.innerText = dateStr;

            label.appendChild(checkbox);
            label.appendChild(nameSpan);

            div.appendChild(label);
            div.appendChild(sizeSpan);
            div.appendChild(dateSpan);

            if (inspectList) inspectList.appendChild(div);
        });

    } catch (e) {
        if (inspectList) inspectList.innerHTML = `<div style="color:red;">Network Error: ${e}</div>`;
    }
}

function closeInspectModal() {
    ensureInspectElements();
    if (inspectModal) inspectModal.classList.add('hidden');
}

async function downloadSelectedFiles() {
    ensureInspectElements();
    if (!inspectList) return;

    const checkboxes = inspectList.querySelectorAll('input[type="checkbox"]:checked');
    const selected = Array.from(checkboxes).map(c => c.value);

    // Get Target Subdirectory if specified
    const subdirInput = document.getElementById('dl-target-subdir');
    const subdir = subdirInput ? subdirInput.value.trim() : null;

    if (selected.length === 0) {
        if (!confirm("No files selected. Download ENTIRE repository?")) return;
        return triggerDownload(currentRepoId, null, subdir);
    }

    triggerDownload(currentRepoId, selected, subdir);
    closeInspectModal();
}

async function triggerDownload(repo, files, subdir = null) {
    const btnGetModel = document.getElementById('btn-get-model');
    if (btnGetModel) {
        btnGetModel.innerText = "Starting...";
        btnGetModel.disabled = true;
    }

    updateStatus('Starting download...', 'info');

    try {
        const payload = { repo_id: repo };
        if (files) payload.filenames = files;
        if (subdir) payload.subdirectory = subdir;

        // Determine target category based on current mode
        if (window.currentMode) {
            const categoryMap = {
                'i2t': 'i2t',
                't2i': 'checkpoints',
                'i2i': 'checkpoints'
            };
            if (categoryMap[window.currentMode]) {
                payload.target_category = categoryMap[window.currentMode];
            }
        }

        const tokenInput = document.getElementById('api-token-input');
        if (tokenInput && tokenInput.value.trim() !== "") {
            payload.token = tokenInput.value.trim();
        }

        const res = await fetch('/api/models/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const d = await res.json();

        if (d.status === 'success') {
            updateStatus(d.message || "Download started in background", 'success');
        } else {
            alert(d.message || d.status);
            updateStatus("Download Error", 'error');
        }
    } catch (e) {
        alert("Error: " + e);
        updateStatus("Network Error", 'error');
    } finally {
        if (btnGetModel) {
            setTimeout(() => {
                btnGetModel.innerText = "Get Model";
                btnGetModel.disabled = false;
            }, 5000);
        }
    }
}

async function downloadModel() {
    const inputEl = document.getElementById('hf-repo-input');
    if (!inputEl) return;

    let raw = inputEl.value;
    if (!raw) return alert("Please enter a Hugging Face Repo ID or Civitai URL");

    const repo = parseRepoId(raw);

    const subdirInput = document.getElementById('dl-target-subdir');
    const subdir = subdirInput ? subdirInput.value.trim() : null;

    triggerDownload(repo, null, subdir);
}

// Export functions to global scope
window.inspectModel = inspectModel;
window.closeInspectModal = closeInspectModal;
window.downloadSelectedFiles = downloadSelectedFiles;
window.downloadModel = downloadModel;

console.log('Model Downloader loaded');
