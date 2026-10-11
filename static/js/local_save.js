// ============================================================================
// Local Directory Save - File System Access API + Fallback
// Saves history items (image + JSON + source images) to a local directory.
// Preferred: File System Access API (Chrome/Edge) → saves to chosen folder.
// Fallback: Classic download (all browsers) → downloads files one by one.
// The chosen directory handle is persisted via IndexedDB.
// ============================================================================

const LS_DB_NAME = 'glm_local_save';
const LS_DB_VERSION = 1;
const LS_STORE_NAME = 'directory_handles';

function openLocalSaveDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(LS_DB_NAME, LS_DB_VERSION);
        req.onerror = () => reject(req.error);
        req.onsuccess = () => resolve(req.result);
        req.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(LS_STORE_NAME)) {
                db.createObjectStore(LS_STORE_NAME);
            }
        };
    });
}

async function saveDirectoryHandle(handle) {
    const db = await openLocalSaveDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(LS_STORE_NAME, 'readwrite');
        const store = tx.objectStore(LS_STORE_NAME);
        const req = store.put(handle, 'default_dir');
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
    });
}

async function getDirectoryHandle() {
    const db = await openLocalSaveDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(LS_STORE_NAME, 'readonly');
        const store = tx.objectStore(LS_STORE_NAME);
        const req = store.get('default_dir');
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

// Classic download fallback (works on all browsers)
function fallbackDownload(name, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

async function downloadBlob(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    return await res.blob();
}

async function saveViaDirectoryPicker(files, btn) {
    // files = [{ name, blob }]
    let dirHandle = await getDirectoryHandle();
    let needsPicker = false;

    if (!dirHandle) {
        needsPicker = true;
    } else {
        try {
            const perm = await dirHandle.requestPermission({ mode: 'readwrite' });
            if (perm !== 'granted') needsPicker = true;
        } catch (e) {
            needsPicker = true;
        }
    }

    if (needsPicker) {
        dirHandle = await window.showDirectoryPicker();
        const perm = await dirHandle.requestPermission({ mode: 'readwrite' });
        if (perm !== 'granted') {
            throw new Error('Directory permission denied');
        }
        await saveDirectoryHandle(dirHandle);
    }

    for (const f of files) {
        const fileHandle = await dirHandle.getFileHandle(f.name, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(f.blob);
        await writable.close();
        console.log('[LocalSave] Saved:', f.name);
    }

    if (window.log) window.log(`Saved ${files.length} file(s) to local directory`, false);
}

async function saveViaClassicDownload(files) {
    // Add a small delay between downloads to avoid browser blocking
    for (let i = 0; i < files.length; i++) {
        const f = files[i];
        fallbackDownload(f.name, f.blob);
        console.log('[LocalSave] Downloaded:', f.name);
        if (i < files.length - 1) {
            await new Promise(r => setTimeout(r, 300));
        }
    }
    if (window.log) window.log(`Downloaded ${files.length} file(s)`, false);
}

function serverPathToUrl(path) {
    if (!path) return null;
    if (path.startsWith('/app/')) return path.replace('/app/', '/');
    if (path.startsWith('/')) return path;
    return '/' + path;
}

async function saveHistoryItemToLocal(item, btnElement) {
    const btn = btnElement;
    if (btn) {
        btn.disabled = true;
        btn.innerText = '⏳';
        btn.title = 'Saving...';
    }

    try {
        const filename = item.filename || (item.image ? item.image.split('/').pop() : null);
        if (!filename) {
            throw new Error('No filename for item');
        }

        const filesToSave = []; // { name, blob }

        // 1. Main image
        const imageUrl = item.image || (item.filename ? '/outputs/' + item.filename : null);
        if (imageUrl) {
            const blob = await downloadBlob(imageUrl);
            filesToSave.push({ name: filename, blob });
        }

        // 2. JSON metadata
        const jsonName = filename.replace(/\.[^.]+$/, '.json');
        const jsonUrl = '/outputs/' + jsonName;
        let jsonData = null;
        try {
            const jsonBlob = await downloadBlob(jsonUrl);
            filesToSave.push({ name: jsonName, blob: jsonBlob });
            const jsonText = await jsonBlob.text();
            jsonData = JSON.parse(jsonText);
        } catch (e) {
            console.warn('[LocalSave] JSON not found:', jsonName);
        }

        // 3. Source images from JSON
        if (jsonData) {
            const sourcePaths = new Set();
            const collectImagePaths = (obj) => {
                if (!obj || typeof obj !== 'object') return;
                for (const [key, val] of Object.entries(obj)) {
                    if (typeof val === 'string') {
                        const lower = val.toLowerCase();
                        if (/\.(png|jpg|jpeg|webp|gif|bmp)$/i.test(lower)) {
                            sourcePaths.add(val);
                        }
                    } else if (Array.isArray(val)) {
                        val.forEach(v => {
                            if (typeof v === 'string' && /\.(png|jpg|jpeg|webp)$/i.test(v)) {
                                sourcePaths.add(v);
                            } else if (v && typeof v === 'object' && v.path) {
                                sourcePaths.add(v.path);
                            }
                        });
                    } else if (val && typeof val === 'object') {
                        if (val.path) sourcePaths.add(val.path);
                        collectImagePaths(val);
                    }
                }
            };
            collectImagePaths(jsonData.inputs);
            collectImagePaths(jsonData.params);

            // Remove output image itself
            sourcePaths.delete('/app/outputs/' + filename);
            sourcePaths.delete('/outputs/' + filename);
            sourcePaths.delete(filename);

            for (const srcPath of sourcePaths) {
                try {
                    const url = serverPathToUrl(srcPath);
                    if (!url) continue;
                    const srcBlob = await downloadBlob(url);
                    const srcName = srcPath.split('/').pop() || 'source_image.png';
                    filesToSave.push({ name: srcName, blob: srcBlob });
                } catch (e) {
                    console.warn('[LocalSave] Skip source:', srcPath, e.message);
                }
            }
        }

        if (filesToSave.length === 0) {
            throw new Error('No files to save');
        }

        // Choose strategy based on browser support
        const hasFSA = typeof window.showDirectoryPicker === 'function';
        if (hasFSA) {
            await saveViaDirectoryPicker(filesToSave, btn);
        } else {
            await saveViaClassicDownload(filesToSave);
            if (window.log) {
                window.log('Browser does not support directory picker. Files downloaded individually.', true);
            }
        }

        if (btn) {
            btn.innerText = '✅';
            btn.title = 'Saved!';
            setTimeout(() => {
                btn.innerText = '⬇️';
                btn.title = hasFSA
                    ? 'Save to local directory (image + JSON + sources)'
                    : 'Download files (image + JSON + sources)';
                btn.disabled = false;
            }, 2000);
        }

    } catch (e) {
        console.error('[LocalSave] Error:', e);
        if (btn) {
            btn.innerText = '❌';
            btn.title = e.message || 'Error';
            setTimeout(() => {
                btn.innerText = '⬇️';
                btn.title = typeof window.showDirectoryPicker === 'function'
                    ? 'Save to local directory'
                    : 'Download files';
                btn.disabled = false;
            }, 3000);
        }
        if (window.log) window.log('Save error: ' + e.message, true);
    }
}

window.saveHistoryItemToLocal = saveHistoryItemToLocal;
