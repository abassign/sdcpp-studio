// JSONViewer.js - Self-contained JSON viewer with popup dialog
// Auto-injects HTML when script loads

(function () {
    // HTML template
    const html = `
<div id="json-viewer-container">
<style>
#json-viewer-container { font-family: 'Segoe UI', system-ui, sans-serif; }

#jsonDialog {
border: 1px solid #444;
border-radius: 8px;
padding: 0;
width: 70%;
height: 70%;
min-width: 400px;
min-height: 300px;
box-shadow: 0 25px 50px rgba(0,0,0,0.6);
resize: both;
overflow: hidden;
background: #2d2d2d; /* Dark background */
color: #e0e0e0;
font-size: 12px; /* Smaller font */
margin: auto; /* Centering */
}

#jsonDialog::backdrop { background: rgba(0,0,0,0.1); backdrop-filter: none; }

.modal-header {
    background: #383838;
    padding: 8px 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #444;
    cursor: move; /* Indicate draggable */
    user-select: none;
}

.modal-title {
    font-weight: 600;
    color: #fff;
    margin-right: auto; /* Push controls to right */
    padding-left: 5px;
    font-size: 13px;
}

.view-controls button {
    padding: 6px 12px;
    margin-right: 5px;
    cursor: pointer;
    border: 1px solid #555;
    background: #444;
    color: #ddd;
    border-radius: 4px;
    font-size: 11px;
    transition: all 0.2s;
}

.view-controls button:hover { background: #555; }
.view-controls button.active { background: #0d47a1; color: white; border-color: #0d47a1; }

.content-area {
    height: calc(100% - 55px);
    overflow-y: auto;
    padding: 20px;
    box-sizing: border-box;
    background: #252525;
}

.view-simple { line-height: 1.5; color: #ccc; word-wrap: break-word; white-space: pre-wrap; font-family: Consolas, monospace; }
.simple-row { margin-bottom: 4px; padding-left: 15px; border-left: 1px solid #444; }
.simple-key { font-weight: 600; color: #ff79c6; margin-right: 8px; }
.simple-val { color: #8be9fd; }

.view-raw {
    display: none;
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 15px;
    margin: 0;
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 11px;
    white-space: pre-wrap;
    tab-size: 2;
}
.k { color: #9cdcfe; } .s { color: #ce9178; } .n { color: #b5cea8; } .b { color: #569cd6; }

.btn-close { background: #d32f2f; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 11px; }
.btn-close:hover { background: #b71c1c; }
</style>

<dialog id="jsonDialog">
<div class="modal-header">
<span class="modal-title">JSON Data</span>
<div class="view-controls">
<button id="btnSimple" class="active" onclick="JSONViewer.switchView('simple')">Vista Semplice</button>
<button id="btnRaw" onclick="JSONViewer.switchView('raw')">Sorgente JSON</button>
</div>
<button class="btn-close" onclick="document.getElementById('jsonDialog').close()">Chiudi</button>
</div>

<div class="content-area">
<div id="simpleContainer" class="view-simple"></div>
<pre id="rawContainer" class="view-raw"></pre>
</div>
</dialog>
</div>`;

    // Inject HTML when DOM is ready
    function injectHTML() {
        if (document.body) {
            document.body.insertAdjacentHTML('beforeend', html);
            console.log('JSONViewer: HTML injected');
            initJSONViewer();
        } else {
            setTimeout(injectHTML, 10);
        }
    }

    // Start injection
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectHTML);
    } else {
        injectHTML();
    }
})();

// Initialize JSONViewer API after HTML is injected
function initJSONViewer() {
    const dialog = document.getElementById('jsonDialog');
    const simpleCont = document.getElementById('simpleContainer');
    const rawCont = document.getElementById('rawContainer');

    if (!dialog || !simpleCont || !rawCont) {
        console.log('JSONViewer: Waiting for elements...');
        setTimeout(initJSONViewer, 50);
        return;
    }

    // Internal render function
    const _render = (data) => {
        try {
            const jsonObj = (typeof data === 'string') ? JSON.parse(data) : data;

            simpleCont.innerHTML = _generateSimpleHTML(jsonObj);
            rawCont.innerHTML = _colorize(jsonObj);

            JSONViewer.switchView('simple');
            if (!dialog.open) dialog.show(); // Modeless
        } catch (e) {
            console.error("JSONViewer Error:", e);
            alert("Error loading JSON data.");
        }
    };

    const _generateSimpleHTML = (obj, indent = 0) => {
        let html = '';
        for (let key in obj) {
            const val = obj[key];
            const isObj = typeof val === 'object' && val !== null;

            html += `<div class="simple-row" style="margin-left: ${indent}px">`;
            html += `<span class="simple-key">${key}:</span>`;

            if (isObj) {
                html += `<div>${_generateSimpleHTML(val, 15)}</div>`;
            } else {
                html += `<span class="simple-val">${val}</span>`;
            }
            html += `</div>`;
        }
        return html || "<em>(Vuoto)</em>";
    };

    const _colorize = (json) => {
        const str = JSON.stringify(json, null, 4);
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g, function (m) {
                let c = 'n';
                if (/^"/.test(m)) c = /:$/.test(m) ? 'k' : 's';
                else if (/true|false/.test(m)) c = 'b';
                return `<span class="${c}">${m}</span>`;
            });
    };

    // Create the JSONViewer object on window
    window.JSONViewer = {
        openFromFile: async function (url) {
            try {
                const response = await fetch(url);
                const data = await response.json();
                _render(data);
            } catch (err) {
                alert("Impossibile recuperare il file: " + url);
            }
        },

        openFromData: function (data) {
            _render(data);
        },

        switchView: function (type) {
            const isSimple = type === 'simple';
            simpleCont.style.display = isSimple ? 'block' : 'none';
            rawCont.style.display = isSimple ? 'none' : 'block';
            document.getElementById('btnSimple').classList.toggle('active', isSimple);
            document.getElementById('btnRaw').classList.toggle('active', !isSimple);
        },

        isOpen: function () {
            return dialog && dialog.open;
        }
    };

    // --- Drag Logic ---
    const header = dialog.querySelector('.modal-header');
    let isDragging = false;
    let startX, startY, initialLeft, initialTop;

    if (header) {
        header.addEventListener('mousedown', (e) => {
            if (e.target.tagName === 'BUTTON') return;

            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;

            const rect = dialog.getBoundingClientRect();

            // Switch to fixed positioning for dragging
            dialog.style.margin = '0';
            dialog.style.left = rect.left + 'px';
            dialog.style.top = rect.top + 'px';
            dialog.style.position = 'fixed';

            initialLeft = rect.left;
            initialTop = rect.top;

            header.setPointerCapture(e.pointerId);
        });

        header.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            dialog.style.left = (initialLeft + dx) + 'px';
            dialog.style.top = (initialTop + dy) + 'px';
        });

        header.addEventListener('mouseup', (e) => {
            if (isDragging) {
                isDragging = false;
                header.releasePointerCapture(e.pointerId);
                saveState(); // Save on drag end
            }
        });
    }

    // --- Persistence Logic ---
    const STORAGE_KEY = 'json_viewer_state';
    let saveTimeout;

    function saveState() {
        if (saveTimeout) clearTimeout(saveTimeout);
        saveTimeout = setTimeout(() => {
            const rect = dialog.getBoundingClientRect();
            // Only save if visible and has reasonable dimensions
            if (rect.width > 50 && rect.height > 50) {
                const state = {
                    left: dialog.style.left || rect.left + 'px',
                    top: dialog.style.top || rect.top + 'px',
                    width: rect.width + 'px',
                    height: rect.height + 'px',
                    moved: true
                };
                localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
            }
        }, 200);
    }

    function restoreState() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) {
                const state = JSON.parse(raw);
                if (state.moved) {
                    dialog.style.margin = '0';
                    dialog.style.position = 'fixed';
                    dialog.style.left = state.left;
                    dialog.style.top = state.top;
                }
                if (state.width) dialog.style.width = state.width;
                if (state.height) dialog.style.height = state.height;
            }
        } catch (e) {
            console.error("JSONViewer state restore error:", e);
        }
    }

    // Save size on resize (MutationObserver is overkill, resize event on window doesn't catch div resize)
    // Using ResizeObserver is best
    if (window.ResizeObserver) {
        new ResizeObserver(saveState).observe(dialog);
    }

    // Restore on init
    restoreState();

    console.log('JSONViewer: API initialized');
}
