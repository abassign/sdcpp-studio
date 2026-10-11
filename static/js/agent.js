/**
 * agent.js - Agentic Refine UI.
 *
 * Workspace: left run parameters, center selected image + prompt/critique,
 * right "Progression" rail (ordered iteration images). While this tab is
 * active the STANDARD History panel of the app (the same one used by T2I/I2I)
 * is docked at the far right of the workspace and restored to the main layout
 * when leaving - no custom History UI is built here.
 *
 * Drives workers/agent_refine.py through api/agent.py (SSE stream).
 */
(function () {
    let agentRunning = false;
    let abortController = null;
    let curIter = '';
    let agentStopFlag = false;    // set when the user hits Stop (batch loop exits)
    let variantLabel = '';        // 'V2 · ' while running batch variants, '' otherwise
    let origTileAdded = false;    // at most one 'original' star tile per cleared rail
    let expOpen = true;           // 🧭 Agent experience panel collapsed?
    let exp = { mode: 'idle', iter: null, adherence: null, fidelity: null, combined: null,
                best: null, strategies: [], diagnosis: '', msg: '' };
    let agentOutDir = null;      // finished run folder (disk path)
    let agentFinalized = false;
    let lastPromptText = '';     // most recent PROMPT event payload
    let sel = null;              // current selected tile {url,label,isIter,iterN}
    const stepPrompts = {};      // image url -> prompt used for that iteration

    function $(id) { return document.getElementById(id); }

    // ----------------------------- log / status -----------------------------
    // Follows the T2I/I2I convention: progress lines go to the GLOBAL console
    // (window.log) and the last one is mirrored into the top [STATUS] bar.
    function elog(msg, isErr) {
        if (window.log) window.log(msg, !!isErr, false);
    }

    function setStatus(msg, kind) {
        const st = window.statusText;
        if (!st) return;
        st.innerText = msg;
        st.className = 'status-text status-' + (kind || 'info');
    }

    // ------- Text-Encoder / VAE / Scheduler-type (same lists as T2I/I2I) -------
    // We clone the options from the GLOBAL T2I selects so the Agent panel
    // always offers exactly the same choices, and [Auto] copies the current
    // T2I value. "auto" => we send null => the worker auto-detects.
    function agentFieldValue(id) {
        const v = $(id)?.value;
        return (v && v !== "auto") ? v : null;
    }
    let _agentFieldsTries = 0;
    function populateAgentModelFields() {
        const pairs = [
            ["agent-llm", window.llmSelect],
            ["agent-vae", window.vaeSelect],
            ["agent-scheduler-type", window.schedulerTypeSelect],
        ];
        let ready = 0;
        pairs.forEach(([id, src]) => {
            const dst = $(id);
            if (!dst || !src || src.options.length <= 1) return;
            ready++;
            const cur = dst.value;
            dst.innerHTML = '<option value="auto">Auto-Detect</option>';
            Array.from(src.options).forEach(o => {
                if (o.value === "" || o.value === "auto" || o.value === "default") return;
                const opt = document.createElement("option");
                opt.value = o.value;
                opt.textContent = o.textContent;
                dst.appendChild(opt);
            });
            dst.value = (cur && Array.from(dst.options).some(o => o.value === cur)) ? cur : "auto";
        });
        if (ready < pairs.length && _agentFieldsTries < 12) {
            _agentFieldsTries++;
            setTimeout(populateAgentModelFields, 500);
        }
        refreshAgentModulesSummary();
        // Keep the VAE / scheduler-type lists filtered to the Agent generator
        if (window.applyAllVaeFilters) window.applyAllVaeFilters();
        if (window.applyAllSchedFilters) window.applyAllSchedFilters();
    }
    // [Auto] — uses the project preferences "database" (data/preferences.json via
    // /api/preferences/frequent): the most-used VAE / text-encoder / sampler for the
    // Generator (T2I) selected here. Same source as the T2I/I2I "✨ Auto" button.
    // Falls back to copying the T2I/I2I selects when the model has no saved stats.
    function agentModelKey() {
        const gen = $("agent-diffusion")?.value || "";
        return gen ? gen.replace(/\\/g, "/").split("/").pop() : "";
    }
    async function fetchFrequentForMode(mode, model) {
        try {
            const url = "/api/preferences/frequent?mode=" + encodeURIComponent(mode) +
                        (model ? "&model=" + encodeURIComponent(model) : "");
            const r = await fetch(url);
            if (!r.ok) return {};
            const d = await r.json();
            return (d && d.values) ? d.values : {};
        } catch (e) { return {}; }
    }
    async function fetchAgentFrequent(model) {
        // Agent-specific stats (mode "agent_refine") win; the fields the agent has
        // never recorded yet are back-filled from the C++ T2I stats. The classic
        // modes only cover older data.
        const agentVals = await fetchFrequentForMode("agent_refine", model);
        if (Object.keys(agentVals).length) {
            elog("\u2728 Auto: stats mode=agent_refine model=" + (model || "?") +
                 " (" + Object.keys(agentVals).join(", ") + ")");
            const base = await fetchFrequentForMode("t2i_cpp", model);
            for (const k of Object.keys(base)) {
                if (agentVals[k] === undefined) agentVals[k] = base[k];
            }
            return agentVals;
        }
        for (const mode of ["t2i_cpp", "t2i", "i2i_cpp", "i2i_edit"]) {
            const v = await fetchFrequentForMode(mode, model);
            if (Object.keys(v).length) {
                elog("\u2728 Auto: stats mode=" + mode + " model=" + (model || "?") +
                     " (" + Object.keys(v).join(", ") + ")");
                return v;
            }
        }
        return {};
    }
    function setAgentSelectByValue(id, wanted) {
        const sel = $(id);
        if (!sel || wanted === undefined || wanted === null || wanted === "") return false;
        const w = String(wanted);
        const hit = Array.from(sel.options).find(o =>
            o.value === w ||
            (o.value && w.indexOf("/") !== -1 && o.value.split("/").pop() === w.split("/").pop()) ||
            (o.textContent || "").trim() === w);
        if (!hit) return false;
        sel.value = hit.value;
        return true;
    }
    // Register the finished run in the preferences DB under its own mode, so the
    // choices made here never mix with the T2I/I2I statistics.
    function agentReportStats() {
        if (!window.Preferences || !window.Preferences.updateStatsAfterGeneration) return;
        const val = (id) => { const el = $(id); return el ? el.value : null; };
        const params = {
            scheduler: val("agent-scheduler"),
            scheduler_type: agentFieldValue("agent-scheduler-type"),
            vae_path: agentFieldValue("agent-vae"),
            text_encoder: agentFieldValue("agent-llm"),
            steps: (v => Number.isFinite(v) ? v : null)(parseInt(val("agent-steps"), 10)),
            guidance: (v => Number.isFinite(v) ? v : null)(parseFloat(val("agent-cfg"))),
        };
        Object.keys(params).forEach(k => {
            if (params[k] === null || params[k] === undefined || params[k] === "") delete params[k];
        });
        if (!Object.keys(params).length) return;
        const gen = val("agent-diffusion");
        window.Preferences.updateStatsAfterGeneration("agent_refine", gen, params);
        elog("\U0001F4CA Preferences: run recorded (mode agent_refine" +
             (gen ? ", " + String(gen).split("/").pop() : "") + ")");
    }
    window.agentAutoModelFields = async function () {
        populateAgentModelFields();               // refresh the option lists first
        const model = agentModelKey();
        const values = await fetchAgentFrequent(model);
        const applied = [];
        if (setAgentSelectByValue("agent-vae", values.vae_path || values.vae)) applied.push("VAE");
        if (setAgentSelectByValue("agent-llm", values.text_encoder)) applied.push("Text encoder");
        if (setAgentSelectByValue("agent-scheduler-type", values.scheduler_type)) applied.push("Scheduler type");
        if (setAgentSelectByValue("agent-scheduler", values.scheduler)) applied.push("Sampler");
        if (!applied.length) {
            [["agent-llm", "llmSelect"], ["agent-vae", "vaeSelect"],
             ["agent-scheduler-type", "schedulerTypeSelect"]].forEach(([id, g]) => {
                const dst = $(id), src = window[g];
                if (!dst || !src) return;
                dst.value = (src.value && Array.from(dst.options).some(o => o.value === src.value)) ? src.value : "auto";
            });
            elog('\u2728 Auto: no saved stats for "' + (model || "?") + '" \u2014 copied from the T2I/I2I panel.');
        } else {
            elog("\u2728 Auto (" + (model || "?") + "): " + applied.join(", ") + " from the preferences DB.");
        }
        refreshAgentModulesSummary();
        return applied;
    };
    // ------- Collapsible "External modules" panel -------
    // Collapsed it shows the 3 selected values; the chevron expands the editors.
    let _agentModulesOpen = false;
    function agentModuleLabel(id, fallback) {
        const sel = $(id);
        if (!sel) return fallback;
        const opt = sel.options[sel.selectedIndex];
        return ((opt ? opt.textContent : sel.value) || fallback).trim();
    }
    function refreshAgentModulesSummary() {
        const set = (id, val) => { const el = $(id); if (el) el.textContent = val; };
        set("agent-sum-llm",   agentModuleLabel("agent-llm", "Auto-Detect"));
        set("agent-sum-vae",   agentModuleLabel("agent-vae", "Auto-Detect"));
        set("agent-sum-stype", agentModuleLabel("agent-scheduler-type", "Auto-Detect"));
    }
    window.refreshAgentModulesSummary = refreshAgentModulesSummary;
    window.agentToggleModules = function (force) {
        const body = $("agent-modules-body");
        const sum = $("agent-modules-summary");
        const chev = $("agent-modules-chevron");
        if (!body) return;
        _agentModulesOpen = (typeof force === "boolean") ? force : !_agentModulesOpen;
        body.classList.toggle("hidden", !_agentModulesOpen);
        if (sum) sum.classList.toggle("hidden", _agentModulesOpen);
        if (chev) {
            chev.textContent = _agentModulesOpen ? "▼" : "▶";
            chev.setAttribute("aria-expanded", String(_agentModulesOpen));
        }
        if (_agentModulesOpen) refreshAgentModulesSummary();
    };
    function initAgentModulesPanel() {
        ["agent-llm", "agent-vae", "agent-scheduler-type"].forEach(id => {
            const sel = $(id);
            if (sel) sel.addEventListener("change", refreshAgentModulesSummary);
        });
        window.agentToggleModules(false);   // start collapsed
        refreshAgentModulesSummary();
    }

    populateAgentModelFields();
    initAgentModulesPanel();

    // ------- LoRA (same slots/picker as T2I/I2I, own container) -------
    function agentLoraContainer() { return document.getElementById("agent-lora-container"); }
    function getAgentLoraConfigs() {
        const container = agentLoraContainer();
        if (!container) return [];
        const out = [];
        container.querySelectorAll(".lora-slot").forEach(slot => {
            const sel = slot.querySelector("select.lora-native-select");
            const w = slot.querySelector('input[type="number"].lora-weight');
            if (sel && sel.value) out.push({ path: sel.value, weight: parseFloat(w && w.value) || 1.0 });
        });
        return out;
    }
    function addAgentLoraSlot() {
        const container = agentLoraContainer();
        if (!container || typeof window.addLoraSlot !== "function") return;
        const prev = window.loraContainer;
        window.loraContainer = container;   // lora.js appends to this global
        try { window.addLoraSlot(); } finally { window.loraContainer = prev; }
        refreshAgentLoraStatus();
    }
    function refreshAgentLoraStatus() {
        const st = document.getElementById("agent-lora-filter-status");
        if (!st) return;
        const total = (window.loraFiles || []).length;
        if (!total) { st.textContent = ""; return; }
        st.textContent = window.loraFilterOn ? ("filter ON · " + total + " loras") : ("all " + total);
    }
    // Convert the two LoRA shapes used by the app ({path,weight}[] legacy array,
    // or {path, files:[[name,weight,active], ...]} written by T2I/I2I and now by
    // the ARF manifest) into the [{path, weight, active}] form of the slots.
    function agentLorasToRestore(raw) {
        if (!raw) return [];
        if (Array.isArray(raw)) {
            return raw.map(l => ({
                path: l.path || '',
                weight: (l.weight !== undefined ? l.weight
                    : (l.strength !== undefined ? l.strength : 1.0)),
                active: l.active
            })).filter(l => l.path);
        }
        if (raw.files && Array.isArray(raw.files)) {
            const base = String(raw.path || '').replace(/\/+$/, '');
            return raw.files.map(f => ({
                path: base ? base + '/' + f[0] : f[0],
                weight: f[1],
                active: f[2]
            })).filter(l => l.path);
        }
        return [];
    }
    // Restore the Agent LoRA slots to exactly the set used for a run/step opened
    // from History/ARF (the selected image + prompt already restore).
    function restoreAgentLoras(raw) {
        const container = agentLoraContainer();
        if (!container) return;
        const list = agentLorasToRestore(raw);
        container.innerHTML = '';
        if (!list.length) { refreshAgentLoraStatus(); return; }
        list.forEach(() => addAgentLoraSlot());
        // Same deferral trick as history.js: give the shared scanner a tick to
        // populate <option>s; a missing option is appended on the fly so the
        // saved LoRA is never silently dropped.
        setTimeout(() => {
            const slots = container.querySelectorAll('.lora-slot');
            list.forEach((lora, idx) => {
                const slot = slots[idx];
                if (!slot) return;
                const sel = slot.querySelector('select.lora-native-select');
                const w = slot.querySelector('input[type="number"].lora-weight');
                if (sel) {
                    let found = false;
                    for (let i = 0; i < sel.options.length; i++) {
                        if (sel.options[i].value === lora.path) { sel.selectedIndex = i; found = true; break; }
                    }
                    if (!found) {
                        const opt = document.createElement('option');
                        opt.value = lora.path;
                        opt.textContent = lora.path;
                        sel.appendChild(opt);
                        sel.value = lora.path;
                    }
                    sel.dispatchEvent(new Event('change'));
                }
                if (w && lora.weight !== undefined && lora.weight !== null) w.value = lora.weight;
            });
        }, 0);
        refreshAgentLoraStatus();
    }
    // Put the run's reference images back into the SHARED I2T/I2I slots:
    // slot 1 = the run's ORIGINAL (anchor, role init), slot 2+ = its extras.
    // Slots are created/removed through the upload.js API so its internal id
    // counter stays consistent (the legacy file-input mapping depends on it).
    function restoreAgentRefImages(list, anchorUrl) {
        if (typeof imageSlots === 'undefined') return;
        const extras = (Array.isArray(list) ? list : []).filter(x => x && x.path);
        const anchorName = String(anchorUrl || '').split('/').pop();
        const wanted = [];
        if (anchorUrl) wanted.push({ path: anchorUrl, role: 'init' });
        extras.forEach(e => {
            const name = String(e.path).split('/').pop();
            if (anchorName && name === anchorName) return;   // already the init slot
            wanted.push({ path: e.path, role: e.role || 'ref' });
        });
        if (!wanted.length) return;
        const toUrl = (p) => {
            const s = String(p || '');
            return (s.startsWith('/outputs/') || s.startsWith('/uploads/') || s.startsWith('http'))
                ? s : '/outputs/' + s.split('/').pop();
        };
        try {
            // Remove extra slots through the app API (keeps _nextSlotId sane).
            while (imageSlots.length > 1) {
                const last = imageSlots[imageSlots.length - 1];
                if (typeof window.removeImageSlot === 'function') window.removeImageSlot(last.id);
                else imageSlots.pop();
            }
            if (!imageSlots.length) {
                if (typeof window.addImageSlot === 'function') window.addImageSlot('init');
                else imageSlots.push({ id: 1, role: 'init', path: '', file: null, previewDataUrl: '', filename: '' });
            }
            // Grow with the app API so each new slot gets a valid unique id.
            while (imageSlots.length < wanted.length && typeof window.addImageSlot === 'function') {
                if (window.addImageSlot('ref') === null) break;
            }
            wanted.forEach((r, idx) => {
                const slot = imageSlots[idx];
                if (!slot) return;
                slot.path = r.path;
                slot.role = r.role || (idx === 0 ? 'init' : 'ref');
                slot.previewDataUrl = toUrl(r.path);
                slot.filename = String(r.path).split('/').pop();
            });
            // stop renderSlots() from overriding the restored roles with the
            // roles previously saved in localStorage for agent mode
            try {
                localStorage.setItem('slot1_role_agent', imageSlots[0].role);
                if (imageSlots[1]) localStorage.setItem('slot2_role_agent', imageSlots[1].role);
            } catch (e) { /* ignore */ }
            if (typeof window.renderSlots === 'function') window.renderSlots();
            // Make the automatic run-reference restore explicit: otherwise the
            // extra slots look like images the user never added.
            if (wanted.length) {
                setStatus('🖼️ ' + wanted.length + ' images restored from the run into the reference slots', 'info');
                elog('🖼️ restoreAgentRefImages: ' + wanted.length + ' images loaded from the run', false);
            }
            const hp = $('agent-image-path');
            if (hp) hp.value = anchorUrl || (wanted[0] && wanted[0].path) || '';
        } catch (e) { console.error('restoreAgentRefImages', e); }
    }
    function initAgentLora() {
        // The shared reader lives in lora.js and scans the WHOLE document; scope
        // it to the T2I container so Agent LoRA slots never leak into T2I/I2I.
        (function scopeSharedLoraReader() {
            const orig = window.getLoraConfigs;
            if (typeof orig !== "function" || orig.__agentScoped) return;
            const scoped = function (scope) {
                const root = scope || window.loraContainer || document;
                const out = [];
                root.querySelectorAll(".lora-slot").forEach(slot => {
                    const sel = slot.querySelector("select.lora-native-select");
                    const w = slot.querySelector('input[type="number"].lora-weight');
                    if (sel && sel.value) out.push({ path: sel.value, weight: parseFloat(w && w.value) || 1.0 });
                });
                return out;
            };
            scoped.__agentScoped = true;
            window.getLoraConfigs = scoped;
        })();
        const add = document.getElementById("agent-btn-add-lora");
        if (add) add.addEventListener("click", (e) => { e.preventDefault(); addAgentLoraSlot(); });
        const scan = document.getElementById("agent-btn-scan");
        if (scan) scan.addEventListener("click", async (e) => {
            e.preventDefault();
            if (typeof window.scanLoras === "function") { await window.scanLoras(); }
            refreshAgentLoraStatus();
        });
        const filt = document.getElementById("agent-btn-lora-filter");
        if (filt) filt.addEventListener("click", (e) => {
            e.preventDefault();
            if (typeof window.toggleLoraFilter === "function") { window.toggleLoraFilter(); }
            refreshAgentLoraStatus();
        });
        const container = agentLoraContainer();
        if (container && !container.querySelector(".lora-slot")) addAgentLoraSlot();
        refreshAgentLoraStatus();
    }
    initAgentLora();

    // The interpreter dropdown uses the SAME model list as the I2T panel
    // (/api/config/i2t-models). Entries without a vision projector are shown
    // disabled: they cannot read the image, so they cannot be an interpreter.
    async function loadAgentInterpreterModels() {
        const iSel = $('agent-interpreter');
        if (!iSel) return;
        try {
            const res = await fetch('/api/config/i2t-models');
            const data = await res.json();
            const keep = iSel.value;
            iSel.innerHTML = '<option value="">Auto (none \u2014 uses the Critic)</option>';
            (data.models || []).forEach(m => {
                const o = document.createElement('option');
                o.value = m.path;
                const vis = m.vision !== false;
                o.innerText = (vis ? '' : '\u26a0\ufe0f ') + m.name + (vis ? '' : ' \u2014 no mmproj');
                if (!vis) o.disabled = true;
                iSel.appendChild(o);
            });
            if (keep) iSel.value = keep;
        } catch (e) { console.error('loadAgentInterpreterModels', e); }
    }
    window.loadAgentInterpreterModels = loadAgentInterpreterModels;

    // ----------------------------- model lists -----------------------------
    async function loadCritics() {
        try {
            const res = await fetch('/api/agent/critics');
            const data = await res.json();
            const selEl = $('agent-critic');
            if (!selEl) return;
            selEl.innerHTML = '';
            const crits = data.critics || [];
            if (!crits.length) {
                selEl.innerHTML = '<option value="">No critic model available</option>';
                return;
            }
            crits.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.innerText = c;
                selEl.appendChild(opt);
            });
            // The server returns the list already ordered best-first and names the
            // one to preselect (CRITIC_PREFERENCE in workers/agent_refine.py:
            // SenseNova-SI -> Qwen3.6-35B-A3B -> Qwen3-VL-8B -> ...).
            const pref = (data.preferred && crits.indexOf(data.preferred) !== -1)
                ? data.preferred
                : crits[0];
            selEl.value = pref;
            refreshAgentCapabilities();
        } catch (e) { console.error('loadCritics', e); }
    }

    // The iterations field defaults to the NUMBER of resolutions in the chosen
    // schedule (a 512->768->1024->1152 ladder with 3 iterations would silently
    // skip the last size). Extra iterations repeat the last resolution.
    window.agentScheduleChanged = function (sel) {
        sel = sel || $('agent-schedule');
        const it = $('agent-iter');
        if (!sel || !it) return;
        const n = (sel.value || '').split(',').filter(s => s.trim()).length;
        if (!n) return;
        const max = parseInt(it.max, 10) || 12;
        it.value = String(Math.min(Math.max(n, 1), max));
    };

    // ------- Resolution: aspect ratio + size ladder -------
    // The ladder sent to the worker is derived from the chosen ratio and from the
    // sizes currently toggled on (512/768/1024 by default). The "size" is the
    // longer side, so 3:4 @1024 -> 768x1024 and 16:9 @1024 -> 1024x576.
    let _agentRatio = "1:1";
    let _agentSizes = [512, 768, 1024];
    const AGENT_SNAP = 32;            // same grid as the width/height sliders

    function agentDims(size) {
        const parts = _agentRatio.split(":");
        const rw = Number(parts[0]) || 1;
        const rh = Number(parts[1]) || 1;
        let w, h;
        if (rw === rh) { w = h = size; }
        else if (rw < rh) { h = size; w = size * rw / rh; }   // portrait
        else { w = size; h = size * rh / rw; }                 // landscape
        const snap = (v) => Math.max(AGENT_SNAP, Math.round(v / AGENT_SNAP) * AGENT_SNAP);
        return [snap(w), snap(h)];
    }

    function agentRecomputeSchedule() {
        const parts = [..._agentSizes].sort((a, b) => a - b)
            .map(sz => { const d = agentDims(sz); return d[0] + "x" + d[1]; });
        const hidden = $("agent-schedule");
        if (hidden) hidden.value = parts.join(",");
        const prev = $("agent-sched-preview");
        if (prev) prev.textContent = parts.length
            ? "\u2192 " + parts.join("   \u2192   ")
            : "select at least one resolution";
        if (window.agentScheduleChanged) window.agentScheduleChanged();
    }

    function agentSyncResolutionButtons() {
        document.querySelectorAll("#agent-ratio-row button").forEach(b => {
            b.classList.toggle("active", b.dataset.ratio === _agentRatio);
        });
        document.querySelectorAll("#agent-size-row button").forEach(b => {
            b.classList.toggle("active", _agentSizes.indexOf(parseInt(b.dataset.size, 10)) !== -1);
        });
    }

    window.agentSetRatio = function (r) {
        if (!r || !/^\d+:\d+$/.test(r)) return;
        _agentRatio = r;
        agentSyncResolutionButtons();
        agentRecomputeSchedule();
    };

    window.agentToggleSize = function (size) {
        size = parseInt(size, 10);
        if (!size) return;
        const i = _agentSizes.indexOf(size);
        if (i === -1) _agentSizes.push(size);
        else if (_agentSizes.length > 1) _agentSizes.splice(i, 1);   // never empty
        agentSyncResolutionButtons();
        agentRecomputeSchedule();
    };

    // Restore ratio + sizes from a schedule string (e.g. a saved run).
    window.agentApplySchedule = function (str) {
        const parts = String(str || "").split(",").map(x => x.trim()).filter(Boolean);
        if (!parts.length) return;
        const sizes = [];
        let ratio = null;
        parts.forEach(pt => {
            const m = pt.match(/^(\d+)\s*x\s*(\d+)$/i);
            if (!m) return;
            const w = parseInt(m[1], 10), h = parseInt(m[2], 10);
            sizes.push(Math.max(w, h));
            if (!ratio) {
                const gcd = (a, b) => (b ? gcd(b, a % b) : a);
                const d = gcd(w, h) || 1;
                ratio = (w / d) + ":" + (h / d);
            }
        });
        if (sizes.length) _agentSizes = sizes;
        if (ratio) _agentRatio = ratio;
        agentSyncResolutionButtons();
        agentRecomputeSchedule();
    };

    (function initAgentResolution() {
        const row = $("agent-ratio-row");
        if (!row) return;                       // panel not present
        agentSyncResolutionButtons();
        agentRecomputeSchedule();
    })();

    async function loadDiffusionModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            const selEl = $('agent-diffusion');
            if (!selEl) return;
            selEl.innerHTML = '';
            const models = (data.models || []).filter(m => {
                const isCk = (m.category === 'checkpoint' || !m.category);
                if (!isCk) return false;
                const b = (m.path || '').toLowerCase();
                if (/_uncond|mmproj/.test(b)) return false;
                if (!/(\.gguf|\.safetensors|\.sft)$/.test(b)) return false;
                return true;
            });
            if (!models.length) {
                selEl.innerHTML = '<option value="">No checkpoint model available</option>';
                return;
            }
            const groups = {};
            models.forEach(m => {
                const parts = (m.path || '').replace(/\\/g, '/').split('/');
                let folder = 'root';
                const idx = parts.indexOf('checkpoints');
                if (idx + 1 < parts.length) folder = parts[idx + 1];
                const name = (m.name || m.path.split('/').pop()).replace(/\.(safetensors|gguf|sft)$/i, '');
                (groups[folder] = groups[folder] || []).push({ path: m.path, name });
            });
            Object.keys(groups).sort().forEach(folder => {
                const g = groups[folder];
                if (g.length === 1) {
                    const opt = document.createElement('option');
                    opt.value = g[0].path;
                    opt.innerText = folder + ' — ' + g[0].name;
                    selEl.appendChild(opt);
                } else {
                    const og = document.createElement('optgroup');
                    og.label = folder;
                    g.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m.path;
                        opt.innerText = m.name;
                        og.appendChild(opt);
                    });
                    selEl.appendChild(og);
                }
            });
            const opts = Array.from(selEl.options);
            const fav = opts.find(o => /z_image_turbo-Q8_0|z-image-turbo-Q8_0|z_image_turbo-Q6_K/i.test(o.value));
            if (fav) selEl.value = fav.value;
            refreshAgentCapabilities();
        } catch (e) { console.error('loadDiffusionModels', e); }
    }

    // ----------------------------- edit capability gating -----------------------------
    // The two checkboxes are clickable ONLY when the selected models can honour
    // them; otherwise they are greyed out with the reason shown (the user must
    // not have to read the log to discover a missing capability).
    let agentCaps = null;

    function _hasAgentRefImage() {
        const p = $('agent-image-path')?.value || '';
        return !!p;
    }

    function _setCapToggle(id, enabled, reason, titleWhenOn) {
        const input = $(id);
        if (!input) return;
        const wrap = input.closest('label');
        input.disabled = !enabled;
        if (wrap) {
            wrap.classList.toggle('cap-disabled', !enabled);
            wrap.title = enabled ? (titleWhenOn || '') : (reason || 'Not supported by the selected model.');
        }
    }

    async function refreshAgentCapabilities() {
        const diff = $('agent-diffusion')?.value || '';
        const crit = $('agent-critic')?.value || '';
        const warn = $('agent-cap-warning');
        let caps = null;
        try {
            const res = await fetch('/api/agent/capabilities?diffusion=' + encodeURIComponent(diff) +
                                    '&critic=' + encodeURIComponent(crit));
            caps = await res.json();
        } catch (e) { console.error('refreshAgentCapabilities', e); }
        agentCaps = caps;
        const gen = (caps && caps.generator) || { edit: false, reason: 'Capability check unavailable.' };
        const cri = (caps && caps.critic) || { vision: false, reason: 'Capability check unavailable.' };

        // 1) Interpret image (critic vision)
        _setCapToggle('agent-use-image', !!cri.vision, cri.reason,
                      'Send the reference image to the VLM critic so it can see what it judges.');
        const useImg = $('agent-use-image');
        if (useImg && !cri.vision) useImg.checked = false;

        // 2) Edit from reference (generator capability + vision encoder)
        _setCapToggle('agent-edit-ref', !!gen.edit, gen.reason,
                      'Pass the reference image to the generator so every iteration is a real EDIT (sd.cpp -r).');
        const editCb = $('agent-edit-ref');
        if (editCb && !gen.edit) editCb.checked = false;

        // 3) Anchor selector (only meaningful when edit is available)
        const anchor = $('agent-edit-anchor');
        if (anchor) {
            if (Array.isArray(gen.anchor_modes) && gen.anchor_modes.length) {
                const cur = anchor.value || 'hybrid';
                anchor.innerHTML = '';
                gen.anchor_modes.forEach(m => {
                    const o = document.createElement('option');
                    o.value = m.id; o.innerText = m.label;
                    anchor.appendChild(o);
                });
                anchor.value = gen.anchor_modes.some(m => m.id === cur) ? cur : gen.anchor_modes[0].id;
            }
            anchor.disabled = !gen.edit;
            anchor.classList.toggle('cap-disabled', !gen.edit);
        }

        // 4) Clear, prominent warning banner (not only in the log)
        const msgs = [];
        if (!cri.vision && cri.reason) msgs.push('🖼️ ' + cri.reason);
        if (!gen.edit && gen.reason) msgs.push('✏️ ' + gen.reason);
        if (gen.edit && gen.edit_only && !_hasAgentRefImage())
            msgs.push('✏️ This generator is <b>edit-only</b>: it needs an Init image, otherwise generation fails.');
        if (warn) {
            if (msgs.length) { warn.innerHTML = msgs.join('<br>'); warn.classList.remove('hidden'); }
            else { warn.innerHTML = ''; warn.classList.add('hidden'); }
        }
    }
    window.refreshAgentCapabilities = refreshAgentCapabilities;

    // ----------------------------- reference image -----------------------------
    // The reference is the FIRST of the shared I2T/I2I slots (role 🎯 Init
    // Image): there is no separate Agent-specific upload box anymore. The
    // hidden agent-image-path mirrors it for the worker/run payload.
    function refUrl(p) {
        const s = String(p || '');
        if (!s) return '';
        // already a browser-usable URL (data/blob/http/outputs/uploads)
        if (s.startsWith('data:') || s.startsWith('blob:') || s.startsWith('http') ||
            s.startsWith('/outputs/') || s.startsWith('/uploads/')) return s;
        // absolute container disk path -> web path
        if (s.startsWith('/app/outputs/')) return s.replace('/app/outputs/', '/outputs/');
        if (s.startsWith('/app/uploads/')) return s.replace('/app/uploads/', '/uploads/');
        // relative path (keep subfolders, do NOT take only the basename)
        return '/outputs/' + s.replace(/^\/+/, '');
    }
    function setRefSlot(url) {
        if (!url || typeof imageSlots === 'undefined') return;
        try {
            if (!imageSlots.length) {
                imageSlots.push({ id: 1, role: 'init', path: '', file: null, previewDataUrl: '', filename: '' });
            }
            const s = imageSlots[0];
            s.role = 'init';
            s.path = url;
            s.previewDataUrl = refUrl(url);
            s.filename = String(url).split('/').pop();
            try { localStorage.setItem('slot1_role_agent', 'init'); } catch (e) { /* ignore */ }
            if (typeof window.renderSlots === 'function') window.renderSlots();
        } catch (e) { console.error('setRefSlot', e); }
    }
    function showRefPreview(url) {
        setRefSlot(url);
        markRefTile(url);
    }

    // url must be browser-resolvable (/outputs/...) OR a disk path; the worker accepts both
    function setRefByUrl(url) {
        if (!url) return;
        const hp = $('agent-image-path');
        if (hp) hp.value = url;
        showRefPreview(url);
        setStatus('Reference image: ' + String(url).split('/').pop(), 'success');
    }
    window.agentUseAsRef = setRefByUrl;

    // Translate the GOAL to English with the LOCAL chat model (backend
    // /api/agent/translate). Toggle: clicking again restores the original text.
    window.translateAgentGoal = async function () {
        const ta = $('agent-goal');
        const btn = $('btn-goal-translate');
        const st = $('goal-translate-status');
        if (!ta) return;
        const current = ta.value;

        // Revert if the box still holds the last translation
        if (ta.dataset.goalOriginal &&
            current.trim() === (ta.dataset.goalTranslated || '').trim()) {
            ta.value = ta.dataset.goalOriginal;
            delete ta.dataset.goalOriginal;
            delete ta.dataset.goalTranslated;
            if (btn) btn.textContent = '🌐 Traduci in English';
            if (st) st.textContent = '↩︎ ripristinato (IT)';
            setStatus('Goal ripristinato in italiano', 'info');
            return;
        }

        const text = current.trim();
        if (!text) { setStatus('Write a goal to translate first.', 'error'); return; }

        // Model: prefer the interpreter, else the critic (the backend falls
        // back to the default critic when empty).
        const model = (typeof window.getTranslateModel === 'function')
            ? window.getTranslateModel()
            : ($('agent-interpreter')?.value || $('agent-critic')?.value || '');
        const oldLabel = btn ? btn.textContent : '';
        if (btn) { btn.disabled = true; btn.textContent = '⏳ traduco…'; }
        if (st) st.textContent = 'avvio del modello locale…';
        try {
            const resp = await fetch('/api/agent/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, model })
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.detail || ('HTTP ' + resp.status));
            const en = (data.text || '').trim();
            if (!en) throw new Error('risposta vuota dal modello');
            ta.dataset.goalOriginal = current;
            ta.dataset.goalTranslated = en;
            ta.value = en;
            if (btn) btn.textContent = '↩︎ Ripristina (IT)';
            if (st) st.textContent = 'EN ✓' + (data.model ? ' (' + data.model + ')' : '');
            setStatus('🎯 Goal translated to English', 'success');
            elog('🌐 Goal translated (IT→EN) with ' + (data.model || 'local model'), false);
        } catch (e) {
            if (st) st.textContent = '';
            setStatus('Translation failed: ' + e.message, 'error');
            elog('❌ Goal translation failed: ' + e.message, true);
            if (btn) btn.textContent = oldLabel || '🌐 Traduci in English';
        } finally {
            if (btn) btn.disabled = false;
        }
    };

    // Anchor sent to the worker. The VISIBLE shared slots win: the hidden
    // agent-image-path can be stale after the user picks new images in the
    // boxes (it is only kept as a fallback when no slot has an image).
    function agentAnchorPath() {
        if (typeof imageSlots !== 'undefined') {
            const init = imageSlots.find(s => s.path && s.role === 'init') || imageSlots.find(s => s.path);
            return init ? init.path : null;
        }
        const hp = $('agent-image-path');
        return (hp && hp.value) || null;
    }

    // highlight the tile(s) that are the current reference
    function markRefTile(url) {
        document.querySelectorAll('.ag-tile.isref').forEach(t => t.classList.remove('isref'));
        if (!url) return;
        document.querySelectorAll('.ag-tile').forEach(t => {
            if (t.dataset.url === url || (url.indexOf('/outputs/') === -1 && t.dataset.url && t.dataset.url.split('/').pop() === url.split('/').pop())) {
                t.classList.add('isref');
            }
        });
    }

    // ----------------------------- Progression rail -----------------------------
    function makeTile(url, label, container) {
        const gal = container || $('agent-gallery');
        if (!gal) return null;
        const isIter = /iter_(\d+)_img\.png/.test(url);
        const iterN = isIter ? parseInt(url.match(/iter_(\d+)_img\.png/)[1], 10) + 1 : 0;
        const tile = document.createElement('div');
        tile.className = 'ag-tile';
        tile.dataset.url = url;
        tile.title = label || '';

        const img = document.createElement('img');
        img.src = url + (url.indexOf('?') === -1 ? '?t=' + Date.now() : '');
        img.loading = 'lazy';
        img.alt = label || '';

        // visible caption with the real image dimensions (also in the tooltip)
        const cap = document.createElement('div');
        cap.className = 'ag-tile-dim loading';
        cap.textContent = '…';
        img.onload = () => {
            if (img.naturalWidth && img.naturalHeight) {
                const label2 = label ? (label + ' — ') : '';
                tile.title = label2 + img.naturalWidth + 'x' + img.naturalHeight + ' px';
                cap.textContent = img.naturalWidth + ' × ' + img.naturalHeight + ' px';
                cap.classList.remove('loading');
            }
        };

        // step badge (numbered overlay so the progression order is clear)
        let badgeTxt = '';
        if (isIter) badgeTxt = '#' + iterN;
        else if (/original/i.test(label || '')) badgeTxt = '★';
        else if (/arf/i.test(label || '')) badgeTxt = 'arf';
        else badgeTxt = '';
        const badge = document.createElement('div');
        badge.className = 'ag-badge';
        badge.innerText = badgeTxt;

        const acts = document.createElement('div');
        acts.className = 'ag-acts';
        const mk = (txt, title, fn) => {
            const b = document.createElement('button');
            b.className = 'ag-act';
            b.textContent = txt;
            b.title = title;
            b.onclick = (e) => { e.stopPropagation(); fn(); };
            return b;
        };
        const bZoom = mk('🔍', 'Open full screen', () => window.open(url, '_blank'));
        // single-step delete: only step tiles of the current Progression rail
        // (images inside an agent_refine_* run folder)
        const canDelete = !container && isIter && url.indexOf('/agent_refine_') !== -1;
        if (canDelete) {
            const bDel = mk('🗑', 'Delete this step from the run (file + manifest)', () => window.agentRemoveStep(url));
            acts.appendChild(bDel);
        }
        const bRef = mk('📌', 'Use as reference image', () => {
            setRefByUrl(url);
            elog('📌 Image set as reference: ' + url.split('/').pop());
        });
        const bUse = mk('🎨', 'Send image + prompt to the I2I tab', () => {
            window.agentImageToI2I(url, label);
        });
        acts.appendChild(bZoom);
        if (isIter) acts.appendChild(bUse);
        acts.appendChild(bRef);

        tile.appendChild(img);
        tile.appendChild(badge);
        tile.appendChild(cap);
        tile.appendChild(acts);

        tile.addEventListener('click', () => selectImage(url, label, { setRef: !agentRunning }));

        gal.appendChild(tile);
        gal.scrollTop = gal.scrollHeight;
        // keep the isref highlight if this tile is the current reference
        if ($('agent-image-path') && $('agent-image-path').value === url) tile.classList.add('isref');
        return tile;
    }

    function addGalleryImage(url, label) {
        return makeTile(url, label);
    }

    // ----------------------------- delete one step of the run -----------------------------
    // Removes iter_NNN_img.png + its _prompt.txt/_critique.txt from the run folder
    // and updates manifest.json, so History / ARF picker / Retry stay consistent.
    window.agentRemoveStep = async function (url) {
        if (agentRunning) {
            alert('A run is in progress: you can only delete steps after it finishes.');
            return;
        }
        const m = url.match(/\/outputs\/(agent_refine_[^/]+)\/([^/]*iter_\d+_img\.png)/);
        if (!m) return;
        const outDir = '/outputs/' + m[1];
        const file = m[2];
        if (!confirm('Delete this step (' + file + ') from the run?\n' +
            'The image file, its critique and its prompt will be removed and the manifest updated.')) return;
        try {
            const res = await fetch('/api/agent/remove-step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ out_dir: outDir, image: file }),
            });
            const d = await res.json();
            if (!res.ok || !d.ok) {
                setStatus('❌ ' + (d.detail || d.error || 'deletion failed'), 'error');
                return;
            }
            // remove the tile from the rail
            document.querySelectorAll('#agent-gallery .ag-tile').forEach(t => {
                if (t.dataset.url === url) t.remove();
            });
            delete stepPrompts[url];
            if (sel && sel.url === url) {
                const tiles = Array.from(document.querySelectorAll('#agent-gallery .ag-tile'));
                if (tiles.length) {
                    const t = tiles[tiles.length - 1];
                    selectImage(t.dataset.url, (t.querySelector('img') || {}).alt || '', { setRef: false });
                } else {
                    resetViewer();
                }
            }
            if (runLast && runLast.url === url) runLast = null;  // getRunLast() falls back to remaining tiles
            setStatus('🗑️ Step deleted: ' + file + ' (' + (d.remaining || 0) + ' image(s) left in the run)', 'success');
            elog('🗑️ Step deleted from the run: ' + url);
            if (d.remaining === 0) {
                setRetryVisible(false);
                setResultActions(false);
            }
            refreshStandardHistory();
        } catch (e) {
            setStatus('Step deletion error: ' + e.message, 'error');
        }
    };

    // ----------------------------- open an ARF run inside this workspace -----------------------------
    // Loads a run picked from History: goal + generation params + critic/diffusion
    // models from the run manifest, the chosen step as reference, all step tiles
    // rebuilt in the Progression rail and the run folder set as the active one so
    // 🔁 Retry / 🗑️ Cleanup operate on it.
    window.agentLoadArfRun = function (item, step) {
        if (agentRunning) { alert('A run is in progress: please wait for it to finish.'); return; }
        if (!item || !item.arf_dir || !step || !step.image) { alert('Invalid run/step.'); return; }
        const p = item.params || {};
        const goal = (item.inputs && item.inputs.goal) || p.goal || '';
        agentOutDir = '/outputs/' + item.arf_dir;
        agentFinalized = false;

        // Restore the exact LoRAs used for the selected step (fallback: the
        // run-level set stored in the manifest).
        restoreAgentLoras((step && step.loras) || p.loras);

        // goal + generation parameters of the run
        const g = $('agent-goal'); if (g) g.value = goal;
        const hasVal = (v) => v !== undefined && v !== null && v !== '';
        if (hasVal(p.seed)) { const s = $('agent-seed'); if (s) s.value = p.seed; }
        if (hasVal(p.steps)) {
            const st = $('agent-steps'); if (st) { st.value = p.steps; const l = $('val-agent-steps'); if (l) l.textContent = p.steps; }
        }
        if (hasVal(p.guidance)) {
            const c = $('agent-cfg'); if (c) { c.value = p.guidance; const l = $('val-agent-cfg'); if (l) l.textContent = Number(p.guidance).toFixed(1); }
        }
        if (p.scheduler) {
            const sch = $('agent-scheduler');
            if (sch && Array.from(sch.options).some(o => o.value === p.scheduler)) sch.value = p.scheduler;
        }
        if (p.model_path) {
            const dm = $('agent-diffusion');
            if (dm) {
                const hit = Array.from(dm.options).find(o => o.value === p.model_path ||
                    o.value.replace(/\\/g, '/').endsWith('/' + String(p.model_path).split('/').pop()));
                if (hit) dm.value = hit.value;
            }
        }
        if (p.critic_model) {
            const cr = $('agent-critic');
            if (cr) {
                const hit = Array.from(cr.options).find(o => o.value === p.critic_model ||
                    o.value.split('/').pop() === p.critic_model || o.textContent.trim() === p.critic_model);
                if (hit) cr.value = hit.value;
            }
        }
        // reference-edit controls carry over from the run
        if (p.use_image !== undefined && $('agent-use-image')) $('agent-use-image').checked = !!p.use_image;
        if (p.edit_from_reference !== undefined && $('agent-edit-ref')) $('agent-edit-ref').checked = !!p.edit_from_reference;
        if (p.edit_anchor && $('agent-edit-anchor')) $('agent-edit-anchor').value = p.edit_anchor;
        if (p.interpreter_model && $('agent-interpreter')) {
            const hit = Array.from($('agent-interpreter').options).find(o => o.value === p.interpreter_model);
            if (hit) $('agent-interpreter').value = hit.value;
        }
        if (window.refreshAgentCapabilities) window.refreshAgentCapabilities();

        // rebuild the Progression rail with the run content
        clearRailForRun();
        if (item.original_image) {
            window.agentOriginalLoaded = true;
            addGalleryImage(item.original_image, 'original');
        }
        (item.arf_steps || []).forEach(st2 => {
            stepPrompts[st2.image] = st2.prompt || '';
            addGalleryImage(st2.image, 'iter ' + st2.iter);
        });
        runLast = { url: step.image, prompt: step.prompt || stepPrompts[step.image] || '' };
        setRefByUrl(step.image);
        selectImage(step.image, 'iter ' + (step.iter || ''), { setRef: false });
        // The shared reference boxes show what the RUN used (original + extras);
        // the hidden anchor becomes the run's ORIGINAL image, not the step.
        restoreAgentRefImages(p.ref_images || (item.inputs && item.inputs.ref_images),
                              item.original_image || step.image);
        setResultActions(true);
        setRetryVisible(true);
        openProgressionRail();
        setStatus('🪄 Run opened in Agent Refine: ' + item.arf_dir + ' — goal and models loaded, 🔁 Retry available', 'success');
        elog('🪄 ARF ' + item.arf_dir + ' opened in the Agent Refine panel (step #' + (step.iter || '') + ')', false, true);
        refreshStandardHistory();
        agentExpLoadFromRun('/outputs/' + item.arf_dir);
    };

    // ----------------------------- load persisted experience of a run -----------------------------
    // When a finished run is opened (from History / ARF), populate the 🧭 panel
    // from the explore_state.json that the worker left in its folder.
    async function agentExpLoadFromRun(outDirUrl) {
        expReset();
        expSetMode('idle');
        try {
            const res = await fetch('/api/agent/experience?out_dir=' + encodeURIComponent(outDirUrl));
            const d = await res.json();
            if (!res.ok || !d || !d.exists) {
                const m = expEl('exp-msg');
                if (m) m.textContent = '(no exploration memory for this run) — start a run or open an explored one.';
                return;
            }
            if (d.best) expApplyBest(d.best);
            if (d.strategies && d.strategies.length) {
                exp.strategies = d.strategies.slice();
            }
            if (d.diagnosis) exp.msg = 'Last diagnosis: ' + d.diagnosis;
            else exp.msg = 'Explored ' + (d.explore_count || 0) + ' round(s) · best score ' +
                (d.best ? d.best.score + '/10' : 'n/a') + '.';
            expRender();
        } catch (e) {
            console.error('agentExpLoadFromRun', e);
        }
    }

    // ----------------------------- Progression rail / standard History coordination -----------------------------
    // While in agent mode the STANDARD #history-panel is moved to the far right
    // of this workspace; only one of {Progression rail, History panel} stays open
    // at a time (opening one collapses the other).
    function stdHistoryOpen() {
        const p = $('history-panel');
        return !!p && !p.classList.contains('collapsed');
    }
    function setPRailIcon(closed) {
        const t = $('agent-rail-toggle');
        if (!t) return;
        t.textContent = closed ? '◀' : '▶';
        t.title = closed ? 'Open the Progression gallery' : 'Collapse the image gallery';
    }
    function openProgressionRail() {
        const r = $('agent-rail');
        if (r) r.classList.remove('closed');
        setPRailIcon(false);
        collapseStdHistory();
    }
    function closeProgressionRail() {
        const r = $('agent-rail');
        if (r) r.classList.add('closed');
        setPRailIcon(true);
    }
    function collapseStdHistory() {
        const p = $('history-panel');
        const t = $('history-toggle-btn');
        if (p && !p.classList.contains('collapsed')) {
            p.classList.add('collapsed');
            if (t) t.classList.remove('hidden');
        }
    }
    function expandStdHistory() {
        const p = $('history-panel');
        const t = $('history-toggle-btn');
        if (p && p.classList.contains('collapsed')) {
            p.classList.remove('collapsed');
            if (t) t.classList.add('hidden');
            if (window.loadHistory) window.loadHistory();
        }
    }
    // refresh the standard History when it is visible (new/finished/deleted runs)
    function refreshStandardHistory() {
        if (window.currentMode === 'agent' && stdHistoryOpen() && window.loadHistory) {
            window.loadHistory();
        }
    }

    function mountStdHistoryIntoAgent() {
        const row = $('agent-workspace-row');
        const p = $('history-panel');
        const t = $('history-toggle-btn');
        if (!row || !p) return;
        if (p.parentElement !== row) {
            row.appendChild(p);
            if (t) row.appendChild(t);
        }
    }
    function restoreStdHistory() {
        const host = document.querySelector('.panel-center-container');
        const p = $('history-panel');
        const t = $('history-toggle-btn');
        if (!host || !p) return;
        if (p.parentElement !== host) host.appendChild(p);
        if (t && t.parentElement !== host) host.appendChild(t);
    }

    // While in agent mode the SHARED I2T/I2I upload panel (dynamic image slots
    // with roles) is docked into the Agent left column: the same boxes the user
    // already knows send the extra reference images to the VLM critic.
    let _uploadPanelHome = null;
    function mountUploadPanelIntoAgent() {
        const host = document.querySelector('.agent-left');
        const up = document.getElementById('upload-panel');
        if (!host || !up) return;
        if (!_uploadPanelHome) {
            _uploadPanelHome = { parent: up.parentElement, next: up.nextElementSibling };
        }
        if (up.parentElement !== host) {
            const goalGroup = host.querySelector('.group');
            if (goalGroup && goalGroup.nextElementSibling) host.insertBefore(up, goalGroup.nextElementSibling);
            else host.insertBefore(up, host.firstChild);
        }
        up.classList.remove('hidden');
        const hb = up.querySelector('.upload-history-btn');
        if (hb) hb.classList.add('hidden');
    }
    function restoreUploadPanel() {
        const up = document.getElementById('upload-panel');
        if (!up || !_uploadPanelHome) return;
        const { parent, next } = _uploadPanelHome;
        if (parent) parent.insertBefore(up, (next && next.parentElement === parent) ? next : null);
        up.classList.add('hidden');
        const hb = up.querySelector('.upload-history-btn');
        if (hb) hb.classList.remove('hidden');
    }

    // called by ui.js switchTab when the Agent tab becomes active
    window.agentEnterWorkspace = function () {
        mountStdHistoryIntoAgent();
        mountUploadPanelIntoAgent();
        if (stdHistoryOpen()) closeProgressionRail();
        else openProgressionRail();
    };
    // called by ui.js switchTab when leaving the Agent tab
    window.agentExitWorkspace = function () {
        restoreStdHistory();
        restoreUploadPanel();
        // reset the Progression rail without closing an open History panel
        if (stdHistoryOpen()) closeProgressionRail();
        else openProgressionRail();
    };

    window.toggleAgentRail = function () {
        const rail = $('agent-rail');
        if (!rail) return;
        rail.classList.toggle('closed');
        setPRailIcon(rail.classList.contains('closed'));
        // opening Progression closes the standard History (only one open at a time)
        if (!rail.classList.contains('closed')) collapseStdHistory();
    };

    // wrap the global History toggle so that, in agent mode, opening History
    // closes the Progression rail (everything else keeps its standard behaviour).
    const _origToggleHistory = window.toggleHistory;
    window.toggleHistory = function () {
        const p = $('history-panel');
        const wasClosed = p ? p.classList.contains('collapsed') : true;
        const ret = (typeof _origToggleHistory === 'function') ? _origToggleHistory() : undefined;
        if (window.currentMode === 'agent' && p && wasClosed && !p.classList.contains('collapsed')) {
            closeProgressionRail();
        }
        return ret;
    };

    window.agentClearRail = function () {
        const gal = $('agent-gallery');
        if (gal) gal.innerHTML = '';
        origTileAdded = false;
        for (const k in stepPrompts) delete stepPrompts[k];
        window.agentOriginalLoaded = false;
        resetViewer();
        // the reference image stays in the upload slot (left)
    };

    // ----------------------------- center viewer / step selection -----------------------------
    function updateViewerButtons() {
        const hasSel = !!sel;
        const hasStep = !!(sel && sel.isIter);
        [['btn-sel-zoom', hasSel], ['btn-sel-ref', hasSel], ['btn-sel-t2i', hasStep], ['btn-sel-i2i', hasSel]].forEach(([id, on]) => {
            const b = $(id);
            if (!b) return;
            b.disabled = !on;
            b.style.opacity = on ? '1' : '0.5';
            b.style.cursor = on ? 'pointer' : 'not-allowed';
        });
    }

    function resetViewer() {
        sel = null;
        const empty = $('agent-sel-empty');
        const img = $('agent-sel-img');
        if (empty) empty.classList.remove('hidden');
        if (img) { img.classList.add('hidden'); img.src = ''; }
        const t = $('agent-sel-title'); if (t) t.innerText = '🪄 No step selected';
        const d = $('agent-sel-dims'); if (d) d.innerText = '';
        const p = $('agent-sel-prompt'); if (p) p.innerText = '…';
        const c = $('agent-sel-critique'); if (c) c.innerText = '…';
        updateViewerButtons();
        updateSelCounter();
    }

    function selectImage(url, label, opts) {
        if (!url) return;
        opts = opts || {};
        // clicking an image from the rail also loads it as the reference in the
        // LEFT upload slot (unless a run is active / it is an automatic selection)
        if (opts.setRef) {
            const hp = $('agent-image-path');
            if (hp) hp.value = url;
            showRefPreview(url);
        }
        const m = url.match(/iter_(\d+)_img\.png/);
        const isIter = !!m;
        const iterN = isIter ? parseInt(m[1], 10) + 1 : 0;
        sel = { url, label, isIter, iterN };

        // canvas: enlarged image
        const empty = $('agent-sel-empty');
        const img = $('agent-sel-img');
        if (empty) empty.classList.add('hidden');
        if (img) {
            img.classList.remove('hidden');
            img.src = url + (url.indexOf('?') === -1 ? '?t=' + Date.now() : '');
            img.onload = () => {
                const d = $('agent-sel-dims');
                if (d && img.naturalWidth) d.innerText = img.naturalWidth + ' × ' + img.naturalHeight + ' px';
            };
        }

        // title
        const title = $('agent-sel-title');
        if (title) title.innerText = isIter
            ? ('🪄 Step ' + iterN + ' — ' + (label || url.split('/').pop()))
            : ('📷 ' + (label || 'image'));

        const pb = $('agent-sel-prompt');
        const cb = $('agent-sel-critique');

        // highlight the tile
        document.querySelectorAll('.ag-tile.sel').forEach(t => t.classList.remove('sel'));
        document.querySelectorAll('.ag-tile').forEach(t => {
            if (t.dataset.url === url) t.classList.add('sel');
        });

        if (isIter) {
            const promptFallback = stepPrompts[url] || lastPromptText || '';
            if (pb) pb.innerText = promptFallback || '…';
            if (cb) cb.innerText = 'Loading critique…';
            updateViewerButtons();

            // artifacts on disk (the worker writes per-iteration files next to the image)
            const base = url.replace(/_img\.png$/, '');
            fetch(base + '_prompt.txt').then(r => (r.ok ? r.text() : Promise.reject()))
                .then(t => { if (t && t.trim() && pb) pb.innerText = t.trim(); })
                .catch(() => { /* keep SSE prompt */ });
            fetch(base + '_critique.txt').then(r => (r.ok ? r.text() : Promise.reject()))
                .then(t => { if (cb) cb.innerText = (t && t.trim()) ? t.trim() : 'Critique not available for this step.'; })
                .catch(() => { if (cb) cb.innerText = 'Critique not available for this step.'; });
        } else {
            if (pb) pb.innerText = '— (reference image, no step prompt) —';
            if (cb) cb.innerText = '—';
        }
        updateViewerButtons();
        updateSelCounter();
    }

    // ----------------------------- selected-step actions -----------------------------
    function galleryTiles() {
        // ordered list of tiles currently in the Progression rail (#agent-gallery)
        return Array.from(document.querySelectorAll('#agent-gallery .ag-tile'));
    }

    function agentStepNav(dir) {
        const tiles = galleryTiles();
        if (!tiles.length) return;
        let idx = tiles.findIndex(t => sel && t.dataset.url === sel.url);
        if (idx === -1) idx = tiles.length - 1;
        idx = Math.min(tiles.length - 1, Math.max(0, idx + dir));
        const t = tiles[idx];
        selectImage(t.dataset.url, (t.querySelector('img') || {}).alt || '', { setRef: false });
    }
    window.agentSelPrev = function () { agentStepNav(-1); };
    window.agentSelNext = function () { agentStepNav(1); };

    function updateSelCounter() {
        const tiles = galleryTiles();
        const c = $('agent-sel-count');
        if (!c) return;
        if (!sel || !tiles.length) { c.innerText = ''; return; }
        const idx = tiles.findIndex(t => t.dataset.url === sel.url);
        c.innerText = idx === -1 ? '' : ((idx + 1) + ' / ' + tiles.length);
    }

    window.agentSelZoom = function () {
        if (sel && sel.url) window.open(sel.url, '_blank');
        else alert('Select an image first.');
    };

    window.agentSelAsRef = function () {
        if (!sel) { alert('Select an image first.'); return; }
        setRefByUrl(sel.url);
        elog('📌 Step image set as reference: ' + sel.url.split('/').pop());
    };

    // send a selected step image + prompt to I2I (no finalize/cleanup of the run)
    window.agentImageToI2I = function (url, label) {
        const prompt = stepPrompts[url] || '';
        const pb = $('agent-sel-prompt');
        // if this step is selected, take the current (possibly file-loaded) prompt
        const txt = (sel && sel.url === url && pb) ? pb.innerText : prompt;
        if (window.switchTab) window.switchTab('i2i');
        setTimeout(() => {
            const promptBox = document.getElementById('prompt');
            if (promptBox && txt) promptBox.value = txt;
            if (window.loadToInput) window.loadToInput(url, 1);
            if (window.updateStatusGen) window.updateStatusGen('Step image + prompt loaded into the I2I tab (' + (label || url.split('/').pop()) + ')', 'success');
            if (window.log) window.log('🎨 Agent step → I2I: ' + url, false, true);
        }, 180);
    };

    window.agentSelToT2I = function () {
        if (!sel || !sel.isIter) { alert('Select a generated step (iteration) first.'); return; }
        const prompt = $('agent-sel-prompt').innerText || '';
        const img = $('agent-sel-img');
        const clamp = (v, mn, mx, st) => Math.min(mx, Math.max(mn, Math.round(v / st) * st));
        if (window.switchTab) window.switchTab('t2i');
        setTimeout(() => {
            const promptBox = document.getElementById('prompt');
            if (promptBox && prompt) promptBox.value = prompt;
            if (img && img.naturalWidth && img.naturalHeight) {
                const w = clamp(img.naturalWidth, 512, 2048, 32);
                const h = clamp(img.naturalHeight, 512, 2048, 32);
                ['width', 'height'].forEach(id => {
                    const s = document.getElementById(id);
                    const v = document.getElementById(id === 'width' ? 'val-w' : 'val-h');
                    if (s) { s.value = (id === 'width' ? w : h); if (v) v.textContent = s.value; }
                });
            }
            if (sel && sel.url && window.showImage) window.showImage(sel.url);
            if (window.updateStatusGen) window.updateStatusGen('Step prompt imported into the T2I tab (' + sel.label + ')', 'success');
            if (window.log) window.log('✅ Agent step → T2I: ' + sel.label + ' (' + (prompt || '').slice(0, 50) + '…)', false, true);
        }, 180);
    };

    window.agentSelToI2I = function () {
        if (!sel) { alert('Select an image first.'); return; }
        const url = sel.url;
        const pb = $('agent-sel-prompt');
        const txt = (sel.isIter && pb) ? (pb.innerText || '') : '';
        if (window.switchTab) window.switchTab('i2i');
        setTimeout(() => {
            const promptBox = document.getElementById('prompt');
            if (promptBox && txt) promptBox.value = txt;
            if (window.loadToInput) window.loadToInput(url, 1);
            if (window.updateStatusGen) window.updateStatusGen('Image (' + sel.label + ') + prompt loaded into the I2I tab', 'success');
            if (window.log) window.log('🎨 Agent step → I2I: ' + url, false, true);
        }, 180);
    };

    // ----------------------------- copy -----------------------------
    window.copyAgentPrompt = function (el) {
        const txt = el.innerText || '';
        if (!txt || txt.startsWith('…')) return;
        navigator.clipboard?.writeText(txt).then(() => {
            el.style.borderColor = '#4ade80';
            setTimeout(() => { el.style.borderColor = '#333'; }, 900);
        }).catch(() => {});
    };

    // ----------------------------- cross-flow: result toolbar -> ARF -----------------------------
    function setArfToolbar(hasImage) {
        const btn = $('btn-arf-toolbar');
        if (!btn) return;
        if (hasImage) {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
            btn.title = 'Send to Agent Refine (refine this image with the iterative loop)';
        } else {
            btn.disabled = true;
            btn.style.opacity = '0.4';
            btn.style.cursor = 'not-allowed';
            btn.title = 'No image available - generate one first';
        }
    }
    window.setArfToolbar = setArfToolbar;

    function setRetryVisible(show) {
        const btn = $('btn-agent-retry');
        if (btn) btn.classList.toggle('hidden', !show);
    }
    window.setRetryVisible = setRetryVisible;

    function setResultActions(enabled) {
        const bar = $('agent-result-actions');
        const ids = ['btn-agent-toT2I', 'btn-agent-toI2I', 'btn-agent-cleanup'];
        if (!bar) return;
        bar.classList.toggle('hidden', !enabled);
        ids.forEach(id => {
            const b = $(id);
            if (!b) return;
            b.disabled = !enabled;
            b.style.opacity = enabled ? '1' : '0.5';
            b.style.cursor = enabled ? 'pointer' : 'not-allowed';
        });
    }

    // Send the current main result image to the Agent Refine panel as reference.
    window.sendToARF = function () {
        const cur = window.currentResultPath || ($('result-img') ? $('result-img').src : '') || '';
        if (!cur || cur === window.location.href) {
            alert('No image to send to Agent Refine.');
            return;
        }
        let path = cur;
        if (/^https?:\/\//.test(path)) {
            try { path = new URL(path).pathname; } catch (e) { /* keep */ }
        }
        setRefByUrl(path);
        // seed the rail with it as 'original' if none yet
        if (!window.agentOriginalLoaded) {
            window.agentOriginalLoaded = true;
            addGalleryImage(path, 'original');
        }
        const nav = document.querySelectorAll('.nav-btn')[3];
        if (window.switchTab) window.switchTab('agent', nav);
        setStatus('Image loaded as the original reference. Write the goal and start.', 'success');
        elog('🪄 Image loaded from result -> ' + path);
    };

    // ----------------------------- reuse the last run step (no export, no files) -----------------------------
    // The last image + prompt of the run live in the run folder; T2I/I2I read them
    // straight from there (/outputs/agent_refine_*/...). No file is written to the
    // outputs root by the agent UI - images land there only via real T2I/I2I runs.
    let runLast = null;  // {url, prompt} filled by the SSE events

    function getRunLast() {
        if (runLast && runLast.url) return runLast;
        // fallback: newest agent_refine tile in the rail
        const tiles = Array.from(document.querySelectorAll('#agent-gallery .ag-tile'))
            .filter(t => (t.dataset.url || '').indexOf('/agent_refine_') !== -1);
        if (!tiles.length) return null;
        const t = tiles[tiles.length - 1];
        const url = t.dataset.url;
        return { url, prompt: stepPrompts[url] || lastPromptText || '' };
    }

    function setPromptBox(text) {
        const pb = document.getElementById('prompt');
        if (pb && text) pb.value = text;
    }

    // clamp helper for the T2I W/H sliders
    function setT2iSizeFromImage(url) {
        const clamp = (v, mn, mx, st) => Math.min(mx, Math.max(mn, Math.round(v / st) * st));
        const img = new Image();
        img.onload = () => {
            if (!img.naturalWidth || !img.naturalHeight) return;
            const w = clamp(img.naturalWidth, 512, 2048, 32);
            const h = clamp(img.naturalHeight, 512, 2048, 32);
            ['width', 'height'].forEach(id => {
                const s = document.getElementById(id);
                const v = document.getElementById(id === 'width' ? 'val-w' : 'val-h');
                if (s) { s.value = (id === 'width' ? w : h); if (v) v.textContent = s.value; }
            });
        };
        img.src = url + (url.indexOf('?') === -1 ? '?t=' + Date.now() : '');
    }

    function agentModelPath() {
        const selEl = $('agent-diffusion');
        return (selEl && selEl.value) || '';
    }

    function setMainModelSelect(path) {
        const selEl = document.getElementById('model-select');
        if (!selEl || !path) return;
        const opts = Array.from(selEl.options);
        const hit = opts.find(o => o.value === path ||
            o.value.replace(/\\/g, '/').endsWith('/' + path.split('/').pop()));
        if (hit) selEl.value = hit.value;
    }

    // ➡️ to T2I: load the final prompt (and resolution/model) in the T2I tab, no file created
    window.agentToT2I = function () {
        const last = getRunLast();
        if (!last) { alert('No run image to use.'); return; }
        if (window.switchTab) window.switchTab('t2i');
        setTimeout(() => {
            setPromptBox(last.prompt);
            setT2iSizeFromImage(last.url);
            const m = agentModelPath();
            if (m) setMainModelSelect(m);
            if (window.showImage) window.showImage(last.url);
            if (window.updateStatusGen) window.updateStatusGen('Final prompt imported into the T2I tab (no file created) — press 🚀 GENERATE', 'success');
            if (window.log) window.log('✅ Agent → T2I: ' + last.url, false, true);
        }, 180);
    };

    // 🎨 to I2I: load the last image as source in the I2I tab, no file created
    window.agentToI2I = function () {
        const last = getRunLast();
        if (!last) { alert('No run image to use.'); return; }
        if (window.switchTab) window.switchTab('i2i');
        setTimeout(() => {
            setPromptBox(last.prompt);
            if (window.loadToInput) window.loadToInput(last.url, 1);
            if (window.updateStatusGen) window.updateStatusGen('Source image + prompt loaded into the I2I tab (no file created) — pick a model and generate', 'success');
            if (window.log) window.log('🎨 Agent → I2I: ' + last.url, false, true);
        }, 180);
    };

    // 🗑️ cleanup: delete the whole run folder (all its steps)
    window.agentCleanup = async function () {
        if (!agentOutDir) { alert('No run to delete.'); return; }
        if (!confirm('Delete the run folder (all its steps)?')) return;
        try {
            const res = await fetch('/api/agent/finalize', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ out_dir: agentOutDir, action: 'cleanup' }),
            });
            const data = await res.json();
            if (!res.ok || !data.ok) {
                setStatus('❌ ' + (data.detail || data.error || 'cleanup failed'), 'error');
                return;
            }
            agentFinalized = true;
            agentOutDir = null;
            runLast = null;
            setResultActions(false);
            setRetryVisible(false);
            document.querySelectorAll('.ag-tile').forEach(t => {
                if ((t.dataset.url || '').indexOf('/agent_refine_') !== -1) t.remove();
            });
            if (sel && sel.url && sel.url.indexOf('/agent_refine_') !== -1) resetViewer();
            expReset();
            setStatus('Folder deleted: ' + (data.deleted || ''), 'success');
            elog('🗑️ Cleanup: agent_refine folder deleted (' + (data.deleted || '') + ')');
            // the standard History still lists the deleted run: refresh it if open
            refreshStandardHistory();
        } catch (e) {
            setStatus('Cleanup error: ' + e.message, 'error');
        }
    };

    // ----------------------------- SSE stream (run & retry) -----------------------------
    async function startAgentStream(url, body) {
        abortController = new AbortController();
        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: abortController.signal,
            });
            if (!resp.ok) {
                const err = await resp.text();
                elog('HTTP ' + resp.status + ': ' + err, true);
                setStatus('Start error: ' + err, 'error');
                return false;
            }
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const ev = line.slice(6);
                    const pipe = ev.indexOf('|');
                    const type = pipe === -1 ? ev : ev.slice(0, pipe);
                    const data = pipe === -1 ? '' : ev.slice(pipe + 1);
                    handleEvent(type, data);
                }
            }
            return true;
        } catch (e) {
            if (e.name !== 'AbortError') {
                elog('Stream error: ' + e.message, true);
                setStatus('Error: ' + e.message, 'error');
            }
            return false;
        } finally {
            abortController = null;
        }
    }

    function setRunButtons(running) {
        const runBtn = $('btn-agent-run');
        const stopBtn = $('btn-agent-stop');
        if (runBtn) { runBtn.disabled = running; runBtn.style.opacity = running ? '0.5' : '1'; }
        if (stopBtn) stopBtn.classList.toggle('hidden', !running);
        if (running) setRetryVisible(false);
    }

    // 🎲 pick a random seed for the base of the run
    window.agentRandomSeed = function () {
        const s = $('agent-seed');
        if (!s) return;
        s.value = Math.floor(Math.random() * 2147483646) + 1;
    };

    // ----------------------------- Agent experience panel -----------------------------
    // Live, in-workspace view of the search: current mode/scores, strategies
    // tried and best prompt. Updated from the SCORE/STRATEGY SSE events during
    // a run, and populated from explore_state.json when a run is reopened.
    function expEl(id) { return $(id); }

    function expSetMode(mode) {
        exp.mode = mode;
        const el = expEl('exp-mode');
        if (el) {
            el.textContent = mode + ' mode';
            el.dataset.mode = (mode === 'explore' || mode === 'reground') ? mode : '';
        }
    }

    function expReset() {
        exp = { mode: 'idle', iter: null, adherence: null, fidelity: null, combined: null,
                best: null, strategies: [], diagnosis: '', msg: '' };
        const m = expEl('exp-mode');
        if (m) { m.textContent = 'mode: —'; delete m.dataset.mode; }
        const it = expEl('exp-iter'); if (it) it.textContent = 'iter —';
        const a = expEl('exp-adv'); if (a) a.textContent = 'adherence —';
        const f = expEl('exp-fid'); if (f) f.textContent = 'fidelity —';
        const c = expEl('exp-comb'); if (c) c.textContent = 'combined —';
        const st = expEl('exp-strategies'); if (st) st.innerHTML = '';
        const b = expEl('exp-best'); if (b) b.textContent = '…';
        const msg = expEl('exp-msg'); if (msg) msg.textContent = '';
    }

    function expRender() {
        const it = expEl('exp-iter');
        if (it && exp.iter != null) it.textContent = 'iter ' + exp.iter;
        const a = expEl('exp-adv');
        if (a) a.textContent = exp.adherence != null ? ('adherence ' + exp.adherence + '/10') : 'adherence —';
        const f = expEl('exp-fid');
        if (f) f.textContent = exp.fidelity != null ? ('fidelity ' + exp.fidelity + '/10') : 'fidelity —';
        const c = expEl('exp-comb');
        if (c) c.textContent = exp.combined != null ? ('combined ' + exp.combined + '/10') : 'combined —';
        // strategies chips (sorted by best score desc)
        const box = expEl('exp-strategies');
        if (box) {
            box.innerHTML = '';
            const list = exp.strategies.slice().sort((x, y) => (y.best_score || -1) - (x.best_score || -1));
            list.forEach(s => {
                const c = document.createElement('span');
                c.className = 'strat-chip';
                const sc = s.best_score;
                if (sc != null && sc >= 7) c.classList.add('best');
                else if (sc != null && sc < 5) c.classList.add('tried-low');
                c.title = s.label + (s.best_fidelity != null ? ' · fidelity ' + s.best_fidelity : '');
                c.textContent = s.label + (sc != null ? ' ' + sc + '/10' : '');
                box.appendChild(c);
            });
        }
        const b = expEl('exp-best');
        if (b) {
            if (exp.best && exp.best.prompt) b.textContent = exp.best.prompt;
            else b.textContent = '…';
        }
        const msg = expEl('exp-msg');
        if (msg) msg.textContent = exp.msg;
    }

    function expApplyScore(o) {
        exp.iter = (o.i != null ? o.i : exp.iter);
        if (o.mode) expSetMode(o.mode);
        if (o.adherence != null) exp.adherence = o.adherence;
        if (o.fidelity != null) exp.fidelity = o.fidelity;
        if (o.combined != null) exp.combined = o.combined;
        expRender();
    }
    function expApplyStrategy(o) {
        // update/create the chip for the strategy being tried
        const found = exp.strategies.find(s => s.label === o.strategy);
        if (found) found.rounds = (found.rounds || 1) + 1;
        else exp.strategies.push({ label: o.strategy, rounds: 1, best_score: exp.adherence, best_fidelity: exp.fidelity });
        if (o.mode === 'reground') expSetMode('reground');
        else expSetMode('explore');
        exp.msg = (o.mode === 'reground' ? '🔄 Re-grounded from the ORIGINAL' : '🧭 Trying new strategy') +
                  ': ' + (o.strategy || '?') + (o.diagnosis ? ' — ' + o.diagnosis : '');
        expRender();
        // Do NOT force the panel open here: the user may have collapsed it and
        // wants to keep it collapsed. The strategy change is already announced
        // in the top status bar + system log.
    }
    function expApplyBest(b) {
        exp.best = b;
        const box = expEl('exp-best');
        if (box) box.textContent = b && b.prompt ? b.prompt : '…';
    }

    window.toggleAgentExp = function () {
        expOpen = !expOpen;
        const body = expEl('agent-exp-body');
        const arrow = expEl('agent-exp-arrow');
        if (body) body.classList.toggle('collapsed', !expOpen);
        if (arrow) arrow.textContent = expOpen ? '▼' : '▶';
    };
    // bring the DOM in line with the stored collapsed state without toggling
    function agentExpSync() {
        const body = expEl('agent-exp-body');
        const arrow = expEl('agent-exp-arrow');
        if (body) body.classList.toggle('collapsed', !expOpen);
        if (arrow) arrow.textContent = expOpen ? '▼' : '▶';
    }

    // reset the panel when a run is started (collapse state is kept by the user)
    function agentExpResetForRun() {
        expReset();
    }

    window.runAgent = async function () {
        if (agentRunning) return;
        const goal = ($('agent-goal')?.value || '').trim();
        if (!goal) { alert('Enter a goal (the requested edit).'); return; }

        const seedBase = parseInt($('agent-seed')?.value) || 42;
        const smVal = ($('agent-seed-mode')?.value) || 'fixed';
        let seed_mode = 'fixed', seed_step = 1;
        if (smVal === 'random') seed_mode = 'random';
        else if (smVal && smVal.indexOf('step') === 0) {
            seed_mode = 'step';
            seed_step = parseInt(smVal.replace('step', ''), 10) || 1;
        }
        const criticT = parseFloat($('agent-creativity')?.value);
        const variants = Math.min(6, Math.max(1, parseInt($('agent-variants')?.value) || 1));
        const exploreWhenStuck = !!($('agent-explore')?.checked);

        const basePayload = {
            goal,
            image: agentAnchorPath(),
            images: (typeof getImageSlots === 'function')
                ? getImageSlots().filter(s => s.path && s.role && s.role !== 'none')
                : [],
            critic: $('agent-critic')?.value || null,
            interpreter: $('agent-interpreter')?.value || null,
            diffusion: $('agent-diffusion')?.value || null,
            schedule: $('agent-schedule')?.value || '512x512,768x768,1024x1024',
            iterations: parseInt($('agent-iter')?.value) || null,
            seed: seedBase,
            seed_mode,
            seed_step,
            critic_temperature: isNaN(criticT) ? 0.5 : criticT,
            explore_when_stuck: exploreWhenStuck,
            two_phase_critic: !!($('agent-twophase')?.checked),
            steps: (v => Number.isFinite(v) ? v : 10)(parseInt($('agent-steps')?.value)),
            guidance: (v => Number.isFinite(v) ? v : 2.0)(parseFloat($('agent-cfg')?.value)),
            scheduler: $('agent-scheduler')?.value || 'euler',
            llm_path: agentFieldValue('agent-llm'),
            vae_path: agentFieldValue('agent-vae'),
            scheduler_type: agentFieldValue('agent-scheduler-type'),
            loras: getAgentLoraConfigs(),
            early_stop: !!($('agent-early')?.checked),
            use_image: !!($('agent-use-image')?.checked),
            edit_from_reference: !!($('agent-edit-ref')?.checked),
            edit_anchor: $('agent-edit-anchor')?.value || 'hybrid',
            no_evict: false,
        };

        agentRunning = true;
        agentOutDir = null;
        agentFinalized = false;
        agentStopFlag = false;
        variantLabel = '';
        setResultActions(false);
        setRunButtons(true);
        clearRailForRun();
        agentExpResetForRun();
        if (variants > 1) {
            const m = expEl('exp-msg');
            if (m) m.textContent = '🧪 Running ' + variants + ' variants…';
        }
        // during a run the user must see the iterations: force Progression open
        openProgressionRail();
        if (window.startTimer) window.startTimer();   // seconds counter in the top status bar
        setStatus('Starting... (the VLM critic will be loaded; VRAM is coordinated automatically)', 'info');
        elog('🪄 Goal: ' + goal);
        if (variants > 1) elog('🧪 Batch: ' + variants + ' variants with seeds ' + seedBase + '..' + (seedBase + variants - 1));
        const empty = $('agent-sel-empty');
        if (empty) empty.innerHTML = '⏳ Run in progress: images will appear here as they are generated…';

        for (let v = 0; v < variants; v++) {
            if (agentStopFlag) break;
            if (variants > 1) {
                variantLabel = 'V' + (v + 1) + ' · ';
                basePayload.seed = (seedBase + v) % 2147483647;
                // hide Retry/result actions until the whole batch has finished
                setResultActions(false);
                setRetryVisible(false);
                elog('🧪 Variant ' + (v + 1) + '/' + variants + ' — base seed ' + basePayload.seed);
                setStatus('🧪 Variant ' + (v + 1) + '/' + variants + ' (seed ' + basePayload.seed + ') — running…', 'info');
            }
            const ok = await startAgentStream('/api/agent/refine', Object.assign({}, basePayload));
            if (!ok && !agentStopFlag) {
                // HTTP-level stream failure: stop the whole batch
                break;
            }
        }

        agentRunning = false;
        agentStopFlag = false;
        setRunButtons(false);
        if (window.stopTimer) window.stopTimer();
        if (!exp.msg) {
            const m = expEl('exp-msg');
            if (m) m.textContent = 'Run finished. The panel above shows the final scores; best prompt and strategies were persisted in the run folder.';
        }
        expRender();
    };

    // Continue a finished run with a chosen size delta:
    //   delta < 0  -> shrink  (−256/−512 px on the longer side)
    //   delta == 0 -> same resolution
    //   delta > 0  -> enlarge (+256/+512 px on the longer side)
    window.agentRetry = async function (delta) {
        if (agentRunning) return;
        if (!agentOutDir || agentFinalized) {
            alert('No finished run to continue (clean it first before starting a new one).');
            return;
        }
        delta = (delta === undefined || delta === null) ? 256 : parseInt(delta, 10);
        // highlight the chosen segment
        document.querySelectorAll('#btn-agent-retry .retry-delta').forEach(b => {
            b.classList.toggle('active', parseInt(b.dataset.delta, 10) === delta);
        });
        const mag = Math.abs(delta);
        const modeTxt = delta < 0 ? `shrinking (−${mag} px on the longer side)`
            : delta > 0 ? `enlarging (+${mag} px on the longer side)`
            : 'keeping the same resolution';
        agentRunning = true;
        setResultActions(false);
        setRunButtons(true);
        if (window.startTimer) window.startTimer();   // restart counter for the continuation
        setStatus('🔁 Retry: continuing the run ' + modeTxt + '...', 'info');
        elog('🔁 Retry requested: one more iteration on this run (' + modeTxt + ').');

        await startAgentStream('/api/agent/resume', { out_dir: agentOutDir, iterations: 1, delta, loras: getAgentLoraConfigs(),
            images: (typeof getImageSlots === 'function') ? getImageSlots().filter(s => s.path && s.role && s.role !== 'none') : [] });
        agentRunning = false;
        setRunButtons(false);
        if (window.stopTimer) window.stopTimer();
        if (agentOutDir && !agentFinalized) setRetryVisible(true);
    };
    // backward-compatible alias (enlarge by default)
    window.retryAgent = function () { return window.agentRetry(256); };

    window.stopAgent = async function () {
        agentStopFlag = true;
        if (window.stopTimer) window.stopTimer();
        if (abortController) { try { abortController.abort(); } catch (e) {} }
        try { await fetch('/api/agent/stop', { method: 'POST' }); } catch (e) {}
        setStatus('Stop requested.', 'warning');
    };

    // ----------------------------- SSE events -----------------------------
    function clearRailForRun() {
        const gal = $('agent-gallery');
        if (gal) gal.innerHTML = '';
        window.agentOriginalLoaded = false;
        origTileAdded = false;
        lastPromptText = '';
        runLast = null;
        // the in-folder copy of the original is added by the ORIG event
        resetViewer();
        const empty = $('agent-sel-empty');
        if (empty) empty.innerHTML = '⏳ Run in progress: images will appear here as they are generated…';
        setResultActions(false);
        setRetryVisible(false);
    }

    function handleEvent(type, data) {
        switch (type) {
            case 'LOG':
                elog(data);
                break;
            case 'ORIG': {
                // Original reference copied into the run folder by the worker;
                // on Retry/Resume the worker re-emits it for an original.png that
                // is ALREADY in the rail -> never add a duplicate tile. In a batch
                // every variant has its own folder copy: show only the first one.
                if (origTileAdded) { markRefTile(data); break; }
                const dup = Array.from(document.querySelectorAll('#agent-gallery .ag-tile'))
                    .some(t => t.dataset.url === data);
                if (dup) { markRefTile(data); break; }
                window.agentOriginalLoaded = true;
                origTileAdded = true;
                addGalleryImage(data, 'original');
                break;
            }
            case 'ITER': {
                curIter = data;
                setStatus('Iteration ' + data, 'info');
                elog('▶ Iteration ' + data);
                break;
            }
            case 'IMG': {
                const label = (variantLabel || '') + (curIter ? ('iter ' + curIter.split('/')[0]) : 'gen');
                stepPrompts[data] = lastPromptText;
                runLast = { url: data, prompt: lastPromptText || '' };
                addGalleryImage(data, label);
                // enlarge the new image in the center with its prompt/critique
                selectImage(data, label);
                setStatus('Image generated (' + label + ') — ' + data, 'success');
                break;
            }
            case 'PROMPT':
                lastPromptText = data;
                break;
            case 'GOAL':
                if (data === 'yes') { setStatus('🎯 Critic: GOAL REACHED', 'success'); elog('🎯 GOAL_REACHED: yes'); }
                else if (data === 'none') elog('⏳ Critic: no evaluation (iter 0)');
                else elog('⏳ GOAL_REACHED: no — fixing the prompt');
                break;
            case 'SCORE':
                // exploration telemetry: {i, adherence, fidelity, combined, goal_reached}
                try {
                    const o = JSON.parse(data);
                    let parts = [];
                    if (o && o.adherence != null) parts.push('adherence ' + o.adherence + '/10');
                    if (o && o.fidelity != null) parts.push('fidelity ' + o.fidelity + '/10');
                    if (parts.length) elog('📊 iter ' + o.i + ' ' + parts.join(' · '));
                    expApplyScore(o || {});
                    // track the best (elitism) on the client using the COMBINED score
                    const sc = (o && o.combined != null) ? o.combined : (o && o.adherence != null ? o.adherence : null);
                    if (sc != null && (!exp.best || sc > exp.best.score)) {
                        expApplyBest({ score: sc, prompt: lastPromptText, iter: o.i, mode: o.mode || 'exploit' });
                    }
                } catch (e) {}
                break;
            case 'STRATEGY':
                // the loop switched to explore/reground with a new strategy
                try {
                    const o = JSON.parse(data);
                    const kind = o.strategy === 'reground_original' ? '🔄 Re-ground from original' : '🧭 Explore';
                    setStatus(kind + ': strategy \u201C' + (o.strategy || '?') + '\u201D (iter ' + o.i + ')', 'info');
                    elog((kind === '🧭 Explore' ? '🧭' : '🔄') + ' [iter ' + o.i + '] ' + (o.strategy || '') + ' · ' + (o.diagnosis || ''));
                    expApplyStrategy(o || {});
                } catch (e) {}
                break;
            case 'FINISH': {
                agentOutDir = data.replace(/\/manifest\.json$/, '');
                agentFinalized = false;
                setStatus('Run finished. Manifest: ' + data, 'success');
                elog('✅ Run finished — continue with 🔁 Retry (+256px) or use ➡️ to T2I / 🎨 to I2I / 🗑️ Cleanup');
                setResultActions(true);
                setRetryVisible(true);
                agentReportStats();
                // the standard History now has a new ARF run: refresh if open
                refreshStandardHistory();
                break;
            }
            case 'DONE':
                setStatus(data, 'success');
                elog('✅ ' + data);
                break;
            case 'ERR':
                setStatus('❌ ' + data, 'error');
                elog('❌ ' + data, true);
                setResultActions(false);
                setRetryVisible(false);
                break;
            case 'PING':
                break;
            default:
                if (data) elog('[' + type + '] ' + data);
        }
    }

    // ----------------------------- init -----------------------------
    function init() {
        loadCritics();
        loadDiffusionModels();
        loadAgentInterpreterModels();
        resetViewer();
        const gal = $('agent-gallery');
        if (gal) gal.innerHTML = '';
        window.agentScheduleChanged();  // iterations follow the default schedule
        // capability gating follows the selected critic / generator
        const _ac = $('agent-critic'); if (_ac) _ac.addEventListener('change', refreshAgentCapabilities);
        const _ad = $('agent-diffusion'); if (_ad) _ad.addEventListener('change', () => {
            refreshAgentCapabilities();
            // the Agent VAE / scheduler lists depend on the generator family
            if (window.applyAllVaeFilters) window.applyAllVaeFilters();
            if (window.applyAllSchedFilters) window.applyAllSchedFilters();
        });
        refreshAgentCapabilities();
        // bind the Agent experience header (single source of truth for the toggle)
        const head = $('agent-exp-head');
        if (head) head.addEventListener('click', window.toggleAgentExp);
        // make sure the DOM matches the stored collapsed state
        agentExpSync();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
