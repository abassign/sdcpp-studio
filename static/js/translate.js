// translate.js - local translation (IT->EN) for EVERY prompt / negative prompt field.
// =============================================================================
// Uses the backend POST /api/agent/translate, which runs on the local llama-server
// (no download, same VRAM as I2T). Each button is a toggle:
//   🌐 EN   -> translate the field to English
//   ↩︎ IT   -> restore the user's original text
// Works with <textarea>/<input> and with div[contenteditable].
// =============================================================================
(function () {
    'use strict';

    function getValue(el) {
        if (!el) return '';
        if (el.isContentEditable) return (el.innerText || el.textContent || '').trim();
        return (el.value || '');
    }

    function setValue(el, text) {
        if (!el) return;
        if (el.isContentEditable) el.innerText = text;
        else el.value = text;
        // Notify the rest of the app (autosave, slot sync, ...)
        try {
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        } catch (e) { /* ignore */ }
    }

    // Model to use for translation: the one chosen in the global selector
    // (🔓 uncensored). If "Auto" ('') the backend picks an uncensored model, so
    // artistic/anatomical terms are not refused.
    function readModel() {
        if (typeof window.getTranslateModel === 'function') {
            return window.getTranslateModel();
        }
        const i = document.getElementById('agent-interpreter');
        const c = document.getElementById('agent-critic');
        return (i && i.value) || (c && c.value) || '';
    }

    // Current translation model (from the header selector, else localStorage).
    window.getTranslateModel = function () {
        try {
            const sel = document.getElementById('translate-model-select');
            if (sel) return sel.value || '';
            return localStorage.getItem('translate_model') || '';
        } catch (e) { return ''; }
    };

    // Populate the global selector with translation models (uncensored first).
    async function loadTranslateModels() {
        const sel = document.getElementById('translate-model-select');
        if (!sel) return;
        try {
            const r = await fetch('/api/agent/translate-models');
            const d = await r.json();
            const saved = localStorage.getItem('translate_model') || '';
            sel.innerHTML = '<option value="">🌐 Trans: Auto (uncensored)</option>';
            (d.models || []).forEach(m => {
                const o = document.createElement('option');
                o.value = m.dir;
                o.textContent = (m.uncensored ? '🔓 ' : '') + m.name;
                sel.appendChild(o);
            });
            if (saved) sel.value = saved;   // '' if not found -> Auto
            sel.addEventListener('change', () => {
                try { localStorage.setItem('translate_model', sel.value || ''); } catch (e) { /* ignore */ }
            });
        } catch (e) { /* ignore */ }
    }

    document.addEventListener('DOMContentLoaded', loadTranslateModels);

    /**
     * Translate (or restore) the content of a field.
     * @param {string} fieldId  field id (textarea/input/contenteditable)
     * @param {string} [btnId]  button id to update (label + disabled)
     */
    window.translateField = async function (fieldId, btnId) {
        const el = document.getElementById(fieldId);
        const btn = btnId ? document.getElementById(btnId) : null;
        if (!el) return;

        const current = getValue(el);

        // Toggle: if the field still holds the last translation, restore the original.
        if (el.dataset.trOriginal !== undefined && current === el.dataset.trTranslated) {
            setValue(el, el.dataset.trOriginal);
            delete el.dataset.trOriginal;
            delete el.dataset.trTranslated;
            if (btn) { btn.textContent = '🌐 EN'; btn.title = 'Translate to English (IT→EN)'; }
            return;
        }

        if (!current) return;   // nothing to translate

        const oldLabel = btn ? btn.textContent : '';
        if (btn) { btn.disabled = true; btn.textContent = '⏳'; }
        try {
            const resp = await fetch('/api/agent/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: current, model: readModel() })
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.detail || ('HTTP ' + resp.status));
            const en = (data.text || '').trim();
            if (!en) throw new Error('empty response from the model');

            el.dataset.trOriginal = current;
            el.dataset.trTranslated = en;
            setValue(el, en);
            if (btn) { btn.textContent = '↩︎ IT'; btn.title = 'Restore the original text'; }
        } catch (e) {
            if (btn) btn.textContent = oldLabel || '🌐 EN';
            if (window.log) window.log('❌ Translation failed: ' + e.message, true, false);
            else console.error('Translation failed:', e);
        } finally {
            if (btn) btn.disabled = false;
        }
    };
})();
