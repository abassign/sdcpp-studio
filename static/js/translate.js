// translate.js — Traduzione locale (IT→EN) per OGNI campo prompt / negative prompt.
// =============================================================================
// Usa il backend POST /api/agent/translate, che gira sul llama-server locale
// (nessun download, stessa VRAM dell'I2T). Ogni bottone ha un toggle:
//   🌐 EN   -> traduce il campo in inglese
//   ↩︎ IT   -> ripristina il testo originale scritto dall'utente
// Funziona sia con <textarea>/<input> che con div[contenteditable].
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
        // Notifica il resto dell'app (autosave, sync slot, ecc.)
        try {
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        } catch (e) { /* ignore */ }
    }

    // Modello da usare per la traduzione: quello scelto nel selettore globale
    // (🔓 uncensored). Se "Auto" ('') il backend sceglie un modello uncensored,
    // così i termini artistici/anatomici non vengono rifiutati.
    function readModel() {
        if (typeof window.getTranslateModel === 'function') {
            return window.getTranslateModel();
        }
        const i = document.getElementById('agent-interpreter');
        const c = document.getElementById('agent-critic');
        return (i && i.value) || (c && c.value) || '';
    }

    // Modello di traduzione corrente (dal selettore header, altrimenti localStorage).
    window.getTranslateModel = function () {
        try {
            const sel = document.getElementById('translate-model-select');
            if (sel) return sel.value || '';
            return localStorage.getItem('translate_model') || '';
        } catch (e) { return ''; }
    };

    // Popola il selettore globale con i modelli di traduzione (uncensored in cima).
    async function loadTranslateModels() {
        const sel = document.getElementById('translate-model-select');
        if (!sel) return;
        try {
            const r = await fetch('/api/agent/translate-models');
            const d = await r.json();
            const saved = localStorage.getItem('translate_model') || '';
            sel.innerHTML = '<option value="">🌐 Trad: Auto (uncensored)</option>';
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
     * Traduce (o ripristina) il contenuto di un campo.
     * @param {string} fieldId  id del campo (textarea/input/contenteditable)
     * @param {string} [btnId]  id del bottone da aggiornare (label + disabled)
     */
    window.translateField = async function (fieldId, btnId) {
        const el = document.getElementById(fieldId);
        const btn = btnId ? document.getElementById(btnId) : null;
        if (!el) return;

        const current = getValue(el);

        // Toggle: se il campo contiene ancora l'ultima traduzione, ripristina l'originale.
        if (el.dataset.trOriginal !== undefined && current === el.dataset.trTranslated) {
            setValue(el, el.dataset.trOriginal);
            delete el.dataset.trOriginal;
            delete el.dataset.trTranslated;
            if (btn) { btn.textContent = '🌐 EN'; btn.title = 'Traduci in inglese (IT→EN)'; }
            return;
        }

        if (!current) return;   // nulla da tradurre

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
            if (!en) throw new Error('risposta vuota dal modello');

            el.dataset.trOriginal = current;
            el.dataset.trTranslated = en;
            setValue(el, en);
            if (btn) { btn.textContent = '↩︎ IT'; btn.title = 'Ripristina il testo originale'; }
        } catch (e) {
            if (btn) btn.textContent = oldLabel || '🌐 EN';
            if (window.log) window.log('❌ Traduzione fallita: ' + e.message, true, false);
            else console.error('Traduzione fallita:', e);
        } finally {
            if (btn) btn.disabled = false;
        }
    };
})();
