// preferences.js - User Preferences/Statistics System
// Analyzes generation history to provide auto-fill suggestions
//
// Usage:
//   window.Preferences.autoFill()            - Auto-fill UI with frequent values
//   window.Preferences.refreshStats()         - Force refresh from server
//   window.Preferences.recomputeStats()       - Force recompute from JSON files
//   window.Preferences.init()                - Initialize on page load
//   window.Preferences.setDebug(true)        - Enable debug logging

window.Preferences = (function() {
    'use strict';

    const CONFIG = {
        AUTO_REFRESH_INTERVAL: 5 * 60 * 1000,
        DEBUG: true
    };

    // ============================================================
    // Notification System
    // ============================================================

    function showNotification(message, type, duration) {
        const existing = document.getElementById('pref-notification');
        if (existing) existing.remove();

        const notif = document.createElement('div');
        notif.id = 'pref-notification';
        
        const colors = {
            'success': 'rgba(76, 175, 80, 0.95)',
            'error': 'rgba(244, 67, 54, 0.95)',
            'info': 'rgba(33, 150, 243, 0.95)',
            'warning': 'rgba(255, 193, 7, 0.95)'
        };
        
        notif.style.cssText = `
            position: fixed;
            top: 60px;
            right: 20px;
            padding: 12px 20px;
            background: ${colors[type] || colors.info};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10000;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 14px;
            max-width: 300px;
        `;
        
        notif.innerText = message;
        document.body.appendChild(notif);
        
        setTimeout(() => { notif.remove(); }, duration || 3000);
    }

    function showNoDataNotification() {
        showNotification('📊 No preference data yet for this mode', 'warning', 3000);
    }

    function showAutoFillSuccessNotification(count) {
        showNotification(`✨ Auto-filled ${count} fields!`, 'success', 3000);
    }

    // ============================================================
    // State
    // ============================================================

    let _currentMode = 't2i';
    let _currentModel = null;

    function log(...args) {
        if (CONFIG.DEBUG) {
            console.log('[Preferences]', ...args);
        }
    }

    // ============================================================
    // Cached DOM Elements (from config.js pattern)
    // ============================================================

    function getVAESelect() {
        if (!window.vaeSelect) {
            window.vaeSelect = document.getElementById('vae-select');
        }
        return window.vaeSelect;
    }

    function getLLMSelect() {
        if (!window.llmSelect) {
            window.llmSelect = document.getElementById('llm-select');
        }
        return window.llmSelect;
    }

    // ============================================================
    // API Functions
    // ============================================================

    async function fetchFrequentValues(mode, model) {
        try {
            let url = `/api/preferences/frequent?mode=${encodeURIComponent(mode)}`;
            if (model) {
                url += `&model=${encodeURIComponent(model)}`;
            }
            log('Fetching:', url);
            log('  Using mode:', mode, 'model:', model);
            log('  _currentMode from module:', _currentMode);
            if (typeof window.currentMode !== 'undefined') {
                log('  window.currentMode:', window.currentMode);
            }
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            log('API Response:', data);
            log('  data.values keys:', Object.keys(data.values || {}));
            return data;
        } catch (error) {
            console.error('[Preferences] Error:', error);
            return { values: {} };
        }
    }

    async function recomputeStats() {
        try {
            const response = await fetch('/api/preferences/recompute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            return await response.json();
        } catch (error) {
            console.error('[Preferences] Recompute error:', error);
            return null;
        }
    }

    async function getPreferencesSummary() {
        try {
            const response = await fetch('/api/preferences/summary');
            return await response.json();
        } catch (error) {
            console.error('[Preferences] Summary error:', error);
            return { modes: [] };
        }
    }

    // ============================================================
    // EXACT copy of restoreVAE from history.js
    // ============================================================

    async function restoreVAE(vaePath) {
        const vaeSelect = getVAESelect();
        
        // Get the VAE from path - only use vae field
        let vaeFilename = vaePath ? vaePath.split('/').pop() : null;
        
        log('=== restoreVAE called ===');
        log('  vaePath input:', vaePath);
        log('  vaeFilename extracted:', vaeFilename);
        
        if (!vaeFilename || !vaeSelect) {
            log('restoreVAE: skipped - vaeFilename:', vaeFilename, 'vaeSelect exists:', !!vaeSelect);
            return 0;
        }
        
        // Clean the filename - remove extension for comparison
        const cleanVaeFilename = vaeFilename.replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
        log('Looking for VAE with filename:', vaeFilename, '(cleaned:', cleanVaeFilename + ')');
        
        // Also try matching against option value directly
        log('  vaeSelect exists:', !!vaeSelect, 'options count:', vaeSelect?.options?.length);
        
        // Wait for options if needed (max 10 seconds)
        if (vaeSelect.options.length <= 0) {
            log('Waiting for VAE selector to be populated...');
            let waitTime = 0;
            while (vaeSelect.options.length <= 0 && waitTime < 10000) {
                await new Promise(resolve => setTimeout(resolve, 100));
                waitTime += 100;
            }
            log('  After waiting, options count:', vaeSelect.options.length);
        }
        
        if (vaeSelect.options.length <= 0) {
            log('VAE selector not populated');
            return 0;
        }
        
        let vaeFound = false;
        for (let i = 0; i < vaeSelect.options.length; i++) {
            const opt = vaeSelect.options[i];
            const optText = opt.innerText;
            const optValue = opt.value;
            log('VAE option[' + i + ']: innerText="' + optText + '", value="' + optValue + '"');
            
            // Try matching against option VALUE first
            if (optValue) {
                const optValueFilename = optValue.split('/').pop().replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
                log('  -> Option value filename:', optValueFilename, '(cleaned)');
                if (optValueFilename === cleanVaeFilename || optValueFilename === vaeFilename) {
                    log('  -> MATCH FOUND via VALUE! Setting selectedIndex to', i);
                    vaeSelect.selectedIndex = i;
                    vaeFound = true;
                    log('VAE restored:', vaeFilename);
                    return 1;
                }
            }
            
            // Extract filename from option text (format: "displayName (filename)")
            const match = optText.match(/\(([^)]+)\)/);
            if (match) {
                const optFilename = match[1];
                const cleanOptFilename = optFilename.replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
                log('  -> Option text filename:', optFilename, '(cleaned:', cleanOptFilename + ')');
                
                // Compare cleaned filenames
                if (cleanOptFilename === cleanVaeFilename || optFilename === vaeFilename) {
                    log('  -> MATCH FOUND via TEXT! Setting selectedIndex to', i);
                    vaeSelect.selectedIndex = i;
                    vaeFound = true;
                    log('VAE restored:', vaeFilename);
                    return 1;
                }
            }
        }
        
        if (!vaeFound) {
            log('VAE not found in selector:', vaeFilename, '- using auto');
            // Set to default (Auto)
            vaeSelect.selectedIndex = 0;
        }
        return 1;
    }

    // ============================================================
    // EXACT copy of restoreTextEncoder from history.js
    // ============================================================

    async function restoreTextEncoder(textEncoderPath) {
        const llmSelect = getLLMSelect();
        
        // Get the text encoder from path - only use text_encoder field
        let textEncoderFilename = textEncoderPath ? textEncoderPath.split('/').pop() : null;
        
        if (!textEncoderFilename || !llmSelect) {
            log('restoreTextEncoder: skipped - text_encoder:', textEncoderFilename, 'llmSelect exists:', !!llmSelect);
            return 0;
        }
        
        // Clean the filename - remove extension for comparison
        const cleanTextEncoderFilename = textEncoderFilename.replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
        log('Looking for text encoder with filename:', textEncoderFilename, '(cleaned:', cleanTextEncoderFilename + ')');
        
        // Wait for options if needed (max 10 seconds)
        if (llmSelect.options.length <= 0) {
            log('Waiting for text encoder selector to be populated...');
            let waitTime = 0;
            while (llmSelect.options.length <= 0 && waitTime < 10000) {
                await new Promise(resolve => setTimeout(resolve, 100));
                waitTime += 100;
            }
        }
        
        if (llmSelect.options.length <= 0) {
            log('Text encoder selector not populated');
            return 0;
        }
        
        let textEncoderFound = false;
        for (let i = 0; i < llmSelect.options.length; i++) {
            const opt = llmSelect.options[i];
            const optText = opt.innerText;
            log('Text encoder option[' + i + ']: innerText="' + optText + '"');
            
            // Extract filename from option text (format: "displayName (filename) - source - type")
            const match = optText.match(/\(([^)]+)\)/);
            if (match) {
                const optFilename = match[1];
                const cleanOptFilename = optFilename.replace(/\.(safetensors|gguf|bin|pt|ckpt|sft)$/i, '').trim();
                log('  -> Option filename:', optFilename, '(cleaned:', cleanOptFilename + ')');
                
                // Compare cleaned filenames
                if (cleanOptFilename === cleanTextEncoderFilename || optFilename === textEncoderFilename) {
                    log('  -> MATCH FOUND! Setting selectedIndex to', i);
                    llmSelect.selectedIndex = i;
                    textEncoderFound = true;
                    log('Text encoder restored:', textEncoderFilename);
                    return 1;
                }
            }
        }
        
        if (!textEncoderFound) {
            log('Text encoder not found in selector:', textEncoderFilename, '- using default');
            // Set to default (first option)
            llmSelect.selectedIndex = 0;
        }
        return 1;
    }

    // ============================================================
    // Auto-Fill Debug Popup
    // ============================================================
    
    function showAutoFillDebugPopup(values, onConfirm) {
        // Remove existing popup if any
        const existing = document.getElementById('autofill-debug-popup');
        if (existing) existing.remove();
        
        // Create popup overlay
        const overlay = document.createElement('div');
        overlay.id = 'autofill-debug-popup';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
        `;
        
        // Create popup content
        const popup = document.createElement('div');
        popup.style.cssText = `
            background: #2a2a2a;
            border: 1px solid #555;
            border-radius: 12px;
            padding: 20px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            color: #fff;
            font-family: monospace;
        `;
        
        popup.innerHTML = `
            <h2 style="margin: 0 0 15px 0; color: #4af;">🔍 Auto-Fill Debug</h2>
            <p style="color: #888; margin-bottom: 15px;">Mode: ${_currentMode}, Model: ${_currentModel || 'any'}</p>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 1px solid #555;">
                        <th style="text-align: left; padding: 8px; color: #4f4;">Campo</th>
                        <th style="text-align: left; padding: 8px; color: #4f4;">Valore trovato</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(values).map(([key, val]) => `
                        <tr style="border-bottom: 1px solid #333;">
                            <td style="padding: 8px; color: #aaa;">${key}</td>
                            <td style="padding: 8px; color: #fff; word-break: break-all;">${val !== null && val !== undefined ? String(val) : '<em style="color:#666">(null)</em>'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
            <div style="margin-top: 20px; display: flex; gap: 10px; justify-content: flex-end;">
                <button id="autofill-cancel-btn" style="
                    padding: 10px 20px;
                    background: #555;
                    color: #fff;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                ">Annulla</button>
                <button id="autofill-confirm-btn" style="
                    padding: 10px 20px;
                    background: #4a4;
                    color: #fff;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                ">Applica</button>
            </div>
        `;
        
        overlay.appendChild(popup);
        document.body.appendChild(overlay);
        
        // Event handlers
        document.getElementById('autofill-cancel-btn').onclick = () => {
            overlay.remove();
            log('Auto-fill cancelled by user');
        };
        
        document.getElementById('autofill-confirm-btn').onclick = () => {
            overlay.remove();
            log('Auto-fill confirmed by user');
            onConfirm();
        };
        
        // Close on overlay click
        overlay.onclick = (e) => {
            if (e.target === overlay) {
                overlay.remove();
                log('Auto-fill cancelled by user (overlay click)');
            }
        };
    }

    // ============================================================
    // Auto-Fill Logic
    // ============================================================

    async function performAutoFill() {
        log('=== PERFORM AUTO-FILL ===');
        log('Mode:', _currentMode, 'Model:', _currentModel);
        
        showNotification('✨ Auto-fill analyzing...', 'info', 2000);
        
        try {
            const data = await fetchFrequentValues(_currentMode, _currentModel);
            const values = data.values || {};
            
            log('Values from API:', values);
            
            if (Object.keys(values).length === 0) {
                log('No values returned');
                showNoDataNotification();
                return;
            }
            
            // Show debug popup first
            showAutoFillDebugPopup(values, async () => {
                // This callback is called when user clicks "Applica"
                let filledCount = 0;
                
                // Sliders
                if (values.steps !== undefined && window.sliderSteps) {
                    window.sliderSteps.value = values.steps;
                    if (window.valSteps) window.valSteps.innerText = values.steps;
                    log('Set steps =', values.steps);
                    filledCount++;
                }
                
                if (values.guidance !== undefined && window.sliderCfg) {
                    window.sliderCfg.value = values.guidance;
                    if (window.valCfg) window.valCfg.innerText = values.guidance;
                    log('Set guidance =', values.guidance);
                    filledCount++;
                }
                
                if (values.width !== undefined && window.sliderW) {
                    window.sliderW.value = values.width;
                    if (window.valW) window.valW.innerText = values.width;
                    log('Set width =', values.width);
                    filledCount++;
                }
                
                if (values.height !== undefined && window.sliderH) {
                    window.sliderH.value = values.height;
                    if (window.valH) window.valH.innerText = values.height;
                    log('Set height =', values.height);
                    filledCount++;
                }
                
                if (values.strength !== undefined && window.sliderStrength) {
                    window.sliderStrength.value = values.strength;
                    if (window.valStrength) window.valStrength.innerText = values.strength;
                    log('Set strength =', values.strength);
                    filledCount++;
                }
                
                if (values.clip_skip !== undefined && window.sliderClip) {
                    window.sliderClip.value = values.clip_skip;
                    if (window.valClip) window.valClip.innerText = values.clip_skip;
                    log('Set clip_skip =', values.clip_skip);
                    filledCount++;
                }
                
                // Scheduler (Sampler)
                if (values.scheduler !== undefined && window.schedulerSelect) {
                    const schedVal = String(values.scheduler).toLowerCase();
                    for (let i = 0; i < window.schedulerSelect.options.length; i++) {
                        if (window.schedulerSelect.options[i].value.toLowerCase() === schedVal) {
                            window.schedulerSelect.selectedIndex = i;
                            log('Set scheduler =', values.scheduler);
                            filledCount++;
                            break;
                        }
                    }
                }
                
                // Scheduler Type
                if (values.scheduler_type !== undefined && window.schedulerTypeSelect) {
                    const schedVal = String(values.scheduler_type).toLowerCase();
                    for (let i = 0; i < window.schedulerTypeSelect.options.length; i++) {
                        if (window.schedulerTypeSelect.options[i].value.toLowerCase() === schedVal) {
                            window.schedulerTypeSelect.selectedIndex = i;
                            log('Set scheduler_type =', values.scheduler_type);
                            filledCount++;
                            break;
                        }
                    }
                }
                
                // VAE - check both vae_path (new) and vae (old) field names
                const vaeValue = values.vae_path || values.vae;
                if (vaeValue && vaeValue !== 'auto' && vaeValue !== '') {
                    log('About to restore VAE with path:', vaeValue);
                    const vaeCount = await restoreVAE(vaeValue);
                    log('VAE restore returned:', vaeCount);
                    filledCount += vaeCount;
                } else {
                    log('VAE skipped - value:', vaeValue);
                }
                
                if (values.text_encoder && String(values.text_encoder).trim() !== '' && values.text_encoder !== 'default') {
                    log('About to restore text_encoder with:', values.text_encoder);
                    const llmCount = await restoreTextEncoder(values.text_encoder);
                    log('Text encoder restore returned:', llmCount);
                    filledCount += llmCount;
                } else {
                    log('text_encoder skipped - value:', values.text_encoder);
                }
                
                log('Auto-fill completed. Fields filled:', filledCount);
                
                if (filledCount > 0) {
                    showAutoFillSuccessNotification(filledCount);
                } else {
                    showNoDataNotification();
                }
            });
            
        } catch (error) {
            console.error('[Preferences] Auto-fill error:', error);
            showNotification(`❌ Auto-fill error: ${error.message}`, 'error', 4000);
        }
    }

    async function autoFill() {
        log('Auto-fill called');
        await performAutoFill();
    }

    // ============================================================
    // Debug: Print VAE dropdown state
    // ============================================================
    function debugVAEDropdown() {
        const vaeSelect = getVAESelect();
        console.log('=== VAE Dropdown Debug ===');
        console.log('vaeSelect exists:', !!vaeSelect);
        if (vaeSelect) {
            console.log('options count:', vaeSelect.options.length);
            console.log('selectedIndex:', vaeSelect.selectedIndex);
            console.log('selectedValue:', vaeSelect.value);
            console.log('selectedText:', vaeSelect.options[vaeSelect.selectedIndex]?.innerText);
            for (let i = 0; i < vaeSelect.options.length; i++) {
                console.log(`  Option[${i}]: value="${vaeSelect.options[i].value}", text="${vaeSelect.options[i].innerText}"`);
            }
        }
        console.log('=========================');
    }

    // Expose for debugging
    window.debugVAEDropdown = debugVAEDropdown;

    // ============================================================
    // Event Handlers
    // ============================================================

    function onModelSelected(modelName) {
        _currentModel = modelName;
        log('Model selected:', modelName);
    }

    function onModeChanged(mode) {
        _currentMode = mode;
        log('Mode changed to:', mode);
    }

    async function updateStatsAfterGeneration(mode, model, params) {
        if (!params || Object.keys(params).length === 0) return;
        try {
            await fetch('/api/preferences/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode, model, params })
            });
        } catch (error) {
            console.error('[Preferences] Error reporting generation:', error);
        }
    }

    // ============================================================
    // Initialization
    // ============================================================

    async function refreshStats() {
        try {
            const summary = await getPreferencesSummary();
            log('Stats summary:', summary);
            return summary;
        } catch (error) {
            console.error('[Preferences] Error refreshing stats:', error);
            return null;
        }
    }

    async function init() {
        log('Initializing preferences...');
        
        if (window.currentMode) {
            _currentMode = window.currentMode;
            log('Set initial mode:', _currentMode);
        }
        
        const modelSelect = document.getElementById('model-select');
        if (modelSelect && modelSelect.value) {
            _currentModel = modelSelect.value;
            log('Set initial model:', _currentModel);
        }
        
        await refreshStats();
        
        // Hook into model-select dropdown
        const tryAttach = () => {
            const ms = document.getElementById('model-select');
            if (ms && !ms._prefHooked) {
                ms._prefHooked = true;
                ms.addEventListener('change', function() {
                    _currentModel = this.value;
                    onModelSelected(this.value);
                });
            }
        };
        tryAttach();
        setInterval(tryAttach, 2000);
        
        // Hook into switchTab
        if (typeof window.switchTab === 'function') {
            const orig = window.switchTab;
            window.switchTab = function(mode, btn) {
                const result = orig.apply(this, arguments);
                onModeChanged(mode);
                return result;
            };
        }
        
        log('Initialized. Mode:', _currentMode, 'Model:', _currentModel);
    }

    // ============================================================
    // Public API
    // ============================================================

    return {
        init,
        autoFill,
        refreshStats,
        recomputeStats,
        onModelSelected,
        onModeChanged,
        updateStatsAfterGeneration,
        getCurrentMode: () => _currentMode,
        getCurrentModel: () => _currentModel,
        setDebug: (enabled) => { CONFIG.DEBUG = enabled; }
    };

})();

// Auto-initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        if (window.Preferences?.init) window.Preferences.init();
    });
} else {
    if (window.Preferences?.init) window.Preferences.init();
}

console.log('Preferences module loaded');
