/**
 * utils.js - Helper Functions
 */

// DOM References for Logging
const consoleDiv = document.getElementById('console');
const timerDisplay = document.getElementById('timer');

window.log = function (msg, isError = false, isSuccess = false) {
    if (!consoleDiv) return;
    const div = document.createElement('div');
    div.className = 'log-line';
    if (isError) div.className += ' log-err';
    if (isSuccess) div.className += ' log-info';
    div.innerText = msg;
    consoleDiv.appendChild(div);
    consoleDiv.scrollTop = consoleDiv.scrollHeight;
};

window.formatDateYYMMDD = function (timestamp) {
    const d = new Date(timestamp * 1000);
    const pad = (n) => n.toString().padStart(2, '0');
    const yy = d.getFullYear().toString().slice(-2);
    const mm = pad(d.getMonth() + 1);
    const dd = pad(d.getDate());
    const hh = pad(d.getHours());
    const min = pad(d.getMinutes());
    const ss = pad(d.getSeconds());
    return `${yy}-${mm}-${dd} ${hh}:${min}:${ss}`;
};

window.startTimer = function () {
    window.startTime = Date.now();
    if (timerDisplay) timerDisplay.innerText = "0.0s";
    window.timerInterval = setInterval(() => {
        if (timerDisplay) timerDisplay.innerText = ((Date.now() - window.startTime) / 1000).toFixed(1) + "s";
    }, 100);
};

window.stopTimer = function () {
    if (window.timerInterval) clearInterval(window.timerInterval);
};

window.setupSlider = function (el, display) {
    if (el && display) {
        el.addEventListener('input', () => display.innerText = el.value);
    }
};

window.parseRepoId = function (input) {
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
};

console.log("Utils module loaded.");

// Clean prompt from special tokens (used by history.js)
window.cleanPrompt = function(text) {
    if (!text) return '';
    return text.replace(/<\|im_start\|>user\n/g, '')
              .replace(/<\|im_start\|>assistant\n/g, '')
              .replace(/<\|im_end\|>/g, '')
              .replace(/<\|im_end\|>/g, '')
              .trim();
};
