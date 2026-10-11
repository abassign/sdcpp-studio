/**
 * state.js - Global State Management
 * Defines variables shared across multiple modules.
 */

// Global State
window.currentMode = "t2i";
window.loraFiles = [];
window.i2tFullResponse = ""; // Buffer for I2T text

// Virtual T2T Mode State
window.i2tConversationHistory = ""; // Accumulates Q&A pairs for virtual T2T
window.i2tVirtualMode = false;      // true when in virtual T2T (no image yet)

// Timer State
window.timerInterval = null;
window.startTime = null;

// Upload State
window.uploadedImageDims = { w: 0, h: 0 };

// Prompt & Image Buffers (Persist across tab switches)
window.promptBuffers = {
    t2i: "",
    i2i: "",
    i2t: ""
};

window.imageBuffers = {
    t2i: "",
    i2i: "",
    i2t: ""
};

// History State
window.lastHistoryData = [];
window.currentPage = 0;
window.pageSize = 10; // Default for 128px
window.historyIconSize = 128;
window.showAllHistory = false;
window.lastValidItemsCount = 0;

// Export relevant references if needed, but window.* is the main bus here.
console.log("State module loaded.");
