// config.js - Costanti e variabili globali
// ==========================================

// State globali
window.loraFiles = [];
window.timerInterval = null;
window.startTime = null;
window.currentMode = "t2i";
window.uploadedImageDims = { w: 0, h: 0 };
window.i2tFullResponse = "";

/**
 * True se il path punta a un modello Qwen-Image-Edit-family
 * (qwen-image-edit-2511, Qwen-Rapid-*, ...). Questi modelli sono
 * diffusion-only e fanno edit semantico: vanno instradati al worker i2i_edit.
 * Il backend (is_qwen_image_edit_model) fa comunque un check autoritativo.
 */
window.isQwenImageEditModel = function (path) {
    const name = (path || '').split('/').pop() || path || '';
    return /qwen-image-edit|qwen-rapid/i.test(name);
};

// Elementi DOM - Sliders
window.sliderW = document.getElementById('width');
window.sliderH = document.getElementById('height');
window.valW = document.getElementById('val-w');
window.valH = document.getElementById('val-h');
window.sliderSteps = document.getElementById('steps');
window.valSteps = document.getElementById('val-steps');
window.sliderCfg = document.getElementById('guidance');
window.valCfg = document.getElementById('val-cfg');
window.sliderTopK = document.getElementById('topk');
window.valTopK = document.getElementById('val-topk');
window.sliderTemp = document.getElementById('temp');
window.valTemp = document.getElementById('val-temp');
window.sliderClip = document.getElementById('clip-skip');
window.chkUseSlg = document.getElementById('use-slg');
window.chkUseApg = document.getElementById('use-apg');
window.valClip = document.getElementById('val-clip');
window.sliderStrength = document.getElementById('strength');
window.valStrength = document.getElementById('val-strength');
window.sliderMix = document.getElementById('mix-ratio');
window.valMix = document.getElementById('val-mix');

// I2T-specific slider elements
window.sliderTopKI2t = document.getElementById('topk-i2t');
window.valTopKI2t = document.getElementById('val-topk-i2t');
window.sliderTempI2t = document.getElementById('temp-i2t');
window.valTempI2t = document.getElementById('val-temp-i2t');
window.sliderStrengthI2t = document.getElementById('strength-i2t');
window.valStrengthI2t = document.getElementById('val-strength-i2t');
window.sliderMixI2t = document.getElementById('mix-ratio-i2t');
window.valMixI2t = document.getElementById('val-mix-i2t');
window.sliderCtxI2t = document.getElementById('ctx-i2t');
window.valCtxI2t = document.getElementById('val-ctx-i2t');
window.sliderMaxImgI2t = document.getElementById('maximg-i2t');
window.valMaxImgI2t = document.getElementById('val-maximg-i2t');
window.sliderMaxImgI2i = document.getElementById('maximg-i2i');
window.valMaxImgI2i = document.getElementById('val-maximg-i2i');

// Elementi DOM - Pannelli
window.i2iParamsPanel = document.getElementById('i2i-specific-params');
window.colMix = document.getElementById('col-mix');
window.promptBox = document.getElementById('prompt');
window.negativePromptBox = document.getElementById('negative-prompt');
window.seedInput = document.getElementById('seed');
window.randomizeCheckbox = document.getElementById('randomize');
window.btnGenerate = document.getElementById('btn-generate');
window.btnEditImage = document.getElementById('btn-edit-image');
window.btnStop = document.getElementById('btn-stop');
window.btnExit = document.getElementById('btn-exit');
window.btnOrig = document.querySelector('.btn-orig');
window.faceEnhanceCheckbox = document.getElementById('face-enhance');

// Elementi DOM - LoRA
window.loraContainer = document.getElementById('lora-container');
window.loraFolderInput = document.getElementById('lora-folder');
window.btnScan = document.getElementById('btn-scan');
window.btnAddLora = document.getElementById('btn-add-lora');
window.uploadPanel = document.getElementById('upload-panel');
window.paramsPanel = document.getElementById('params-panel');

// Elementi DOM - Upload
window.uploadZone1 = document.getElementById('upload-zone-1');
window.fileInput = document.getElementById('file-input');
window.uploadedPathInput = document.getElementById('uploaded-path');
window.preview1 = document.getElementById('preview-1');
window.btnDel1 = document.getElementById('btn-del-1');
window.slot2Container = document.getElementById('slot-2-container');
window.swapContainer = document.getElementById('swap-container');
window.uploadZone2 = document.getElementById('upload-zone-2');
window.fileInput2 = document.getElementById('file-input-2');
window.uploadedPathInput2 = document.getElementById('uploaded-path-2');
window.preview2 = document.getElementById('preview-2');
window.btnDel2 = document.getElementById('btn-del-2');

// Elementi DOM - Output
window.imgArea = document.getElementById('image-area');
window.placeholder = document.querySelector('.placeholder');
window.resultImg = document.getElementById('result-img');
window.textResultArea = document.getElementById('text-result-area');
window.thinkingBox = document.getElementById('thinking-box');
window.answerBox = document.getElementById('answer-box');

// Elementi DOM - System
window.statusBar = document.getElementById('status-bar');
window.statusText = document.getElementById('status-text');
window.timerDisplay = document.getElementById('timer');
window.consoleDiv = document.getElementById('console');

// Model elements
window.modelSelect = document.getElementById('model-select');
window.schedulerSelect = document.getElementById('scheduler-select');
window.schedulerTypeSelect = document.getElementById('scheduler-type-select');
window.vaeSelect = document.getElementById('vae-select');
window.btnAddModel = document.getElementById('btn-add-model');
window.modelAddRow = document.getElementById('model-add-row');
window.hfRepoInput = document.getElementById('hf-repo-input');
window.btnGetModel = document.getElementById('btn-get-model');
window.llmSelect = document.getElementById('llm-select');
window.llmSelectContainer = document.getElementById('llm-select-container');

console.log('Config loaded');
