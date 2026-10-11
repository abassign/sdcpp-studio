"""
Shared utilities for configuration, generation, and output management.
"""
import os
import json
import logging
import unicodedata
from pathlib import Path

# Load settings first so all dirs are resolved from settings
from core.config import load_settings

settings = load_settings()

# Resolve models directory
custom_models_dir = settings.get("models_dir")
if custom_models_dir and os.path.exists(custom_models_dir):
    MODELS_DIR = custom_models_dir
    print(f"--> [Config] Loading custom models dir: {MODELS_DIR}", flush=True)
elif os.getenv("MODELS_DIR"):
    MODELS_DIR = os.getenv("MODELS_DIR")
    print(f"--> [Config] Using MODELS_DIR from environment: {MODELS_DIR}", flush=True)
else:
    # Default Docker path
    MODELS_DIR = "/app/models"
    print(f"--> [Config] Using default models dir: {MODELS_DIR}", flush=True)

# Resolve outputs directory
custom_outputs_dir = settings.get("outputs_dir")
if custom_outputs_dir and os.path.exists(custom_outputs_dir):
    OUTPUT_DIR = custom_outputs_dir
    print(f"--> [Config] Loading custom outputs dir: {OUTPUT_DIR}", flush=True)
elif os.getenv("OUTPUTS_DIR"):
    OUTPUT_DIR = os.getenv("OUTPUTS_DIR")
    print(f"--> [Config] Using OUTPUTS_DIR from environment: {OUTPUT_DIR}", flush=True)
else:
    # Default Docker path
    OUTPUT_DIR = "/app/outputs"
    print(f"--> [Config] Using default outputs dir: {OUTPUT_DIR}", flush=True)

# Organized model subdirectories
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
I2T_DIR = os.path.join(MODELS_DIR, "i2t")
LORAS_DIR = os.path.join(MODELS_DIR, "loras")
VAE_DIR = os.path.join(MODELS_DIR, "vae")
SAMPLERS_DIR = os.path.join(MODELS_DIR, "samplers")
TEXT_ENCODERS_DIR = os.path.join(MODELS_DIR, "text_encoders")

# Ensure directories exist
for dir_path in [CHECKPOINTS_DIR, I2T_DIR, LORAS_DIR, VAE_DIR, SAMPLERS_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def setup_logging():
    """Configures logging to suppress verbose library warnings."""
    logging.getLogger("transformers").setLevel(logging.ERROR)


def sanitize_prompt(text: str) -> str:
    """
    Remove or replace problematic characters from prompt text.

    Handles four layers of encoding issues:
    1. Double-encoding recovery: when UTF-8 bytes (e.g. E2 80 94 for —)
       are misinterpreted as Latin-1 (â\x80\x94), re-encode as Latin-1
       then re-decode as UTF-8 to recover the original character.
       Trigger: presence of C1 control chars (0x80-0x9F) — these are
       UTF-8 continuation bytes that never appear in normal text.
    2. Typographic replacements: smart quotes, em/en dashes, ellipsis,
       bullets → ASCII equivalents.
    3. Unicode decomposition: decompose accented chars (é→e, ü→u, Ä→A)
       via NFKD + strip combining marks, so only pure ASCII remains.
    4. UTF-8 safety net: encode/decode with 'replace' error mode for any
       remaining undecodable byte sequences.

    Order matters: recovery → typographic → decomposition → safety.
    """
    if not text:
        return text
    try:
        # --- STEP 1: Double-encoding recovery ---
        if isinstance(text, str) and any(0x80 <= ord(c) <= 0x9F for c in text):
            try:
                recovered = text.encode('latin-1').decode('utf-8', errors='replace')
                if recovered != text:
                    diff_count = sum(1 for a, b in zip(text, recovered) if a != b)
                    print(f"--> [Sanitize] Fixed double-encoding: recovered {diff_count} chars", flush=True)
                    text = recovered
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass

        # --- STEP 2: Replace typographic Unicode with ASCII ---
        replacements = {
            '\u2018': "'",  '\u2019': "'",
            '\u201c': '"',  '\u201d': '"',
            '\u201e': '"',
            '\u2013': '-',  '\u2014': '--',
            '\u2026': '...',
            '\u00ab': '"',  '\u00bb': '"',
            '\u2032': "'",  '\u2033': '"',
            '\u00b7': '*',  '\u2022': '*',
            '\ufeff': '',   '\u00a0': ' ',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        # --- STEP 3: Unicode decomposition to pure ASCII ---
        # Use NFKD to decompose accented chars (é→e+◌́), then strip combining
        # marks. This handles: Ä→A, é→e, ü→u, ç→c, ñ→n, ô→o, etc.
        # CJK chars decompose to themselves (no combining marks), so they're
        # caught by the final ASCII check and replaced with '?'.
        text = unicodedata.normalize('NFKD', text)
        # Strip combining diacritical marks (Category: Mn, Mc, Me)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        # Any remaining non-ASCII char (including CJK) → '?'
        text = ''.join(c if ord(c) < 128 else '?' for c in text)

        # --- STEP 4: Final UTF-8 safety net ---
        if isinstance(text, str):
            return text.encode('utf-8', errors='replace').decode('utf-8')
        return text.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"--> [Sanitize] Warning: prompt sanitization failed ({e}), returning original", flush=True)
        return text


def format_loras_for_json(loras_raw):
    """
    Transform Loras from [[path, strength, active], ...]
    to {"path": base_path, "files": [[filename, strength, active], ...]}
    """
    if not loras_raw or not isinstance(loras_raw, list):
        return {"path": "", "files": []}

    files = []
    common_prefix = ""

    for item in loras_raw:
        if isinstance(item, list) and len(item) >= 3:
            path = item[0]
            strength = item[1]
            active = item[2]
            filename = os.path.basename(path)
            files.append([filename, strength, active])
            if not common_prefix:
                common_prefix = os.path.dirname(path)

    return {"path": common_prefix, "files": files}

def save_json(full_path, data, prefix="System"):
    """
    Generic function to save data to a JSON file.
    """
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"--> [{prefix}] JSON Log saved: {full_path}", flush=True)
    except Exception as e:
        print(f"--> [{prefix} Warning] Failed to save JSON: {e}", flush=True)

def save_generation_log(mode, inputs, params, outputs, image_path_for_filename=None, model_name="unknown"):
    """
    Unified V2 JSON Saver.
    mode: 't2i' | 'i2i' | 'i2t'
    inputs: {'prompt': ..., 'loras': {...} } (image input paths for i2i)
    params: {'width': ..., 'height': ..., 'steps': ..., 'guidance': ..., ...}
    outputs: {'files': [...], 'gpu_vram_usage_mb': ..., 'exec_time': ...}
    """
    try:
        if not image_path_for_filename and outputs.get('files'):
            # take first output file
            image_path_for_filename = outputs['files'][0]
        if not image_path_for_filename:
            ts = outputs.get('exec_time', 0)
            image_path_for_filename = f"{mode}_{int(ts)}.png"
        else:
            image_path_for_filename = os.path.basename(image_path_for_filename)

        base_name = os.path.splitext(image_path_for_filename)[0]
        json_path = os.path.join(OUTPUT_DIR, base_name + ".json")

        metadata = {
            "app": "sdcpp-studio",
            "version": "2.1",
            "mode": mode,
            "model": model_name if model_name else "unknown",
            "created_at": outputs.get('exec_time', 0),  # seconds since epoch
            "inputs": inputs,
            "params": params,
            "outputs": outputs
        }
        save_json(json_path, metadata, prefix=model_name)
    except Exception as e:
        print(f"--> [SaveJSON] Error saving generation log: {e}", flush=True)

def save_output_log(mode, inputs_data, params_data, outputs_data, image_path=None, model_name="sd-cpp"):
    """
    Save output log for generation (T2I, I2I).
    """
    try:
        # Extract model name from model_path if available
        model_path = params_data.get("model", "")
        model_short = os.path.basename(model_path) if model_path else model_name

        # Build output entry
        output_entry = {
            "mode": mode,
            "model": model_short,
            "timestamp": outputs_data.get("exec_time", 0),
            "inputs": inputs_data,
            "params": params_data,
            "outputs": outputs_data
        }

        # Save to JSON
        if image_path:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            base_name = f"{mode}_{int(outputs_data.get('exec_time', 0))}"
        json_path = os.path.join(OUTPUT_DIR, base_name + ".json")
        save_json(json_path, output_entry, prefix=model_name)

        # Also log to project log
        project_log = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "project_log.md")
        # Ideally log to it:
        # TODO: integrate with project_log system
    except Exception as e:
        print(f"--> [SaveOutputLog] Error: {e}", flush=True)
