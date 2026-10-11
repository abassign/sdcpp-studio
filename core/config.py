"""
Core Configuration - Secrets and settings management
"""
import os
import json

# Determine base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths - check local directory first, then fallback to /app (for Docker)
LOCAL_SETTINGS = os.path.join(BASE_DIR, "settings.json")
LOCAL_SECRETS = os.path.join(BASE_DIR, "secrets.json")
DOCKER_SETTINGS = "/app/settings.json"
DOCKER_SECRETS = "/app/secrets.json"

# Use local files if they exist, otherwise use Docker paths
SECRETS_FILE = LOCAL_SECRETS if os.path.exists(LOCAL_SECRETS) else DOCKER_SECRETS
SETTINGS_FILE = LOCAL_SETTINGS if os.path.exists(LOCAL_SETTINGS) else DOCKER_SETTINGS

# Defaults
DEFAULT_SETTINGS = {
    "vae_tiling_enabled": True,      # Enable VAE tiling for high resolutions
    "auto_fix_models": True,          # Enable automatic fix for GGUF models
    "auto_prefer_bf16_zimage": True,  # RDNA3: auto-upgrade quantized Z-Image GGUF to BF16 safetensors (~2x faster, matmul via tensor cores)
    "vae_tiling_threshold_mp": 2,     # MP threshold for tiling (default 2MP)
    "models_dir": "/app/models",      # External models directory
    "outputs_dir": "/app/outputs",    # External outputs directory
    "flux1_repo": "https://huggingface.co/unsloth/FLUX.1-dev-GGUF",
    "flux2_repo": "https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF",
    "flux_vae_repo": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    "flux_clip_repo": "https://huggingface.co/comfyanonymous/flux_text_encoders",
    # Qwen Image Edit settings
    "qwen_image_edit_repo": "https://huggingface.co/city96/Qwen2.5-VL-7B-Image-Edit-GGUF",
    "qwen_vl_repo": "https://huggingface.co/city96/Qwen2.5-VL-7B-Instruct-GGUF",
    "qwen_image_edit_model_path": "/app/models/checkpoints/qwen-image-edit-2511-Q4_K_M.gguf",  # Full path to the default Qwen Image Edit model
    # Agent Refine - promote a critic (VLM) to the default WITHOUT touching code.
    # Comma-separated patterns (regex or plain substring), matched against the
    # model directory name or its main GGUF filename; they are tried BEFORE the
    # built-in CRITIC_PREFERENCE. Examples: "sensenova", "qwen3.6.*35b.*a3b".
    # Empty = keep the built-in order (SenseNova-SI first).
    "agent_critic_prefer": ""
}


def load_secrets() -> dict:
    """Load secrets from file"""
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_secrets(data: dict) -> bool:
    """Save secrets"""
    try:
        current = load_secrets()
        current.update(data)
        with open(SECRETS_FILE, 'w') as f:
            json.dump(current, f, indent=4)
        return True
    except IOError:
        return False


def load_settings() -> dict:
    """Load user settings"""
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                user_settings = json.load(f)
                settings.update(user_settings)
        except (json.JSONDecodeError, IOError):
            pass
    return settings


def save_settings(data: dict) -> bool:
    """Save settings"""
    try:
        current = load_settings()
        
        # Remove keys with None value (explicit reset)
        for key, value in list(data.items()):
            if value is None:
                if key in current:
                    del current[key]
            else:
                current[key] = value
        
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(current, f, indent=4)
        return True
    except IOError:
        return False


def get_secret(key: str, default=None):
    """Get a specific secret"""
    secrets = load_secrets()
    return secrets.get(key, default)


def get_setting(key: str, default=None):
    """Get a specific setting"""
    settings = load_settings()
    return settings.get(key, default)
