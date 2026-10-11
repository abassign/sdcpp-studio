"""
Core Model Config - per-model `dataModels.json` support.

A model folder (e.g. `checkpoints/qwen-2.1/`) can ship a `dataModels.json`
file that declares:

  * the default diffusion weight among the several present in the folder
  * the VAE / text-encoder / vision-encoder components to use
  * the recommended generation defaults (steps, guidance, sampler, size)

Example (`checkpoints/qwen-2.1/dataModels.json`):

    {
      "name": "Qwen-Image 2.1",
      "type": "qwen-image-2.1",
      "diffusion_model": "qwen_image_2.1-Q6_K.gguf",
      "vae": "vae/qwen_image_2.1_vae_bf16.safetensors",
      "llm": "text_encoders/qwen3-vl-8b-instruct-q4_k_m.gguf",
      "llm_vision": "text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
      "defaults": {"steps": 20, "guidance": 6.0, "sampler": "euler",
                    "width": 1024, "height": 1024}
    }

Paths may be absolute, or relative to the model folder, to `MODELS_DIR` or to
`checkpoints/` / `vae/` / `text_encoders/`.

NOTE: `shared_utils` is imported lazily so this module can be unit-tested
outside the container (where `/app/models` does not exist).
"""
import os
import json
from typing import Any, Dict, Optional

# Global UI defaults for the T2I sliders (see static/index.html).
# Used to detect "the user did not touch this field", so that a model's
# recommended defaults can be applied transparently.
UI_PARAM_DEFAULTS = {
    "steps": 15,
    "guidance": 1.5,
    "width": 1024,
    "height": 1024,
    "sampler": "euler",
}


def _models_dir() -> Optional[str]:
    try:
        import shared_utils
        return getattr(shared_utils, "MODELS_DIR", None)
    except Exception:
        return None


def find_model_config_path(model_path: str, max_up: int = 3) -> Optional[str]:
    """Walk up from the model file/folder looking for a `dataModels.json`."""
    if not model_path:
        return None
    base = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    if not base:
        return None
    cur = os.path.abspath(base)
    for _ in range(max_up + 1):
        cand = os.path.join(cur, "dataModels.json")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def load_model_config(model_path: str) -> Dict[str, Any]:
    """Load the `dataModels.json` associated with a model (empty dict if none)."""
    path = find_model_config_path(model_path)
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}
    cfg["_config_path"] = path
    return cfg


def resolve_config_path(value: Optional[str], model_path: str,
                        config_path: Optional[str] = None) -> Optional[str]:
    """Resolve a component path declared in `dataModels.json`."""
    if not value or not isinstance(value, str):
        return None
    if os.path.isabs(value):
        return value if os.path.exists(value) else None

    model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    config_dir = os.path.dirname(config_path) if config_path else model_dir
    models_dir = _models_dir()

    candidates = [
        os.path.join(config_dir, value),
        os.path.join(model_dir, value),
    ]
    if models_dir:
        candidates.append(os.path.join(models_dir, value))
        candidates.append(os.path.join(models_dir, os.path.basename(value)))
        # Allow short labels such as "vae/foo.safetensors" relative to MODELS_DIR
        candidates.append(os.path.join(models_dir, "checkpoints", value))
    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def get_component(model_path: str, key: str) -> Optional[str]:
    """Return a resolved component path (vae, llm, llm_vision, diffusion_model...)."""
    cfg = load_model_config(model_path)
    if not cfg:
        return None
    return resolve_config_path(cfg.get(key), model_path, cfg.get("_config_path"))


def get_model_defaults(model_path: str) -> Dict[str, Any]:
    """Return the `defaults` block of a model config (empty dict if none)."""
    cfg = load_model_config(model_path)
    defaults = cfg.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def get_effective_params(model_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a model's recommended defaults into the requested parameters.

    A default only wins when the incoming value is the global UI default
    (i.e. the user did not change the field), so manual overrides are kept.
    """
    defaults = get_model_defaults(model_path)
    if not defaults:
        return params

    effective = dict(params)
    applied = {}
    for key, value in defaults.items():
        if key not in effective:
            continue
        if effective[key] == UI_PARAM_DEFAULTS.get(key) and effective[key] != value:
            effective[key] = value
            applied[key] = value
    if applied:
        effective["_applied_defaults"] = applied
    return effective
