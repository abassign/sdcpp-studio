"""
Reference-editing capabilities of a diffusion model (Agent Refine "Edit" mode).

stable-diffusion.cpp accepts reference images uniformly with `-r` / `--ref-image`
and auto-selects the per-architecture preset (see `docs/edit.md`). Some editing
models additionally need a VLM vision encoder (`--llm_vision`) to encode the
reference; without it sd.cpp aborts, e.g.:

    Qwen Image 2.1 editing requires Qwen3-VL vision weights

This module is the single source of truth used by
  * the API (`/api/agent/capabilities`) to enable/disable the UI controls,
  * the workers to pick the invocation and the vision encoder.

It is import-light on purpose (shared_utils is loaded lazily) so it can be used
from the API process as well.
"""
import os
from typing import Any, Dict, List, Optional

# Rules are evaluated in order on the lowercased model basename.
# (all-of these substrings, family, needs explicit vision encoder)
_RULES = [
    (["qwen", "2.1"], "qwen_image_2_1", True),
    (["qwen", "2_1"], "qwen_image_2_1", True),
    (["qwen", "edit"], "qwen_image_edit", True),
    (["qwen", "rapid"], "qwen_image_edit", True),
    (["qwen", "layered"], "qwen_layered", True),
    (["kontext"], "flux_kontext", False),
    (["flux2"], "flux2", False),
    (["flux.2"], "flux2", False),
    (["longcat", "edit"], "longcat", False),
    (["boogu"], "boogu", False),
    # NB: "mage" alone would also match "z-iMAGE": require the full family name
    (["mage-flow"], "mage_flow", False),
    (["mage_flow"], "mage_flow", False),
    (["llada"], "llada_image", False),
]

# Families that accept repeated `-r` (multiple reference images).
MULTI_REF_FAMILIES = {"qwen_image_2_1", "qwen_image_edit", "qwen_layered"}

# Families that work through the edit-only worker (diffusion-only GGUF).
EDIT_WORKER_FAMILIES = {"qwen_image_edit"}

ANCHOR_MODES = [
    {"id": "hybrid", "label": "Ibrida (originale + precedente)"},
    {"id": "original", "label": "Originale (nessun drift)"},
    {"id": "chain", "label": "Catena (dalla precedente)"},
]


def _models_dir() -> Optional[str]:
    try:
        import shared_utils
        return getattr(shared_utils, "MODELS_DIR", None)
    except Exception:
        return None


def detect_family(model_path: str) -> Optional[str]:
    """Return the editing family id, or None if the model is not reference-capable."""
    if not model_path or not isinstance(model_path, str):
        return None
    b = os.path.basename(model_path).replace("\\", "/").lower()
    for needles, family, _vis in _RULES:
        if all(n in b for n in needles):
            return family
    return None


def _vision_required(family: Optional[str]) -> bool:
    for _needles, fam, vis in _RULES:
        if fam == family:
            return vis
    return False


def _scan_text_encoders() -> List[str]:
    """Reuse the project's encoder discovery (with a local fallback)."""
    try:
        from core.text_encoders import scan_text_encoders
        base = _models_dir()
        ents = scan_text_encoders(base or "")
        return [e["path"] for e in ents]
    except Exception:
        pass
    out = []
    md = _models_dir()
    if md:
        d = os.path.join(md, "text_encoders")
        if os.path.isdir(d):
            out = [os.path.join(d, f) for f in sorted(os.listdir(d))]
    return out


def resolve_vision_encoder(family: Optional[str], model_path: str) -> Optional[str]:
    """Resolve the mmproj / vision encoder required by *family* (None if not needed)."""
    if not family or not _vision_required(family):
        return None

    # 1) explicit declaration in the model folder's dataModels.json
    if family == "qwen_image_2_1":
        try:
            from core.model_config import get_component
            p = get_component(model_path, "llm_vision")
            if p and os.path.exists(p):
                return p
        except Exception:
            pass

    # 2) match the family's vision tower
    want = []
    if family == "qwen_image_2_1":
        want = [("qwen3", "mmproj"), ("qwen3vl", "mmproj")]
    elif family in ("qwen_image_edit", "qwen_layered"):
        want = [("qwen2.5", "mmproj"), ("qwen2_5", "mmproj"), ("qwen2.5vl", "mmproj")]

    candidates = []
    for path in _scan_text_encoders():
        b = os.path.basename(path).lower()
        if not b.endswith(".gguf") or "mmproj" not in b:
            continue
        for a, c in want:
            if a in b and c in b:
                candidates.append(path)
                break
    if not candidates:
        return None
    # deterministic: prefer the smallest filename (F16/Q8 variants all work)
    candidates.sort(key=lambda p: (len(os.path.basename(p)), p))
    return candidates[0]


def detect_edit(model_path: str) -> Dict[str, Any]:
    """Capability report for the Agent Refine "Edit from reference" control."""
    family = detect_family(model_path)
    if not family:
        return {
            "edit": False,
            "family": None,
            "edit_only": False,
            "vision_required": False,
            "vision_encoder": None,
            "vision_ready": False,
            "multi_ref": False,
            "worker": "t2i_cpp",
            "anchor_modes": ANCHOR_MODES,
            "reason": "This generator does not support reference images (no edit preset in stable-diffusion.cpp).",
        }

    vision_required = _vision_required(family)
    vis = resolve_vision_encoder(family, model_path)
    vision_ready = (not vision_required) or bool(vis)
    reason = None
    if not vision_ready:
        model = "Qwen3-VL (mmproj)" if family == "qwen_image_2_1" else "Qwen2.5-VL (mmproj)"
        reason = f"Reference editing needs the {model} vision encoder, which is not installed under text_encoders/."

    return {
        "edit": bool(vision_ready),
        "family": family,
        "edit_only": family in EDIT_WORKER_FAMILIES,
        "vision_required": vision_required,
        "vision_encoder": vis,
        "vision_ready": vision_ready,
        "multi_ref": family in MULTI_REF_FAMILIES,
        "worker": "i2i_edit" if family in EDIT_WORKER_FAMILIES else "t2i_cpp",
        "anchor_modes": ANCHOR_MODES,
        "reason": reason,
    }
