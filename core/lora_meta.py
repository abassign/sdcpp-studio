"""
Core LoRA metadata - sidecar JSON management for LoRA files.

For every LoRA file we keep a sidecar JSON ("<lora-file>.json") that stores
user curated info used to filter the LoRA picker:

    {
        "display_name":        "name shown in the LoRA list (default = file stem)",
        "trigger_words":       ["list", "of", "trigger", "words"],
        "compatible_models":   ["/abs/path/to/checkpoint/one.safetensors", ...],
        "base_arch":           "informational auto-detected arch (zimage, krea, flux, ...)",
        "updated_at":          "ISO timestamp"
    }
"""
import os
import re
import json
import datetime
from typing import List, Optional

# ---------------------------------------------------------------------------
# Arch / family detection (heuristic, best-effort)
# ---------------------------------------------------------------------------
# (family_key, display_label, [substring tokens checked on normalized text])
FAMILIES = [
    ("flux",    "Flux",       ["flux"]),
    ("zimage",  "Z-Image",    ["zimage", "z_image", "z-image", "z imageturbo", "z image"]),
    ("ideogram","Ideogram",   ["ideogram"]),
    ("krea",    "Krea",       ["krea"]),
    ("qwen",    "Qwen",       ["qwen"]),
    ("glm",     "GLM-Image",  ["glm"]),
    ("pony",    "Pony (SDXL)",["pony"]),
    ("sdxl",    "SDXL",       ["sdxl", "sd_xl"]),
    ("sd15",    "SD 1.5",     ["sd15", "sd_1.5", "sd1.5", "v15", "1.5"]),
    ("sd3",     "SD3",        ["sd3", "sd_3"]),
]

FAMILY_LABEL = {k: lbl for k, lbl, _ in FAMILIES}

GENERIC_NAMES = {"lora", "pro_8", "training", "unet", "diffusion_model"}

_SD_KEYS = ("ss_base_model_version", "base_model", "ss_sd_model_name",
            "ss_network_module", "ss_network_args", "ss_output_name")


def _norm(text: Optional[str]) -> str:
    """Lowercase and strip separators so substring checks are forgiving."""
    if not text:
        return ""
    t = text.lower()
    for sep in ("_", "-", ".", " ", "/", "\\", ":"):
        t = t.replace(sep, "")
    return t


def detect_families(*texts) -> List[str]:
    """Returns the family keys matched by any of the given texts (best effort)."""
    hay = " ".join(_norm(t) for t in texts if t)
    if not hay:
        return []
    found = []
    for key, _lbl, tokens in FAMILIES:
        if any(tok.replace("_", "").replace("-", "").replace(" ", "") in hay for tok in tokens):
            if key not in found:
                found.append(key)
    return found


def family_label(key: Optional[str]) -> str:
    return FAMILY_LABEL.get(key or "", key or "")


# ---------------------------------------------------------------------------
# Safetensors metadata reading
# ---------------------------------------------------------------------------
def read_safetensors_metadata(path: str, max_header: int = 128 * 1024 * 1024) -> dict:
    """Reads the __metadata__ block of a .safetensors file (empty dict on failure)."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n <= 0 or n > max_header:
                return {}
            hdr = json.loads(f.read(n).decode("utf-8", errors="replace"))
            meta = hdr.get("__metadata__")
            return dict(meta) if isinstance(meta, dict) else {}
    except Exception:
        return {}


def find_civitai_info(lora_path: str) -> Optional[dict]:
    """Looks for a civitai sidecar info json (ModelName/TrainedWords/BaseModel)."""
    base, ext = os.path.splitext(lora_path)
    candidates = [
        lora_path + ".cm-info.json",
        base + ".cm-info.json",
        base + ".civitai.info.json",
        base + ".civitai.json",
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return None


def _clean_display_candidate(s: Optional[str]) -> Optional[str]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    low = s.lower()
    if len(s) < 2 or low.startswith("training_") or low in GENERIC_NAMES:
        return None
    if re.search(r"^\d{6,}$", s):  # plain numeric training id
        return None
    return s


def detect_lora_info(lora_path: str) -> dict:
    """
    Best-effort automatic detection for one LoRA file.

    Returns:
        {
          "arch":            first detected family key or None
          "arch_sources":    raw text that produced the family
          "display_name":    suggested display name or None
          "trigger_words":   trigger words found (civitai TrainedWords etc.)
        }
    """
    info = {"arch": None, "arch_sources": [], "display_name": None, "trigger_words": []}

    if not os.path.isfile(lora_path):
        return info

    stem = os.path.splitext(os.path.basename(lora_path))[0]
    meta = {}
    if lora_path.lower().endswith(".safetensors"):
        meta = read_safetensors_metadata(lora_path)

    civ = find_civitai_info(lora_path)

    # --- arch sources -------------------------------------------------
    arch_sources = []
    for key in _SD_KEYS:
        val = meta.get(key)
        if isinstance(val, str) and val and not val.lower().startswith("none"):
            arch_sources.append(val)
    if civ:
        if civ.get("BaseModel"):
            arch_sources.append(str(civ["BaseModel"]))

    fams = detect_families(*arch_sources)
    # name-based fallback only if metadata gave no answer
    if not fams:
        fams = detect_families(stem)

    if fams:
        info["arch"] = fams[0]
    info["arch_sources"] = [s for s in arch_sources if s][:3]

    # --- display name --------------------------------------------------
    candidate = None
    if civ and civ.get("VersionName"):
        # skip civitai version labels like "v4.9" / "1.0.3" - not a name
        vn = str(civ["VersionName"]).strip()
        if not re.match(r"^v?\d+([\.\-]\d+)*$", vn, re.IGNORECASE):
            candidate = _clean_display_candidate(vn)
    if not candidate:
        candidate = _clean_display_candidate(meta.get("ss_output_name"))
    if not candidate and civ and civ.get("ModelName"):
        candidate = _clean_display_candidate(str(civ["ModelName"]))
    info["display_name"] = candidate or None

    # --- trigger words --------------------------------------------------
    tw = []
    if civ and isinstance(civ.get("TrainedWords"), list):
        tw = [str(w).strip() for w in civ["TrainedWords"] if str(w).strip()]
    info["trigger_words"] = tw[:50]

    return info


# ---------------------------------------------------------------------------
# Sidecar JSON management
# ---------------------------------------------------------------------------
def sidecar_path(lora_path: str) -> str:
    """Path of the metadata JSON stored next to the LoRA file."""
    return lora_path + ".json"


def default_meta(lora_path: str) -> dict:
    """Fresh default metadata (does not touch disk)."""
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    detected = detect_lora_info(lora_path)
    return {
        "display_name": detected["display_name"] or stem,
        "trigger_words": list(detected["trigger_words"] or []),
        "compatible_models": [],
        "base_arch": detected["arch"] or "",
        "updated_at": None,
    }


def load_meta(lora_path: str) -> Optional[dict]:
    """Loads the sidecar JSON (merged with defaults) or None when absent/corrupt."""
    sp = sidecar_path(lora_path)
    if not os.path.isfile(sp):
        return None
    try:
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
    except Exception:
        return None
    base = default_meta(lora_path)
    base.update(data)
    # ensure lists
    for key in ("trigger_words", "compatible_models"):
        if not isinstance(base.get(key), list):
            base[key] = []
    base["has_meta"] = True
    return base


def load_meta_or_default(lora_path: str) -> dict:
    m = load_meta(lora_path)
    if m is not None:
        return m
    d = default_meta(lora_path)
    d["has_meta"] = False
    return d


def save_meta(lora_path: str, meta: dict) -> dict:
    """Validates, cleans and writes the sidecar JSON. Returns the saved dict."""
    if not os.path.isfile(lora_path):
        raise ValueError(f"LoRA file not found: {lora_path}")

    def clean_str(v, fallback):
        v = (v or "").strip() if isinstance(v, str) else ""
        v = " ".join(v.split())
        return v if v else fallback

    def clean_str_list(v):
        out = []
        for item in (v or []) if isinstance(v, list) else []:
            if isinstance(item, str):
                s = " ".join(item.split())
                if s and s not in out:
                    out.append(s)
        return out

    stem = os.path.splitext(os.path.basename(lora_path))[0]
    detected = detect_lora_info(lora_path)

    cleaned = {
        "display_name": clean_str(meta.get("display_name"), stem),
        "trigger_words": clean_str_list(meta.get("trigger_words")),
        "compatible_models": clean_str_list(meta.get("compatible_models")),
        "base_arch": clean_str(meta.get("base_arch"), detected.get("arch") or ""),
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    sp = sidecar_path(lora_path)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    cleaned["has_meta"] = True
    return cleaned


def summarize(lora_path: str) -> dict:
    """Compact info used by the LoRA list endpoint (no sidecar writes)."""
    m = load_meta_or_default(lora_path)
    detected = detect_lora_info(lora_path)
    return {
        "display_name": m.get("display_name") or os.path.splitext(os.path.basename(lora_path))[0],
        "trigger_words": m.get("trigger_words") or [],
        "compatible_models": m.get("compatible_models") or [],
        "base_arch": m.get("base_arch") or detected.get("arch") or "",
        "has_meta": bool(m.get("has_meta")),
    }


# ---------------------------------------------------------------------------
# Checkpoint suggestions
# ---------------------------------------------------------------------------
def checkpoint_family(cp_path: str, cp_name: str = "") -> str:
    """Heuristic family of a checkpoint from its path/name."""
    fams = detect_families(cp_path, cp_name)
    return fams[0] if fams else ""


def build_checkpoint_list(lora_path: str):
    """
    Returns (checkpoints, suggested_paths):
      checkpoints: checkpoint models available on disk (dedup by path)
      suggested_paths: paths to pre-check for this LoRA
                        (existing compatible_models, or auto-detected family match)
    """
    from core.models import list_all_models

    meta = load_meta_or_default(lora_path)
    existing = set(meta.get("compatible_models") or [])
    detected = detect_lora_info(lora_path)
    lora_arch = (meta.get("base_arch") or detected.get("arch") or "").lower()

    checkpoints = []
    seen = set()
    for m in list_all_models():
        p = m.get("path")
        if not p or p in seen:
            continue
        if m.get("category") == "checkpoint" and m.get("role") in ("checkpoint", None):
            seen.add(p)
            name = m.get("name") or os.path.basename(p)
            fam = checkpoint_family(p, name)
            checkpoints.append({
                "path": p,
                "name": name,
                "folder": _folder_of(p),
                "family": fam,
                "family_label": family_label(fam),
            })

    suggested = sorted(existing)
    if not suggested and lora_arch:
        # auto suggest checkpoints of the same detected family
        for cp in checkpoints:
            if cp["family"] == lora_arch or lora_arch in (cp["family"] or ""):
                suggested.append(cp["path"])

    checkpoints.sort(key=lambda c: (c["folder"] or "", c["name"] or ""))
    return checkpoints, suggested, lora_arch


def _folder_of(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return ""
