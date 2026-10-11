"""
API Preferences - User preferences/statistics management
========================================================

New structure:
- MODEL: the base key
  - TAB (mode): t2i, i2i, ei2i, i2t, etc.
    - fields: steps, guidance, scheduler_type, vae, etc. (with recency weighting)

This module analyzes JSON output files from generations to build user preference statistics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import os
import json
import time
import math
from datetime import datetime
from collections import defaultdict
import threading

router = APIRouter()

from shared_utils import OUTPUT_DIR

# ============================================================
# Constants & Configuration
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PREFERENCES_FILE = os.path.join(DATA_DIR, "preferences.json")

# Parameters to track (excluding prompt, negative_prompt, loras)
TRACKED_PARAMS = [
    'steps', 'guidance', 'scheduler_type', 'scheduler',
    'vae_path', 'clip_skip', 'width', 'height', 'strength',
    'temperature', 'top_k', 'text_encoder'
]

# Decay factor per hour (0.95 = 5% decay per hour)
DECAY_FACTOR_HOUR = 0.95

# Minimum weight threshold
MIN_WEIGHT_THRESHOLD = 0.01

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# Data Models
# ============================================================

class UpdateRequest(BaseModel):
    model: Optional[str] = None
    mode: str = "t2i"
    params: Dict[str, Any] = {}


# ============================================================
# Helper Functions
# ============================================================

def get_default_preferences() -> Dict:
    """Returns the default preferences structure."""
    return {
        "version": "2.0",
        "last_updated": datetime.now().isoformat() + "Z",
        "models": {}
    }


def load_preferences() -> Dict:
    """Load preferences from disk or return default structure."""
    if os.path.exists(PREFERENCES_FILE):
        try:
            with open(PREFERENCES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Preferences] Error loading preferences: {e}")
    
    prefs = get_default_preferences()
    save_preferences(prefs)
    return prefs


def save_preferences(prefs: Dict) -> None:
    """Save preferences to disk."""
    try:
        prefs["last_updated"] = datetime.now().isoformat() + "Z"
        with open(PREFERENCES_FILE, 'w') as f:
            json.dump(prefs, f, indent=4)
    except Exception as e:
        print(f"[Preferences] Error saving preferences: {e}")


def extract_model_name(model_path: Optional[str]) -> Optional[str]:
    """Extract just the model filename from a full path."""
    if not model_path:
        return None
    return os.path.basename(model_path.rstrip('/\\'))


def discretize_value(key: str, value: float) -> Any:
    """Discretize continuous values into bins for better statistics."""
    if value is None:
        return None
    
    if key == 'steps':
        return round(value / 5) * 5
    elif key in ['guidance', 'strength', 'temperature']:
        return round(value * 2) / 2
    elif key == 'clip_skip':
        return int(value)
    elif key in ['width', 'height']:
        return round(value / 64) * 64
    elif key == 'top_k':
        return round(value * 10) / 10
    
    return value


def get_recency_weight(timestamp: float, current_time: float) -> float:
    """Calculate recency weight using exponential decay."""
    hours_elapsed = (current_time - timestamp) / 3600
    return math.pow(DECAY_FACTOR_HOUR, hours_elapsed)


def analyze_json_file(json_path: str) -> Optional[Dict[str, Any]]:
    """Analyze a single JSON file and extract relevant parameters."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return None
    
    result = {
        'timestamp': os.path.getmtime(json_path),
        'mode': None,
        'model': None,
        'params': {}
    }
    
    meta = data.get('meta', {})
    inputs = data.get('inputs', {})
    params = data.get('params', {}) or data.get('parameters', {})
    
    # Mode: check meta first, then infer from filename
    result['mode'] = meta.get('mode')
    if not result['mode']:
        filename = os.path.basename(json_path).lower()
        if filename.startswith('i2t_'): result['mode'] = 'i2t'
        elif filename.startswith('t2i_'): result['mode'] = 't2i'
        elif filename.startswith('i2i_'): result['mode'] = 'i2i'
        elif filename.startswith('ei2i_'): result['mode'] = 'ei2i'
        elif filename.startswith('upscale_'): result['mode'] = 'upscale'
        elif filename.startswith('restore_'): result['mode'] = 'restore'
        else: result['mode'] = 'unknown'
    
    # Model
    result['model'] = extract_model_name(meta.get('model') or params.get('model_path'))
    
    # Extract tracked parameters
    for param in TRACKED_PARAMS:
        value = None
        if param in meta: value = meta[param]
        elif param in params: value = params[param]
        elif param in inputs: value = inputs[param]
        
        if value is not None:
            if isinstance(value, (int, float)):
                value = discretize_value(param, value)
            result['params'][param] = value
    
    return result


# ============================================================
# Statistics Computation
# ============================================================

def recompute_all_stats() -> Dict:
    """
    Recompute all statistics by scanning all JSON files in outputs directory.
    Structure: MODEL -> MODE -> fields with recency weighting
    """
    print("[Preferences] Starting full recompute of statistics...")

    # The agent worker deletes its intermediate sidecar JSONs, so its statistics
    # (mode "agent_refine") cannot be rebuilt from outputs/: carry them over so a
    # recompute does not silently discard them.
    try:
        _prev_models = (load_preferences() or {}).get("models", {}) or {}
    except Exception:
        _prev_models = {}
    agent_refine_modes = {}
    for _m, _d in _prev_models.items():
        _modes = (_d or {}).get("modes") or {}
        if "agent_refine" in _modes:
            agent_refine_modes[_m] = _modes["agent_refine"]
    
    prefs = get_default_preferences()
    current_time = time.time()
    
    if not os.path.exists(OUTPUT_DIR):
        print("[Preferences] Outputs directory does not exist")
        save_preferences(prefs)
        return prefs
    
    json_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.json')]
    print(f"[Preferences] Found {len(json_files)} JSON files to analyze")
    
    # Aggregate: (model, mode, param) -> list of (timestamp, weight, value)
    aggregated: Dict[tuple, List[tuple]] = defaultdict(list)
    
    for json_file in json_files:
        json_path = os.path.join(OUTPUT_DIR, json_file)
        data = analyze_json_file(json_path)
        
        if not data or not data['model']:
            continue
        
        model = data['model']
        mode = data['mode']
        timestamp = data['timestamp']
        
        # Initialize model if needed
        if model not in prefs["models"]:
            prefs["models"][model] = {"modes": {}, "total_uses": 0}
        
        # Initialize mode under model
        if mode not in prefs["models"][model]["modes"]:
            prefs["models"][model]["modes"][mode] = {
                "total_uses": 0,
                "last_used": 0,
                "fields": {}
            }
        
        model_mode = prefs["models"][model]["modes"][mode]
        model_mode["total_uses"] += 1
        model_mode["last_used"] = timestamp
        prefs["models"][model]["total_uses"] += 1
        
        # Aggregate parameter values
        for param, value in data['params'].items():
            if value is None:
                continue
            
            weight = get_recency_weight(timestamp, current_time)
            key = (model, mode, param, str(value))
            aggregated[key].append((timestamp, weight))
    
    # Build final statistics
    for (model, mode, param, value_str), entries in aggregated.items():
        if model not in prefs["models"]:
            prefs["models"][model] = {"modes": {}, "total_uses": 0}
        if mode not in prefs["models"][model]["modes"]:
            prefs["models"][model]["modes"][mode] = {"total_uses": 0, "last_used": 0, "fields": {}}
        if param not in prefs["models"][model]["modes"][mode]["fields"]:
            prefs["models"][model]["modes"][mode]["fields"][param] = {
                "values": [], "weights": [], "total_weight": 0
            }
        
        field = prefs["models"][model]["modes"][mode]["fields"][param]
        total_weight = sum(w for _, w in entries)
        
        # Check if value already exists
        if value_str in field["values"]:
            idx = field["values"].index(value_str)
            field["weights"][idx] += total_weight
        else:
            field["values"].append(value_str)
            field["weights"].append(total_weight)
        
        field["total_weight"] = sum(field["weights"])
    
    # Normalize weights
    for model_data in prefs["models"].values():
        for mode_data in model_data.get("modes", {}).values():
            for field in mode_data.get("fields", {}).values():
                total = field.get("total_weight", 0)
                if total > 0 and field.get("weights"):
                    field["weights"] = [w / total for w in field["weights"]]
    
    # Re-inject the carried-over agent statistics BEFORE saving, otherwise the
    # recompute would silently discard them.
    for _m, _mode in agent_refine_modes.items():
        _entry = prefs["models"].setdefault(_m, {"modes": {}, "total_uses": 0})
        _entry.setdefault("modes", {})["agent_refine"] = _mode
        _entry["total_uses"] = _entry.get("total_uses", 0) + int((_mode or {}).get("total_uses", 0) or 0)

    save_preferences(prefs)
    print(f"[Preferences] Recompute complete. Models: {len(prefs['models'])}")
    return prefs


def update_stats_from_generation(model: str, mode: str, params: Dict[str, Any]) -> Dict:
    """Update statistics from a single generation event."""
    prefs = load_preferences()
    current_time = time.time()
    
    # Initialize model if needed
    if model not in prefs["models"]:
        prefs["models"][model] = {"modes": {}, "total_uses": 0}
    
    # Initialize mode under model
    if mode not in prefs["models"][model]["modes"]:
        prefs["models"][model]["modes"][mode] = {
            "total_uses": 0, "last_used": 0, "fields": {}
        }
    
    model_mode = prefs["models"][model]["modes"][mode]
    model_mode["total_uses"] += 1
    model_mode["last_used"] = current_time
    prefs["models"][model]["total_uses"] += 1
    
    # Update tracked parameters
    for param in TRACKED_PARAMS:
        if param not in params or params[param] is None:
            continue
        
        value = params[param]
        if isinstance(value, (int, float)):
            value = discretize_value(param, value)
        value_str = str(value)
        
        # Initialize field if needed
        if param not in model_mode["fields"]:
            model_mode["fields"][param] = {
                "values": [], "weights": [], "total_weight": 0
            }
        
        field = model_mode["fields"][param]
        new_weight = 1.0  # Current generation has full weight
        
        # Check if value exists
        if value_str in field["values"]:
            idx = field["values"].index(value_str)
            # Apply decay to existing weight and add new weight
            old_weight = field["weights"][idx]
            decayed_weight = old_weight * get_recency_weight(model_mode["last_used"], current_time)
            field["weights"][idx] = decayed_weight + new_weight
        else:
            field["values"].append(value_str)
            field["weights"].append(new_weight)
        
        # Recalculate total and normalize
        field["total_weight"] = sum(field["weights"])
        if field["total_weight"] > 0:
            field["weights"] = [w / field["total_weight"] for w in field["weights"]]
        
        # Remove negligible weights
        new_values = []
        new_weights = []
        for v, w in zip(field["values"], field["weights"]):
            if w >= MIN_WEIGHT_THRESHOLD:
                new_values.append(v)
                new_weights.append(w)
        field["values"] = new_values
        field["weights"] = new_weights
        if field["weights"]:
            total = sum(field["weights"])
            field["weights"] = [w / total for w in field["weights"]]
        field["total_weight"] = sum(field["weights"]) if field["weights"] else 0
    
    save_preferences(prefs)
    return prefs


def get_frequent_values(model: str, mode: str) -> Dict[str, Any]:
    """
    Get the most frequent values for a given model and mode.
    Returns the top value for each tracked parameter.
    """
    prefs = load_preferences()
    
    result = {}
    
    # Try model-specific stats first
    if model and model in prefs["models"]:
        model_data = prefs["models"][model]
        if mode in model_data.get("modes", {}):
            mode_data = model_data["modes"][mode]
            result = _extract_top_values(mode_data.get("fields", {}))
    
    # If no model result, try to find ANY model that has this mode with data
    if not result:
        for model_name, model_data in prefs.get("models", {}).items():
            if mode in model_data.get("modes", {}):
                mode_data = model_data["modes"][mode]
                if mode_data.get("fields"):
                    result = _extract_top_values(mode_data.get("fields", {}))
                    break
    
    return result


def _extract_top_values(fields: Dict) -> Dict[str, Any]:
    """Extract the top value (by weight) from each field."""
    result = {}
    for param, data in fields.items():
        if not data.get("values") or not data.get("weights"):
            continue
        
        weights = data.get("weights", [])
        values = data.get("values", [])
        
        if not weights or not values:
            continue
        
        max_idx = weights.index(max(weights))
        result[param] = values[max_idx]
    
    return result


# ============================================================
# API Endpoints
# ============================================================

@router.get("/", tags=["Preferences"])
async def get_preferences():
    """Get all preferences/statistics."""
    return load_preferences()


@router.get("/frequent", tags=["Preferences"])
async def get_frequent(model: Optional[str] = None, mode: str = "t2i"):
    """
    Get the most frequent values for auto-fill.
    
    Args:
        model: Model name to get model-specific stats
        mode: Generation mode (t2i, i2i, ei2i, etc.)
    
    Returns:
        Dict mapping parameter names to their most frequent value
    """
    values = get_frequent_values(model, mode)
    return {
        "model": model,
        "mode": mode,
        "values": values
    }


@router.post("/update", tags=["Preferences"])
async def update_preferences(req: UpdateRequest):
    """Update statistics from a generation event."""
    model_name = extract_model_name(req.model)
    print(f"[Preferences] Updating stats: model={model_name}, mode={req.mode}")
    
    update_stats_from_generation(
        model=model_name or "_unknown_",
        mode=req.mode,
        params=req.params
    )
    
    return {"status": "updated", "model": model_name, "mode": req.mode}


@router.post("/recompute", tags=["Preferences"])
async def recompute_preferences():
    """Force a full recompute of all statistics from JSON files."""
    print("[Preferences] Manual recompute requested")
    prefs = recompute_all_stats()
    
    return {
        "status": "recomputed",
        "total_models": len(prefs.get("models", {}))
    }


@router.get("/summary", tags=["Preferences"])
async def get_summary():
    """Get a quick summary of preferences."""
    prefs = load_preferences()
    
    summary = {
        "total_models": len(prefs.get("models", {})),
        "models": []
    }
    
    for model_name, model_data in prefs.get("models", {}).items():
        modes = list(model_data.get("modes", {}).keys())
        summary["models"].append({
            "model": model_name,
            "total_uses": model_data.get("total_uses", 0),
            "modes": modes
        })
    
    return summary


# ============================================================
# Background Tasks
# ============================================================

_recompute_lock = threading.Lock()
_last_auto_recompute = 0
AUTO_RECOMPUTE_INTERVAL = 3600  # 1 hour


def auto_recompute_if_needed():
    """Called periodically to update statistics."""
    global _last_auto_recompute
    current_time = time.time()
    
    if current_time - _last_auto_recompute < AUTO_RECOMPUTE_INTERVAL:
        return None
    
    with _recompute_lock:
        if current_time - _last_auto_recompute < AUTO_RECOMPUTE_INTERVAL:
            return None
        
        _last_auto_recompute = current_time
        return recompute_all_stats()


def initialize_preferences():
    """Initialize preferences on server startup."""
    global _last_auto_recompute
    
    if not os.path.exists(PREFERENCES_FILE):
        print("[Preferences] No preferences file found, doing initial recompute...")
        recompute_all_stats()
    else:
        prefs = load_preferences()
        if not prefs.get("models"):
            print("[Preferences] Empty preferences, doing initial recompute...")
            recompute_all_stats()
        else:
            _last_auto_recompute = time.time()
            print(f"[Preferences] Loaded existing preferences, models: {len(prefs.get('models', {}))}")


def init_async():
    thread = threading.Thread(target=initialize_preferences, daemon=True)
    thread.start()


init_async()
