"""
Core Models - Model scanning and management
"""
import os
import json
from typing import Optional, List, Dict

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Import MODELS_DIR and subdirectories from shared_utils
try:
    from shared_utils import MODELS_DIR, CHECKPOINTS_DIR, I2T_DIR, LORAS_DIR, VAE_DIR, SAMPLERS_DIR
except ImportError:
    MODELS_DIR = "/app/models"
    CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
    I2T_DIR = os.path.join(MODELS_DIR, "i2t")
    LORAS_DIR = os.path.join(MODELS_DIR, "loras")
    VAE_DIR = os.path.join(MODELS_DIR, "vae")
    SAMPLERS_DIR = os.path.join(MODELS_DIR, "samplers")


def detect_model_role(path: str, category: str, file_type: str = None) -> str:
    """
    Detect the model role based on:
    - Position in the directory (text_encoders, loras)
    - Folder content (model_index.json = Diffusers)
    - File extension (.gguf, .safetensors)
    - File name
    
    Args:
        path: Model path (file or directory)
        category: Model category (checkpoint, i2t, lora)
        file_type: File type if known (e.g. 'gguf', 'diffusers', 'custom_file')
    
    Returns:
        'text_encoder' | 'lora' | 'checkpoint'
    """
    path_lower = path.lower()
    
    # Already in text_encoders/ or a text_encoder folder
    if 'text_encoders' in path_lower or '/text_encoder' in path_lower:
        return 'text_encoder'
    
    # Already in loras/
    if 'loras' in path_lower or '/loras' in path_lower:
        return 'lora'
    
    # If it is a directory, inspect its contents
    if os.path.isdir(path):
        # Diffusers: ha model_index.json
        if os.path.exists(os.path.join(path, 'model_index.json')):
            return 'checkpoint'
        # Cerca sottocartelle tipiche di un modello diffusion
        for subdir in ['text_encoder', 'transformer', 'vae', 'tokenizer']:
            if os.path.isdir(os.path.join(path, subdir)):
                return 'checkpoint'
        # If it is a directory with no model indicators, use the category
        return category
    
    # È un file singolo
    basename = os.path.basename(path).lower()
    
    # File GGUF sono sempre checkpoint
    if path.endswith('.gguf'):
        return 'checkpoint'
    
    # Safetensors/Bin
    if path.endswith(('.safetensors', '.bin')):
        # Controlla il nome per LoRA
        if 'lora' in basename or 'lycoris' in basename or 'lokr' in basename or 'loha' in basename:
            return 'lora'
        # Check if it is a text encoder
        if 'text_encoder' in basename or 'clip' in basename or 't5' in basename:
            return 'text_encoder'
        return 'checkpoint'
    
    # Default: usa la categoria
    return category


def scan_models(model_type_filter: str = None) -> Optional[str]:
    """
    Recursive scan for dataModels.json
    
    Args:
        model_type_filter: model type e.g. "z-image"
        
    Returns:
        Path to the model if found
    """
    if not os.path.exists(MODELS_DIR):
        return None
    
    for root, dirs, files in os.walk(MODELS_DIR):
        if "dataModels.json" in files:
            try:
                json_path = os.path.join(root, "dataModels.json")
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                if model_type_filter and data.get("type") == model_type_filter:
                    raw_path = data.get("path", ".")
                    if raw_path.startswith("./"):
                        abs_path = os.path.join(root, raw_path[2:])
                    elif raw_path.startswith("/"):
                        abs_path = raw_path
                    else:
                        abs_path = os.path.join(root, raw_path)
                    
                    # Create specific LoRA directory
                    model_dir = os.path.dirname(abs_path) if os.path.isfile(abs_path) else abs_path
                    lora_dir = os.path.join(model_dir, "loras")
                    if not os.path.exists(lora_dir):
                        os.makedirs(lora_dir, exist_ok=True)
                        os.chmod(lora_dir, 0o777)
                    
                    # Auto-download for Z-Image
                    if not has_content(abs_path) and data.get("type") == "z-image":
                        auto_download_zimage(abs_path)
                    
                    if os.path.exists(abs_path):
                        return abs_path
            except Exception:
                pass
    return None


def has_content(path: str) -> bool:
    """Checks if the directory contains weights"""
    if not os.path.exists(path):
        return False
    content = os.listdir(path)
    weights = [x for x in content if x.endswith((".safetensors", ".bin")) or x == "model_index.json"]
    return len(weights) > 0


def auto_download_zimage(abs_path: str):
    """Automatic Z-Image download"""
    if not HF_AVAILABLE:
        return
    try:
        print(f"--> Auto-downloading Z-Image to {abs_path}")
        snapshot_download(
            repo_id="Tongyi-MAI/Z-Image",
            local_dir=abs_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
    except Exception as e:
        print(f"--> Auto-download failed: {e}")


def get_model_created_at(path: str) -> float:
    """Returns creation time or modification time as fallback"""
    try:
        if os.path.isfile(path):
            return os.path.getctime(path)
        else:
            # For directories, use the newest file's mtime as proxy
            latest_mtime = 0
            for root, dirs, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(fp)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                    except:
                        pass
            return latest_mtime if latest_mtime > 0 else os.path.getctime(path)
    except:
        return 0

def list_all_models() -> List[Dict]:
    """
    Lists all models in organized directories (checkpoints, i2t, loras)
    Also scans root MODELS_DIR for backward compatibility
    
    Returns:
        List of dicts with id, name, path, type, category, created_at
    """
    models = []
    seen_paths = set()  # Track already added paths to avoid duplicates
    
    # Scan checkpoints directory (for t2i, i2i)
    if os.path.exists(CHECKPOINTS_DIR):
        _scan_directory(CHECKPOINTS_DIR, models, category="checkpoint", seen_paths=seen_paths)
    
    # Scan i2t directory (for image-to-text models)
    if os.path.exists(I2T_DIR):
        _scan_directory(I2T_DIR, models, category="i2t", seen_paths=seen_paths)
    
    # Scan loras directory (for lora files)
    if os.path.exists(LORAS_DIR):
        _scan_directory(LORAS_DIR, models, category="lora", seen_paths=seen_paths)
    
    # Scan text_encoders directory
    text_encoders_dir = os.path.join(MODELS_DIR, "text_encoders")
    if os.path.exists(text_encoders_dir):
        _scan_directory(text_encoders_dir, models, category="text_encoder", seen_paths=seen_paths)
    
    # Scan vae directory (for VAE models like vae-ft-mse-840000-ema-pruned)
    vae_dir = os.path.join(MODELS_DIR, "vae")
    if os.path.exists(vae_dir):
        _scan_directory(vae_dir, models, category="vae", seen_paths=seen_paths)
    
    # Sort by created_at descending (most recent first)
    models.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    
    return models


def _scan_directory(base_dir: str, models: List[Dict], category: str, seen_paths: set = None):
    """
    Helper to scan a directory for models
    
    Args:
        base_dir: Directory to scan
        models: List to append results to
        category: Category label (checkpoint, i2t, lora, text_encoder)
        seen_paths: Set of already added paths to avoid duplicates
    """
    if not os.path.exists(base_dir):
        return
    
    if seen_paths is None:
        seen_paths = set()
    
    # Find Diffusers roots
    diffusers_roots = set()
    for r, d, f in os.walk(base_dir):
        if "model_index.json" in f:
            diffusers_roots.add(r)
    
    for r, d, f in os.walk(base_dir):
        rel_path = os.path.relpath(r, base_dir)
        if rel_path == ".":
            rel_path = ""
        
        # Skip internals
        is_internal = False
        for root in diffusers_roots:
            if r != root and r.startswith(root):
                is_internal = True
                break
        
        # Diffusers folders
        if "model_index.json" in f:
            try:
                # Skip if already added
                if r in seen_paths:
                    continue
                seen_paths.add(r)
                
                display_name = rel_path or os.path.basename(r)
                json_path = os.path.join(r, "dataModels.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as jf:
                        data = json.load(jf)
                        if "name" in data:
                            display_name = data["name"]
                
                created_at = get_model_created_at(r)
                role = detect_model_role(r, category, "diffusers")
                
                models.append({
                    "id": os.path.join(category, rel_path or os.path.basename(r)),
                    "name": display_name,
                    "path": r,
                    "type": "diffusers",
                    "category": category,
                    "role": role,
                    "repo_id": rel_path,
                    "created_at": created_at
                })
            except Exception:
                pass
        
        # GGUF files
        for file in f:
            if file.endswith(".gguf"):
                # Skip text encoders
                if "text_encoder" in rel_path or "text_encoder" in file:
                    continue
                # Skip PhotoMaker adapter models (used internally by face-swap)
                if "photomaker" in file.lower() or "pmid" in file.lower():
                    continue
                file_path = os.path.join(r, file)
                # Skip if already added
                if file_path in seen_paths:
                    continue
                seen_paths.add(file_path)
                role = detect_model_role(file_path, category, "gguf")
                models.append({
                    "id": os.path.join(category, rel_path, file) if rel_path else os.path.join(category, file),
                    "name": f"{rel_path}/{file}" if rel_path else file,
                    "path": file_path,
                    "type": "gguf",
                    "category": category,
                    "role": role,
                    "repo_id": rel_path,
                    "created_at": get_model_created_at(file_path)
                })
            
            # Safetensors
            elif file.endswith(".safetensors"):
                # Skip PhotoMaker adapter models (used internally by face-swap)
                if "photomaker" in file.lower() or "pmid" in file.lower():
                    continue
                if is_internal:
                    continue
                if "model_index.json" in f:
                    continue
                file_path = os.path.join(r, file)
                # Skip if already added
                if file_path in seen_paths:
                    continue
                seen_paths.add(file_path)
                role = detect_model_role(file_path, category, "custom_file")
                models.append({
                    "id": os.path.join(category, rel_path, file) if rel_path else os.path.join(category, file),
                    "name": f"{rel_path}/{file}" if rel_path else file,
                    "path": file_path,
                    "type": "custom_file",
                    "category": category,
                    "role": role,
                    "repo_id": rel_path,
                    "created_at": get_model_created_at(file_path)
                })
