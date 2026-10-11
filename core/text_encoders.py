"""
Core Text Encoders - Text encoder search and management
"""
import os
import sys
from typing import List, Dict, Optional, Tuple

# Cache for selected encoders per model
SELECTED_ENCODERS = {}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_utils


def scan_text_encoders(model_path: str) -> List[Dict]:
    """
    Finds text encoders associated with a model
    
    Args:
        model_path: path to the model
        
    Returns:
        List of dicts with path, name, source, type
    """
    if not model_path or not os.path.exists(model_path):
        return []
    
    encoders = []
    seen = set()
    
    def add_enc(path: str, source: str, enc_type: str = "qwen"):
        if path not in seen:
            encoders.append({
                "path": path,
                "name": os.path.basename(path),
                "source": source,
                "type": enc_type
            })
            seen.add(path)
    
    # 1. text_encoders subfolder
    sub_te = os.path.join(model_path, "text_encoders")
    if os.path.isdir(sub_te):
        for f in os.listdir(sub_te):
            if f.endswith(".gguf"):
                if "qwen" in f.lower():
                    add_enc(os.path.join(sub_te, f), "Embedded", "qwen")
                elif "t5" in f.lower():
                    add_enc(os.path.join(sub_te, f), "Embedded", "t5")
                elif "clip" in f.lower():
                    add_enc(os.path.join(sub_te, f), "Embedded", "clip")
    
    # 2. Sibling folder
    parent = os.path.dirname(model_path)
    sibling_te = os.path.join(parent, "text_encoders")
    if os.path.isdir(sibling_te):
        for f in os.listdir(sibling_te):
            if f.endswith(".gguf"):
                if "qwen" in f.lower():
                    add_enc(os.path.join(sibling_te, f), "Folder", "qwen")
                elif "t5" in f.lower():
                    add_enc(os.path.join(sibling_te, f), "Folder", "t5")
                elif "clip" in f.lower():
                    add_enc(os.path.join(sibling_te, f), "Folder", "clip")
    
    # 3. Z-Image-Turbo-GGUF
    common_te = os.path.join(shared_utils.MODELS_DIR, "Z-Image-Turbo-GGUF", "text_encoders")
    if os.path.isdir(common_te):
        for f in os.listdir(common_te):
            if f.endswith(".gguf"):
                if "qwen" in f.lower():
                    add_enc(os.path.join(common_te, f), "Shared", "qwen")
                elif "t5" in f.lower():
                    add_enc(os.path.join(common_te, f), "Shared", "t5")
                elif "clip" in f.lower():
                    add_enc(os.path.join(common_te, f), "Shared", "clip")
    
    # 1. text_encoders subfolder
    sub_te = os.path.join(model_path, "text_encoders")
    if os.path.isdir(sub_te):
        for f in os.listdir(sub_te):
            if f.endswith((".gguf", ".safetensors")):
                if "qwen" in f.lower():
                    add_enc(os.path.join(sub_te, f), "Embedded", "qwen")
                elif "mistral" in f.lower():
                    add_enc(os.path.join(sub_te, f), "Embedded", "mistral")
                elif "t5" in f.lower():
                    add_enc(os.path.join(sub_te, f), "Embedded", "t5")
                elif "clip" in f.lower():
                    add_enc(os.path.join(sub_te, f), "Embedded", "clip")
    
    # 2. Sibling folder
    parent = os.path.dirname(model_path)
    sibling_te = os.path.join(parent, "text_encoders")
    if os.path.isdir(sibling_te) and sibling_te != sub_te:
        for f in os.listdir(sibling_te):
            if f.endswith((".gguf", ".safetensors")):
                if "qwen" in f.lower():
                    add_enc(os.path.join(sibling_te, f), "Folder", "qwen")
                elif "mistral" in f.lower():
                    add_enc(os.path.join(sibling_te, f), "Folder", "mistral")
                elif "t5" in f.lower():
                    add_enc(os.path.join(sibling_te, f), "Folder", "t5")
                elif "clip" in f.lower():
                    add_enc(os.path.join(sibling_te, f), "Folder", "clip")
    
    # 3. Z-Image-Turbo-GGUF (Legacy check)
    common_te = os.path.join(shared_utils.MODELS_DIR, "Z-Image-Turbo-GGUF", "text_encoders")
    if os.path.isdir(common_te):
        for f in os.listdir(common_te):
            if f.endswith((".gguf", ".safetensors")):
                if "qwen" in f.lower():
                    add_enc(os.path.join(common_te, f), "Shared", "qwen")
                elif "mistral" in f.lower():
                    add_enc(os.path.join(common_te, f), "Shared", "mistral")
                elif "t5" in f.lower():
                    add_enc(os.path.join(common_te, f), "Shared", "t5")
                elif "clip" in f.lower():
                    add_enc(os.path.join(common_te, f), "Shared", "clip")
                    
    # 4. Global text_encoders folder
    global_te = os.path.join(shared_utils.MODELS_DIR, "text_encoders")
    if os.path.isdir(global_te):
        for f in os.listdir(global_te):
            if f.endswith((".gguf", ".safetensors")):
                if "qwen" in f.lower():
                    add_enc(os.path.join(global_te, f), "Global", "qwen")
                elif "mistral" in f.lower():
                    add_enc(os.path.join(global_te, f), "Global", "mistral")
                elif "t5" in f.lower():
                    add_enc(os.path.join(global_te, f), "Global", "t5")
                elif "clip" in f.lower():
                    add_enc(os.path.join(global_te, f), "Global", "clip")
    
    return encoders


def find_text_encoders_for_gguf(gguf_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds clip_l and t5xxl for Pony/Flux models
    
    Args:
        gguf_path: path to the GGUF file
        
    Returns:
        (clip_l_path, t5xxl_path)
    """
    search_dirs = [
        os.path.join(shared_utils.MODELS_DIR, "text_encoders"),
        shared_utils.MODELS_DIR,
        os.path.dirname(gguf_path)
    ]
    
    clip_l = None
    t5xxl = None
    
    # Search Clip L
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if "clip" in f.lower() and "_l" in f.lower() and f.endswith((".safetensors", ".gguf")):
                clip_l = os.path.join(d, f)
                break
        if clip_l:
            break
    
    # Search T5XXL
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if "t5" in f.lower() and "xxl" in f.lower() and f.endswith((".safetensors", ".gguf")):
                t5xxl = os.path.join(d, f)
                break
        if t5xxl:
            break
    
    return clip_l, t5xxl


def get_selected_encoder(model_path: str) -> Optional[str]:
    """
    Get the selected text encoder for a specific model
    
    Args:
        model_path: path to the model
        
    Returns:
        Path to the selected encoder or None if using default
    """
    return SELECTED_ENCODERS.get(model_path)


def set_selected_encoder(model_path: str, encoder_path: Optional[str]):
    """
    Set the selected text encoder for a specific model
    
    Args:
        model_path: path to the model
        encoder_path: path to the encoder or None to use default
    """
    if encoder_path:
        SELECTED_ENCODERS[model_path] = encoder_path
    else:
        SELECTED_ENCODERS.pop(model_path, None)


def clear_selected_encoders():
    """Clear all selected encoders"""
    SELECTED_ENCODERS.clear()


def get_all_selected_encoders() -> Dict[str, Optional[str]]:
    """Get all selected encoders"""
    return SELECTED_ENCODERS.copy()
