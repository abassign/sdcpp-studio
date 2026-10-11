"""
Workers Tag - AI-powered semantic tagging for images
Uses existing GLM-4.1V or llama-cli models to analyze images and generate tags.
Uses LlamaServerManager on port 8002 for persistent server-based inference.
"""
import os
import sys
import json
import time
import subprocess
import re
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.getcwd())
import shared_utils

# Configuration
TAGS_DB_PATH = "/app/data/semantic_tags.json"
CACHE_DIR = "/app/.tag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TAGS_DB_PATH), exist_ok=True)

# Global stop flag
tagging_stop_flag = False

# Global reference to the active tagging subprocess (for external kill)
active_tagging_process = None

# VRAM Arbiter integration
def _get_arbiter():
    from api.vram_arbiter import get_arbiter
    return get_arbiter()

def _check_arbiter_stop():
    """Check if arbiter wants us to stop (IA needs VRAM)"""
    try:
        arbiter = _get_arbiter()
        status = arbiter.get_status()
        state = status.get('state', 'idle')
        # Stop if handoff requested or IA wants VRAM while we're tagging
        if state == 'handoff_tag_to_ia':
            return True
        if status.get('ia_wants_vram') and status.get('token_holder') == 'tag':
            return True
    except:
        pass
    return False

# Model types
MODEL_TYPE_CPP = "cpp"  # Uses sd (stable-diffusion.cpp) binary with GGUF models
MODEL_TYPE_LLAMA = "llama"  # Uses llama-cli with GGUF + mmproj
MODEL_TYPE_SERVER = "server"  # Uses persistent LlamaServerManager on port 8002

# Server-based tagging model (uses LlamaServerManager on port 8002)
TAGGING_SERVER_MODEL = {
    "server": {
        "name": "Tagging Server (Port 8002)",
        "prompt": "Analyze this image and provide a concise description covering: subject, style, content, mood. Format as a single paragraph.",
        "type": MODEL_TYPE_SERVER,
        "path": None  # Auto-detects from I2T_DIR
    }
}


def discover_i2t_models() -> Dict[str, Dict]:
    """
    Discover all available I2T models in the models/i2t directory.
    Scans for GGUF files and builds the SUPPORTED_MODELS dict dynamically.
    Only includes models that are actually suitable for image-to-text (multimodal).
    
    Returns:
        Dict with model_id -> model_info mapping
    """
    models = {}
    i2t_dir = shared_utils.I2T_DIR
    
    print(f"--> [discover_i2t_models] Scanning directory: {i2t_dir}", flush=True)
    
    if not os.path.exists(i2t_dir):
        print(f"--> [discover_i2t_models] Directory does not exist", flush=True)
        return models
    
    # Scan for GGUF files (excluding mmproj files)
    for root, dirs, files in os.walk(i2t_dir):
        for f in files:
            if f.endswith(".gguf") and not f.startswith("mmproj"):
                full_path = os.path.join(root, f)
                # Use parent directory for the model, so workers can search within it
                model_dir = root
                rel_path = os.path.relpath(full_path, i2t_dir)
                
                # Generate a unique model ID from filename
                model_id = os.path.splitext(rel_path)[0]
                model_id = model_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
                
                # Check if there's a corresponding mmproj file in the same directory
                mmproj_path = None
                for mf in files:
                    if mf.startswith("mmproj") and mf.endswith(".gguf"):
                        mmproj_path = os.path.join(root, mf)
                        break
                
                # Determine model type based on naming AND presence of mmproj
                f_lower = f.lower()
                
                # Determine model type based on presence of mmproj (vision projector)
                if mmproj_path:
                    model_type = MODEL_TYPE_LLAMA
                else:
                    model_type = MODEL_TYPE_CPP
                
                models[model_id] = {
                    "name": rel_path,
                    "path": model_dir,  # Return directory, not file path
                    "file": full_path,  # Keep actual file path too
                    "mmproj": mmproj_path,
                    "type": model_type,
                    "prompt": "Describe this image concisely: subject, style, content, mood."
                }
                print(f"--> [discover_i2t_models] Found model: {model_id} -> {model_type} in {model_dir} (mmproj: {'yes' if mmproj_path else 'no'})", flush=True)
    
    print(f"--> [discover_i2t_models] Total multimodal models found: {len(models)}", flush=True)
    return models


# Supported models for tagging - built dynamically from discovered models
SUPPORTED_MODELS = {
    "glm_4v": {
        "name": "GLM-4.1V (Default)",
        "worker": "workers/analyze_cpp.py",
        "prompt": "Analyze this image and provide a concise description covering: subject, style, content, mood. Format as a single paragraph.",
        "type": MODEL_TYPE_CPP,
        "path": None  # Uses default I2T_DIR
    },
    "llama": {
        "name": "Llama CLI (Default)",
        "worker": "workers/i2t_llama.py",
        "prompt": "Describe this image concisely: subject, style, content, mood.",
        "type": MODEL_TYPE_LLAMA,
        "path": None  # Auto-detects from I2T_DIR
    }
}


def get_available_models() -> Dict[str, Dict]:
    """
    Get all available models including dynamically discovered ones.
    Merges default models with discovered GGUF models.
    """
    # Start with defaults
    all_models = dict(SUPPORTED_MODELS)
    
    # Add discovered models
    discovered = discover_i2t_models()
    for model_id, model_info in discovered.items():
        # Skip if already in defaults (by similar name)
        skip = False
        for default_id in ["glm_4v", "llama"]:
            if default_id in model_id.lower() or model_id.lower() in default_id:
                skip = True
                break
        
        if not skip:
            all_models[f"gguf_{model_id}"] = model_info
    
    return all_models


def load_tags_db() -> Dict:
    """Load tags database"""
    if os.path.exists(TAGS_DB_PATH):
        try:
            with open(TAGS_DB_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"version": "1.0", "tags": {}, "models_used": {}, "last_updated": None}


def save_tags_db(db: Dict):
    """Save tags database"""
    db["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(os.path.dirname(TAGS_DB_PATH), exist_ok=True)
    with open(TAGS_DB_PATH, 'w') as f:
        json.dump(db, f, indent=2)


def cleanup_missing_images(tags_db: Dict, output_dir: str) -> int:
    """
    Remove entries from tags_db whose image files no longer exist in output_dir.
    Uses efficient set-based lookups to minimize file system calls.
    
    Returns the number of entries removed.
    """
    if "tags" not in tags_db or not tags_db["tags"]:
        return 0
    
    # Build set of existing images in output_dir (single directory scan)
    existing_images = set()
    existing_bases = set()
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    
    try:
        for f in os.listdir(output_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                existing_images.add(f)
                existing_bases.add(os.path.splitext(f)[0])
    except OSError:
        return 0
    
    # Find keys to remove (images that don't exist)
    tags_to_remove = []
    for key in tags_db["tags"]:
        # Check exact match or basename match
        if key not in existing_images and os.path.splitext(key)[0] not in existing_bases:
            tags_to_remove.append(key)
    
    # Remove non-existent entries
    for key in tags_to_remove:
        del tags_db["tags"][key]
    
    return len(tags_to_remove)


def extract_keywords(text: str) -> List[Dict]:
    """
    Extract keywords from text and categorize them.
    Simple keyword extraction based on common categories.
    """
    tags = []
    text_lower = text.lower()
    
    # Subject categories
    subjects = ["landscape", "portrait", "animal", "person", "building", "nature", 
                "city", "mountain", "ocean", "forest", "sky", "water", "flower",
                "car", "vehicle", "food", "object", "abstract", "sport", "athlete",
                "running", "racing", "swimming", "cycling", "hiking"]
    
    for subj in subjects:
        if subj in text_lower:
            tags.append({"category": "subject", "value": subj, "confidence": 0.85})
    
    # Style categories
    styles = ["photorealistic", "anime", "cartoon", "painting", "drawing", 
              "digital", "sketch", "watercolor", "oil", "3d", "realistic",
              "stylized", "minimalist", "detailed", "abstract"]
    
    for style in styles:
        if style in text_lower:
            tags.append({"category": "style", "value": style, "confidence": 0.80})
    
    # Mood categories
    moods = ["serene", "dramatic", "peaceful", "energetic", "dark", "bright",
             "moody", "cheerful", "melancholic", "mysterious", "romantic",
             "epic", "calm", "intense"]
    
    for mood in moods:
        if mood in text_lower:
            tags.append({"category": "mood", "value": mood, "confidence": 0.75})
    
    # Content categories
    contents = ["sunset", "sunrise", "night", "day", "rain", "snow", "cloud",
                "tree", "grass", "rock", "sand", "beach", "river", "lake",
                "road", "bridge", "tower", "house", "castle", "temple"]
    
    for content in contents:
        if content in text_lower:
            tags.append({"category": "content", "value": content, "confidence": 0.80})
    
    # If no specific tags found, add a general one
    if not tags:
        # Use first few words as general tags
        words = text.split()[:5]
        for word in words:
            word = word.strip(".,!?;:()[]{}\"'").lower()
            if len(word) > 3:
                tags.append({"category": "general", "value": word, "confidence": 0.60})
    
    return tags


def analyze_with_glm(image_path: str, model_path: str = None) -> str:
    """
    Analyze image using GLM-4.1V via stable-diffusion.cpp
    Uses the existing analyze_cpp.py worker
    
    Args:
        image_path: Path to the image to analyze
        model_path: Optional specific model path (directory or GGUF file)
    """
    cmd = [
        sys.executable, "-u", "workers/analyze_cpp.py",
        "--image_path", image_path,
        "--prompt", "Describe this image concisely: subject, style, content, mood.",
        "--top_k", "1.0",
        "--temperature", "0.6",
        "--no-log"
    ]
    
    # Add model path if specified
    if model_path:
        cmd.extend(["--model_path", model_path])
    
    print(f"--> [analyze_with_glm] Running cmd: {' '.join(cmd[:5])}...", flush=True)
    
    try:
        global active_tagging_process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "0"}
        )
        active_tagging_process = process
        
        start_time = time.time()
        try:
            while True:
                if tagging_stop_flag:
                    try:
                        process.kill()
                        process.wait(timeout=5)
                    except:
                        pass
                    return "Error: Tagging interrupted by user"
                    
                if process.poll() is not None:
                    break
                    
                if time.time() - start_time > 120:
                    try:
                        process.kill()
                        process.wait(timeout=5)
                    except:
                        pass
                    return "Error: Analysis timed out"
                    
                time.sleep(0.1)
                
            stdout, stderr = process.communicate()
            print(f"--> [analyze_with_glm] returncode={process.returncode}", flush=True)
            
            if process.returncode == 0:
                output = stdout
                if "TEXT_OUTPUT_START" in output and "TEXT_OUTPUT_END" in output:
                    start = output.find("TEXT_OUTPUT_START") + len("TEXT_OUTPUT_START")
                    end = output.find("TEXT_OUTPUT_END")
                    return output[start:end].strip()
                return output.strip()[:500]
            else:
                return f"Error: {stderr[:200] if stderr else 'No stderr'}"
        finally:
            active_tagging_process = None
    except Exception as e:
        active_tagging_process = None
        print(f"--> [analyze_with_glm] Exception: {str(e)}", flush=True)
        return f"Error: {str(e)}"


def analyze_with_llama(image_path: str, model_path: str = None) -> str:
    """
    Analyze image using llama-cli
    Uses the existing i2t_llama.py worker
    
    Args:
        image_path: Path to the image to analyze
        model_path: Optional specific model path (directory containing GGUF and mmproj)
    """
    cmd = [
        sys.executable, "-u", "workers/i2t_llama.py",
        image_path,
        "--prompt", "Describe this image concisely: subject, style, content, mood.",
        "--temperature", "0.7",
        "--max-tokens", "512",
        "--no-log"
    ]
    
    # Add model path if specified
    if model_path:
        cmd.extend(["--model-dir", model_path])
    
    print(f"--> [analyze_with_llama] Running cmd: {' '.join(cmd[:5])}...", flush=True)
    
    global active_tagging_process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "0"}
        )
        active_tagging_process = process
        
        start_time = time.time()
        try:
            while True:
                if tagging_stop_flag:
                    try:
                        process.kill()
                        process.wait(timeout=5)
                    except:
                        pass
                    return "Error: Tagging interrupted by user"
                    
                if process.poll() is not None:
                    break
                    
                if time.time() - start_time > 120:
                    try:
                        process.kill()
                        process.wait(timeout=5)
                    except:
                        pass
                    return "Error: Analysis timed out"
                    
                time.sleep(0.1)
                
            stdout, stderr = process.communicate()
            print(f"--> [analyze_with_llama] returncode={process.returncode}", flush=True)
            
            if process.returncode == 0:
                output = stdout
                if "TEXT_OUTPUT_START" in output and "TEXT_OUTPUT_END" in output:
                    start = output.find("TEXT_OUTPUT_START") + len("TEXT_OUTPUT_START")
                    end = output.find("TEXT_OUTPUT_END")
                    return output[start:end].strip()
                return output.strip()[:500]
            else:
                return f"Error: {stderr[:200] if stderr else 'No stderr'}"
        finally:
            active_tagging_process = None
    except Exception as e:
        active_tagging_process = None
        print(f"--> [analyze_with_llama] Exception: {str(e)}", flush=True)
        return f"Error: {str(e)}"


def analyze_with_server(image_path: str, model_path: str = None) -> str:
    """
    Analyze image using the persistent LlamaServerManager on port 8002.
    This avoids loading/unloading the model for every single image.
    
    Args:
        image_path: Path to the image to analyze
        model_path: Optional specific model directory path.
    
    Returns:
        Summary text of the image analysis.
    """
    try:
        from api.llama_server_manager import get_vram_manager
        vram = get_vram_manager()
        
        # Determine model directory
        if model_path is None:
            model_path = shared_utils.I2T_DIR
        
        # Ensure the tagging server is running (port 8002)
        vram.ensure_tag_server(model_path)
        
        # Query the persistent server
        prompt = "Analyze this image and provide a concise description covering: subject, style, content, mood. Format as a single paragraph."
        result = vram.query_tag(image_path, prompt, temperature=0.6, max_tokens=512)
        
        response = result.get("response", "")
        if not response:
            response = result.get("raw_output", "")
        
        return response.strip()[:500] if response else "Error: Empty response from server"
    except RuntimeError as e:
        error_msg = str(e)
        if "IA server is active" in error_msg:
            print(f"--> [analyze_with_server] IA server active, tagging suspended: {error_msg}", flush=True)
            return f"Error: Tagging suspended - {error_msg}"
        print(f"--> [analyze_with_server] RuntimeError: {error_msg}", flush=True)
        return f"Error: {error_msg}"
    except Exception as e:
        print(f"--> [analyze_with_server] Exception: {str(e)}", flush=True)
        return f"Error: {str(e)}"


def _is_model_path(model_id: str) -> bool:
    """Check if model_id is a file path rather than a key name."""
    if not model_id:
        return False
    return (model_id.endswith(".gguf") or model_id.endswith(".safetensors") or
            "/" in model_id or "\\" in model_id)


def _resolve_model_path(model_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve a model_id (which may be a key or a full path) to (model_type, model_path, display_name).
    Returns (None, None, None) if not found.
    """
    # If it looks like a file path, use it directly
    if _is_model_path(model_id):
        if os.path.isfile(model_id):
            model_dir = os.path.dirname(model_id)
            # Check for mmproj in the same directory
            has_mmproj = False
            if os.path.isdir(model_dir):
                for f in os.listdir(model_dir):
                    if "mmproj" in f.lower() and f.endswith(".gguf"):
                        has_mmproj = True
                        break
            model_type = MODEL_TYPE_LLAMA if has_mmproj else MODEL_TYPE_CPP
            return model_type, model_id, os.path.basename(model_id)
        else:
            print(f"--> [_resolve_model_path] Model file not found: {model_id}", flush=True)
            return None, None, None
    
    # Otherwise treat as a key and look it up
    all_models = get_available_models()
    if model_id in all_models:
        info = all_models[model_id]
        return info.get("type", MODEL_TYPE_CPP), info.get("path"), model_id
    
    # Try matching gguf_ prefix
    if model_id.startswith("gguf_"):
        discovered = discover_i2t_models()
        for dk, dv in discovered.items():
            if f"gguf_{dk}" == model_id:
                return dv.get("type", MODEL_TYPE_CPP), dv.get("path"), model_id
    
    return None, None, None


def analyze_with_model(image_path: str, model_id: str = None) -> Tuple[str, str]:
    """
    Analyze image using a specific model or auto-detected model.
    
    Args:
        image_path: Path to the image to analyze
        model_id: Optional model ID/key from SUPPORTED_MODELS, or a full file path
                  (e.g., /app/models/i2t/qwen36/model.gguf).
                  If None, auto-discovers available models.
                  Use "server" to use the persistent LlamaServerManager on port 8002.
    
    Returns:
        Tuple of (summary_text, model_id_used)
    """
    print(f"--> [analyze_with_model] Called with model_id={model_id}, image={image_path}", flush=True)
    
    # Server mode
    if model_id == "server":
        print(f"--> [analyze_with_model] Using persistent server on port 8002", flush=True)
        summary = analyze_with_server(image_path, None)
        return summary, "server"
    
    # Resolve the model (handles both keys and full paths)
    if model_id:
        model_type, model_path, display_name = _resolve_model_path(model_id)
        
        if model_type and model_path:
            print(f"--> [analyze_with_model] Resolved: type={model_type}, path={model_path}", flush=True)
            if model_type == MODEL_TYPE_LLAMA:
                summary = analyze_with_llama(image_path, model_path)
            else:
                summary = analyze_with_glm(image_path, model_path)
            return summary, display_name or model_id
        
        print(f"--> [analyze_with_model] Model '{model_id}' not found, trying auto-discovery...", flush=True)
    
    # No model specified or not found - try to find a usable local model
    discovered = discover_i2t_models()
    best_model = None
    for dk, dv in discovered.items():
        if dv.get("type") == MODEL_TYPE_LLAMA:
            best_model = dv
            break
    if not best_model:
        for dk, dv in discovered.items():
            if dv.get("type") == MODEL_TYPE_CPP:
                best_model = dv
                break
    
    if best_model:
        model_type = best_model.get("type", MODEL_TYPE_CPP)
        model_path = best_model.get("path")
        print(f"--> [analyze_with_model] Using discovered model: type={model_type}, path={model_path}", flush=True)
        if model_type == MODEL_TYPE_LLAMA:
            summary = analyze_with_llama(image_path, model_path)
        else:
            summary = analyze_with_glm(image_path, model_path)
        return summary, best_model.get("name", "discovered")
    
    # Last resort: try server
    print(f"--> [analyze_with_model] No local model found, trying server...", flush=True)
    summary = analyze_with_server(image_path, None)
    return summary, "server"


def tag_single_image(image_path: str, model: str = "llama", force: bool = False) -> Dict:
    """
    Tag a single image with semantic analysis.
    
    Args:
        image_path: Path to the image file
        model: Model to use ("glm_4v", "llama", "server", or any discovered model/path)
        force: Force re-tagging even if already tagged
    
    Returns:
        Dict with tags, metadata, and elapsed_time
    """
    t_start = time.time()
    
    # Validate image path
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}", "elapsed_time": 0}
    
    # Get image basename for cache key
    image_basename = os.path.basename(image_path)
    
    # Load existing tags DB
    tags_db = load_tags_db()
    
    # Check cache (unless force)
    if not force and image_basename in tags_db.get("tags", {}):
        cached = tags_db["tags"][image_basename].copy()
        cached["elapsed_time"] = 0
        return cached
    
    # Select analyzer based on model ID (supports any discovered model)
    print(f"--> [tag_single_image] Calling analyze_with_model for {image_basename} with model={model}", flush=True)
    summary, model_used = analyze_with_model(image_path, model)
    print(f"--> [tag_single_image] Got summary, model_used={model_used}", flush=True)
    
    # Check for errors
    if summary and summary.startswith("Error:"):
        elapsed = time.time() - t_start
        print(f"--> [tag_single_image] Error in summary: {summary}", flush=True)
        return {"error": summary, "image": image_path, "elapsed_time": round(elapsed, 1)}
    
    # Extract tags from summary
    tags = extract_keywords(summary)
    
    elapsed = time.time() - t_start
    
    # Build result
    result = {
        "image": image_path,
        "generated_by": model_used,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tags": tags,
        "summary": summary,
        "manual_tags": [],
        "elapsed_time": round(elapsed, 1)
    }
    
    # Save to DB
    tags_db["tags"][image_basename] = result
    tags_db["models_used"][model_used] = time.strftime("%Y-%m-%d")
    save_tags_db(tags_db)
    
    return result


def tag_batch_images(image_paths: List[str], model: str = "server",
                     progress_callback=None, start_from: int = 0) -> Dict:
    """
    Tag multiple images with progress reporting.
    Uses persistent server (port 8002) by default for efficiency.
    Saves checkpoints for resumption if interrupted by IA server.
    
    Args:
        image_paths: List of image paths to tag
        model: Model to use (default: "server" for persistent LlamaServerManager)
        progress_callback: Optional callback(current, total, path)
        start_from: Index to start from (for resuming after IA interruption)
    
    Returns:
        Dict with results per image
    """
    results = {}
    total = len(image_paths)
    
    for idx in range(start_from, total):
        img_path = image_paths[idx]
        
        # Check for stop flag
        if tagging_stop_flag:
            print(f"--> [tag_batch_images] Stopped at idx {idx}/{total}", flush=True)
            # Save checkpoint
            try:
                from api.llama_server_manager import get_vram_manager
                vram = get_vram_manager()
                vram.save_checkpoint({
                    "current": idx,
                    "total": total,
                    "model": model,
                    "status": "stopped"
                })
            except Exception:
                pass
            break
        
        try:
            result = tag_single_image(img_path, model)
            results[img_path] = result
            
            # Check if result indicates IA suspension
            if "error" in result and "Tagging suspended" in result.get("error", ""):
                print(f"--> [tag_batch_images] IA suspension detected at idx {idx}", flush=True)
                # Save checkpoint for resumption
                try:
                    from api.llama_server_manager import get_vram_manager
                    vram = get_vram_manager()
                    vram.save_checkpoint({
                        "current": idx,
                        "total": total,
                        "model": model,
                        "status": "suspended_by_ia"
                    })
                except Exception:
                    pass
                break
            
            if progress_callback:
                progress_callback(idx + 1, total, img_path)
                
        except Exception as e:
            results[img_path] = {"error": str(e), "image": img_path}
    
    # Save final checkpoint if completed
    if len(results) + start_from >= total:
        try:
            from api.llama_server_manager import get_vram_manager
            vram = get_vram_manager()
            vram.save_checkpoint({})  # Clear checkpoint on completion
        except Exception:
            pass
    
    return results


def get_image_tags(image_name: str) -> Optional[Dict]:
    """Get tags for a specific image"""
    tags_db = load_tags_db()
    return tags_db.get("tags", {}).get(image_name)


def add_manual_tags(image_name: str, tags: List[str]) -> Dict:
    """Add manual tags to an image"""
    tags_db = load_tags_db()
    
    if image_name not in tags_db.get("tags", {}):
        return {"error": "Image not found in tags DB"}
    
    existing = set(tags_db["tags"][image_name].get("manual_tags", []))
    existing.update(tags)
    tags_db["tags"][image_name]["manual_tags"] = list(existing)
    
    save_tags_db(tags_db)
    return tags_db["tags"][image_name]


def remove_manual_tag(image_name: str, tag: str) -> Dict:
    """Remove a manual tag from an image"""
    tags_db = load_tags_db()
    
    if image_name not in tags_db.get("tags", {}):
        return {"error": "Image not found in tags DB"}
    
    manual = tags_db["tags"][image_name].get("manual_tags", [])
    if tag in manual:
        manual.remove(tag)
        tags_db["tags"][image_name]["manual_tags"] = manual
        save_tags_db(tags_db)
    
    return tags_db["tags"][image_name]


def get_tags_status() -> Dict:
    """Get tagging system status"""
    tags_db = load_tags_db()
    return {
        "total_tagged": len(tags_db.get("tags", {})),
        "models_used": tags_db.get("models_used", {}),
        "last_updated": tags_db.get("last_updated")
    }


def list_available_models() -> Dict[str, Dict]:
    """
    List all available models for tagging.
    Used by API to populate model selection dropdown.
    """
    return get_available_models()


if __name__ == "__main__":
    # Test mode
    import argparse
    
    # Get available models dynamically
    available_models = list(get_available_models().keys())
    
    parser = argparse.ArgumentParser(description="Tag images with AI")
    parser.add_argument("image", nargs="?", help="Image path to tag")
    parser.add_argument("--model", default="glm_4v", choices=available_models,
                        help="Model to use for tagging")
    parser.add_argument("--force", action="store_true", help="Force re-tagging")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models for tagging:")
        for mid, minfo in get_available_models().items():
            print(f"  {mid}: {minfo['name']} (type: {minfo.get('type', 'unknown')})")
    elif args.status:
        status = get_tags_status()
        print(f"Tagged images: {status['total_tagged']}")
        print(f"Models used: {status['models_used']}")
        print(f"Last updated: {status['last_updated']}")
    elif args.image:
        result = tag_single_image(args.image, args.model, args.force)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
