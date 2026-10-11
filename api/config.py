"""
API Config - Secrets and Settings
"""
import os
import shutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.config import (
    load_secrets, save_secrets,
    load_settings, save_settings
)

router = APIRouter()


class SecretsRequest(BaseModel):
    hf_token: Optional[str] = None
    civitai_token: Optional[str] = None


class SettingsRequest(BaseModel):
    vae_tiling_enabled: Optional[bool] = None
    auto_fix_models: Optional[bool] = None
    vae_tiling_threshold_mp: Optional[int] = None
    models_dir: Optional[str] = None
    outputs_dir: Optional[str] = None
    qwen_image_edit_repo: Optional[str] = None
    qwen_image_edit_model_path: Optional[str] = None


@router.get("/secrets")
async def get_secrets():
    """Returns secrets (masked)"""
    secrets = load_secrets()
    return {
        "hf_token": "******" if secrets.get("hf_token") else None,
        "civitai_token": "******" if secrets.get("civitai_token") else None
    }


@router.post("/secrets")
async def update_secrets(req: SecretsRequest):
    """Update secrets"""
    data = {}
    if req.hf_token is not None:
        data["hf_token"] = req.hf_token
    if req.civitai_token is not None:
        data["civitai_token"] = req.civitai_token
    
    if save_secrets(data):
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Failed to save secrets")


@router.get("/settings")
async def get_settings():
    """Returns settings"""
    from shared_utils import MODELS_DIR, OUTPUT_DIR
    data = load_settings()
    data["models_dir_resolved"] = MODELS_DIR
    data["outputs_dir_resolved"] = OUTPUT_DIR
    return data


@router.get("/qwen-edit-models")
async def get_qwen_edit_models():
    """Returns list of available Qwen Image Edit GGUF models"""
    from shared_utils import CHECKPOINTS_DIR
    import os
    
    models = []
    if os.path.exists(CHECKPOINTS_DIR):
        for f in os.listdir(CHECKPOINTS_DIR):
            if "qwen-image-edit" in f.lower() and f.endswith(".gguf"):
                models.append({
                    "name": f,
                    "path": os.path.join(CHECKPOINTS_DIR, f)
                })
    
    # Sort by name
    models.sort(key=lambda x: x["name"])
    return {"models": models}


@router.get("/i2t-models")
async def get_i2t_models():
    """Returns list of available I2T models.
    
    Phase 8C: Filters out auxiliary files (mmproj, projectors, tokenizers)
    that are not directly usable as the main I2T model. Only returns
    the actual chat models that the user can select.

    Each entry also carries `vision` / `mmproj`: whether the model has a
    vision projector in its folder (i.e. it can actually READ an image).
    Consumers that need image understanding (Agent Refine interpreter) grey
    out the entries where vision is false.
    """
    from shared_utils import I2T_DIR
    import os
    
    # Auxiliary file patterns to exclude (vision projectors, tokenizers, etc.)
    AUXILIARY_PATTERNS = ["mmproj", "projection", "proj.gguf", "tokenizer", "embed"]
    
    def is_auxiliary_file(filename: str) -> bool:
        """Check if a file is an auxiliary (non-main-model) file."""
        fn_lower = filename.lower()
        return any(pattern in fn_lower for pattern in AUXILIARY_PATTERNS)

    def _vision_info(full_path: str):
        """(vision, mmproj_basename) for a model file, via the same matcher the
        llama-server manager uses (so a 12B model never pairs a 26B mmproj)."""
        try:
            from api.llama_server_manager import find_gguf_model
            _g, mm, _fn = find_gguf_model(full_path)
            return bool(mm), (os.path.basename(mm) if mm else None)
        except Exception:
            return False, None
    
    models = []
    if os.path.exists(I2T_DIR):
        # Recursively scan for model files in subdirectories
        for root, dirs, files in os.walk(I2T_DIR):
            for f in files:
                # Only include main model files (gguf, safetensors, bin)
                if f.endswith((".gguf", ".safetensors", ".bin")):
                    # Skip auxiliary files like mmproj-*.gguf
                    if is_auxiliary_file(f):
                        continue
                    # Use relative path from I2T_DIR for display name
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, I2T_DIR)
                    vision, mmproj = _vision_info(full_path)
                    models.append({
                        "name": rel_path,
                        "path": full_path,
                        "dir": root,
                        "vision": vision,
                        "mmproj": mmproj,
                    })
    
    # Sort by name
    models.sort(key=lambda x: x["name"])
    return {"models": models}


def update_docker_script(models_dir: str = None, outputs_dir: str = None) -> dict:
    """
    Update go.sh with new storage paths.
    This allows Docker to mount external directories for models and outputs.
    """
    import re
    
    result = {"status": "success", "updated": []}
    
    # Scripts to update (go.sh auto-detects the AMD/NVIDIA GPU)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts = ["go.sh"]
    
    for script_name in scripts:
        script_path = os.path.join(base_dir, script_name)
        
        try:
            # Read the script
            with open(script_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Update OUTPUTS_DIR
            if outputs_dir is not None:
                # Match: OUTPUTS_DIR="..." or OUTPUTS_DIR="${OUTPUTS_DIR:-...}"
                pattern = r'(OUTPUTS_DIR[^\n]*)'
                match = re.search(pattern, content)
                if match:
                    old_line = match.group(1)
                    # Replace the path
                    new_line = f'OUTPUTS_DIR="{outputs_dir}"'
                    content = content.replace(old_line, new_line)
            
            # Update MODELS_DIR
            if models_dir is not None:
                # Match: MODELS_DIR="${MODELS_DIR:-...}" or MODELS_DIR="..."
                pattern = r'(MODELS_DIR[^\n]*)'
                match = re.search(pattern, content)
                if match:
                    old_line = match.group(1)
                    # Replace with the new path, preserving any variable reference
                    if '${MODELS_DIR:-' in old_line:
                        new_line = f'MODELS_DIR="${{MODELS_DIR:-{models_dir}}}"'
                    else:
                        new_line = f'MODELS_DIR="{models_dir}"'
                    content = content.replace(old_line, new_line)
            
            # Only write if content changed
            if content != original_content:
                with open(script_path, 'w') as f:
                    f.write(content)
                result["updated"].append(script_name)
                
        except Exception as e:
            result["status"] = "partial"
            result["error"] = f"Failed to update {script_name}: {str(e)}"
    
    return result


@router.post("/settings")
async def update_settings(req: SettingsRequest):
    """Update settings"""
    data = {}
    if req.vae_tiling_enabled is not None:
        data["vae_tiling_enabled"] = req.vae_tiling_enabled
    if req.auto_fix_models is not None:
        data["auto_fix_models"] = req.auto_fix_models
    if req.vae_tiling_threshold_mp is not None:
        data["vae_tiling_threshold_mp"] = req.vae_tiling_threshold_mp
    
    # Only save models_dir if it's a non-empty string (user explicitly set a path)
    if req.models_dir and req.models_dir.strip():
        data["models_dir"] = req.models_dir.strip()
    else:
        # Remove any existing custom models_dir to use default
        data["models_dir"] = None
    
    # Only save outputs_dir if it's a non-empty string (user explicitly set a path)
    if req.outputs_dir and req.outputs_dir.strip():
        data["outputs_dir"] = req.outputs_dir.strip()
    else:
        # Remove any existing custom outputs_dir to use default
        data["outputs_dir"] = None
    # Handle Qwen repos
    if req.qwen_image_edit_repo is not None:
        if req.qwen_image_edit_repo.strip():
            data["qwen_image_edit_repo"] = req.qwen_image_edit_repo.strip()
        else:
            data["qwen_image_edit_repo"] = None
    
    # Handle Qwen Image Edit model path
    if req.qwen_image_edit_model_path is not None:
        if req.qwen_image_edit_model_path.strip():
            data["qwen_image_edit_model_path"] = req.qwen_image_edit_model_path.strip()
        else:
            data["qwen_image_edit_model_path"] = None
    
    if save_settings(data):
        # NOTE: We intentionally do NOT auto-update go.sh from here.
        # The 'Models Directory' / 'Outputs Directory' fields in the popup
        # are *container* paths (read by shared_utils.py). go.sh's
        # OUTPUTS_DIR/MODELS_DIR are *host* bind-mount sources. They are
        # independent: changing the container path here does not move data
        # on the host, and rewriting go.sh with container paths would
        # silently break the next `./go.sh` invocation.
        # To change the host location of models/outputs, edit go.sh manually.
        return {"status": "success", "docker_scripts_updated": []}
    raise HTTPException(status_code=500, detail="Failed to save settings")


class VerifyRequest(BaseModel):
    type: str # 'hf' or 'civitai'
    token: str


@router.post("/verify")
async def verify_token(req: VerifyRequest):
    """Verify a token"""
    import verify_keys
    
    if req.type == "hf":
        is_valid = verify_keys.verify_hf(req.token)
        return {"valid": is_valid, "type": "hf"}
    elif req.type == "civitai":
        is_valid = verify_keys.verify_civitai(req.token)
        return {"valid": is_valid, "type": "civitai"}
    else:
        raise HTTPException(status_code=400, detail="Invalid token type")


class PathValidateRequest(BaseModel):
    path: str


@router.post("/validate_path")
async def validate_storage_path(req: PathValidateRequest):
    """Validate a storage path (check existence, permissions, free space)"""
    target_path = req.path.strip()
    
    if not target_path:
        raise HTTPException(status_code=400, detail="Path is empty")
    
    # Check if path is absolute (required for Docker volume mounts)
    if not os.path.isabs(target_path):
        raise HTTPException(
            status_code=400,
            detail="Path must be absolute (e.g., /home/user/models). Relative paths are not supported for Docker volume mounts."
        )
    
    # Check if path exists or can be created
    try:
        # Check if we are in Docker (heuristic: check for /.dockerenv)
        is_docker = os.path.exists('/.dockerenv')

        if not os.path.exists(target_path):
            # Path itself does not exist inside the container.
            # In Docker this almost always means the user entered a *host* path
            # (e.g. /home/user/models) that is not bind-mounted, OR the bind
            # mount in go.sh is missing/wrong. In either case the path is NOT
            # usable from inside the container, so report it as invalid instead
            # of silently allowing it.
            parent = os.path.dirname(target_path)
            if is_docker:
                if not parent or parent == target_path:
                    detail = f"Path '{target_path}' not found inside container. Enter the container path (e.g. /app/models) that is bind-mounted in go.sh."
                else:
                    detail = (
                        f"Path '{target_path}' not found inside container. "
                        f"If this is a host path, set the matching container "
                        f"path (e.g. /home/user/models on host -> /app/models "
                        f"in container) in the 'Models Directory' / 'Outputs "
                        f"Directory' field, or fix the -v bind mount in go.sh."
                    )
                raise HTTPException(status_code=400, detail=detail)
            # Non-Docker: require the parent to exist
            if not os.path.exists(parent):
                raise HTTPException(status_code=400, detail=f"Parent directory {parent} does not exist")
            if not os.access(parent, os.W_OK):
                raise HTTPException(status_code=400, detail=f"No write permission for {parent}")
        else:
            if not os.access(target_path, os.W_OK):
                raise HTTPException(status_code=400, detail=f"No write permission for {target_path}")

        # Check free space (at least 1GB required for basic operations)
        try:
            if os.path.exists(target_path):
                usage = shutil.disk_usage(target_path)
            else:
                usage = shutil.disk_usage(os.path.dirname(target_path))

            free_gb = usage.free / (1024**3)

            return {
                "valid": True,
                "free_gb": round(free_gb, 2),
                "message": f"Path is valid with {free_gb:.2f}GB free"
            }
        except Exception as e:
            # If disk_usage fails but the path was verified, report it as a warning
            return {
                "valid": True,
                "message": f"Path exists and is writable, but disk space check failed: {e}",
                "warning": True
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Path validation error: {e}")


class DockerRequest(BaseModel):
    action: str  # 'start' or 'stop'


@router.get("/docker/status")
async def docker_status():
    """Get Docker container status and socket availability"""
    import subprocess
    
    result = {
        "socket_available": False,
        "container_running": False,
        "container_exists": False,
        "container_name": None,
        "gpu_type": None
    }
    
    # Check if Docker socket is available
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=5,
            check=True
        )
        result["socket_available"] = True
    except Exception:
        return result
    
    # Check running containers
    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        running = [c for c in proc.stdout.strip().split('\n') if c]
    except Exception:
        running = []
    
    # Check all containers
    try:
        proc = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        all_containers = [c for c in proc.stdout.strip().split('\n') if c]
    except Exception:
        all_containers = []
    
    if "sdcpp_runner_nvidia" in running:
        result["container_running"] = True
        result["container_name"] = "sdcpp_runner_nvidia"
        result["gpu_type"] = "nvidia"
    elif "sdcpp_runner" in running:
        result["container_running"] = True
        result["container_name"] = "sdcpp_runner"
        result["gpu_type"] = "rocm"
    elif "sdcpp_runner_nvidia" in all_containers:
        result["container_exists"] = True
        result["container_name"] = "sdcpp_runner_nvidia"
        result["gpu_type"] = "nvidia"
    elif "sdcpp_runner" in all_containers:
        result["container_exists"] = True
        result["container_name"] = "sdcpp_runner"
        result["gpu_type"] = "rocm"
    
    return result


@router.post("/docker")
async def docker_control(req: DockerRequest):
    """Control Docker container (stop/start)"""
    import subprocess
    
    # Get status first
    status = await docker_status()
    
    if not status["socket_available"]:
        raise HTTPException(
            status_code=503,
            detail="Docker socket not available. Please restart Docker manually."
        )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    go_path = os.path.join(base_dir, "go.sh")
    
    container_name = status["container_name"]
    # go.sh rileva automaticamente la GPU (AMD/NVIDIA) e ricrea il container corretto.
    script_path = go_path
    
    if req.action == "stop":
        if not container_name or not status["container_running"]:
            return {"status": "success", "message": "No container running"}
        
        try:
            subprocess.run(["docker", "stop", container_name], check=True, timeout=30)
            return {"status": "success", "message": f"Container {container_name} stopped"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to stop container: {e}")
    
    elif req.action == "start":
        if not script_path or not os.path.exists(script_path):
            raise HTTPException(
                status_code=404,
                detail="Docker startup script not found. Please restart manually."
            )
        
        try:
            # Run the script in background
            subprocess.Popen(
                ["bash", script_path],
                cwd=base_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return {"status": "success", "message": "Docker container starting..."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start container: {e}")
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'start' or 'stop'")
