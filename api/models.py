"""
API Models - Model management
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import subprocess
import threading

from core.models import list_all_models, scan_models
from core.text_encoders import scan_text_encoders

router = APIRouter()

from shared_utils import MODELS_DIR, CHECKPOINTS_DIR, I2T_DIR, LORAS_DIR, OUTPUT_DIR


class DeleteModelRequest(BaseModel):
    model_id: str


class InspectModelRequest(BaseModel):
    repo_id: str
    token: Optional[str] = None


class DownloadModelRequest(BaseModel):
    repo_id: str
    filenames: Optional[List[str]] = None
    subdirectory: Optional[str] = None
    token: Optional[str] = None
    target_category: Optional[str] = None  # "checkpoints", "i2t", "loras"


@router.get("/")
async def get_models():
    """List all models"""
    return {"models": list_all_models()}


@router.post("/delete")
async def delete_model(req: DeleteModelRequest):
    """Delete a model"""
    if ".." in req.model_id or req.model_id.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid model ID")
    
    target = os.path.join(MODELS_DIR, req.model_id)
    target = os.path.abspath(target)
    
    if not target.startswith(MODELS_DIR):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
        return {"status": "success", "message": f"Deleted {req.model_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inspect")
async def inspect_model(req: InspectModelRequest):
    """HF model info"""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=req.token)
        info = api.model_info(req.repo_id, files_metadata=True)
        files = []
        for s in info.siblings:
            file_info = {
                "name": s.rfilename,
                "size": s.size or 0
            }
            # Include creation time if available
            if hasattr(s, 'last_modified') and s.last_modified:
                file_info["created_at"] = s.last_modified.timestamp()
            files.append(file_info)
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}


@router.post("/download")
async def download_model(req: DownloadModelRequest):
    """Download model (HF or Civitai)"""
    from core.config import load_secrets
    secrets = load_secrets()
    
    # Resolve target directory based on category
    target_dir = MODELS_DIR
    if req.target_category:
        category_map = {
            "checkpoints": CHECKPOINTS_DIR,
            "i2t": I2T_DIR,
            "loras": LORAS_DIR
        }
        target_dir = category_map.get(req.target_category, MODELS_DIR)
    
    # 1. Civitai Logic
    if "civitai.com" in req.repo_id:
        try:
            url = req.repo_id
            token = req.token or secrets.get("civitai_token")
            
            # Construct download command (wget)
            # We use wget because it handles content-disposition filenames well
            cmd = ["wget", "--content-disposition", "-P", target_dir]
            
            if token:
                # Civitai uses Authorization: Bearer <token> OR ?token=...
                # Appending param is safer for wget if not using headers param carefully
                if "?" in url:
                    url += f"&token={token}"
                else:
                    url += f"?token={token}"
            
            cmd.append(url)
            
            # Run in background? Ideally yes, but for now blocking or async
            # The user expects a "Started" message.
            # We'll launch it as a subprocess and return success.
            subprocess.Popen(cmd)
            
            return {"status": "success", "message": "Civitai download started in background check server logs"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # 2. Hugging Face Logic
    else:
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            
            token = req.token or secrets.get("hf_token")
            # Build local_dir: if subdirectory provided, append to target_dir; otherwise use target_dir/repo_name
            if req.subdirectory:
                local_dir = os.path.join(target_dir, req.subdirectory)
            else:
                model_name = req.repo_id.split("/")[-1]
                local_dir = os.path.join(target_dir, model_name)
            
            # Background task wrapper
            def run_hf_download():
                try:
                    if req.filenames:
                        for f in req.filenames:
                            print(f"--> Downloading {f} from {req.repo_id}...")
                            hf_hub_download(
                                repo_id=req.repo_id,
                                filename=f,
                                local_dir=local_dir,
                                token=token,
                                local_dir_use_symlinks=False
                            )
                    else:
                        print(f"--> Downloading full repo {req.repo_id}...")
                        snapshot_download(
                            repo_id=req.repo_id,
                            local_dir=local_dir,
                            token=token,
                            local_dir_use_symlinks=False
                        )
                    print("--> Download completed.")
                except Exception as e:
                    print(f"--> Download failed: {e}")

            # Run in separate thread/process
            import threading
            t = threading.Thread(target=run_hf_download)
            t.start()
            
            return {"status": "success", "message": f"HF Download started for {req.repo_id}"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


@router.post("/text_encoders")
async def get_text_encoders(payload: dict):
    """Find text encoders for a model"""
    model_path = payload.get("model_path")
    encoders = scan_text_encoders(model_path)
    return {"encoders": encoders}

@router.post("/text_encoders/select")
async def select_text_encoder(payload: dict):
    """Select a text encoder for a model"""
    model_path = payload.get("model_path")
    encoder_path = payload.get("encoder_path")
    
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Invalid model path")
    
    if encoder_path and not os.path.exists(encoder_path):
        raise HTTPException(status_code=400, detail="Encoder not found")
    
    from core.text_encoders import set_selected_encoder
    set_selected_encoder(model_path, encoder_path)
    
    return {"status": "success", "message": "Text encoder selected"}

@router.get("/text_encoders/selected")
async def get_selected_encoder(model_path: str):
    """Get the selected text encoder for a model"""
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Invalid model path")
    
    from core.text_encoders import get_selected_encoder
    encoder_path = get_selected_encoder(model_path)
    
    if encoder_path:
        return {"encoder_path": encoder_path}
    else:
        return {"encoder_path": None, "message": "Using model default text encoder"}

@router.post("/text_encoders/select")
async def select_text_encoder(payload: dict):
    """Select a text encoder for a model"""
    model_path = payload.get("model_path")
    encoder_path = payload.get("encoder_path")
    
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Invalid model path")
    
    if encoder_path and not os.path.exists(encoder_path):
        raise HTTPException(status_code=400, detail="Encoder not found")
    
    from core.text_encoders import set_selected_encoder
    set_selected_encoder(model_path, encoder_path)
    
    return {"status": "success", "message": "Text encoder selected"}

@router.get("/text_encoders/selected")
async def get_selected_encoder(model_path: str):
    """Get the selected text encoder for a model"""
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Invalid model path")
    
    from core.text_encoders import get_selected_encoder
    encoder_path = get_selected_encoder(model_path)
    
    if encoder_path:
        return {"encoder_path": encoder_path}
    else:
        return {"encoder_path": None, "message": "Using model default text encoder"}


@router.get("/loras")
async def get_loras(folder: str = LORAS_DIR):
    """List LoRA files in a folder, enriched with sidecar metadata"""
    from core.lora_meta import summarize
    files = []
    try:
        if os.path.exists(folder):
            for f in sorted(os.listdir(folder)):
                path = os.path.join(folder, f)
                if os.path.isfile(path) and f.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                    entry = {
                        "name": f,
                        "path": path,
                        "size": os.path.getsize(path)
                    }
                    try:
                        entry["meta"] = summarize(path)
                    except Exception as e:
                        print(f"--> [LoRA meta] summary failed for {f}: {e}", flush=True)
                        entry["meta"] = None
                    files.append(entry)
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}


class LoraMetaReadRequest(BaseModel):
    lora_path: str


class LoraMetaSaveRequest(BaseModel):
    lora_path: str
    meta: dict


def _validate_lora_path(lora_path: str) -> str:
    """Sanity checks: existing LoRA-ish file, no traversal escapes."""
    if not lora_path or ".." in lora_path.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="Invalid LoRA path")
    if not os.path.isfile(lora_path):
        raise HTTPException(status_code=404, detail="LoRA file not found")
    if not lora_path.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
        raise HTTPException(status_code=400, detail="Not a LoRA file")
    return lora_path


@router.post("/loras/meta/read")
async def read_lora_meta(req: LoraMetaReadRequest):
    """Read (or auto-generate) the metadata JSON for one LoRA, plus the list of
    checkpoint models to tag compatibility."""
    from core.lora_meta import load_meta, default_meta, detect_lora_info, build_checkpoint_list
    lora_path = _validate_lora_path(req.lora_path)

    existing = load_meta(lora_path)
    meta = existing if existing is not None else default_meta(lora_path)
    detected = detect_lora_info(lora_path)
    checkpoints, suggested, arch = build_checkpoint_list(lora_path)

    return {
        "file": {"name": os.path.basename(lora_path), "path": lora_path},
        "exists": existing is not None,
        "meta": meta,
        "detected": {
            "arch": detected.get("arch"),
            "arch_label": detected.get("arch"),
            "arch_sources": detected.get("arch_sources", []),
        },
        "checkpoints": checkpoints,
        "suggested_paths": suggested,
        "current_model": None,
    }


@router.post("/loras/meta/save")
async def save_lora_meta(req: LoraMetaSaveRequest):
    """Validate + persist the metadata JSON next to the LoRA file."""
    from core.lora_meta import save_meta
    lora_path = _validate_lora_path(req.lora_path)
    try:
        saved = save_meta(lora_path, req.meta or {})
        return {"status": "success", "meta": saved, "path": lora_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


