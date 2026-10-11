"""
Workers Base - Base class for workers
"""
import os
import sys
import subprocess
import time
import json
import shutil
import re
from typing import Optional, Dict, List

# Add current directory to path to import shared_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_utils


def get_gpu_memory() -> float:
    """Returns VRAM used in MB"""
    try:
        rocm_smi = shutil.which("rocm-smi")
        if not rocm_smi:
            for path in ["/opt/rocm/bin/rocm-smi", "/usr/bin/rocm-smi"]:
                if os.path.exists(path):
                    rocm_smi = path
                    break
        
        if rocm_smi:
            result = subprocess.run(
                [rocm_smi, "--showmeminfo", "vram", "--json"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                total_used = 0
                for card in data:
                    if "VRAM Total Used Memory (B)" in data[card]:
                        total_used += int(data[card]["VRAM Total Used Memory (B)"])
                return total_used / (1024 * 1024)
        
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return sum(int(x) for x in lines)
    except Exception:
        pass
    return 0.0


def get_total_gpu_memory() -> float:
    """Returns total VRAM in MB"""
    try:
        rocm_smi = shutil.which("rocm-smi")
        if not rocm_smi:
            for path in ["/opt/rocm/bin/rocm-smi", "/usr/bin/rocm-smi"]:
                if os.path.exists(path):
                    rocm_smi = path
                    break
        
        if rocm_smi:
            result = subprocess.run(
                [rocm_smi, "--showmeminfo", "vram", "--json"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                total = 0
                for card in data:
                    if "VRAM Total Memory (B)" in data[card]:
                        total += int(data[card]["VRAM Total Memory (B)"])
                if total > 0:
                    return total / (1024 * 1024)
        
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return sum(float(x) for x in lines)
    except Exception:
        pass
    return 8192.0 # Default fallback


def find_model_file(model_path: str) -> Optional[str]:
    """Finds the diffusion model file"""
    if not model_path:
        return None
    
    if os.path.isfile(model_path):
        return model_path
    
    exclude_dirs = {"text_encoder", "text_encoders", "vae", "tokenizer", "scheduler", "model"}
    
    if os.path.isdir(model_path):
        for f in sorted(os.listdir(model_path)):
            if f.endswith(".gguf") and "qwen" not in f.lower():
                return os.path.join(model_path, f)
        
        for f in sorted(os.listdir(model_path)):
            if f.endswith(".safetensors") and "qwen" not in f.lower():
                if "text_encoder" not in f.lower():
                    return os.path.join(model_path, f)
        
        for root, dirs, files in os.walk(model_path):
            dirs[:] = [d for d in dirs if d.lower() not in exclude_dirs]
            for f in sorted(files):
                if f.endswith(".gguf") and "qwen" not in f.lower():
                    return os.path.join(root, f)
            for f in sorted(files):
                if f.endswith(".safetensors") and "qwen" not in f.lower():
                    if "text_encoder" not in f.lower():
                        return os.path.join(root, f)
    
    return None


def find_bf16_zimage_alternative(model_path: str) -> Optional[str]:
    """On RDNA3 (AMD gfx11), Q8_0 GGUF matmul kernels (vec-dot/dp4a path)
    are ~2x slower than rocBLAS WMMA on BF16/F16 safetensors models
    (which use the RDNA3 tensor cores).

    If the user selected a Z-Image Q8_0 (or other quantized) GGUF and a
    BF16/F16 .safetensors sibling exists in the same directory, return it
    so the worker can transparently upgrade to the faster format.
    """
    if not model_path or not model_path.endswith(".gguf"):
        return None
    low = model_path.lower()
    if not any(x in low for x in ["z_image", "z-image", "zimage", "lumina", "glm-4", "z-image-turbo"]):
        return None
    d = os.path.dirname(model_path)
    if not os.path.isdir(d):
        return None
    candidates = []
    for f in os.listdir(d):
        if not f.endswith(".safetensors"):
            continue
        fl = f.lower()
        # Skip non-diffusion files (LLMs, encoders, VAEs)
        if any(x in fl for x in ["qwen", "text_encoder", "clip", "t5", "vae", "tokenizer"]):
            continue
        if any(x in fl for x in ["bf16", "fp16", "bfloat16", "float16", "f16"]):
            candidates.append(f)
    if not candidates:
        return None
    # Prefer BF16 over F16, then deterministic by name
    candidates.sort(key=lambda f: (0 if ("bf16" in f.lower() or "bfloat16" in f.lower()) else 1, f))
    best = os.path.join(d, candidates[0])
    if os.path.exists(best):
        return best
    return None


def find_vae(model_path: str) -> Optional[str]:
    """Finds VAE file"""
    parent = os.path.dirname(model_path)
    models_root = shared_utils.MODELS_DIR
    
    search_paths = [
        os.path.join(models_root, "Z-Image-Turbo", "vae", "diffusion_pytorch_model.safetensors"),
        os.path.join(models_root, "ae.sft"),
        os.path.join(os.path.dirname(model_path), "ae.sft"),
    ]
    
    for p in search_paths:
        if os.path.exists(p):
            return p
    return None


def find_llm(model_path: str, override: str = None) -> Optional[str]:
    """Finds LLM (Qwen) for Z-Image"""
    if override and os.path.exists(override):
        return override
    
    models_root = shared_utils.MODELS_DIR
    parent = os.path.dirname(model_path)
    
    search_dirs = [
        os.path.join(models_root, "Z-Image-GGUF"),
        os.path.join(models_root, "Z-Image-Turbo-GGUF"),
        models_root,
        os.path.join(parent, "text_encoders"),
        parent,
    ]
    
    found = []
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".gguf") and any(x in f.lower() for x in ["qwen", "text_encoder", "llm"]):
                found.append(os.path.join(d, f))
    
    if not found:
        return None
    
    for p in found:
        if "qwen_zimage_text_encoder" in os.path.basename(p).lower():
            return p
    for p in found:
        if "4b" in os.path.basename(p).lower():
            return p
    return found[0] if found else None


def apply_prompt_template(prompt: str) -> str:
    """Applies Qwen prompt template for Z-Image"""
    if "<|im_start|>" not in prompt:
        return f"<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"
    return prompt


def is_zimage_model(model_path: str) -> bool:
    """Detects if a model is Z-Image"""
    if not model_path:
        return False
    path_lower = model_path.lower()
    if any(x in path_lower for x in ["z_image", "z-image", "glm-4", "lumina", "zimage"]):
        return True
        
    # Fallback: check inside safetensors/gguf for cap_embedder
    try:
        with open(model_path, "rb") as f:
            if model_path.lower().endswith(".safetensors"):
                import struct, json
                # Read header size (8 bytes, little integer)
                header_data = f.read(8)
                if len(header_data) == 8:
                    header_size = struct.unpack('<Q', header_data)[0]
                    # Protect against insanely large headers
                    if header_size < 15000000:
                        header_string = f.read(header_size).decode('utf-8', errors='ignore')
                        if "cap_embedder" in header_string:
                            return True
            elif model_path.lower().endswith(".gguf"):
                # Fast string scan for GGUF
                content = f.read(1024 * 1024) # Read first MB
                if b"cap_embedder" in content:
                    return True
    except Exception:
        pass
        
    return False


def is_qwen_image_2_1_model(model_path) -> bool:
    """Detects Qwen-Image-2.1 models (T2I + optional editing).

    Qwen-Image-2.1 is a different architecture from Qwen-Image/Edit
    (32 layers, patch_size=1, Qwen3-VL-8B text encoder, its own VAE) and is
    loaded with `--diffusion-model` + `--llm` + `--vae`.
    """
    if not model_path or not isinstance(model_path, str):
        return False
    p = model_path.replace("\\", "/").lower()
    if any(x in p for x in ["qwen_image_2.1", "qwen-image-2.1", "qwen_image_2_1",
                            "qwen-image-2_1", "qwen_image_2-1"]):
        return True
    if "/qwen-2.1/" in p or p.endswith("/qwen-2.1"):
        return True
    # Folder-level marker from dataModels.json
    try:
        from core.model_config import load_model_config
        cfg = load_model_config(model_path)
        if str(cfg.get("type", "")).lower() in ("qwen-image-2.1", "qwen_image_2_1"):
            return True
    except Exception:
        pass
    return False


def is_qwen3vl_8b_encoder(path) -> bool:
    """True if *path* looks like a Qwen3-VL-8B text encoder.

    Qwen-Image-2.1 hard-requires hidden size 4096 (Qwen3-VL-8B). Passing a
    4B encoder (hidden 2560) makes sd.cpp abort with
    "Qwen Image 2.1 requires an image latent and text conditioning with batch
    size 1" — the real cause is the context_dim mismatch, not the batch.
    """
    if not path or not isinstance(path, str):
        return False
    b = os.path.basename(path).lower()
    if "mmproj" in b:
        return False
    if not b.endswith((".gguf", ".safetensors")):
        return False
    if "qwen3" not in b and "qwen-3" not in b:
        return False
    if "vl" not in b:
        return False
    return "8b" in b


def is_qwen_image_edit_model(model_path) -> bool:
    """Detects Qwen-Image-Edit-family diffusion models (qwen-image-edit-2511,
    Qwen-Rapid-*, ...). These are *diffusion-only* GGUF (bare tensor names like
    img_in.* / transformer_blocks.*, no text encoder embedded) that perform
    semantic edits via prompt + reference image (Qwen2.5-VL LLM + zero-cond-t).
    sd.cpp cannot open them as a full checkpoint (-m) - they must run through
    workers/i2i_edit.py with --diffusion-model + --llm + -r reference.

    Signature: same 1934-tensor layout as qwen-image-edit-2511, including the
    '__index_timestep_zero__' reference-conditioning marker.
    """
    if not model_path or not isinstance(model_path, str):
        return False
    # Qwen-Image-2.1 also supports editing, but it is a full model (T2I too)
    # and must NOT be treated as the diffusion-only Edit family.
    if is_qwen_image_2_1_model(model_path):
        return False
    base = os.path.basename(model_path).lower()
    is_model_file = base.endswith((".gguf", ".safetensors", ".sft"))
    # Fast path: known naming conventions (only on real model files, so the
    # .is_fixed.json / .log sidecars of a rapid model are not misdetected)
    if is_model_file and ("qwen-image-edit" in base or ("qwen" in base and "rapid" in base)):
        return True
    # Content signature: the '__index_timestep_zero__' reference marker +
    # bare 'img_in.weight' (diffusion-only) identify the edit family reliably.
    if base.endswith(".gguf") and os.path.isfile(model_path):
        try:
            with open(model_path, "rb") as f:
                head = f.read(8 * 1024 * 1024)
            if b"__index_timestep_zero__" in head and b"img_in.weight" in head:
                return True
        except Exception:
            pass
    return False


def get_scheduler_arg(scheduler: str, is_zimage: bool = False) -> str:
    """Maps scheduler name to CLI argument"""
    if is_zimage:
        return "euler_a"
    mapping = {
        "euler_a": "euler_a",
        "euler": "euler",
        "dpm++_2m_karras": "dpm++2m",
        "dpm++_sde_karras": "dpm++2s_a",
        "ddim": "euler"
    }
    return mapping.get(scheduler, "euler")


class TilingConfig:
    """VAE Tiling configuration calibrated for VRAM
    
    NOTE: VAE tiling forces sd.cpp to use CPU backend for the VAE,
    which is extremely slow (~1.5s per tile instead of ~0.5s total).
    On high-VRAM cards (>12GB), disable tiling for normal resolutions
    so the VAE stays on GPU for maximum performance.
    """
    
    def __init__(self, enabled: bool = True, threshold_mp: Optional[float] = None):
        self.enabled = enabled
        
        total_vram = get_total_gpu_memory()
        
        # Adaptive defaults based on total VRAM (Tile sizes are in LATENT space)
        # Tiling forces VAE to CPU, so prefer disabling for high-VRAM cards
        if threshold_mp is None:
            if total_vram <= 4200: # 4GB
                self.threshold_mp = 0.25 # Tile almost everything
                self.tile_size = 32 # 256x256 pixels
            elif total_vram <= 8200: # 8GB
                self.threshold_mp = 0.5
                self.tile_size = 64 # 512x512 pixels
            elif total_vram <= 12000: # 12GB
                self.threshold_mp = 2.0 # Only tile >2MP (e.g. 1920x1080+)
                self.tile_size = 96 # 768x768 pixels
            else: # 16GB+ (e.g. 7900 XTX 24GB)
                self.threshold_mp = 6.0 # Only tile >6MP (e.g. 3072x2048+)
                self.tile_size = 128 # 1024x1024 pixels (only used for very large images)
        else:
            self.threshold_mp = threshold_mp
            # Even if threshold is manual, cap tile_size for low VRAM
            if total_vram <= 4200:
                self.tile_size = 32
            elif total_vram <= 8200:
                self.tile_size = 64
            elif total_vram <= 12000:
                self.tile_size = 96
            else:
                self.tile_size = 128
            
        self.threshold = self.threshold_mp * 1024 * 1024
    
    def should_tile(self, width: int, height: int) -> bool:
        if not self.enabled:
            return False
        return (width * height) >= self.threshold
    
    def get_args(self, width: int, height: int) -> List[str]:
        if self.should_tile(width, height):
            return ["--vae-tiling", "--vae-tile-size", str(self.tile_size)]
        return []


def save_output_log(mode: str, inputs: dict, params: dict, outputs: dict, 
                   image_path: str = None, model_name: str = "unknown"):
    """Saves JSON metadata for output"""
    if image_path is None:
        return
    try:
        import shared_utils
        shared_utils.save_generation_log(
            mode, inputs, params, outputs,
            image_path_for_filename=image_path,
            model_name=model_name
        )
    except Exception as e:
        print(f"Warning: Could not save log: {e}")
