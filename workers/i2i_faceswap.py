"""
Workers I2I Face Swap - Face swapping using PhotoMaker via stable-diffusion.cpp
=================================================================================
Takes an init image (the scene) and a face image (source of the face) and uses
PhotoMaker to generate a new image where the face from the face image is placed
in the scene of the init image.

Reference: https://github.com/TencentARC/PhotoMaker
CLI flag: --pmid-dir <dir_with_id_images>  --pm-style-strength <float>
"""
import os
import sys
import argparse
import subprocess
import time
import json
import shutil
import uuid
import tempfile
import re

DEFAULT_SD_BINARY = "/usr/local/bin/sd"

sys.path.insert(0, os.getcwd())

import shared_utils
from workers.base import save_output_log


def download_photomaker_model():
    """Download PhotoMaker v1 model from HuggingFace if not present.

    IMPORTANT: sd.cpp ONLY works with the safetensors release from
    bssrdf/PhotoMaker. The official .bin (TencentARC) is in PyTorch
    torch-zip format and fails with 'pmid.vision_model.* not in model
    metadata' (torch zip load then metadata validation error).
    """
    checkpoints_dir = shared_utils.CHECKPOINTS_DIR
    target_path = os.path.join(checkpoints_dir, "photomaker-v1.safetensors")

    if os.path.exists(target_path):
        return target_path

    print("--> [I2I FaceSwap] PhotoMaker model not found. Downloading from HuggingFace...", flush=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        token = None
        try:
            from core.config import get_secret
            token = get_secret("hf_token")
        except Exception:
            pass

        downloaded = hf_hub_download(
            repo_id="bssrdf/PhotoMaker",
            filename="photomaker-v1.safetensors",
            local_dir=checkpoints_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        # If huggingface_hub places it in a subdirectory, move it to checkpoints_dir
        if downloaded != target_path and os.path.exists(downloaded):
            import shutil
            shutil.move(downloaded, target_path)
            # Clean up any empty subdirectories
            repo_dir = os.path.join(checkpoints_dir, "bssrdf")
            if os.path.isdir(repo_dir):
                shutil.rmtree(repo_dir, ignore_errors=True)

        # Remove any stale broken .bin from the old source (TencentARC)
        stale_bin = os.path.join(checkpoints_dir, "photomaker-v1.bin")
        if os.path.exists(stale_bin):
            try:
                os.remove(stale_bin)
                print(f"--> [I2I FaceSwap] Removed stale incompatible photomaker-v1.bin", flush=True)
            except OSError:
                pass

        if os.path.exists(target_path):
            print(f"--> [I2I FaceSwap] PhotoMaker model downloaded to: {target_path}", flush=True)
            return target_path
        else:
            print("--> [I2I FaceSwap] Download completed but file not at expected path.", flush=True)
            return None
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Install it with: pip install huggingface_hub", flush=True)
        return None
    except Exception as e:
        print(f"ERROR: Failed to download PhotoMaker model: {e}", flush=True)
        return None


def find_photomaker_model():
    """Find PhotoMaker model file (pmid) in the checkpoints directory.
    If not found, attempt to download it automatically from HuggingFace."""
    from core.config import get_setting
    checkpoints_dir = shared_utils.CHECKPOINTS_DIR

    # First, check user setting
    model_path = get_setting("photomaker_model_path")
    if model_path and os.path.exists(model_path):
        return model_path

    # Auto-detect (prefer the sd.cpp-compatible safetensors from bssrdf/PhotoMaker)
    possible_names = [
        "photomaker-v1.safetensors",
        "PhotoMaker-V1.safetensors",
        "photomaker-v2.safetensors",
        "photomaker-v1.bin",
        "PhotoMaker-V1.bin",
        "photomaker.bin",
        "pmid.bin",
        "photomaker-v2.bin",
    ]
    for name in possible_names:
        path = os.path.join(checkpoints_dir, name)
        if os.path.exists(path):
            return path

    # Recursive search
    for root, dirs, files in os.walk(checkpoints_dir):
        for f in files:
            low = f.lower()
            if "photomaker" in low or "pmid" in low:
                if f.endswith(".bin") or f.endswith(".safetensors") or f.endswith(".gguf"):
                    return os.path.join(root, f)

    # Not found — try to download
    return download_photomaker_model()


def find_photomaker_diffusion_model():
    """Find a base diffusion model compatible with PhotoMaker (SDXL/SD1.5)."""
    from core.config import get_setting
    checkpoints_dir = shared_utils.CHECKPOINTS_DIR

    # Check user setting
    model_path = get_setting("photomaker_diffusion_model")
    if model_path and os.path.exists(model_path):
        return model_path

    # Auto-detect: prefer SDXL models. Scan checkpoints + models_external + MODELS_DIR root.
    candidates = []
    scan_dirs = [checkpoints_dir]
    for extra in ["models_external", shared_utils.MODELS_DIR]:
        if os.path.isdir(extra) and extra not in scan_dirs:
            scan_dirs.append(extra)
    for base in scan_dirs:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                low = f.lower()
                # SDXL-family names (RealVisXL picked by the worker already, add others)
                if ("sdxl" in low or "sd_xl" in low or "realvisxl" in low or "juggernautxl" in low
                        or "dreamshaperxl" in low or "pony" in low or "sd1.5" in low or "sd_1.5" in low) and (
                    f.endswith(".gguf") or f.endswith(".safetensors")
                ):
                    # Skip qwen, z-image, and other non-SD models
                    if "qwen" in low or "z-image" in low or "zimage" in low or "flux" in low:
                        continue
                    candidates.append(os.path.join(root, f))

    # Prefer gguf, prefer Q4/Q5
    def score(p):
        s = 0
        low = p.lower()
        if "gguf" in low: s += 10
        if "q4" in low: s += 5
        if "q5" in low: s += 3
        if "q8" in low: s += 1
        if "sdxl" in low or "realvisxl" in low: s += 2
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0] if candidates else None


def find_vae_for_model(model_path):
    """Find a VAE compatible with the model."""
    if not model_path:
        return None
    vae_dir = shared_utils.VAE_DIR
    model_dir = os.path.dirname(model_path)
    # Check model dir
    for f in os.listdir(model_dir) if os.path.isdir(model_dir) else []:
        if f.lower().endswith(('.safetensors', '.gguf', '.sft')) and 'vae' in f.lower():
            return os.path.join(model_dir, f)
    # Check VAE dir
    for f in os.listdir(vae_dir) if os.path.exists(vae_dir) else []:
        if 'sdxl' in f.lower() or 'xl' in f.lower():
            return os.path.join(vae_dir, f)
    return None


def prepare_pmid_dir(face_img, pmid_dir):
    """
    Copy the face image into the pmid_dir so sd can read it.
    PhotoMaker expects a directory of ID images.
    """
    os.makedirs(pmid_dir, exist_ok=True)
    # Clear any previous files
    for f in os.listdir(pmid_dir):
        try:
            os.remove(os.path.join(pmid_dir, f))
        except OSError:
            pass

    # Copy face image with safe filename
    ext = os.path.splitext(face_img)[1] or '.png'
    dst = os.path.join(pmid_dir, f"face_{uuid.uuid4().hex[:8]}{ext}")
    shutil.copy2(face_img, dst)
    return dst




def _resolve_image_path(path: str) -> str:
    """Resolve a possibly-relative or browser-style image path to an absolute filesystem path.

    Handles:
        - /app/...       -> /app/...        (already absolute, no change)
        - /outputs/...   -> /app/outputs/... (browser URL, needs /app prefix)
        - /uploads/...   -> /app/uploads/... (browser URL, needs /app prefix)
        - bare filename   -> /app/outputs/bare_filename  (assume outputs dir)
        - relative path   -> os.path.abspath of it
    """
    if not path:
        return path
    # Already absolute
    if os.path.isabs(path):
        # If path starts with /outputs/ or /uploads/ but file doesn't exist, try /app/...
        if not os.path.exists(path) and (path.startswith("/outputs/") or path.startswith("/uploads/")):
            alt = "/app" + path
            if os.path.exists(alt):
                return alt
        return path
    # Bare filename: assume outputs dir
    if "/" not in path:
        # Look in outputs first
        for base in ["/app/outputs", "/app/uploads", os.getcwd()]:
            candidate = os.path.join(base, path)
            if os.path.exists(candidate):
                return candidate
    # Relative path
    return os.path.abspath(path)


def _validate_model_compatibility(model_path: str):
    """Check if a diffusion model is compatible with PhotoMaker (SDXL/SD1.5).
    Exits with a clear error if the model is known to be incompatible.
    """
    low = os.path.basename(model_path).lower()
    incompatible_patterns = [
        ("z-image", "Z-Image models use an LLM text encoder, not compatible with PhotoMaker/SD.cpp"),
        ("zimage", "Z-Image models use an LLM text encoder, not compatible with PhotoMaker/SD.cpp"),
        ("qwen", "Qwen-VL based models are not compatible with PhotoMaker"),
        ("flux", "FLUX models are not compatible with PhotoMaker (SDXL required)"),
        ("sd3", "Stable Diffusion 3 models are not compatible with PhotoMaker (SDXL/SD1.5 required)"),
        ("sd-3", "Stable Diffusion 3 models are not compatible with PhotoMaker (SDXL/SD1.5 required)"),
        ("hunyuan", "Hunyuan models are not compatible with PhotoMaker"),
        ("auraflow", "AuraFlow models are not compatible with PhotoMaker"),
        ("kolors", "Kolors models are not compatible with PhotoMaker"),
        ("pixart", "PixArt models are not compatible with PhotoMaker"),
    ]
    for pattern, reason in incompatible_patterns:
        if pattern in low:
            print(f"ERROR: Incompatible model for PhotoMaker: {os.path.basename(model_path)}")
            print(f"   Reason: {reason}")
            print(f"   Please select an SDXL or SD1.5 model, or leave model empty for auto-detect.")
            sys.exit(1)


def run_i2i_faceswap(prompt: str, init_img: str, face_img: str, seed: int = -1,
                     steps: int = 30, guidance: float = 5.0,
                     scheduler: str = "euler", width: int = 1024, height: int = 1024,
                     negative_prompt: str = "", model_path: str = None,
                     pmid_path: str = None, pmid_embed_path: str = None,
                     vae_path: str = None, clip_skip: int = 1,
                     scheduler_type: str = "auto",
                     pm_style_strength: float = 20.0,
                     lora_config: str = "",
                     vae_tiling: bool = True,
                     llm_path: str = None):
    """Run face swap using PhotoMaker + stable-diffusion.cpp."""
    start_time = time.time()
    print(f"--> [I2I FaceSwap] Starting... PID: {os.getpid()}")

    # Validate inputs
    # Phase 9: Resolve possibly-relative/browser-style paths to absolute
    init_img = _resolve_image_path(init_img)
    face_img = _resolve_image_path(face_img)
    if not os.path.exists(init_img):
        print(f"ERROR: Init image not found: {init_img}")
        sys.exit(1)
    if not os.path.exists(face_img):
        print(f"ERROR: Face image not found: {face_img}")
        sys.exit(1)

    # Find binary
    sd_binary = DEFAULT_SD_BINARY
    if not os.path.exists(sd_binary):
        local_sd = os.path.join(os.getcwd(), "sd")
        if os.path.exists(local_sd):
            sd_binary = local_sd
        else:
            print(f"ERROR: sd binary not found at {sd_binary}")
            sys.exit(1)

    # Output path
    out_filename = f"i2i_faceswap_{int(time.time())}.png"
    save_path = os.path.join(shared_utils.OUTPUT_DIR, out_filename)
    os.makedirs(shared_utils.OUTPUT_DIR, exist_ok=True)

    # FIRST: find and validate a compatible diffusion model (SDXL/SD1.5)
    # BEFORE downloading PhotoMaker, so users with only Z-Image/Qwen/Flux
    # models don't wait for a useless 300MB download just to get an error.
    final_model = model_path or find_photomaker_diffusion_model()
    if not final_model or not os.path.exists(final_model):
        print(f"ERROR: No compatible diffusion model found for PhotoMaker (SDXL required).", flush=True)
        print(f"", flush=True)
        print(f"   PhotoMaker face-swap requires an SDXL or SD1.5 diffusion model.", flush=True)
        print(f"   Your current checkpoints folder contains ONLY:", flush=True)
        checkpoints_dir = shared_utils.CHECKPOINTS_DIR
        if os.path.isdir(checkpoints_dir):
            found = []
            for root, dirs, files in os.walk(checkpoints_dir):
                for f in files:
                    if f.endswith((".gguf", ".safetensors", ".bin")) and "photomaker" not in f.lower():
                        found.append(os.path.relpath(os.path.join(root, f), checkpoints_dir))
            if found:
                for f in sorted(found)[:10]:
                    print(f"      - {f}")
            else:
                print(f"      (checkpoints folder is empty)")
        print(f"", flush=True)
        print(f"   To enable face swap, download an SDXL model from the UI", flush=True)
        print(f"   (Model Management) e.g. 'RealVisXL V5' or 'SDXL base' in GGUF format.", flush=True)
        sys.exit(1)

    # Validate model compatibility: reject non-SDXL models (Z-Image, Qwen, Flux, etc.)
    _validate_model_compatibility(final_model)
    print(f"--> [I2I FaceSwap] Using diffusion model: {final_model}")

    # Now that we know we have a usable SDXL model, find/download PhotoMaker.
    pmid_model = pmid_path or find_photomaker_model()
    if not pmid_model or not os.path.exists(pmid_model):
        print(f"ERROR: PhotoMaker model not found. Please run Setup to download it.")
        print(f"   Expected: photomaker-v1.bin or PhotoMaker-V1.bin in checkpoints/")
        sys.exit(1)
    print(f"--> [I2I FaceSwap] Using PhotoMaker: {pmid_model}")

    # Find VAE
    vae_used = vae_path or find_vae_for_model(final_model)
    if vae_used:
        print(f"--> [I2I FaceSwap] Using VAE: {vae_used}")

    # Prepare pmid directory with face image
    pmid_dir = os.path.join(shared_utils.OUTPUT_DIR, f"pmid_{uuid.uuid4().hex[:8]}")
    try:
        prepare_pmid_dir(face_img, pmid_dir)
        print(f"--> [I2I FaceSwap] Prepared PMID dir: {pmid_dir}")

        # Build command
        # NOTE: this build of sd.cpp uses --photo-maker + --pm-id-images-dir.
        # (older builds used the now-removed --pmid-dir flag)
        cmd = [
            sd_binary,
            "-t", str(os.cpu_count() or 4),
            "--mode", "img_gen",
            "--diffusion-model", final_model,
            "-m", final_model,
            "-i", init_img,
            "--photo-maker", pmid_model,
            "--pm-id-images-dir", pmid_dir,
            "--pm-style-strength", str(pm_style_strength),
            "--prompt", prompt,
            "-W", str(width),
            "-H", str(height),
            "--steps", str(steps),
            "--cfg-scale", str(guidance),
            "-o", save_path,
            "-v",
            "--fa",
        ]
        if vae_used:
            cmd.extend(["--vae", vae_used])
        if vae_tiling:
            cmd.extend(["--vae-tiling"])
        if llm_path and os.path.exists(llm_path):
            cmd.extend(["--llm", llm_path])
        if pmid_embed_path and os.path.exists(pmid_embed_path):
            cmd.extend(["--pm-id-embed-path", pmid_embed_path])
        if scheduler:
            cmd.extend(["--sampling-method", scheduler])
        # "auto"/empty => sd.cpp picks the model-native default scheduler.
        if scheduler_type and scheduler_type not in ("auto", "default"):
            cmd.extend(["--scheduler", scheduler_type])
        if seed >= 0:
            cmd.extend(["--seed", str(seed)])
        if negative_prompt:
            cmd.extend(["--negative-prompt", negative_prompt])
        # Optional init strength
        if '--strength' not in cmd:
            cmd.extend(["--strength", "0.65"])

        # Add LoRA if configured
        if lora_config and os.path.exists(lora_config):
            try:
                with open(lora_config, 'r') as f:
                    loras = json.load(f)
                    for lora in loras:
                        path = lora[0]
                        strength = float(lora[1])
                        active = lora[2]
                        if active and os.path.exists(path):
                            lora_dir = os.path.dirname(path)
                            if "--lora-model-dir" not in cmd:
                                cmd.extend(["--lora-model-dir", lora_dir])
                            stem = os.path.splitext(os.path.basename(path))[0]
                            esc_stem = re.escape(stem)
                            pattern = rf"<lora:{esc_stem}:([0-9.]+)>"
                            if re.search(pattern, prompt):
                                prompt = re.sub(pattern, f"<lora:{stem}:{strength}>", prompt)
                            else:
                                prompt += f" <lora:{stem}:{strength}>"
                            try:
                                p_idx = cmd.index("--prompt")
                                if p_idx + 1 < len(cmd):
                                    cmd[p_idx + 1] = prompt
                            except ValueError:
                                pass
            except Exception as e:
                print(f"--> [I2I FaceSwap] LoRA error: {e}", flush=True)

        print(f"--> [I2I FaceSwap] Running: {' '.join(cmd)}", flush=True)

        # Execute
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=os.getcwd(), env=env
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                if "%" in line or "step" in line.lower():
                    print(f"data: LOG|{line}", flush=True)
                else:
                    print(f"    {line}", flush=True)

        process.wait()
        if process.returncode != 0:
            print(f"ERROR: sd-cli exited with code {process.returncode}")
            sys.exit(1)

        if os.path.exists(save_path):
            elapsed = time.time() - start_time
            print(f"--> [I2I FaceSwap] Done in {elapsed:.1f}s -> {save_path}")
            print(f"SUCCESS_OUTPUT:{save_path}")
            sys.stdout.flush()

            # Save JSON metadata
            lora_list = []
            inputs_data = {
                "prompt": prompt,
                "source_image_1": os.path.basename(init_img),
                "source_image_2": os.path.basename(face_img) + " (face source)",
                "image_role": "face_swap",
                "loras": shared_utils.format_loras_for_json(lora_list)
            }
            params_data = {
                "width": width, "height": height, "steps": steps,
                "guidance": guidance, "seed": seed,
                "backend": "stable-diffusion.cpp",
                "mode": "i2i_faceswap",
                "model": os.path.basename(final_model),
                "photomaker": os.path.basename(pmid_model),
                "pm_style_strength": pm_style_strength,
                "vae": os.path.basename(vae_used) if vae_used else "",
                "scheduler": scheduler,
                "scheduler_type": scheduler_type,
                "loras": shared_utils.format_loras_for_json(lora_list)
            }
            outputs_data = {
                "type": "image",
                "files": [out_filename],
                "exec_time": round(elapsed, 2)
            }
            shared_utils.save_generation_log(
                "i2i_faceswap", inputs_data, params_data, outputs_data,
                image_path_for_filename=save_path, model_name="photomaker"
            )
        else:
            print(f"ERROR: Output file not created: {save_path}")
            sys.exit(1)
    finally:
        # Cleanup pmid directory
        try:
            if os.path.exists(pmid_dir):
                shutil.rmtree(pmid_dir)
                print(f"--> [I2I FaceSwap] Cleaned up pmid dir: {pmid_dir}")
        except Exception as e:
            print(f"--> [I2I FaceSwap] Cleanup warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="I2I Face Swap Worker (PhotoMaker)")
    # Sentinel mode: no-op used by the API to surface a clean error in the UI
    # without downloading PhotoMaker (which needs an SDXL model to be useful).
    parser.add_argument("--error_only", action="store_true",
                        help="Emit the standard help/error text and exit (used by API pre-check)")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--init_img", type=str, required=False, help="Scene image (where to place the face)")
    parser.add_argument("--face_img", type=str, required=False, help="Face source image")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--scheduler", type=str, default="euler")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--pmid_path", type=str, default=None, help="Path to PhotoMaker .bin file")
    parser.add_argument("--pmid_embed_path", type=str, default=None, help="Path to pre-computed PM v2 ID embed")
    parser.add_argument("--pm_style_strength", type=float, default=20.0)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--clip_skip", type=int, default=1)
    parser.add_argument("--scheduler_type", type=str, default="auto")
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--lora_config", type=str, default="")
    parser.add_argument("--vae_tiling", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--llm_path", type=str, default=None,
                        help="Path to text encoder (LLM) model for SDXL/Flux-style models")
    args = parser.parse_args()

    if args.error_only:
        print("ERROR: I2I FaceSwap (PhotoMaker) requires an SDXL/SD1.5 diffusion model.", flush=True)
        print("No SDXL/SD1.5 checkpoint was found on disk, so PhotoMaker cannot be used.", flush=True)
        print("Download an SDXL model (e.g. RealVisXL V5 GGUF) via Model Management, then retry.", flush=True)
        sys.exit(2)

    # Normal execution requires init & face images
    if not args.init_img or not args.face_img:
        print("ERROR: --init_img and --face_img are required.", flush=True)
        sys.exit(2)

    run_i2i_faceswap(
        prompt=args.prompt,
        init_img=args.init_img,
        face_img=args.face_img,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        scheduler=args.scheduler,
        width=args.width,
        height=args.height,
        negative_prompt=args.negative_prompt,
        model_path=args.model_path,
        pmid_path=args.pmid_path,
        pmid_embed_path=args.pmid_embed_path,
        vae_path=args.vae_path,
        clip_skip=args.clip_skip,
        scheduler_type=args.scheduler_type,
        pm_style_strength=args.pm_style_strength,
        lora_config=args.lora_config,
        vae_tiling=args.vae_tiling,
        llm_path=args.llm_path,
    )


if __name__ == "__main__":
    main()
