import sys
import os
import argparse
import subprocess
import time
import json
import select
import re
from PIL import Image

# Add current directory to Python path so we can import shared_utils
sys.path.insert(0, os.getcwd())

import shared_utils

# ensuring fully compatible output logging and JSON metadata generation
# for sdcpp-studio.

from workers.base import (
    get_gpu_memory, find_model_file, find_vae, find_llm,
    apply_prompt_template, is_zimage_model, is_qwen_image_edit_model, get_scheduler_arg,
    TilingConfig, save_output_log, get_total_gpu_memory, find_bf16_zimage_alternative
)
from core.config import load_settings
from core.text_encoders import find_text_encoders_for_gguf
from workers.t2i_cpp_helpers import find_llm_for_flux2

import shutil
import time

# Try to import fix_models
try:
    sys.path.append(os.getcwd())
    import fix_models
except ImportError:
    fix_models = None

# VRAM helpers now imported from workers.base

def ensure_model_fixed(model_path):
    """
    Checks if GGUF model has a .is_fixed.json sidecar.
    If not, runs fix_models.run() on it and creates the sidecar.
    """
    if not model_path.endswith(".gguf") or not os.path.exists(model_path):
        return model_path
    
    if fix_models is None:
        return model_path

    fixed_json = model_path + ".is_fixed.json"
    if os.path.exists(fixed_json):
        # assume fixed
        return model_path
        
    print(f"--> [CPP I2I Worker] Auto-Fix: Checking/Repairing GGUF tensor names for {os.path.basename(model_path)}...", flush=True)
    temp_fixed = model_path + ".tmp_fix.gguf"
    
    try:
        # Run fix
        # fix_models.run prints logic, so we'll see it in logs
        fix_models.run(model_path, temp_fixed)
        
        # If success, replace original
        if os.path.exists(temp_fixed):
            # Check size to be sure it's valid? 
            # Assume fix_models works if it returns specific size 
            shutil.move(temp_fixed, model_path)
            
            # Write marker
            with open(fixed_json, "w") as f:
                json.dump({"fixed": True, "timestamp": time.time()}, f)
            print(f"--> [CPP I2I Worker] Auto-Fix: Success. Model updated.", flush=True)
        else:
            print(f"--> [CPP I2I Worker] Auto-Fix: No output file created. Skipping.", flush=True)
            
    except Exception as e:
        print(f"--> [CPP I2I Worker] Auto-Fix: Failed with error: {e}", flush=True)
        if os.path.exists(temp_fixed):
            os.remove(temp_fixed)
            
    return model_path

def get_text_encoders_args(model_path):
    """
    Returns list of args [--clip_l, path, --t5xxl, path] if encoders found.
    Simple heuristic: check widely known paths.
    """
    args = []
    
    # Common locations: using shared_utils.MODELS_DIR
    search_dirs = [
        os.path.join(shared_utils.MODELS_DIR, "text_encoders"),
        shared_utils.MODELS_DIR,
        os.path.dirname(model_path)
    ]
    
    clip_l = None
    t5xxl = None
    
    # Search for Clip L
    for d in search_dirs:
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if "clip" in f.lower() and "_l" in f.lower() and (f.endswith(".safetensors") or f.endswith(".gguf")):
                clip_l = os.path.join(d, f)
                break
        if clip_l: break
        
    # Search for T5XXL
    for d in search_dirs:
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if "t5" in f.lower() and "xxl" in f.lower() and (f.endswith(".safetensors") or f.endswith(".gguf")):
                t5xxl = os.path.join(d, f)
                break
        if t5xxl: break
        
    # --- AUTO-DOWNLOAD (New Feature) ---
    if not clip_l or not t5xxl:
        try:
            print(f"--> [CPP I2I Worker] Auto-Download: Text Encoders missing. Downloading from Hugging Face...", flush=True)
            print(f"data: LOG|⏳ IMPORTANT: First-time setup detected for Pony/Flux.", flush=True)
            print(f"data: LOG|⏳ Downloading missing Text Encoders (~10GB Total). This will take a while, please be patient...", flush=True)
            
            from huggingface_hub import hf_hub_download
            
            target_dir = os.path.join(shared_utils.MODELS_DIR, "text_encoders")
            os.makedirs(target_dir, exist_ok=True)
            
            repo_id = "comfyanonymous/flux_text_encoders"
            
            if not clip_l:
                print(f"data: LOG|⬇️  Downloading clip_l.safetensors (~240MB)...", flush=True)
                clip_l = hf_hub_download(repo_id=repo_id, filename="clip_l.safetensors", local_dir=target_dir)
                
            if not t5xxl:
                print(f"data: LOG|⬇️  Downloading t5xxl_fp16.safetensors (~9.5GB)... DO NOT CLOSE.", flush=True)
                t5xxl = hf_hub_download(repo_id=repo_id, filename="t5xxl_fp16.safetensors", local_dir=target_dir)
                
            print(f"data: LOG|✅ Download complete! Starting generation...", flush=True)
                
        except Exception as e:
            print(f"--> [CPP I2I Worker] Auto-Download Failed: {e}", flush=True)
            print(f"data: LOG|❌ Auto-Download Failed: {str(e)}", flush=True)
            print(f"--> [CPP I2I Worker] Please manually download clip_l.safetensors and t5xxl_fp16.safetensors to {target_dir}", flush=True)

    if clip_l:
        print(f"--> [CPP I2I Worker] Auto-Inject: Found Clip-L: {clip_l}", flush=True)
        args.extend(["--clip_l", clip_l])
        
    if t5xxl:
        print(f"--> [CPP I2I Worker] Auto-Inject: Found T5-XXL: {t5xxl}", flush=True)
        args.extend(["--t5xxl", t5xxl])
        
    return args

SD_BINARY_PATH = "/usr/local/bin/sd"

def run_sd_cpp_i2i(prompt, width, height, steps, guidance, seed, init_img, strength, lora_config=None, top_k=1, temperature=0.6, model_path=None, vae_path=None, clip_skip=1, scheduler=None, scheduler_type="auto", llm_override=None, vae_tiling=True, negative_prompt: str = "", init_img_2: str = None, mix_ratio: float = 0.5, ref_images=None, mask_image=None, control_image=None, control_strength=0.8, image_role=None, no_auto_resize_ref=False, max_img_size=1024, use_slg=False, use_apg=False):
    global SD_BINARY_PATH
    start_time = time.time()
    llm_path = None # Initialize to avoid UnboundLocalError
    print(f"--> [CPP I2I Worker] Starting process PID: {os.getpid()}", flush=True)
    
    # Ensure init_img is an absolute path so the C++ sd binary can find it
    if init_img and not os.path.isabs(init_img):
        # Try OUTPUT_DIR first (most likely location for generated images)
        abs_init = os.path.join(shared_utils.OUTPUT_DIR, init_img)
        if os.path.exists(abs_init):
            init_img = abs_init
        else:
            # Fall back to current working directory
            abs_init = os.path.join(os.getcwd(), init_img)
            if os.path.exists(abs_init):
                init_img = abs_init
    
    # Also resolve init_img_2 if it's a relative path
    if init_img_2 and not os.path.isabs(init_img_2):
        abs_init2 = os.path.join(shared_utils.OUTPUT_DIR, init_img_2)
        if os.path.exists(abs_init2):
            init_img_2 = abs_init2
        else:
            abs_init2 = os.path.join(os.getcwd(), init_img_2)
            if os.path.exists(abs_init2):
                init_img_2 = abs_init2
    
    # Save original source image paths for JSON metadata (before blending)
    original_source_1 = init_img
    original_source_2 = init_img_2

    # --- Image Resize Logic (saves VRAM for large input images) ---
    def _resize_one_image(src_path):
        """Resize a single image if larger than max_img_size. Saves to /tmp. Returns new path."""
        if not max_img_size or max_img_size <= 0 or not src_path or not os.path.exists(src_path):
            return src_path
        try:
            from PIL import Image
            img = Image.open(src_path)
            w, h = img.size
            if w > max_img_size or h > max_img_size:
                if w >= h:
                    new_w = max_img_size
                    new_h = int(h * max_img_size / w)
                else:
                    new_h = max_img_size
                    new_w = int(w * max_img_size / h)
                img = img.resize((new_w, new_h), Image.BILINEAR)
                # Save in /tmp to avoid cluttering output directories
                base = os.path.splitext(os.path.basename(src_path))[0]
                resized_path = f"/tmp/{base}_resized_{os.getpid()}.png"
                img.save(resized_path)
                print(f"--> [CPP I2I Worker] Resized {os.path.basename(src_path)} from {w}x{h} to {new_w}x{new_h} (max_dim={max_img_size})", flush=True)
                return resized_path
        except Exception as e:
            print(f"--> [CPP I2I Worker] Warning: Could not resize {src_path}: {e}", flush=True)
        return src_path
    
    # Apply resize to all input images
    init_img = _resize_one_image(init_img)
    init_img_2 = _resize_one_image(init_img_2)
    if ref_images:
        ref_images = [_resize_one_image(p) for p in ref_images]
    if mask_image:
        mask_image = _resize_one_image(mask_image)
    if control_image:
        control_image = _resize_one_image(control_image)
    
    # --- Image Blending Logic (if second image provided) ---
    if init_img_2 and os.path.exists(init_img_2):
        if init_img and os.path.exists(init_img):
            print(f"--> [CPP I2I Worker] Blending images with mix_ratio={mix_ratio}", flush=True)
            img1 = Image.open(init_img).convert("RGB")
            img2 = Image.open(init_img_2).convert("RGB")
            # Resize both to target generation size
            img1 = img1.resize((width, height), Image.BILINEAR)
            img2 = img2.resize((width, height), Image.BILINEAR)
            # Blend: out = img1 * (1 - mix_ratio) + img2 * mix_ratio
            blended = Image.blend(img1, img2, alpha=float(mix_ratio))
            # Save to outputs directory (same folder as generated images)
            blend_filename = f"i2i_blend_{int(time.time())}_{os.getpid()}.png"
            blend_path = os.path.join(shared_utils.OUTPUT_DIR, blend_filename)
            blended.save(blend_path)
            print(f"--> [CPP I2I Worker] Blended image saved to {blend_path}", flush=True)
            # Override init_img with the blended result
            init_img = blend_path
        else:
            print(f"--> [CPP I2I Worker] WARNING: init_img_2 provided but init_img missing. Ignoring second image.", flush=True)

    if not os.path.exists(SD_BINARY_PATH):
        # Check local path as fallback if not in /usr/local/bin
        local_sd = os.path.join(os.getcwd(), "sd")
        if os.path.exists(local_sd):
            SD_BINARY_PATH = local_sd
        else:
            print(f"--> [CPP I2I Worker] ERROR: C++ Binary not found at {SD_BINARY_PATH}. Please rebuild container.", flush=True)

    # 1. Determine Output Filename
    out_filename = f"i2i_cpp_{int(time.time())}.png"
    save_path = os.path.join(shared_utils.OUTPUT_DIR, out_filename)
    os.makedirs(shared_utils.OUTPUT_DIR, exist_ok=True)

    # 2. Determine Model Path
    final_model = model_path
    if model_path and os.path.isdir(model_path):
         # Try to find a specific weight file if a directory was passed
         # We MUST ignore "qwen" files as they are LLMs, not diffusion models
         cands = [f for f in sorted(os.listdir(model_path)) if f.endswith(".gguf") and "qwen" not in f.lower()]
         if not cands: 
             cands = [f for f in sorted(os.listdir(model_path)) if f.endswith(".safetensors") and "qwen" not in f.lower()]
         
         if cands:
             final_model = os.path.join(model_path, cands[0])
         else:
             # Deep search for Z-Image
             if "z_image" in model_path.lower() or "z-image" in model_path.lower():
                 # Look for a GGUF variant folder
                 parent = os.path.dirname(model_path)
                 for d in os.listdir(parent):
                     if "z" in d.lower() and "image" in d.lower() and "gguf" in d.lower():
                         gguf_path = os.path.join(parent, d)
                         if os.path.isdir(gguf_path):
                             cands_gguf = [f for f in sorted(os.listdir(gguf_path)) if f.endswith(".gguf") and "qwen" not in f.lower()]
                             if cands_gguf:
                                 final_model = os.path.join(gguf_path, cands_gguf[0])
                                 break

    if not final_model or (not os.path.exists(final_model)):
        print(f"data: LOG|❌ ERROR: Model file or directory not found: {final_model}", flush=True)
        sys.exit(1)

    # --- RDNA3 OPTIMIZATION: quantized GGUF is ~2x slower than BF16 ---
    # On AMD RDNA3 (gfx1100) the Q8_0 GGUF matmul kernels use the slow
    # vec-dot/dp4a path, while BF16 safetensors go through rocBLAS WMMA
    # (tensor cores). If a BF16 sibling exists, upgrade transparently.
    settings = load_settings()
    if settings.get("auto_prefer_bf16_zimage", True):
        bf16_alt = find_bf16_zimage_alternative(final_model)
        if bf16_alt:
            print("--> [CPP I2I] [OPTIMIZATION] Q8_0/quantized GGUF matmul is ~2x slower on RDNA3")
            print(f"--> [CPP I2I] [OPTIMIZATION] Auto-upgraded model to BF16: {os.path.basename(bf16_alt)}")
            print("--> [CPP I2I] [OPTIMIZATION] (set auto_prefer_bf16_zimage=false in settings.json to force quantized)")
            final_model = bf16_alt

    # === Guard: Qwen-Image-Edit is a diffusion-only edit model ===
    # It cannot be opened as a full checkpoint (-m): sd.cpp fails with
    # "get sd version from file failed". api/generate.py routes these
    # requests to workers/i2i_edit.py; this guard catches direct/legacy
    # calls and fails with a clear message instead of a cryptic crash.
    # NOTE: placed BEFORE the auto-fix so fix_models never rewrites a
    # qwen-image-edit diffusion GGUF (its bare tensor names are correct).
    if final_model and is_qwen_image_edit_model(final_model):
        print(f"--> [CPP I2I Worker] ERROR: Qwen-Image-Edit model '{os.path.basename(final_model)}' cannot be used with the generic I2I (img2img) backend.", flush=True)
        print("--> [CPP I2I Worker] Qwen-Image-Edit performs semantic edits via prompt + reference image (Qwen2.5-VL text encoder).", flush=True)
        print("--> [CPP I2I Worker] Select it in the I2I tab: the app now routes it to the dedicated Edit worker automatically.", flush=True)
        sys.exit(1)

    # --- AUTO-FIX LOGIC ---
    # Attempt to auto-fix GGUF tensors if needed
    if final_model.endswith(".gguf"):
        final_model = ensure_model_fixed(final_model)

    # 3. Construct Command
    # Check if it's a Z-Image model (Lumina2 architecture)
    is_zimage = is_zimage_model(final_model)

    # Check for Ideogram4 model
    is_ideogram4 = "ideogram4" in os.path.basename(final_model).lower() and final_model.endswith(".gguf")
    
    # Check if it's a FLUX model (needs --diffusion-model instead of -m for GGUF)
    is_flux2 = "flux2" in final_model.lower() and final_model.endswith((".gguf", ".safetensors"))
    is_flux = ("flux" in final_model.lower() and not is_flux2 and not is_ideogram4) and final_model.endswith(".gguf")
    
    # Check for FLUX manifest (created by setup module)
    flux_manifest = None
    if is_flux:
        model_dir = os.path.dirname(final_model)
        manifest_path = os.path.join(model_dir, "flux_setup.json")
        if not os.path.exists(manifest_path):
            # Also check parent directories
            manifest_path = os.path.join(os.path.dirname(model_dir), "flux_setup.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                flux_manifest = json.load(f)
            print(f"--> [CPP I2I Worker] Found FLUX manifest: {manifest_path}", flush=True)

    cmd = [
        SD_BINARY_PATH,
        "--mode", "img_gen", # I2I is handled by img_gen + init_img
        # NOTE: -t 0 tells sd.cpp to auto-detect physical core count.
        # Using logical cores (os.cpu_count()) causes thread contention and
        # degrades performance on systems with hyperthreading.
        "-t", "0",
        "--prompt", prompt,
        "-W", str(width),
        "-H", str(height),
        "--steps", str(steps),
        "--cfg-scale", str(guidance),
        "--seed", str(seed),
        "-i", init_img, # Init Image
        "--strength", str(strength), # Denoise Strength
        "-o", save_path,
        "-v",
        "--fa" # Force Flash Attention (Correct Flag)
    ]
    
    # Add negative prompt if provided
    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])
    
    # Add explicit VAE path if provided (for non Z-Image models or overriding default)
    custom_vae = vae_path  # Save user-provided VAE
    if not is_zimage and custom_vae and os.path.exists(custom_vae):
        cmd.extend(["--vae", custom_vae])

    if is_flux2:
        print("--> [CPP I2I Worker] Detected FLUX.2 model. Using --diffusion-model flag and offload-to-cpu.", flush=True)
        cmd.extend(["--diffusion-model", final_model, "--offload-to-cpu", "--diffusion-fa"])
        
        vae_path = find_vae(final_model)
        if not vae_path or "flux2" not in os.path.basename(vae_path).lower():
            for d in [os.path.dirname(final_model), os.path.join(shared_utils.MODELS_DIR, "vae")]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if "flux2_ae" in f.lower() or "flux2-vae" in f.lower() or "flux2" in f.lower() and "vae" in f.lower():
                            vae_path = os.path.join(d, f)
                            break
        
        if vae_path and os.path.exists(vae_path):
             cmd.extend(["--vae", vae_path])
        else:
             print("--> [CPP I2I Worker] WARNING: VAE not found for FLUX.2 model. Generation may fail.", flush=True)
             
        llm_search = find_llm_for_flux2(final_model, llm_override)
        if llm_search:
            print(f"--> [CPP I2I Worker] Using LLM for FLUX.2: {llm_search}", flush=True)
            cmd.extend(["--llm", llm_search, "--clip-on-cpu"])
        else:
            print("--> [CPP I2I Worker] WARNING: Mistral text encoder not found for FLUX.2 model. Generation may fail.", flush=True)

    elif is_flux:
        print(f"--> [CPP I2I Worker] Detected FLUX GGUF model. Using --diffusion-model flag.", flush=True)
        # Must pass -m as well for VAE loading to work
        cmd.extend(["-m", final_model])
        cmd.extend(["--diffusion-model", final_model])
        
        # FLUX requires VAE, clip_l, and t5xxl
        # Use manifest if available, otherwise auto-detect
        if flux_manifest:
            vae_path = flux_manifest.get("vae")
            clip_l = flux_manifest.get("clip_l")
            t5xxl = flux_manifest.get("t5xxl")
            print("--> [CPP I2I Worker] Using FLUX manifest for component paths.", flush=True)
            # The manifest may hold paths from another environment (e.g. /workspace/...):
            # ignore any entry missing on disk and fall back to auto-detection,
            # otherwise FLUX runs with NO text encoders -> "failed to encode prompt".
            if not (vae_path and os.path.exists(vae_path)):
                vae_path = None if (custom_vae and os.path.exists(custom_vae)) else find_vae(final_model)
            if not (clip_l and os.path.exists(clip_l)) or not (t5xxl and os.path.exists(t5xxl)):
                auto_clip_l, auto_t5xxl = find_text_encoders_for_gguf(final_model)
                if not (clip_l and os.path.exists(clip_l)):
                    clip_l = auto_clip_l
                if not (t5xxl and os.path.exists(t5xxl)):
                    t5xxl = auto_t5xxl
                print("--> [CPP I2I Worker] Manifest encoder paths missing on disk: using auto-detected encoders.", flush=True)
        else:
            vae_path = find_vae(final_model)
            clip_l, t5xxl = find_text_encoders_for_gguf(final_model)
        
        if vae_path and os.path.exists(vae_path):
            cmd.extend(["--vae", vae_path])
        else:
            print(f"data: LOG|⚠️ WARNING: VAE not found for FLUX model. Generation may fail.", flush=True)
        
        if clip_l and os.path.exists(clip_l):
            cmd.extend(["--clip_l", clip_l])
        if t5xxl and os.path.exists(t5xxl):
            cmd.extend(["--t5xxl", t5xxl])
        if not clip_l or not t5xxl:
            print(f"data: LOG|⚠️ WARNING: Text encoders not found for FLUX model. Generation may fail.", flush=True)
    elif is_ideogram4:
        print(f"--> [CPP I2I Worker] Detected Ideogram4 model. Using --diffusion-model + --uncond-diffusion-model.", flush=True)
        cmd.extend(["--diffusion-model", final_model])
        # Find unconditional model (same dir, with _uncond in name)
        ideogram_dir = os.path.dirname(final_model)
        uncond_model = os.path.join(ideogram_dir, "ideogram4_uncond-Q4_0.gguf")
        if os.path.exists(uncond_model):
            cmd.extend(["--uncond-diffusion-model", uncond_model])
        else:
            print(f"--> [CPP I2I Worker] WARNING: Unconditional model not found at {uncond_model}", flush=True)
        # Use flux2 VAE
        vae_path = find_vae(final_model)
        if not vae_path or "flux2" not in os.path.basename(vae_path).lower():
            for d in [os.path.dirname(final_model), os.path.join(shared_utils.MODELS_DIR, "vae")]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if "flux2_ae" in f.lower() or "flux2-vae" in f.lower() or ("flux2" in f.lower() and "vae" in f.lower()):
                            vae_path = os.path.join(d, f)
                            break
        if vae_path and os.path.exists(vae_path):
            cmd.extend(["--vae", vae_path])
        else:
            print("--> [CPP I2I Worker] WARNING: VAE not found for Ideogram4. Generation may fail.", flush=True)
        # Ideogram4 uses Qwen3VL as text encoder
        llm_search = None
        i2t_dir = os.path.join(shared_utils.MODELS_DIR, "i2t")
        if os.path.exists(i2t_dir):
            for root, dirs, files in os.walk(i2t_dir):
                for f in files:
                    if f.endswith(".gguf") and "qwen3" in f.lower() and "vl" in f.lower():
                        llm_search = os.path.join(root, f)
                        break
                if llm_search:
                    break
        if llm_search:
            print(f"--> [CPP I2I Worker] Using LLM for Ideogram4: {llm_search}", flush=True)
            cmd.extend(["--llm", llm_search, "--offload-to-cpu"])
        else:
            print("--> [CPP I2I Worker] WARNING: Qwen3VL text encoder not found. Please download it.", flush=True)
    elif is_zimage:
        print(f"--> [CPP I2I Worker] Detected Z-Image Architecture. Using specialized flags.", flush=True)
        cmd.extend(["-m", final_model])
        cmd.extend(["--diffusion-model", final_model])
        # NOTE: --vae-on-cpu removed — on 24GB 7900 XTX the VAE fits easily in VRAM.
        # VAE on CPU is ~1.5s per tile vs ~0.05s on GPU.
        # Only re-add --vae-on-cpu if running on <12GB VRAM cards.
        
        # Try to find VAE and LLM (Same logic as T2I)
        models_root = shared_utils.MODELS_DIR
        zimage_base = os.path.join(models_root, "Z-Image-Turbo")
        
        # 1. Look for VAE (ae.sft or from Z-Image folder)
        vae_path = None
        # Use custom VAE if provided, otherwise auto-detect
        if custom_vae and os.path.exists(custom_vae):
            vae_path = custom_vae
        else:
            vae_search_paths = [
                os.path.join(zimage_base, "vae/diffusion_pytorch_model.safetensors"),
                os.path.join(models_root, "ae.sft"),
                os.path.join(os.path.dirname(final_model), "ae.sft")
            ]
            for p in vae_search_paths:
                if os.path.exists(p):
                    vae_path = p
                    break
        
        if vae_path:
            cmd.extend(["--vae", vae_path])
        else:
            print(f"data: LOG|⚠️ WARNING: Z-Image VAE (ae.sft) not found. Generation might fail.", flush=True)
        
        # Z-Image uses a 3D VAE that requires ~10GB compute buffer.
        # Force VAE tiling at all resolutions to prevent OOM even on 24GB cards.
        # NOTE: Do NOT specify --vae-tile-size — let the binary auto-calculate
        # the optimal tile size. Hardcoding 128px causes excessive tiling overlap
        # and GPU segfaults (SIGSEGV -11).
        cmd.extend(["--vae-tiling"])

        # 2. Look for LLM (Qwen GGUF)
        llm_path = None
        
        if llm_override and os.path.exists(llm_override):
             print(f"--> [CPP I2I Worker] Using Manual LLM Override: {llm_override}", flush=True)
             llm_path = llm_override
        else:
            # Auto-detect logic
            llm_search_dirs = [
                os.path.join(models_root, "Z-Image-Turbo-GGUF"),
                models_root,
                os.path.join(zimage_base, "text_encoder"),
                os.path.join(os.path.dirname(final_model), "text_encoders"),
                os.path.dirname(final_model) # Add model dir itself
            ]
            
            found_llms = []
            for d in llm_search_dirs:
                if not os.path.exists(d): continue
                for f in os.listdir(d):
                    if f.endswith(".gguf") and ("qwen" in f.lower()):
                        full_path = os.path.join(d, f)
                        size_mb = os.path.getsize(full_path) / (1024*1024)
                        print(f"[DEBUG] Found LLM candidate: {full_path} ({size_mb:.1f} MB)", flush=True)
                        found_llms.append(full_path)
            
            print(f"[DEBUG] All found LLMs: {found_llms}", flush=True)
            
            if found_llms:
                # Prioritize 4B
                for p in found_llms:
                    if "4b" in os.path.basename(p).lower():
                        llm_path = p
                        break
                if not llm_path:
                    llm_path = found_llms[0]
                print(f"[DEBUG] Selected LLM: {llm_path} (size: {os.path.getsize(llm_path)/(1024*1024):.1f} MB)", flush=True)

        if llm_path:
            print(f"--> [CPP I2I Worker] Using LLM: {llm_path}", flush=True)
            # Keep LLM in RAM to leave VRAM for Z-Image compute buffers.
            # Without --clip-on-cpu the 7.5GB Qwen3 fills VRAM, leaving only ~4.6GB
            # for compute — not enough for the Z-Image diffusion graph (crashes SIGSEGV -11).
            cmd.extend(["--llm", llm_path, "--clip-on-cpu"])
        else:
            print(f"data: LOG|⚠️ WARNING: Qwen LLM GGUF not found for Z-Image. Attempting with standard flags.", flush=True)
            print(f"[DEBUG] Searched dirs: {llm_search_dirs}", flush=True)
            print(f"[DEBUG] Found LLMs: {found_llms}", flush=True)
            
        # Optional DiT acceleration parameters for huge resolutions
        # Enable cache-dit for DiT models (Z-Image) to skip redundant diffusion steps
        cmd.extend(["--cache-mode", "cache-dit"])
    else:
        cmd.extend(["-m", final_model])

        # Auto-inject text encoders for Pony/Flux
        if "pony" in final_model.lower() or "flux" in final_model.lower():
             print(f"--> [CPP I2I Worker] Detected Pony/Flux model. Checking for text encoders...", flush=True)
             enc_args = get_text_encoders_args(final_model)
             cmd.extend(enc_args)

    # === Multi-Image Conditioning (Phase 1A / N-Slot) ===
    # Forward ref-images, mask, control to the sd binary
    # NOTE: --ref-image is supported ONLY by Flux Kontext-family models
    # (FLUX, FLUX.2, Qwen-Image, LongCat). Other architectures (Z-Image,
    # SD1.5, SDXL, SD3/Pony, ...) do NOT support it and sd.cpp segfaults
    # with SIGSEGV (-11). Skip with a warning in those cases; suggest
    # the 'blend' slot as an alternative.
    if ref_images:
        model_basename = os.path.basename(final_model).lower()
        supports_ref = (
            is_flux or is_flux2 or
            "qwen-image" in model_basename or
            "longcat" in model_basename
        )
        if not supports_ref:
            print(f"--> [CPP I2I Worker] WARNING: Model '{os.path.basename(final_model)}' does not support "
                  f"reference images (--ref-image is only for Flux Kontext-family models). "
                  f"Skipping {len(ref_images)} ref image(s). Use the 'blend' slot instead.", flush=True)
        else:
            for ref_path in ref_images:
                if ref_path and os.path.exists(ref_path):
                    print(f"--> [CPP I2I Worker] Adding Reference Image: {os.path.basename(ref_path)}", flush=True)
                    cmd.extend(["--ref-image", ref_path])
    # Mask (inpainting) and ControlNet are also architecture-specific:
    # sd.cpp applies --mask only for inpaint-family models (SD1/SD2/SDXL inpaint,
    # FLUX_FILL, FLEX_2) and --control-image only for FLUX_CONTROLS/FLEX_2.
    # Z-Image (and most others) silently ignore them -> warn and skip.
    model_basename_low = os.path.basename(final_model).lower()
    mask_supported = is_flux or is_flux2 or "flux.fill" in model_basename_low
    control_supported = is_flux or is_flux2 or "flux.controls" in model_basename_low
    if mask_image and os.path.exists(mask_image):
        if not mask_supported:
            print(f"--> [CPP I2I Worker] WARNING: Mask/inpainting is not supported by model "
                  f"'{os.path.basename(final_model)}' (only SD-inpaint/FLUX-FILL/FLEX-2 models). "
                  f"Skipping mask. The result will be a normal img2img edit.", flush=True)
        else:
            print(f"--> [CPP I2I Worker] Adding Mask Image: {os.path.basename(mask_image)}", flush=True)
            cmd.extend(["--mask", mask_image])
    if control_image and os.path.exists(control_image):
        if not control_supported:
            print(f"--> [CPP I2I Worker] WARNING: ControlNet is not supported by model "
                  f"'{os.path.basename(final_model)}' (only FLUX-controls/FLEX-2 models). "
                  f"Skipping control image.", flush=True)
        else:
            print(f"--> [CPP I2I Worker] Adding Control Image: {os.path.basename(control_image)} (strength={control_strength})", flush=True)
            cmd.extend(["--control-image", control_image])
            cmd.extend(["--control-strength", str(control_strength)])
    if no_auto_resize_ref and (is_flux or is_flux2 or "qwen-image" in os.path.basename(final_model).lower() or "longcat" in os.path.basename(final_model).lower()):
        cmd.extend(["--no-auto-resize-ref-image"])

    # Scheduler Mapping
    sched_map = {
        "euler_a": "euler_a",
        "euler": "euler",
        "dpm++_2m_karras": "dpm++2m",
        "dpm++_sde_karras": "dpm++2s_a",
        "ddim": "euler",
        "res_multistep": "res_multistep"
    }
    if scheduler and scheduler in sched_map:
        cmd.extend(["--sampling-method", sched_map[scheduler]])
    
    # Scheduler Type: "auto"/empty => sd.cpp picks the model-native default.
    if scheduler_type and scheduler_type not in ("auto", "default"):
        cmd.extend(["--scheduler", scheduler_type])

    # Guidance Optimization: SLG (Skip Layer Guidance) for DiT models
    if use_slg:
        print(f"--> [CPP I2I Worker] Enabling Skip Layer Guidance (SLG) for faster CFG", flush=True)
        cmd.extend(["--slg-scale", "2.5", "--skip-layers", "[7,8,9]"])

    # Guidance Optimization: APG (Adaptive Projected Guidance)
    if use_apg:
        print(f"--> [CPP I2I Worker] Enabling Adaptive Projected Guidance (APG)", flush=True)
        cmd.extend(["--extra-sample-args", "apg_eta=1.0,apg_momentum=0.0"])

    # 4. LoRA Configuration (Gather and Link Strategy)
    temp_lora_dir = None

    if lora_config and os.path.exists(lora_config):
        try:
            with open(lora_config, 'r') as f:
                loras = json.load(f)
            
            # Create Temp Directory for this generation
            temp_lora_dir = f"/tmp/lora_ctx_{os.getpid()}_{int(time.time())}"
            os.makedirs(temp_lora_dir, exist_ok=True)
            print(f"--> [CPP I2I Worker] Created temp LoRA dir: {temp_lora_dir}", flush=True)

            cmd.extend(["--lora-model-dir", temp_lora_dir, "--lora-apply-mode", "at_runtime"])
            
            # Process LoRAs
            seen_names = {} # name -> count to handle collisions
            
            for lora in loras:
                # Format: [path, strength, active]
                raw_path = lora[0]
                strength = float(lora[1])
                active = lora[2]
                
                # Resolve Path
                if not raw_path.startswith("/"):
                    path = os.path.join(shared_utils.LORAS_DIR, raw_path)
                else:
                    path = raw_path

                if active and os.path.exists(path):
                    filename = os.path.basename(path)
                    name_stem, ext = os.path.splitext(filename)
                    
                    # Handle Collisions
                    if name_stem in seen_names:
                        seen_names[name_stem] += 1
                        link_name = f"{name_stem}_{seen_names[name_stem]}{ext}"
                        link_stem = f"{name_stem}_{seen_names[name_stem]}"
                    else:
                        seen_names[name_stem] = 0
                        link_name = filename
                        link_stem = name_stem
                    
                    link_path = os.path.join(temp_lora_dir, link_name)
                    
                    # Create Symlink
                    try:
                        if not os.path.exists(link_path):
                            os.symlink(path, link_path)
                            print(f"--> [CPP I2I Worker] Linked LoRA: {filename} -> {link_name} ({strength})", flush=True)
                    except Exception as le:
                        print(f"--> [CPP I2I Worker] Error linking LoRA {filename}: {le}", flush=True)
                        continue

                    # Update Prompt
                    esc_stem = re.escape(name_stem) # Match original name in prompt
                    # We look for <lora:OriginalName:Value>
                    pattern = rf"<lora:{esc_stem}:([0-9.]+)>"
                    
                    if re.search(pattern, prompt):
                        print(f"--> [CPP I2I Worker] Updating tag for {name_stem} -> {link_stem}", flush=True)
                        prompt = re.sub(pattern, f"<lora:{link_stem}:{strength}>", prompt)
                    else:
                        prompt += f" <lora:{link_stem}:{strength}>"

        except Exception as e:
            print(f"--> [CPP I2I Worker] Error processing LoRA config: {e}", flush=True)

    # 4b. Handle Prompt-Only LoRAs (if no UI LoRAs selected)
    if "--lora-model-dir" not in cmd and "<lora:" in prompt:
        print(f"--> [CPP I2I Worker] Detected LoRA tags in prompt but no LoRA directory set.", flush=True)
        # Try to find a default LoRA directory
        # 1. Z-Image Turbo Default
        default_lora_dir = os.path.join(shared_utils.MODELS_DIR, "Z-Image-Turbo-GGUF", "loras")
        # 2. General LORAS_DIR
        default_lora_dir_alt = shared_utils.LORAS_DIR
        if not os.path.exists(default_lora_dir):
            default_lora_dir = "/app/loras"
        
        # 3. If model path is deep, check relative 'loras' folder
        if os.path.exists(final_model):
            model_dir_loras = os.path.join(os.path.dirname(final_model), "loras")
            if os.path.exists(model_dir_loras):
                default_lora_dir = model_dir_loras

        if os.path.exists(default_lora_dir):
            print(f"--> [CPP I2I Worker] Auto-setting LoRA directory to: {default_lora_dir}", flush=True)
            cmd.extend(["--lora-model-dir", default_lora_dir])
        else:
            print(f"--> [CPP I2I Worker] Warning: Could not find a default LoRA directory.", flush=True)

    # UPDATE PROMPT IN CMD
    # cmd[4] is prompt because:
    # 0: binary
    # 1: --mode
    # 2: img_gen
    # 3: -t
    # 4: threads
    # 5: --prompt
    # 6: PROMPT <--- Index 6
        
        # safely find --prompt index
        try:
            p_idx = cmd.index("--prompt")
            if p_idx + 1 < len(cmd):
                cmd[p_idx + 1] = prompt
                print(f"--> [CPP I2I Worker] Updated Prompt with LoRA: {prompt}", flush=True)
        except:
             print("--> [CPP I2I Worker] Warning: Could not update prompt in command list", flush=True)


    # 5. Execution Loop (Adaptive Tiling)
    settings = load_settings()
    current_threshold = settings.get("vae_tiling_threshold_mp", 2)
    if current_threshold == 2: current_threshold = None # Enable adaptive logic
    tiling_config = TilingConfig(enabled=vae_tiling, threshold_mp=current_threshold)
    
    current_cmd = cmd.copy()
    current_cmd.extend(tiling_config.get_args(width, height))
    
    using_tiling = tiling_config.should_tile(width, height)
    if using_tiling:
        print(f"--> [CPP I2I Worker] Adaptive VAE Tiling enabled (Tile Size: {tiling_config.tile_size})", flush=True)

    # Retry Loop
    max_retries = 2
    final_returncode = -1
    for attempt in range(max_retries):
        print(f"--> [CPP I2I Worker] Execution Attempt {attempt+1}/{max_retries} (Tiling: {using_tiling})", flush=True)
        print(f"--> [CPP I2I Worker] Executing: {' '.join(current_cmd)}", flush=True)
        
        # Reset metrics for new attempt
        max_vram = 0.0
        compute_buffer_mb = 0.0
        last_vram_check = 0.0
        
        print(f"--> [CPP Worker] I2I Generation Attempt {attempt + 1}/{max_retries}...", flush=True)
        print(f"--> [CPP Worker] Running command: {' '.join(current_cmd)}", flush=True)
        
        try:
            process = subprocess.Popen(current_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors='replace')
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                # Check VRAM every 5 seconds instead of every 1 second.
                # rocm-smi is a subprocess that competes with the sd binary for
                # GPU driver access. Polling more often than 5s causes measurable
                # throughput loss on AMD GPUs (3-12s overhead per generation).
                if time.time() - last_vram_check > 5.0:
                     current_vram = get_gpu_memory()
                     if current_vram > max_vram: max_vram = current_vram
                     last_vram_check = time.time()
                
                if line:
                    raw = line.strip()
                    
                    # Parse Compute Buffer Size
                    if "z_image compute buffer size:" in raw:
                        try:
                            parts = raw.split("size:")
                            if len(parts) > 1:
                                val_part = parts[1].strip().split(" ")[0]
                                compute_buffer_mb = float(val_part)
                        except: pass

                    if "%" in raw or "step" in raw.lower():
                         print(f"data: LOG|{raw}", flush=True)
                    else:
                         print(raw, flush=True)

            final_returncode = process.poll()
            
            if final_returncode == 0:
                print("--> [CPP I2I Worker] Process finished successfully.", flush=True)
                break # Success!
            else:
                print(f"--> [CPP I2I Worker] Process failed with code {final_returncode}", flush=True)
                
                # If failed and NOT using tiling, try enabling it for next attempt
                if not using_tiling and attempt < max_retries - 1:
                     print(f"--> [CPP I2I Worker] ⚠️ Failure detected. Retrying with VAE Tiling enabled...", flush=True)
                     if "--vae-tiling" not in current_cmd:
                         current_cmd.extend(["--vae-tiling", "--vae-tile-size", "256"])
                     using_tiling = True
                     continue # Retry
                else:
                     break # Failed even with tiling or other error

        except Exception as e:
            print(f"--> [CPP I2I Worker] Execution Error: {e}", flush=True)
            break

    if final_returncode == 0:
        print(f"SUCCESS_OUTPUT:{save_path}", flush=True)
        
        # Load Loras for JSON
        lora_list = []
        if lora_config and os.path.exists(lora_config):
            try:
                with open(lora_config, 'r') as f:
                    lora_list = json.load(f)
            except: pass
        
        # Determine VAE used
        vae_used = None
        if custom_vae and os.path.exists(custom_vae):
            vae_used = custom_vae
        elif vae_path and os.path.exists(vae_path):
            vae_used = vae_path
            
        # SAVE JSON METADATA
        # Use original filenames (before blending) for source_image_1 and source_image_2
        inputs_data = {
            "prompt": prompt,
            "source_image_1": os.path.basename(original_source_1) if original_source_1 else "",
            "source_image_2": os.path.basename(original_source_2) if original_source_2 else None,
            "loras": shared_utils.format_loras_for_json(lora_list)
        }
        params_data = {
            "width": width, "height": height, "steps": steps,
            "guidance": guidance, "seed": seed,
            "strength": strength,
            "backend": "stable-diffusion.cpp",
            "mode": "i2i",
            "model": os.path.basename(final_model) + " (Diffusion)",
            "model_path": final_model or "",
            "negative_prompt": negative_prompt,
            "vae": os.path.basename(vae_used) if vae_used else "",
            "text_encoder": os.path.basename(llm_path) if llm_path else "",
            "clip_skip": clip_skip,
            "sampler": scheduler,
            "scheduler_type": scheduler_type,
            "loras": shared_utils.format_loras_for_json(lora_list),
            "max_img_size": max_img_size
        }
        outputs_data = {
            "type": "image",
            "files": [out_filename],
            "exec_time": round(time.time() - start_time, 2),
            "gpu_vram_usage_mb": round(max_vram, 2),
            "compute_buffer_mb": round(compute_buffer_mb, 2)
        }
        shared_utils.save_generation_log("i2i", inputs_data, params_data, outputs_data, image_path_for_filename=save_path, model_name="sd-cpp")
        
    else:
         print(f"--> [CPP I2I Worker] Process failed with code {final_returncode}")


    # Cleanup
    if temp_lora_dir and os.path.exists(temp_lora_dir):
        try:
            shutil.rmtree(temp_lora_dir)
            print(f"--> [CPP I2I Worker] Cleaned up temp LoRA dir: {temp_lora_dir}", flush=True)
        except Exception as e:
             print(f"--> [CPP I2I Worker] Warning: Failed to cleanup temp LoRA dir: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_config", type=str, default="")
    parser.add_argument("--top_k", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None, help="Path to VAE model (e.g. vae-ft-mse-840000-ema-pruned)")
    parser.add_argument("--clip_skip", type=int, default=1)
    parser.add_argument("--scheduler", type=str, default="euler")
    parser.add_argument("--scheduler_type", type=str, default="auto", help="Scheduler type: auto (model default), discrete, karras, exponential, ays, gits, sgm_uniform, simple, smoothstep, kl_optimal, lcm, bong_tangent, flux, flux2, logit_normal, beta, ltx2, llada_image")
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--vae_tiling", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    # I2I Specific
    parser.add_argument("--init_img", type=str, required=True, help="Path to the initial image")
    parser.add_argument("--init_img_2", type=str, default=None, help="Path to second image for blending")
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--mix_ratio", type=float, default=0.5, help="Blending ratio (0.0 = img1 only, 1.0 = img2 only)")
    # === Multi-Image Conditioning (Phase 1A / N-Slot) ===
    parser.add_argument("--ref-image", dest="ref_images", action="append", default=None,
                        help="Path to reference image (repeatable for N-slot)")
    parser.add_argument("--mask-image", type=str, default=None, help="Path to mask image (inpainting)")
    parser.add_argument("--control-image", type=str, default=None, help="Path to control image (ControlNet)")
    parser.add_argument("--control-strength", type=float, default=0.8, help="ControlNet strength (0-1)")
    parser.add_argument("--image_role", type=str, default=None, help="Role of init image (init/face/ref)")
    parser.add_argument("--no_auto_resize_ref", action="store_true", help="Disable auto-resize of reference images")
    # === Max Image Dimension (saves VRAM) ===
    parser.add_argument("--max_img_size", type=int, default=1024,
                        help="Max image dimension in pixels. Images larger than this are resized to save VRAM (default: 1024)")
    parser.add_argument("--use_slg", type=lambda x: x.lower() == "true", default=False,
                        help="Enable Skip Layer Guidance (SLG) for faster CFG with DiT models")
    parser.add_argument("--use_apg", type=lambda x: x.lower() == "true", default=False,
                        help="Enable Adaptive Projected Guidance (APG) as alternative to standard CFG")
    
    args = parser.parse_args()
    
    run_sd_cpp_i2i(
        args.prompt, args.width, args.height, args.steps, args.guidance,
        args.seed, args.init_img, args.strength,
        args.lora_config, args.top_k, args.temperature,
        args.model_path, args.vae_path, args.clip_skip, args.scheduler,
        args.scheduler_type, args.llm_path, args.vae_tiling, args.negative_prompt,
        args.init_img_2, args.mix_ratio,
        ref_images=args.ref_images,
        mask_image=getattr(args, 'mask_image', None),
        control_image=getattr(args, 'control_image', None),
        control_strength=getattr(args, 'control_strength', 0.8),
        image_role=getattr(args, 'image_role', None),
        no_auto_resize_ref=getattr(args, 'no_auto_resize_ref', False),
        max_img_size=args.max_img_size,
        use_slg=args.use_slg,
        use_apg=args.use_apg
    )


# Alias for compatibility
run_i2i_cpp = run_sd_cpp_i2i
