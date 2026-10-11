"""
T2I CPP worker - Text-to-Image with stable-diffusion.cpp
"""
import os
import sys
import argparse
import subprocess
import time

DEFAULT_SD_BINARY = "/usr/local/bin/sd"

sys.path.insert(0, os.getcwd())

from workers.base import (
    get_gpu_memory, find_model_file,
    find_vae, find_llm, apply_prompt_template,
    is_zimage_model, is_qwen_image_edit_model, is_qwen_image_2_1_model, is_qwen3vl_8b_encoder,
    get_scheduler_arg, TilingConfig, save_output_log,
    find_bf16_zimage_alternative
)
from workers.t2i_cpp_helpers import find_diffusion_model, find_llm_for_zimage, find_llm_for_flux2, SCHED_MAP
from core.config import load_settings
from core.fix_gguf import ensure_fixed
from core.text_encoders import find_text_encoders_for_gguf
import shared_utils


# ---------------------------------------------------------------------------
# Krea family: explicit base + VAE resolution
# ---------------------------------------------------------------------------
# "Krea" is a FAMILY of checkpoints built on DIFFERENT base architectures, and
# each base needs a DIFFERENT VAE. Picking the wrong one silently breaks the
# image (wrong latent space / channel count / normalisation):
#   flux   -> FLUX VAE        (FLUX.1 Krea [dev] / "Krea 1")
#   wan2.1 -> Wan 2.1 VAE     (Krea 2 / Krea Realtime; 16-channel latent)
#   wan2.2 -> Wan 2.2 VAE     (Krea 2.x on Wan 2.2)
# The old code matched ANY file containing "wan" + "vae", so with both
# wan_2.1_vae and wan2.2_vae in models/vae/ it could silently pick the wrong one.
KREA_VAE_PATTERNS = {
    "wan2.1": ("wan_2.1_vae", "wan2.1_vae", "wan_21_vae", "wan_2.1", "wan2.1"),
    "wan2.2": ("wan2.2_vae", "wan_2.2_vae", "wan22_vae", "wan_2.2", "wan2.2"),
}
KREA_BASE_LABEL = {
    "flux":   "FLUX (FLUX.1 Krea / Krea 1)",
    "wan2.1": "Wan 2.1 (Krea 2 / Krea Realtime)",
    "wan2.2": "Wan 2.2 (Krea 2.x)",
}


def detect_krea_base(model_path: str) -> str:
    """Classify a Krea checkpoint by its base architecture.

    Explicit tokens win. A bare 'krea' defaults to the Wan 2.1 line (the one
    this app ships as 'Krea 2'); FLUX-based Krea is recognised by 'flux' (or an
    explicit krea1 token) so it is never given a Wan VAE.
    """
    n = os.path.basename(model_path or "").lower()
    if any(t in n for t in ("wan2.2", "wan_2.2", "wan-2.2", "wan 2.2", "wan22",
                            "krea2.2", "krea_2.2", "krea-2.2", "krea 2.2")):
        return "wan2.2"
    if any(t in n for t in ("flux", "krea1", "krea_1", "krea-1", "krea 1")):
        return "flux"
    return "wan2.1"


def krea_vae_matches(vae_file: str, base: str) -> bool:
    """True when a VAE filename belongs to the given Krea base lineage."""
    patterns = KREA_VAE_PATTERNS.get(base)
    if not patterns:
        return False
    low = os.path.basename(vae_file or "").lower()
    return any(p in low for p in patterns)


def resolve_krea_vae(model_path: str, override: str = None):
    """Strictly resolve the VAE for a Wan-based Krea checkpoint.

    Returns (vae_path, base, searched_dirs). Only a VAE matching the detected
    base is accepted, so several VAE files in models/vae/ cannot make it pick
    the wrong lineage. An explicit `override` is honoured only if it matches.
    """
    base = detect_krea_base(model_path)
    dirs = []
    for d in (os.path.dirname(model_path or ""), "/app/models/vae",
              os.path.join(shared_utils.MODELS_DIR, "vae")):
        if d and d not in dirs:
            dirs.append(d)
    if override and os.path.isfile(override) and krea_vae_matches(override, base):
        return override, base, dirs
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".safetensors", ".sft", ".gguf", ".pth")):
                continue
            if krea_vae_matches(f, base):
                return os.path.join(d, f), base, dirs
    return None, base, dirs


def run_t2i_cpp(prompt: str, width: int, height: int, steps: int, guidance: float,
                seed: int, lora_config: str = None, top_k: float = 1.0,
                temperature: float = 0.6, model_path: str = None,
                vae_path: str = None, clip_skip: int = 1, scheduler: str = "euler",
                scheduler_type: str = "auto", llm_path: str = None,
                vae_tiling: bool = True, negative_prompt: str = "",
                use_slg: bool = False, use_apg: bool = False,
                ref_images: list = None):
    
    start_time = time.time()
    print(f"--> [T2I CPP] Starting... PID: {os.getpid()}")

    # Reference images for edit-capable models (sd.cpp `-r`/`--ref-image`).
    ref_images = [p for p in (ref_images or []) if p and os.path.exists(p)]
    ref_added = False
    
    settings = load_settings()
    # Pass None as threshold to trigger adaptive VRAM-based logic if setting is the old default (2)
    current_threshold = settings.get("vae_tiling_threshold_mp", 2)
    if current_threshold == 2: current_threshold = None
    
    tiling_config = TilingConfig(enabled=vae_tiling, threshold_mp=current_threshold)
    
    # Binary
    sd_binary = DEFAULT_SD_BINARY
    if not os.path.exists(sd_binary):
        local_sd = os.path.join(os.getcwd(), "sd")
        if os.path.exists(local_sd):
            sd_binary = local_sd
        else:
            print(f"ERROR: Binary not found at {sd_binary}")
            sys.exit(1)
    
    # Output
    out_filename = f"t2i_cpp_{int(time.time())}.png"
    save_path = os.path.join(shared_utils.OUTPUT_DIR, out_filename)
    os.makedirs(shared_utils.OUTPUT_DIR, exist_ok=True)
    
    # Model
    final_model = find_diffusion_model(model_path)
    if not final_model or not os.path.exists(final_model):
        print(f"ERROR: Model not found: {final_model}")
        sys.exit(1)
    
    # --- RDNA3 OPTIMIZATION: quantized GGUF is ~2x slower than BF16 ---
    # On AMD RDNA3 (gfx1100) the Q8_0 GGUF matmul kernels use the slow
    # vec-dot/dp4a path, while BF16 safetensors go through rocBLAS WMMA
    # (tensor cores). If a BF16 sibling exists, upgrade transparently.
    if settings.get("auto_prefer_bf16_zimage", True):
        bf16_alt = find_bf16_zimage_alternative(final_model)
        if bf16_alt:
            print("--> [T2I CPP] [OPTIMIZATION] Q8_0/quantized GGUF matmul is ~2x slower on RDNA3")
            print(f"--> [T2I CPP] [OPTIMIZATION] Auto-upgraded model to BF16: {os.path.basename(bf16_alt)}")
            print("--> [T2I CPP] [OPTIMIZATION] (set auto_prefer_bf16_zimage=false in settings.json to force quantized)")
            final_model = bf16_alt
    
    print(f"--> [T2I CPP] Using model: {final_model}")

    # Qwen-Image-2.1: full T2I model (also supports editing) loaded with
    # --diffusion-model + --llm (Qwen3-VL-8B) + --vae. Detect it BEFORE the
    # Edit-only guard below, because the guard would otherwise reject it.
    is_qwen21 = is_qwen_image_2_1_model(final_model)

    # === Guard: Qwen-Image-Edit is a diffusion-only *edit* model ===
    # It requires a reference image + Qwen2.5-VL (workers/i2i_edit.py) and
    # cannot run as a text-to-image checkpoint. Fail clearly instead of
    # letting sd.cpp die with "get sd version from file failed".
    if (not is_qwen21) and is_qwen_image_edit_model(final_model):
        print(f"--> [T2I CPP] ERROR: Qwen-Image-Edit model '{os.path.basename(final_model)}' cannot be used for Text-to-Image.", flush=True)
        print("--> [T2I CPP] It edits an existing image via prompt + reference. Switch to the I2I tab and load an input image.", flush=True)
        sys.exit(1)

    # Auto-fix GGUF
    if settings.get("auto_fix_models", True) and final_model.endswith(".gguf"):
        final_model = ensure_fixed(final_model)
    
    # Detect model type
    is_zimage = is_zimage_model(final_model)
    
    # Detect Ideogram4 model
    is_ideogram4 = "ideogram4" in os.path.basename(final_model).lower() and final_model.endswith(".gguf")
    
    # Detect FLUX2 model
    is_flux2 = "flux2" in final_model.lower() and final_model.endswith((".gguf", ".safetensors"))
    
    # Detect FLUX model (needs --diffusion-model instead of -m for GGUF)
    is_flux = ("flux" in final_model.lower() and not is_flux2 and not is_ideogram4) and final_model.endswith(".gguf")

    # Detect Krea model. Krea spans several bases (see helpers above); the VAE
    # and text encoder depend on the base, so classify it explicitly here.
    is_krea = ("krea" in final_model.lower()) and final_model.endswith((".gguf", ".safetensors"))
    krea_base = detect_krea_base(final_model) if is_krea else None
    is_krea_wan = is_krea and krea_base in ("wan2.1", "wan2.2")
    if is_krea:
        print(f"--> [T2I CPP] Krea detected: base = {KREA_BASE_LABEL.get(krea_base, krea_base)}")
        if krea_base == "flux":
            print("--> [T2I CPP] FLUX-based Krea -> uses the FLUX VAE/components (no Wan VAE).")

    # === Per-model recommended defaults (dataModels.json) ===
    # A model folder may ship a dataModels.json declaring its recommended
    # steps/guidance/sampler/size. Defaults win only when the incoming value is
    # still the global UI default (i.e. the user did not touch the slider).
    try:
        from core.model_config import get_effective_params
        _incoming = {"steps": steps, "guidance": guidance,
                     "width": width, "height": height, "sampler": scheduler}
        _eff = get_effective_params(final_model, _incoming)
        if _eff.get("_applied_defaults"):
            print(f"--> [T2I CPP] Applied model defaults from dataModels.json: {_eff['_applied_defaults']}", flush=True)
        steps    = _eff["steps"]
        guidance = _eff["guidance"]
        width    = _eff["width"]
        height   = _eff["height"]
        scheduler = _eff["sampler"]
    except Exception as _ce:
        print(f"--> [T2I CPP] WARNING: could not read model config: {_ce}", flush=True)
    
    # Check for FLUX manifest (created by setup module)
    flux_manifest = None
    if is_flux:
        model_dir = os.path.dirname(final_model)
        manifest_path = os.path.join(model_dir, "flux_setup.json")
        if not os.path.exists(manifest_path):
            # Also check parent directories
            manifest_path = os.path.join(os.path.dirname(model_dir), "flux_setup.json")
        if os.path.exists(manifest_path):
            import json
            with open(manifest_path, 'r') as f:
                flux_manifest = json.load(f)
            print(f"--> [T2I CPP] Found FLUX manifest: {manifest_path}", flush=True)
    
    original_prompt = prompt
    if is_zimage:
        prompt = apply_prompt_template(prompt)
    
    # Build command
    # NOTE: -t 0 tells sd.cpp to auto-detect physical core count.
    # Using logical cores (os.cpu_count()) causes thread contention and
    # degrades performance on systems with hyperthreading.
    cmd = [
        sd_binary, "--mode", "img_gen", "-t", "0",
        "--prompt", prompt, "-W", str(width), "-H", str(height),
        "--steps", str(steps), "--cfg-scale", str(guidance),
        "--seed", str(seed), "--fa", "-o", save_path, "-v"
    ]
    
    # Add negative prompt if provided
    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])
    
    # Reference-editing needs a lighter VAE tiling threshold: the Qwen 2.1 VAE
    # uses ~9.3 GB of compute buffer per megapixel.
    if is_qwen21 and ref_images:
        tiling_config = TilingConfig(enabled=vae_tiling, threshold_mp=2.0)
    cmd.extend(tiling_config.get_args(width, height))
    
    # Model-specific args
    if is_krea_wan:
        print(f"--> [T2I CPP] Detected Wan-based Krea ({KREA_BASE_LABEL[krea_base]}): "
              f"--diffusion-model + --llm (Qwen3VL 4B) + explicit --vae.")
        cmd.extend(["--diffusion-model", final_model, "--diffusion-fa"])
        # Wan VAE full-res decode buffer (~1.9GB) can OOM on top of the diffusion
        # model + LLM encoder, so force VAE tiling for every Wan-based Krea.
        cmd.extend(["--vae-tiling", "--vae-tile-size", "64"])

        # --- Explicit VAE selection: match the Wan 2.1 / 2.2 lineage EXACTLY ---
        # (never fall back to the other Wan version or to a FLUX VAE).
        krea_vae, krea_base, krea_vae_dirs = resolve_krea_vae(final_model, vae_path)
        if krea_vae and os.path.exists(krea_vae):
            print(f"--> [T2I CPP] Krea VAE (explicit {krea_base} match): {krea_vae}")
            cmd.extend(["--vae", krea_vae])
        else:
            expected = KREA_VAE_PATTERNS[krea_base][0] + ".safetensors"
            avail = []
            for d in krea_vae_dirs:
                if os.path.isdir(d):
                    avail += [f for f in os.listdir(d)
                              if f.lower().endswith((".safetensors", ".sft", ".gguf", ".pth"))]
            print(f"--> [T2I CPP] ERROR: no {KREA_BASE_LABEL[krea_base]} VAE found for this Krea model.")
            print(f"--> [T2I CPP] Expected e.g. '{expected}' in {krea_vae_dirs}.")
            if avail:
                print(f"--> [T2I CPP] VAE files present but none match the base: {sorted(set(avail))}")
            print("--> [T2I CPP] Refusing to fall back to a mismatching VAE (it would silently break the image).")

        llm_krea = None
        # Krea2 REQUIRES Qwen3-VL 4B (text_dim=2560). Other Qwen3-VL sizes
        # (e.g. 8B, hidden 4096) cause GGML_ASSERT shape mismatch at sampling.
        # Always prefer the 4B encoder in the krea subdir; only accept llm_path
        # if it is actually a 4B model.
        llm_dirs = ["/app/models/text_encoders/krea", os.path.join(shared_utils.MODELS_DIR, "text_encoders"),
                    shared_utils.MODELS_DIR]
        if llm_path and os.path.exists(llm_path) and "4b" in os.path.basename(llm_path).lower():
            llm_krea = llm_path
        else:
            # First pass: prefer abliterated Qwen3-VL 4B (less censored / more elastic)
            for d in llm_dirs:
                if os.path.isdir(d):
                    for root, _, files in os.walk(d):
                        for f in files:
                            if f.endswith(".gguf") and ("qwen3vl" in f.lower() or "qwen3-vl" in f.lower()) \
                               and "4b" in f.lower() and "mmproj" not in f.lower() \
                               and "abliterated" in f.lower():
                                llm_krea = os.path.join(root, f)
                                break
                        if llm_krea:
                            break
                if llm_krea:
                    break
            # Second pass: any matching Qwen3-VL 4B
            if not llm_krea:
                for d in llm_dirs:
                    if os.path.isdir(d):
                        for root, _, files in os.walk(d):
                            for f in files:
                                if f.endswith(".gguf") and ("qwen3vl" in f.lower() or "qwen3-vl" in f.lower()) \
                                   and "4b" in f.lower() and "mmproj" not in f.lower():
                                    llm_krea = os.path.join(root, f)
                                    break
                            if llm_krea:
                                break
                    if llm_krea:
                        break
        if llm_krea and os.path.exists(llm_krea):
            print(f"--> [T2I CPP] Using LLM for {KREA_BASE_LABEL[krea_base]}: {llm_krea}")
            cmd.extend(["--llm", llm_krea, "--clip-on-cpu"])
            # LLM safetensors (HF bf16, ~8.8GB) usano piu' ram/VRAM del Q4 GGUF:
            # decodifica VAE su CPU per evitare OOM sullo split VRAM col diffusion model.
            if llm_krea.lower().endswith(".safetensors"):
                print("--> [T2I CPP] safetensors LLM detected: forcing VAE decode on CPU to save VRAM")
                cmd.extend(["--vae-on-cpu"])
        else:
            print(f"--> [T2I CPP] WARNING: Qwen3VL text encoder not found for {KREA_BASE_LABEL[krea_base]} model. Generation may fail.")
    elif is_qwen21:
        print("--> [T2I CPP] Detected Qwen-Image 2.1 model. Using --diffusion-model + --llm (Qwen3-VL-8B) + --vae.")
        from core.model_config import get_component

        # --- VAE (2.1 has its OWN VAE; the old qwen_image_vae is NOT compatible) ---
        vae21 = vae_path if (vae_path and os.path.exists(vae_path)) else get_component(final_model, "vae")
        if not vae21:
            for d in [os.path.dirname(final_model), os.path.join(shared_utils.MODELS_DIR, "vae")]:
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if f.endswith((".safetensors", ".sft")) and "qwen_image_2.1" in f.lower() and "vae" in f.lower():
                            vae21 = os.path.join(d, f)
                            break
                if vae21:
                    break

        # --- Text encoder: MUST be Qwen3-VL-8B (context dim 4096) ---
        # The caller (e.g. Agent Refine, whose encoder dropdown is shared with
        # Z-Image) may pass a Qwen3-VL-4B encoder: sd.cpp would then abort with
        # "requires an image latent and text conditioning with batch size 1".
        # Ignore anything that is not an 8B VL encoder and fall back to the
        # model config / auto-detection.
        llm21 = None
        if llm_path and os.path.exists(llm_path):
            if is_qwen3vl_8b_encoder(llm_path):
                llm21 = llm_path
            else:
                print(f"--> [T2I CPP] WARNING: text encoder '{os.path.basename(llm_path)}' is not "
                      f"compatible with Qwen-Image-2.1 (needs Qwen3-VL-8B, hidden 4096). Ignoring it.",
                      flush=True)
        if not llm21:
            llm21 = get_component(final_model, "llm")
        if not llm21:
            enc_dir = os.path.join(shared_utils.MODELS_DIR, "text_encoders")
            if os.path.isdir(enc_dir):
                for f in sorted(os.listdir(enc_dir)):
                    if is_qwen3vl_8b_encoder(f):
                        llm21 = os.path.join(enc_dir, f)
                        break

        cmd.extend(["--diffusion-model", final_model, "--offload-to-cpu", "--diffusion-fa"])
        if vae21 and os.path.exists(vae21):
            print(f"--> [T2I CPP] Qwen-2.1 VAE: {vae21}")
            cmd.extend(["--vae", vae21])
        else:
            print("--> [T2I CPP] WARNING: Qwen-Image-2.1 VAE not found (expected vae/qwen_image_2.1_vae_bf16.safetensors).")
        if llm21 and os.path.exists(llm21):
            print(f"--> [T2I CPP] Qwen-2.1 text encoder: {llm21}")
            cmd.extend(["--llm", llm21])
        else:
            print("--> [T2I CPP] WARNING: Qwen3-VL-8B text encoder not found for Qwen-Image-2.1.")
        # Expose the resolved components so the generation log / history records them
        vae_path = vae21 if (vae21 and os.path.exists(vae21)) else vae_path
        llm_path = llm21 if (llm21 and os.path.exists(llm21)) else llm_path

        # ---- reference edit (sd.cpp -r + --llm_vision) ----
        if ref_images:
            from core.edit_capabilities import resolve_vision_encoder
            vis = resolve_vision_encoder("qwen_image_2_1", final_model)
            if vis and os.path.exists(vis):
                for r in ref_images:
                    cmd.extend(["-r", r])
                cmd.extend(["--llm_vision", vis])
                ref_added = True
                print(f"--> [T2I CPP] Reference edit: {len(ref_images)} ref image(s), "
                      f"vision={os.path.basename(vis)}", flush=True)
            else:
                print("--> [T2I CPP] WARNING: Qwen-2.1 reference edit needs the Qwen3-VL mmproj "
                      "(text_encoders/mmproj-Qwen3VL-8B-*.gguf). Reference ignored -> text-to-image.",
                      flush=True)
    elif is_flux2:
        print("--> [T2I CPP] Detected FLUX.2 model. Using --diffusion-model flag and offload-to-cpu.")
        cmd.extend(["--diffusion-model", final_model, "--offload-to-cpu", "--diffusion-fa"])
        
        vae_path = find_vae(final_model)
        if not vae_path or "flux2" not in os.path.basename(vae_path).lower():
            # Try to force find flux2 ae
            for d in [os.path.dirname(final_model), os.path.join(shared_utils.MODELS_DIR, "vae")]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if "flux2_ae" in f.lower() or "flux2-vae" in f.lower() or "flux2" in f.lower() and "vae" in f.lower():
                            vae_path = os.path.join(d, f)
                            break
        
        if vae_path and os.path.exists(vae_path):
             cmd.extend(["--vae", vae_path])
        else:
             print("--> [T2I CPP] WARNING: VAE not found for FLUX.2 model. Generation may fail.")
             
        llm_search = find_llm_for_flux2(final_model, llm_path)
        if llm_search:
            print(f"--> [T2I CPP] Using LLM for FLUX.2: {llm_search}")
            cmd.extend(["--llm", llm_search, "--clip-on-cpu"])
        else:
            print("--> [T2I CPP] WARNING: Mistral text encoder not found for FLUX.2 model. Generation may fail.")

        # FLUX.2 supports reference images (flux2 preset)
        if ref_images:
            for r in ref_images:
                cmd.extend(["-r", r])
            ref_added = True
            print(f"--> [T2I CPP] Reference edit: {len(ref_images)} ref image(s) (FLUX.2 preset).", flush=True)
    elif is_flux:
        print("--> [T2I CPP] Detected FLUX GGUF model. Using --diffusion-model flag.")
        # Must pass -m as well for VAE loading to work
        cmd.extend(["-m", final_model])
        cmd.extend(["--diffusion-model", final_model])
        
        # FLUX requires VAE, clip_l, and t5xxl
        # Use manifest if available, otherwise auto-detect
        if flux_manifest:
            vae_path = flux_manifest.get("vae")
            clip_l = flux_manifest.get("clip_l")
            t5xxl = flux_manifest.get("t5xxl")
            print("--> [T2I CPP] Using FLUX manifest for component paths.")
            # The manifest may hold paths from another environment (e.g. /workspace/...):
            # ignore any entry missing on disk and fall back to auto-detection,
            # otherwise FLUX runs with NO text encoders -> "failed to encode prompt".
            if not (vae_path and os.path.exists(vae_path)):
                vae_path = find_vae(final_model)
            if not (clip_l and os.path.exists(clip_l)) or not (t5xxl and os.path.exists(t5xxl)):
                auto_clip_l, auto_t5xxl = find_text_encoders_for_gguf(final_model)
                if not (clip_l and os.path.exists(clip_l)):
                    clip_l = auto_clip_l
                if not (t5xxl and os.path.exists(t5xxl)):
                    t5xxl = auto_t5xxl
                print("--> [T2I CPP] Manifest encoder paths missing on disk: using auto-detected encoders.")
        else:
            vae_path = find_vae(final_model)
            clip_l, t5xxl = find_text_encoders_for_gguf(final_model)
        
        if vae_path and os.path.exists(vae_path):
            cmd.extend(["--vae", vae_path])
        else:
            print("--> [T2I CPP] WARNING: VAE not found for FLUX model. Generation may fail.")
        
        if clip_l and os.path.exists(clip_l):
            cmd.extend(["--clip_l", clip_l])
        if t5xxl and os.path.exists(t5xxl):
            cmd.extend(["--t5xxl", t5xxl])
        if not clip_l or not t5xxl:
            print("--> [T2I CPP] WARNING: Text encoders not found for FLUX model. Generation may fail.")

        # FLUX Kontext supports reference images; plain FLUX.1-dev does not.
        if ref_images:
            if "kontext" in final_model.lower():
                for r in ref_images:
                    cmd.extend(["-r", r])
                ref_added = True
                print(f"--> [T2I CPP] Reference edit: {len(ref_images)} ref image(s) (kontext preset).", flush=True)
            else:
                print("--> [T2I CPP] WARNING: plain FLUX.1 does not support reference editing "
                      "(only Kontext). Reference ignored.", flush=True)
    elif is_zimage:
        print("--> [T2I CPP] Z-Image architecture detected")
        cmd.extend(["-m", final_model, "--diffusion-model", final_model])
        
        # User VAE has priority, otherwise try to find it
        if not vae_path or not os.path.exists(vae_path):
            vae_path = find_vae(final_model)
            
        if vae_path and os.path.exists(vae_path):
            cmd.extend(["--vae", vae_path])
        else:
            print("--> [T2I CPP] WARNING: Z-Image requires a VAE but none was found or selected.")
        
        # Z-Image uses a 3D VAE that requires ~10GB compute buffer at 1024x1376.
        # Force VAE tiling at all resolutions to prevent OOM even on 24GB cards.
        # NOTE: Do NOT specify --vae-tile-size — let the binary auto-calculate
        # the optimal tile size. Hardcoding 128px causes excessive tiling overlap
        # and GPU segfaults (SIGSEGV -11).
        cmd.extend(["--vae-tiling"])
        
        llm_search = find_llm_for_zimage(final_model, llm_path)
        if llm_search:
            print(f"--> [T2I CPP] Using LLM: {llm_search}")
            cmd.extend(["--llm", llm_search])
            # Keep LLM in RAM to leave VRAM for Z-Image compute buffers.
            # Without --clip-on-cpu the Qwen3 (7.5GB) fills VRAM, leaving only ~4.6GB
            # for compute — not enough for the Z-Image diffusion graph.
            cmd.extend(["--clip-on-cpu"])
            
        # Enable cache-dit for DiT models (Z-Image) to skip redundant diffusion steps
        cmd.extend(["--cache-mode", "cache-dit"])
    elif is_ideogram4:
        print("--> [T2I CPP] Detected Ideogram4 model. Using --diffusion-model + --uncond-diffusion-model + --llm (Qwen3VL).")
        cmd.extend(["--diffusion-model", final_model])
        # Find unconditional model (same dir, with _uncond in name)
        ideogram_dir = os.path.dirname(final_model)
        uncond_model = os.path.join(ideogram_dir, "ideogram4_uncond-Q4_0.gguf")
        if os.path.exists(uncond_model):
            cmd.extend(["--uncond-diffusion-model", uncond_model])
        else:
            print(f"--> [T2I CPP] WARNING: Unconditional model not found at {uncond_model}")
        # Use flux2 VAE (same as FLUX.2)
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
            print("--> [T2I CPP] WARNING: VAE not found for Ideogram4. Generation may fail.")
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
            print(f"--> [T2I CPP] Using LLM for Ideogram4: {llm_search}")
            cmd.extend(["--llm", llm_search, "--offload-to-cpu"])
        else:
            print("--> [T2I CPP] WARNING: Qwen3VL text encoder not found. Please download it.")
    else:
        cmd.extend(["-m", final_model])
        # Use custom VAE if provided, otherwise try to find one for pony/flux models
        if vae_path and os.path.exists(vae_path):
            cmd.extend(["--vae", vae_path])
        elif "pony" in final_model.lower() or "flux" in final_model.lower():
            # Try to find VAE for pony/flux models
            auto_vae = find_vae(final_model)
            if auto_vae:
                cmd.extend(["--vae", auto_vae])
            clip_l, t5xxl = find_text_encoders_for_gguf(final_model)
            if clip_l: cmd.extend(["--clip_l", clip_l])
            if t5xxl: cmd.extend(["--t5xxl", t5xxl])

    if ref_images and not ref_added:
        print("--> [T2I CPP] WARNING: this model has no reference-edit support; "
              "ignoring the reference image(s).", flush=True)

    # Scheduler
    cmd.extend(["--sampling-method", SCHED_MAP.get(scheduler, "euler")])
    # Scheduler Type: "auto"/empty => let sd.cpp pick the model-native default
    # (flux for FLUX/Qwen-2.1, flux2 for FLUX.2, logit_normal for Ideogram4, ...).
    if scheduler_type and scheduler_type not in ("auto", "default"):
        cmd.extend(["--scheduler", scheduler_type])
    
    # Guidance Optimization: SLG (Skip Layer Guidance) for DiT models
    if use_slg:
        print("--> [T2I CPP] Enabling Skip Layer Guidance (SLG) for faster CFG")
        cmd.extend(["--slg-scale", "2.5", "--skip-layers", "[7,8,9]"])
    
    # Guidance Optimization: APG (Adaptive Projected Guidance)
    if use_apg:
        print("--> [T2I CPP] Enabling Adaptive Projected Guidance (APG)")
        cmd.extend(["--extra-sample-args", "apg_eta=1.0,apg_momentum=0.0"])
    
    # LoRA Configuration (Gather and Link Strategy)
    temp_lora_dir = None
    if lora_config and os.path.exists(lora_config):
        try:
            import json
            import re
            with open(lora_config, "r") as f:
                loras = json.load(f)
            
            # Create Temp Directory for this generation
            temp_lora_dir = f"/tmp/lora_ctx_{os.getpid()}_{int(time.time())}"
            os.makedirs(temp_lora_dir, exist_ok=True)
            print(f"--> [T2I CPP] Created temp LoRA dir: {temp_lora_dir}", flush=True)

            cmd.extend(["--lora-model-dir", temp_lora_dir, "--lora-apply-mode", "at_runtime"])
            
            # Process LoRAs
            seen_names = {} # name -> count to handle collisions
            
            for lora_item in loras:
                # Format: [path, strength, active]
                raw_path = lora_item[0]
                strength = float(lora_item[1])
                active = lora_item[2]
                
                # Resolve Path
                if not raw_path.startswith("/"):
                    path = os.path.join(shared_utils.LORAS_DIR, raw_path)
                else:
                    path = raw_path

                if active and strength != 0 and os.path.exists(path):
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
                            print(f"--> [T2I CPP] Linked LoRA: {filename} -> {link_name} ({strength})", flush=True)
                    except Exception as le:
                        print(f"--> [T2I CPP] Error linking LoRA {filename}: {le}", flush=True)
                        continue

                    # Update Prompt
                    esc_stem = re.escape(name_stem) # Match original name in prompt
                    # We look for <lora:OriginalName:Value>
                    pattern = rf"<lora:{esc_stem}:([0-9.]+)>"
                    
                    if re.search(pattern, prompt):
                        print(f"--> [T2I CPP] Updating tag for {name_stem} -> {link_stem}", flush=True)
                        prompt = re.sub(pattern, f"<lora:{link_stem}:{strength}>", prompt)
                    else:
                        prompt += f" <lora:{link_stem}:{strength}>"
            
            # Update the prompt argument which was already set
            try:
                prompt_idx = cmd.index("--prompt") + 1
                cmd[prompt_idx] = prompt
            except ValueError:
                pass
            print(f"--> [T2I CPP] Applied LoRAs to prompt: {prompt}")

        except Exception as e:
            print(f"--> [T2I CPP] Error applying LoRA config: {e}")
    
    print(f"--> [T2I CPP] Executing: {' '.join(cmd[:10])}...")
    
    max_vram = 0.0
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors='replace')
        last_check = 0
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            # Check VRAM every 5 seconds instead of every 1 second.
            # rocm-smi is a subprocess that competes with the sd binary for
            # GPU driver access. Polling more often than 5s causes measurable
            # throughput loss on AMD GPUs (3-12s overhead per generation).
            if time.time() - last_check > 5.0:
                current_vram = get_gpu_memory()
                if current_vram > max_vram:
                    max_vram = current_vram
                last_check = time.time()
            
            if line:
                raw = line.strip()
                if "%" in raw or "step" in raw.lower():
                    print(f"data: LOG|{raw}")
                else:
                    print(raw)
            
            if process.poll() is not None:
                break
        
        returncode = process.poll()
        
        if returncode == 0 and os.path.exists(save_path):
            print(f"SUCCESS_OUTPUT:{save_path}")
            loras_raw = loras if 'loras' in dir() and loras else []
            inputs_data = {"prompt": original_prompt, "loras": shared_utils.format_loras_for_json(loras_raw)}
            vae_used = None
            if vae_path and os.path.exists(vae_path):
                vae_used = vae_path
                
            params_data = {
                "width": width, "height": height, "steps": steps,
                "guidance": guidance, "seed": seed,
                "backend": "stable-diffusion.cpp",
                "model": os.path.basename(final_model),
                "negative_prompt": negative_prompt,
                "vae": os.path.basename(vae_used) if vae_used else "",
                "clip_skip": clip_skip,
                "sampler": scheduler,
                "scheduler_type": scheduler_type,
                "text_encoder": os.path.basename(llm_path) if llm_path else "",
                "loras": shared_utils.format_loras_for_json(loras_raw)
            }
            outputs_data = {
                "type": "image", "files": [out_filename],
                "gpu_vram_usage_mb": round(max_vram, 2),
                "exec_time": round(time.time() - start_time, 2)
            }
            save_output_log("t2i", inputs_data, params_data, outputs_data,
                          image_path=save_path, model_name="sd-cpp")
        else:
            print(f"ERROR: Process failed with code {returncode}")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        if temp_lora_dir and os.path.exists(temp_lora_dir):
            try:
                import shutil
                shutil.rmtree(temp_lora_dir)
                print(f"--> [T2I CPP] Cleaned up temp LoRA dir: {temp_lora_dir}", flush=True)
            except Exception as e:
                print(f"--> [T2I CPP] Warning: Failed to cleanup temp LoRA dir: {e}", flush=True)


def main():
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
    parser.add_argument("--ref_image", dest="ref_images", action="append", default=None,
                        help="Reference image for edit-capable models (repeatable); "
                             "forwarded to sd.cpp as -r")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--use_slg", type=lambda x: x.lower() == "true", default=False,
                        help="Enable Skip Layer Guidance (SLG) for faster CFG with DiT models")
    parser.add_argument("--use_apg", type=lambda x: x.lower() == "true", default=False,
                        help="Enable Adaptive Projected Guidance (APG) as alternative to standard CFG")
    
    args = parser.parse_args()
    run_t2i_cpp(args.prompt, args.width, args.height, args.steps, args.guidance,
                args.seed, args.lora_config, args.top_k, args.temperature,
                args.model_path, args.vae_path, args.clip_skip, args.scheduler,
                args.scheduler_type, args.llm_path, args.vae_tiling, args.negative_prompt,
                args.use_slg, args.use_apg, args.ref_images)


if __name__ == "__main__":
    main()
