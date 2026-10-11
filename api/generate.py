"""
API Generate - Dispatcher for generation
"""
import os
import sys
import subprocess
import asyncio
import time
import random
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from core.lora import create_lora_config
from shared_utils import OUTPUT_DIR, sanitize_prompt
from workers.base import is_zimage_model, is_qwen_image_edit_model, is_qwen_image_2_1_model

router = APIRouter()

current_process = None


# Models
class LoraItem(BaseModel):
    path: str  # Full path to LoRA file
    weight: float = 1.0  # LoRA strength (same as strength)


class ImageSlot(BaseModel):
    """Phase 5: A single image with its semantic role in the generation.

    Used to build N-slot image arrays for I2I workflows.
    Roles:
        - 'init'    : the main input image (denoising source)
        - 'ref'     : style/content reference (--ref-image, repeatable)
        - 'face'    : face source for PhotoMaker (--pmid-dir)
        - 'mask'    : inpainting mask (--mask-image)
        - 'control' : ControlNet guide (--control-image)
        - 'blend'   : PIL pixel blend (legacy mix_ratio)
    """
    path: str
    role: str = "ref"


class GenRequest(BaseModel):
    mode: str = "t2i"
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    randomize: bool
    # === Phase 5: N-Slot Multi-Image Array ===
    images: Optional[List[ImageSlot]] = None
    # === Phase 1B: 2-Slot Backward-Compat Fields ===
    init_image: Optional[str] = None
    init_image_2: Optional[str] = None
    loras: List[LoraItem] = []
    top_k: Optional[float] = 1.0
    temperature: Optional[float] = 0.6
    strength: Optional[float] = 0.75
    mix_ratio: Optional[float] = 0.5
    noise_level: Optional[int] = 20
    scale: Optional[float] = 4.0
    face_enhance: Optional[bool] = False
    model_path: Optional[str] = None
    clip_skip: Optional[int] = 1
    scheduler: Optional[str] = "euler"
    llm_path: Optional[str] = None
    text_encoder: Optional[str] = None
    vae_path: Optional[str] = None
    scheduler_type: Optional[str] = "auto"
    save_log: Optional[bool] = False
    vae_tiling: Optional[bool] = True
    analyze_backend: Optional[str] = None  # "cpp" or "llama" for analyze modes
    i2t_model_path: Optional[str] = None  # Path to I2T model for analyze modes
    image_analysis: Optional[dict] = None  # IA export data to inject into generated JSON
    # === Multi-Image Roles (Phase 1B) ===
    # Each slot has a role that determines how the image is used by the backend.
    # Possible values for slot1_role / slot2_role:
    #   "init"   - the main input image (default for slot 1)
    #   "ref"    - reference image (Flux Kontext / Edit models) -> --ref-image
    #   "face"   - face source for face swap (PhotoMaker) -> --pmid-dir
    #   "mask"   - inpainting mask (white = modify, black = preserve) -> --mask-image
    #   "control" - ControlNet structural guidance -> --control-image
    #   "blend"  - PIL pixel blend (legacy)
    #   "none"   - slot disabled
    slot1_role: Optional[str] = "init"
    slot2_role: Optional[str] = "none"
    face_image: Optional[str] = None  # Alias for slot2 when role=face
    mask_image: Optional[str] = None  # Alias for slot2 when role=mask
    control_image: Optional[str] = None  # Alias for slot2 when role=control
    control_strength: Optional[float] = 0.8
    ref_image: Optional[str] = None  # Alias for slot2 when role=ref
    no_auto_resize_ref: Optional[bool] = False
    pm_style_strength: Optional[float] = 20.0  # PhotoMaker style strength
    use_slg: Optional[bool] = False  # Skip Layer Guidance for faster CFG > 1 (DiT models)
    use_apg: Optional[bool] = False  # Adaptive Projected Guidance (alternative to standard CFG)
    pmid_embed_path: Optional[str] = None  # Optional pre-computed PM ID embed
    max_img_size: Optional[int] = 1024  # Max image dimension for VRAM saving (0 = disabled)


@router.post("/")
async def generate(req: GenRequest):
    """Generate image.

    VRAM coordination (the whole point of this block):
      The T2I / I2I / restore / upscale worker subprocesses load their
      own C++ models into VRAM (the Z-Image model alone needs ~12 GB,
      plus VAE/LLM caches on top). If the IA llama-server (port 8001)
      or the tagging llama-server (port 8002) is already loaded with
      a model, cudaMalloc fails with OOM.

      To avoid this we use the VramManager.prepare_for_external_vram()
      helper, which:
        1. Records which servers (IA / tag) were running and with which
           model, so we can restart them when generation finishes.
        2. Stops the IA server (port 8001) and the tag server (port 8002)
           - they release their VRAM.
        3. Tracks that an external VRAM user (this generation) is active
           so other code paths (i2t_query, auto-tag batch, ...) refuse to
           start a server while we run.

      After generation finishes, the wrapped_generator's `finally` block
      calls vram.restore_external_vram() to put the servers back into the
      state they were in (e.g. resume the tag batch for the index
      generation the user originally requested).
    """
    global current_process

    # Mutual exclusion: never run a diffusion job while an Agent Refine loop
    # is active - the agent run holds the SAME in-process VramManager eviction
    # token, so the two would overwrite each other's state and double-load VRAM.
    # Lazy dynamic import: server.py loads api.agent later and its module-global
    # agent_process is reassigned at runtime, so a plain `from ... import` would
    # capture a stale value.
    try:
        import api.agent as _agent_mod
        _agent_proc = getattr(_agent_mod, "agent_process", None)
        if _agent_proc is not None and _agent_proc.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="Agent Refine is running: stop it from the 🪄 Agent Refine tab first.",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    # Seed
    if req.randomize or req.seed == -1:
        final_seed = random.randint(0, 2**32 - 1)
    else:
        final_seed = req.seed

    print(f"--> [Generate] Mode: {req.mode} | Model: {req.model_path}")

    # 1) Stop the IA server (port 8001) and tag server (port 8002) so the
    #    T2I subprocess has the full VRAM budget.
    from api.llama_server_manager import get_vram_manager
    vram = get_vram_manager()
    server_state = vram.prepare_for_external_vram("t2i_generation")
    print(
        f"--> [Generate] VRAM state: IA_was_running={server_state.get('ia_was_running')}, "
        f"Tag_was_running={server_state.get('tag_was_running')}",
        flush=True,
    )

    # 2) Stop any active tagging SUBPROCESS (if tag model is the legacy
    #    analyze_with_llama/analyze_with_glm path which spawns a fresh
    #    subprocess per image, the server might not be loaded but a
    #    subprocess could still be holding VRAM).
    try:
        from api.tags import tagging_in_progress, tagging_stop_flag, tagging_interrupted
        import workers.tag_worker
        if tagging_in_progress:
            print(f"--> [Generate] Stopping active tagging subprocess to free VRAM...")
            tagging_stop_flag = True
            workers.tag_worker.tagging_stop_flag = True
            tagging_interrupted = True

            # Force kill the active subprocess immediately
            active_proc = workers.tag_worker.active_tagging_process
            if active_proc and active_proc.poll() is None:
                print(f"--> [Generate] Force killing active tagging subprocess PID {active_proc.pid}...")
                active_proc.kill()
                try:
                    active_proc.wait(timeout=10)
                    print(f"--> [Generate] Tagging subprocess killed")
                except:
                    print(f"--> [Generate] Subprocess kill timeout, proceeding anyway")

            # Save a checkpoint for the auto-tag loop to resume from,
            # preserving the existing 'current' index so we don't
            # re-tag images that were already processed before the
            # user kicked off the generation.
            try:
                existing_cp = dict(vram.checkpoint or {})
                existing_cp["status"] = "interrupted_by_generation"
                vram.save_checkpoint(existing_cp)
                print(f"--> [Generate] Saved tag checkpoint: resume from {existing_cp.get('current', 0)}", flush=True)
            except Exception as e:
                print(f"--> [Generate] Could not save tag checkpoint: {e}", flush=True)
    except Exception as e:
        print(f"--> [Generate] Error stopping tagging subprocess: {e}")

    # 3) Wait for VRAM to be released by the GPU driver / llama-server
    #    runtime. The ROCm/HIP runtime keeps tensors mapped for several
    #    seconds after process exit, so a fixed sleep is not enough on
    #    slow kernels - we poll the free VRAM via rocm-smi instead.
    import asyncio
    from api.llama_server_manager import get_free_vram_mb

    # Reduced from 14 GB to 12 GB to avoid unnecessary waiting.
    # Z-Image (12 GB) + VAE fits in 24 GB card with 12 GB free.
    # The worker itself will handle OOM with tiling fallback if needed.
    REQUIRED_FREE_MB = 12 * 1024  # 12 GB
    MAX_WAIT_S = 30  # Reduced from 60s to 30s max wait
    POLL_S = 1.0
    waited = 0.0
    free_mb = get_free_vram_mb()
    print(
        f"--> [Generate] Initial free VRAM: {free_mb} MB (need >= {REQUIRED_FREE_MB} MB)",
        flush=True,
    )
    while free_mb < REQUIRED_FREE_MB and waited < MAX_WAIT_S:
        await asyncio.sleep(POLL_S)
        waited += POLL_S
        free_mb = get_free_vram_mb()
        if int(waited) % 5 == 0:
            print(
                f"--> [Generate] Waiting for VRAM... {free_mb} MB free after {waited:.0f}s",
                flush=True,
            )
    if free_mb < REQUIRED_FREE_MB:
        print(
            f"--> [Generate] WARNING: only {free_mb} MB free after {MAX_WAIT_S}s, "
            "T2I may OOM or segfault", flush=True,
        )
    else:
        print(
            f"--> [Generate] VRAM OK: {free_mb} MB free after {waited:.1f}s wait",
            flush=True,
        )

    # LoRA config - use path and weight from frontend
    lora_data = [{"path": l.path, "strength": l.weight} for l in req.loras]
    config_path = create_lora_config(lora_data)

    # Build command
    cmd = build_command(req, final_seed, config_path)

    if not cmd:
        # Restore servers before raising - we already freed their VRAM
        # and we don't want the user to have to manually restart them.
        try:
            vram.restore_external_vram(server_state)
        except Exception as e:
            print(f"--> [Generate] Error restoring servers after invalid mode: {e}")
        raise HTTPException(status_code=400, detail="Invalid mode")

    # Wrap the event generator to:
    #   - signal the frontend to auto-resume tagging when T2I finishes
    #     (if the tag batch was running before generation)
    #   - restore the IA and tag servers that were running before gen
    #   - release the legacy vram_arbiter token
    async def wrapped_generator():
        try:
            async for chunk in event_generator(cmd, req):
                yield chunk
            # If the tag batch was running before generation, signal the
            # frontend to auto-resume. The flag is consumed by the
            # frontend on the next poll of /api/tags/vram/auto-resume-pending.
            if server_state.get("tag_was_running"):
                try:
                    tag_cp = vram.checkpoint or {}
                    resume_from = int(tag_cp.get("current", 0))
                    vram.mark_auto_resume_pending(resume_from)
                    yield f"data: LOG|Tagging will automatically resume from index {resume_from}\n\n"
                    # Custom event the frontend listens for. TAGGING_RESUME|<from>
                    yield f"data: TAGGING_RESUME|from={resume_from}\n\n"
                    print(f"--> [Generate] Signaled frontend to auto-resume tagging from {resume_from}", flush=True)
                except Exception as e:
                    print(f"--> [Generate] Could not signal auto-resume: {e}", flush=True)
        finally:
            # Restore IA and tag servers that we stopped before generation
            try:
                vram.restore_external_vram(server_state)
                print(f"--> [Generate] IA/tag servers restored", flush=True)
            except Exception as e:
                print(f"--> [Generate] Error restoring servers: {e}")
            try:
                from api.vram_arbiter import get_arbiter
                arbiter = get_arbiter()
                arbiter.release_vram("ia")
                print(f"--> [Generate] VRAM token released", flush=True)
            except Exception as e:
                print(f"--> [Generate] Error releasing VRAM: {e}")

    return StreamingResponse(wrapped_generator(), media_type="text/event-stream")


def normalize_images(req: GenRequest) -> List[ImageSlot]:
    """Phase 5: Merge N-slot `images` array and 2-slot legacy fields into a single list.

    Priority:
        1. If `req.images` is provided and non-empty, use it (after filtering)
        2. Otherwise, build from legacy fields (init_image, ref_image, face_image, mask_image,
           control_image, init_image_2, slot1_role, slot2_role)

    Returns:
        List of ImageSlot with role != 'none' and path != ''.
    """
    out: List[ImageSlot] = []

    # 1. Try array first
    if req.images:
        for img in req.images:
            if not img or not img.path:
                continue
            role = (img.role or "ref").lower()
            if role == "none":
                continue
            out.append(ImageSlot(path=img.path, role=role))

    # 2. Fallback to legacy 2-slot fields
    if not out:
        # Slot 1: init_image with slot1_role (default 'init')
        if req.init_image:
            role1 = (req.slot1_role or "init").lower()
            if role1 != "none":
                out.append(ImageSlot(path=req.init_image, role=role1))

        # Slot 2: try ref/face/mask/control/init_image_2 in priority order
        slot2_path = (
            req.ref_image or req.face_image or req.mask_image
            or req.control_image or req.init_image_2
        )
        if slot2_path:
            role2 = (req.slot2_role or "none").lower()
            if role2 == "none":
                # Infer from which field is set
                if req.ref_image:       role2 = "ref"
                elif req.face_image:    role2 = "face"
                elif req.mask_image:    role2 = "mask"
                elif req.control_image: role2 = "control"
                elif req.init_image_2:  role2 = "blend"
            if role2 != "none":
                out.append(ImageSlot(path=slot2_path, role=role2))

    return out


def build_command(req: GenRequest, seed: int, lora_config: str) -> Optional[List[str]]:
    """Builds command for the appropriate worker"""
    python = sys.executable
    
    # Sanitize all prompt strings to ensure valid UTF-8 for subprocess/worker transmission
    # This prevents 'invalid continuation byte' errors in C++ layers
    safe_prompt = sanitize_prompt(req.prompt)
    safe_neg_prompt = sanitize_prompt(req.negative_prompt) if req.negative_prompt else ""

    # === Qwen-Image-Edit routing guard ===
    # The qwen-image-edit GGUF is a *diffusion-only* edit model (bare tensor
    # names like img_in.*/transformer_blocks.*, no text encoder embedded).
    # sd.cpp cannot open it as a full checkpoint (-m), so the generic
    # t2i_cpp/i2i_cpp workers die with "get sd version from file failed".
    # These models must run through workers/i2i_edit.py which uses the proper
    # split invocation: --diffusion-model + --llm (Qwen2.5-VL) + -r reference.
    qwen_edit_selected = False
    if req.model_path:
        qwen_edit_selected = is_qwen_image_edit_model(req.model_path)
    if qwen_edit_selected:
        if req.mode == "i2i_cpp":
            # An I2I request against an edit model IS a semantic edit:
            # route it to the dedicated worker so selecting the model in the
            # I2I tab "just works" (instead of a cryptic sd.cpp crash).
            print("--> [Generate] Qwen-Image-Edit model selected -> routing i2i_cpp to i2i_edit", flush=True)
            req.mode = "i2i_edit"
        elif req.mode == "t2i_cpp":
            raise HTTPException(status_code=400, detail=(
                f"'{os.path.basename(req.model_path)}' is a Qwen-Image-Edit model: it edits an "
                "existing image via prompt + reference, so it requires the I2I tab with an input "
                "image (not Text-to-Image)."
            ))

    # Qwen-Image-2.1 supports editing as well: in the I2I tab route it to the
    # dedicated edit worker, because the generic i2i_cpp worker cannot load a
    # --diffusion-model + --llm model.
    if (not qwen_edit_selected) and req.model_path and req.mode == "i2i_cpp" \
            and is_qwen_image_2_1_model(req.model_path):
        print("--> [Generate] Qwen-Image-2.1 model selected in I2I -> routing to i2i_edit", flush=True)
        req.mode = "i2i_edit"

    if req.mode == "t2i_cpp":
        cmd = [
            python, "-u", "workers/t2i_cpp.py",
            "--prompt", safe_prompt,
        ]
        if safe_neg_prompt:
            cmd.extend(["--negative_prompt", safe_neg_prompt])
        cmd.extend([
            "--width", str(req.width),
            "--height", str(req.height),
            "--steps", str(req.steps),
            "--guidance", str(req.guidance),
            "--seed", str(seed),
            "--lora_config", lora_config,
            "--model_path", req.model_path or "",
            "--scheduler", req.scheduler or "euler",
            "--scheduler_type", req.scheduler_type or "auto",
            "--vae_tiling", str(req.vae_tiling).lower()
        ])
        if req.use_slg:
            cmd.extend(["--use_slg", "true"])
        if req.use_apg:
            cmd.extend(["--use_apg", "true"])
        if req.vae_path:
            cmd.extend(["--vae_path", req.vae_path])
        if req.text_encoder:
            cmd.extend(["--llm_path", req.text_encoder])
        elif req.llm_path:
            cmd.extend(["--llm_path", req.llm_path])
        return cmd
    
    elif req.mode == "i2i_cpp":
        # Phase 5: N-Slot Multi-Image support
        # Normalize images array (handles both new array and legacy 2-slot fields)
        normalized_images = normalize_images(req)
        # Find init image (first slot with role='init', or first slot)
        init_img_slot = next((im for im in normalized_images if im.role == "init"), None)
        if not init_img_slot and normalized_images:
            # No explicit init - use first slot as init
            init_img_slot = normalized_images[0]
        if not init_img_slot:
            return None
        # Remove init from multi-image list (it's used as --init_img separately)
        multi_images = [im for im in normalized_images if im.path != init_img_slot.path or im.role != "init"]
        # If no init role but first slot was used as init, keep role info for that slot
        if not any(im.role == "init" for im in normalized_images) and normalized_images:
            # Mark the first slot as init in the multi-image list (it'll be skipped below)
            pass
        cmd = [
            python, "-u", "workers/i2i_cpp.py",
            "--prompt", safe_prompt,
        ]
        if safe_neg_prompt:
            cmd.extend(["--negative_prompt", safe_neg_prompt])
        cmd.extend([
            "--width", str(req.width),
            "--height", str(req.height),
            "--steps", str(req.steps),
            "--guidance", str(req.guidance),
            "--seed", str(seed),
            "--init_img", init_img_slot.path,
            "--strength", str(req.strength or 0.75),
            "--lora_config", lora_config,
            "--model_path", req.model_path or "",
            "--scheduler", req.scheduler or "euler",
            "--scheduler_type", req.scheduler_type or "auto",
            "--vae_tiling", str(req.vae_tiling).lower()
        ])
        # === Phase 5: N-Slot Multi-Image Role-Based Argument Passing ===
        # Iterate over all images (excluding init) and emit the correct C++ flag
        # NOTE: --ref-image is only supported for Flux Kontext-family models.
        # Z-Image (and other non-Kontext architectures) segfault in sd.cpp;
        # the worker skips them, so warn the user here for better UX.
        has_ref = False
        model_is_zimage = is_zimage_model(req.model_path or "")
        ref_supported = model_is_zimage is False
        for img in multi_images:
            role = (img.role or "ref").lower()
            if role == "ref":
                if ref_supported:
                    cmd.extend(["--ref-image", img.path])
                    has_ref = True
                else:
                    print(f"--> [API] WARNING: Model Z-Image does not support reference images. "
                          f"Skipping ref image: {os.path.basename(img.path)}. Use the 'blend' slot instead.", flush=True)
            elif role == "face":
                # Face swap: best with dedicated worker (i2i_faceswap mode),
                # but pass --ref_image as fallback for in-place I2I
                if ref_supported:
                    cmd.extend(["--ref-image", img.path])
                    cmd.extend(["--image_role", "face"])
                    has_ref = True
                    print(f"--> [API] Note: For best face swap, use i2i_faceswap mode (workers/i2i_faceswap.py)", flush=True)
                else:
                    print(f"--> [API] WARNING: Model Z-Image does not support face-swap reference images. "
                          f"Skipping face image: {os.path.basename(img.path)}. Use the 'blend' slot instead.", flush=True)
            elif role == "mask":
                if ref_supported:
                    cmd.extend(["--mask", img.path])
                else:
                    print(f"--> [API] WARNING: Model Z-Image does not support inpainting masks. "
                          f"Skipping mask image: {os.path.basename(img.path)}. The result will be a normal img2img edit.", flush=True)
            elif role == "control":
                if ref_supported:
                    cmd.extend(["--control-image", img.path])
                    cmd.extend(["--control-strength", str(req.control_strength or 0.8)])
                else:
                    print(f"--> [API] WARNING: Model Z-Image does not support ControlNet. "
                          f"Skipping control image: {os.path.basename(img.path)}.", flush=True)
            elif role == "blend":
                # Legacy PIL blend - pass via init_img_2 + mix_ratio
                # Note: only the first blend slot is used (worker limitation)
                if "--init_img_2" not in cmd:
                    cmd.extend(["--init_img_2", img.path])
                    cmd.extend(["--mix_ratio", str(req.mix_ratio or 0.5)])
        # Auto-resize toggle for ref images
        if req.no_auto_resize_ref:
            cmd.extend(["--no_auto_resize_ref"])
        # Log
        if len(multi_images) > 0:
            print(f"--> [API] i2i_cpp: {len(multi_images)} extra images, roles={[im.role for im in multi_images]}", flush=True)
        _ = has_ref  # silence unused
        # Pass max image size for VRAM-saving resize (same as I2T slider)
        if req.max_img_size and req.max_img_size > 0:
            cmd.extend(["--max_img_size", str(req.max_img_size)])
        if req.use_slg:
            cmd.extend(["--use_slg", "true"])
        if req.use_apg:
            cmd.extend(["--use_apg", "true"])
        if req.vae_path:
            cmd.extend(["--vae_path", req.vae_path])
        if req.text_encoder:
            cmd.extend(["--llm_path", req.text_encoder])
        elif req.llm_path:
            cmd.extend(["--llm_path", req.llm_path])
        return cmd

    elif req.mode == "i2i_faceswap":
        # Dedicated face swap mode - uses PhotoMaker via --pmid-dir
        if not req.init_image:
            return None
        face_img = req.face_image or req.init_image_2
        if not face_img:
            return None
        # PhotoMaker requires SDXL/SD1.5 — filter out incompatible models (Z-Image, Qwen, Flux, etc.)
        model_path = req.model_path or ""
        if model_path:
            low = model_path.lower()
            incompatible = ["z-image", "zimage", "qwen", "flux", "sd3", "sd-3", "hunyuan", "auraflow", "kolors", "pixart"]
            if any(p in low for p in incompatible):
                print(f"--> [API] Model {model_path} is incompatible with PhotoMaker (SDXL required). Ignoring and using auto-detect.", flush=True)
                model_path = ""
        # Early check: before spawning the worker (which may download PhotoMaker =
        # ~300MB), verify that an SDXL/SD1.5 diffusion model actually exists.
        # If the user only has Z-Image/Qwen/Flux, fail fast with a helpful message
        # instead of downloading PhotoMaker and then erroring out.
        if not model_path:
            try:
                from workers.i2i_faceswap import find_photomaker_diffusion_model
                sdxl_model = find_photomaker_diffusion_model()
            except Exception:
                sdxl_model = None
            if not sdxl_model:
                from shared_utils import CHECKPOINTS_DIR
                import glob
                avail = [
                    os.path.basename(p) for p in glob.glob(os.path.join(CHECKPOINTS_DIR, "**", "*.*"), recursive=True)
                    if p.lower().endswith((".gguf", ".safetensors"))
                    and "photomaker" not in p.lower()
                ]
                print("--> [API] ERROR: I2I FaceSwap (PhotoMaker) requires an SDXL/SD1.5 diffusion model.", flush=True)
                print(f"--> [API] No SDXL/SD1.5 model found on disk to run PhotoMaker.", flush=True)
                if avail:
                    print(f"--> [API] Your available models are: {', '.join(sorted(set(avail))[:6])}", flush=True)
                print("--> [API] Download an SDXL checkpoint (e.g. RealVisXL V5 GGUF) from Model Management first,", flush=True)
                print("--> [API] then retry. With Z-Image models, face-swap cannot work (SDXL required).", flush=True)
                # Emit a clear SSE error so the UI shows it (return a sentinel command list
                # that the caller will treat as an immediate error-only stream).
                return ["python", "-u", "workers/i2i_faceswap.py", "--error_only"]
        cmd = [
            python, "-u", "workers/i2i_faceswap.py",
            "--prompt", safe_prompt,
            "--init_img", req.init_image,
            "--face_img", face_img,
            "--width", str(req.width),
            "--height", str(req.height),
            "--steps", str(req.steps),
            "--guidance", str(req.guidance),
            "--seed", str(seed),
            "--strength", str(req.strength or 0.6),
            "--lora_config", lora_config,
            "--model_path", model_path,
            "--scheduler", req.scheduler or "euler",
            "--vae_tiling", str(req.vae_tiling).lower(),
            "--pm_style_strength", str(req.pm_style_strength or 20.0),
        ]
        if safe_neg_prompt:
            cmd.extend(["--negative_prompt", safe_neg_prompt])
        if req.vae_path:
            cmd.extend(["--vae_path", req.vae_path])
        if req.text_encoder:
            cmd.extend(["--llm_path", req.text_encoder])
        elif req.llm_path:
            cmd.extend(["--llm_path", req.llm_path])
        if req.pmid_embed_path:
            cmd.extend(["--pmid_embed_path", req.pmid_embed_path])
        return cmd
    
    elif req.mode == "i2i_edit":
        # Qwen-Image-Edit mode - uses sd-cli with special flags
        if not req.init_image:
            return None
        # Resolve relative paths: the frontend normalizes away the
        # /outputs|/uploads URL prefix, so the backend must re-resolve
        # against OUTPUT_DIR / cwd before falling back to base64 decoding.
        init_path = req.init_image
        if not os.path.isabs(init_path):
            for base in (OUTPUT_DIR, os.getcwd()):
                cand = os.path.join(base, init_path)
                if os.path.isfile(cand):
                    init_path = cand
                    break
        # If it is still not a file on disk, try decoding a base64 data URL
        if not os.path.isfile(init_path):
            try:
                import base64, uuid
                # Strip data URL prefix if present
                header, data = init_path.split(',', 1)
                decoded = base64.b64decode(data)
                tmp_name = f"i2i_edit_input_{uuid.uuid4().hex[:8]}.png"
                init_path = os.path.join(OUTPUT_DIR, tmp_name)
                with open(init_path, "wb") as f:
                    f.write(decoded)
            except Exception as e:
                print(f"ERROR: Failed to decode init_image: {e}", flush=True)
                return None
        cmd = [
            python, "-u", "workers/i2i_edit.py",
            "--prompt", safe_prompt,
            "--init_img", init_path,
            "--seed", str(seed),
            "--steps", str(req.steps),
            "--guidance", str(req.guidance),
            "--scheduler", req.scheduler or "euler",
            "--vae_tiling", str(req.vae_tiling).lower(),
        ]
        if safe_neg_prompt:
            cmd.extend(["--negative_prompt", safe_neg_prompt])
        if req.width:
            cmd.extend(["--width", str(req.width)])
        if req.height:
            cmd.extend(["--height", str(req.height)])
        # Pass model selection parameters (like i2i_cpp). NOTE: the VAE is
        # intentionally NOT forwarded here - the i2i_edit worker always
        # auto-detects the Qwen VAE (qwen_image_vae.safetensors). Forwarding
        # the UI's vae select (e.g. ae.safetensors for Z-Image/Flux) would
        # feed the wrong VAE to the Qwen pipeline.
        if req.model_path:
            cmd.extend(["--model_path", req.model_path])
        if req.clip_skip:
            cmd.extend(["--clip_skip", str(req.clip_skip)])
        return cmd
    
    
    elif req.mode == "analyze_cpp":
        if not req.init_image:
            return None
        cmd = [
            python, "-u", "workers/analyze_cpp.py",
            "--image_path", req.init_image,
            "--prompt", safe_prompt,
            "--top_k", str(req.top_k or 1),
            "--temperature", str(req.temperature or 0.6),
            "--model_path", req.model_path or ""
        ]
        return cmd

    elif req.mode == "analyze_llama":
        # Use llama-cli for multimodal image analysis
        if not req.init_image:
            return None
        cmd = [
            python, "-u", "workers/i2t_llama.py",
            req.init_image,
            "--model-dir", req.i2t_model_path or "",
            "--prompt", safe_prompt or "Describe this image in detail.",
            "--temperature", str(req.temperature or 0.7),
            "--max-tokens", "1024",
            # Default to disabled reasoning so the model emits the
            # direct answer. Per-model params.json can re-enable it.
            "--enable-thinking", "false",
            # Grammar file disabled - interferes with media marker parsing
            # "--grammar-file", "/app/workers/i2t_response.gbnf"
        ]
        return cmd
    
    return None


async def event_generator(cmd: List[str], req: GenRequest):
    """Async generator for SSE — non-blocking, does not freeze the event loop.
    
    Uses asyncio.create_subprocess_exec so that other API endpoints
    (history, tags, etc.) can serve requests concurrently during generation.
    """
    global current_process
    
    try:
        # Create async subprocess
        current_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy(),
        )
        
        start_t = time.time()
        buffer = b""
        text_capture_mode = False
        captured_text = []
        last_image_path = None  # Track the last SUCCESS_OUTPUT path for IA injection
        
        # Read loop — non-blocking because readline is async
        while True:
            if current_process is None or current_process.stdout is None:
                break
            
            # Read one line asynchronously (yields control to event loop)
            try:
                line_bytes = await asyncio.wait_for(
                    current_process.stdout.readline(),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                # No data yet — check if process is still alive
                if current_process.returncode is not None:
                    if buffer:
                        text = buffer.decode('utf-8', errors='replace').strip()
                        if text:
                            yield f"data: LOG|{text.strip()}\n\n"
                    break
                
                # Ping every 2s instead of 0.5s to reduce SSE overhead
                yield "data: PING\n\n"
                continue
            
            if not line_bytes:
                # EOF — process closed stdout
                if buffer:
                    text = buffer.decode('utf-8', errors='replace').strip()
                    if text:
                        yield f"data: LOG|{text.strip()}\n\n"
                break
            
            line = line_bytes
            buffer += line
            
            while True:
                idx = buffer.find(b'\n')
                if idx == -1:
                    break
                
                line = buffer[:idx+1]
                buffer = buffer[idx+1:]
                
                try:
                    text = line.decode('utf-8', errors='replace').strip()
                    if not text:
                        continue
                    
                    # Check for text output markers
                    if "TEXT_OUTPUT_START" in text:
                        text_capture_mode = True
                        captured_text = []
                        continue
                    elif "TEXT_OUTPUT_END" in text:
                        text_capture_mode = False
                        if captured_text:
                            full_text = "\\n".join(captured_text).strip()
                            yield f"data: TXT|{full_text}\n\n"
                        continue
                    
                    # Also capture lines that look like JSON (for grammar output)
                    if text_capture_mode:
                        captured_text.append(text)
                        if len(captured_text) == 1 and captured_text[0].strip().startswith('{') and 'thinking' in captured_text[0]:
                            print(f"[DEBUG] Sending JSON directly: {captured_text[0][:100]}...", flush=True)
                            yield f"data: TXT|{captured_text[0].strip()}\n\n"
                            text_capture_mode = False
                            captured_text = []
                        continue
                    
                    if text.strip().startswith('{') and 'thinking' in text:
                        print(f"[DEBUG] Sending JSON from direct match: {text[:100]}...", flush=True)
                        yield f"data: TXT|{text.strip()}\n\n"
                    elif "SUCCESS_OUTPUT:" in text:
                        path = text.split("SUCCESS_OUTPUT:")[1].strip()
                        if path.startswith(OUTPUT_DIR):
                            url = path.replace(OUTPUT_DIR, "/outputs", 1)
                        else:
                            url = path
                        last_image_path = path
                        yield f"data: IMG|{url}\n\n"
                    elif "ERROR_OUTPUT:" in text:
                        error_str = text.split("ERROR_OUTPUT:", 1)[1].strip()
                        try:
                            error_data = json.loads(error_str)
                            yield f"data: ERROR|{json.dumps(error_data)}\n\n"
                        except json.JSONDecodeError:
                            yield f"data: ERROR|{json.dumps({'error': error_str, 'code': 'UNKNOWN_ERROR'})}\n\n"
                    elif "data: LOG|" in text:
                        yield f"data: {text}\n\n"
                    elif text:
                        print(f"[DEBUG] LOG: {text[:80]}...", flush=True)
                        elapsed = time.time() - start_t
                        yield f"data: LOG|{elapsed:08.3f} | {text}\n\n"
                
                except Exception:
                    pass
        
        elapsed = time.time() - start_t
        if current_process and current_process.returncode == 0:
            if req.image_analysis and last_image_path:
                try:
                    json_path = os.path.splitext(last_image_path)[0] + ".json"
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        json_data["image_analysis"] = req.image_analysis
                        with open(json_path, 'w') as f:
                            json.dump(json_data, f, indent=4, ensure_ascii=False)
                        print(f"--> [Generate] IA data injected into: {json_path}", flush=True)
                    else:
                        print(f"--> [Generate] JSON not found for IA injection: {json_path}", flush=True)
                except Exception as e:
                    print(f"--> [Generate] Warning: Could not inject IA data: {e}", flush=True)
            
            yield f"data: DONE|Finished in {elapsed:.1f}s\n\n"
        else:
            buffer_text = buffer.decode('utf-8', errors='replace') if buffer else ""
            if "ERROR_OUTPUT:" not in buffer_text:
                print(f"[DEBUG] Process failed with code {current_process.returncode if current_process else -1}", flush=True)
                print(f"[DEBUG] Buffer content: {buffer_text[:500]}", flush=True)
                yield f"data: ERR|Process exited with code {current_process.returncode if current_process else -1}\n\n"
    
    except Exception as e:
        yield f"data: ERR|{str(e)}\n\n"
    
    finally:
        current_process = None


@router.post("/stop")
async def stop():
    """Stop generation"""
    global current_process
    import signal
    if current_process:
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            print(f"--> [Stopped] Killed process group {current_process.pid}", flush=True)
            current_process = None
            return {"status": "stopped"}
        except Exception as e:
            print(f"--> [Stop Error] {e}", flush=True)
            try:
                current_process.terminate()
            except:
                pass
            current_process = None
            return {"status": "error", "message": str(e)}
    return {"status": "no_process"}
