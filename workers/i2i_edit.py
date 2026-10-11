"""
Workers I2I Edit - Image Editing with Qwen-Image-Edit using stable-diffusion.cpp
"""
import os
import sys
import argparse
import subprocess
import time
import json
import shutil

DEFAULT_SD_BINARY = "/usr/local/bin/sd"

sys.path.insert(0, os.getcwd())

import shared_utils
from workers.base import save_output_log, is_qwen_image_edit_model, is_qwen_image_2_1_model, TilingConfig


def find_qwen_edit_model():
    """Find the Qwen-Image-Edit model in the checkpoints directory."""
    from core.config import get_setting
    checkpoints_dir = shared_utils.CHECKPOINTS_DIR
    
    # First, check if user has explicitly set a model path
    model_path = get_setting("qwen_image_edit_model_path")
    if model_path:
        if os.path.exists(model_path):
            return model_path
        else:
            # Try to resolve relative to checkpoints_dir (in case the saved path is from a different environment)
            filename = os.path.basename(model_path)
            full_path = os.path.join(checkpoints_dir, filename)
            if os.path.exists(full_path):
                return full_path
    
    # Fallback: search for any qwen-image-edit GGUF model
    possible_names = [
        "qwen-image-edit-2511-Q4_K_M.gguf",
        "qwen-image-edit-2511-Q8_0.gguf",
        "qwen-image-edit-2511-Q5_K_M.gguf",
        "qwen-image-edit-2511.gguf"
    ]
    
    for name in possible_names:
        path = os.path.join(checkpoints_dir, name)
        if os.path.exists(path):
            return path
    
    # Also check subdirectories as a fallback (any Qwen-Image-Edit-family
    # diffusion model: qwen-image-edit-2511, Qwen-Rapid-*, ... detected via
    # the shared is_qwen_image_edit_model signature).
    for root, dirs, files in os.walk(checkpoints_dir):
        for f in files:
            if f.endswith(".gguf"):
                cand = os.path.join(root, f)
                if is_qwen_image_edit_model(cand):
                    return cand
    
    return None


def find_qwen_vae():
    """Find the Qwen VAE in the VAE directory."""
    vae_dir = shared_utils.VAE_DIR
    
    possible_names = [
        "qwen_image_vae.safetensors",
        "qwen_image_vae.sft",
        "vae_qwen_image.safetensors"
    ]
    
    for name in possible_names:
        path = os.path.join(vae_dir, name)
        if os.path.exists(path):
            return path
    
    return None


def find_qwen_llm():
    """Find the Qwen LLM (text encoder + vision encoder) in the text_encoders directory."""
    enc_dir = shared_utils.TEXT_ENCODERS_DIR
    
    possible_names = [
        "qwen_2.5_vl_7b.safetensors",
        "Qwen2.5-VL-7B-Instruct.safetensors",
        "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        "Qwen2.5-VL-7B-Instruct-Q4_K_XL.gguf"
    ]
    
    for name in possible_names:
        path = os.path.join(enc_dir, name)
        if os.path.exists(path):
            return path
    
    return None


def find_qwen_llm_vision():
    """Find the Qwen vision encoder projection."""
    enc_dir = shared_utils.TEXT_ENCODERS_DIR
    
    possible_names = [
        "Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf",
        "Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf"
    ]
    
    for name in possible_names:
        path = os.path.join(enc_dir, name)
        if os.path.exists(path):
            return path
    
    return None


def run_i2i_edit(prompt: str, init_img: str, seed: int = -1, steps: int = 20,
                 guidance: float = 2.5, scheduler: str = "euler", width: int = 1024,
                 height: int = 1024, negative_prompt: str = "", model_path: str = None,
                 vae_path: str = None, clip_skip: int = 1, scheduler_type: str = "discrete",
                 vae_tiling: bool = True):
    """Run Qwen-Image-Edit to modify an image based on a text prompt."""
    
    start_time = time.time()
    print(f"--> [I2I Edit] Starting... PID: {os.getpid()}")
    
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
    out_filename = f"i2i_edit_{int(time.time())}.png"
    save_path = os.path.join(shared_utils.OUTPUT_DIR, out_filename)
    os.makedirs(shared_utils.OUTPUT_DIR, exist_ok=True)
    
    # Determine model path (use provided or auto-detect)
    final_model = model_path
    if not final_model or not os.path.exists(final_model):
        final_model = find_qwen_edit_model()
        if not final_model:
            print(f"ERROR: Qwen-Image-Edit model not found. Please run Setup to download it.")
            sys.exit(1)
    
    print(f"--> [I2I Edit] Using model: {final_model}")

    # Qwen-Image-2.1 is a different family: different VAE, Qwen3-VL-8B encoder,
    # auto flow schedule and no zero-cond-t reference hack.
    is_qwen21 = is_qwen_image_2_1_model(final_model)

    # Find VAE
    if is_qwen21:
        from core.model_config import get_component
        vae_used = vae_path if vae_path and os.path.exists(vae_path) else get_component(final_model, "vae")
        if not vae_used:
            cand = os.path.join(shared_utils.VAE_DIR, "qwen_image_2.1_vae_bf16.safetensors")
            vae_used = cand if os.path.exists(cand) else None
        if vae_used:
            print(f"--> [I2I Edit] Using Qwen-2.1 VAE: {vae_used}")
        else:
            print("ERROR: Qwen-Image-2.1 VAE not found (expected vae/qwen_image_2.1_vae_bf16.safetensors).")
            sys.exit(1)
    else:
        vae_used = vae_path if vae_path and os.path.exists(vae_path) else find_qwen_vae()
        if vae_used:
            print(f"--> [I2I Edit] Using VAE: {vae_used}")
        else:
            print(f"ERROR: VAE not found. Please ensure qwen_image_vae.safetensors is in checkpoints directory or set qwen_image_vae_path in settings.")
            sys.exit(1)

    # Find LLM (text encoder + vision)
    if is_qwen21:
        from core.model_config import get_component
        llm_path = get_component(final_model, "llm")
        if not llm_path:
            enc_dir = shared_utils.TEXT_ENCODERS_DIR
            for f in (sorted(os.listdir(enc_dir)) if os.path.isdir(enc_dir) else []):
                fl = f.lower()
                if ("qwen3" in fl and "vl" in fl and "8b" in fl and "mmproj" not in fl
                        and f.endswith((".gguf", ".safetensors"))):
                    llm_path = os.path.join(enc_dir, f)
                    break
        llm_vision_path = get_component(final_model, "llm_vision")
        if not llm_vision_path:
            enc_dir = shared_utils.TEXT_ENCODERS_DIR
            for f in (sorted(os.listdir(enc_dir)) if os.path.isdir(enc_dir) else []):
                if "mmproj" in f.lower() and "qwen3" in f.lower():
                    llm_vision_path = os.path.join(enc_dir, f)
                    break
    else:
        llm_path = find_qwen_llm()
        llm_vision_path = find_qwen_llm_vision()

    if llm_path:
        print(f"--> [I2I Edit] Using LLM: {llm_path}")
    if llm_vision_path and llm_path and llm_path.endswith(".gguf"):
        print(f"--> [I2I Edit] Using Vision LLM: {llm_vision_path}")
    
    # Validate input image
    if not os.path.exists(init_img):
        print(f"ERROR: Input image not found: {init_img}")
        sys.exit(1)
    
    # Build command based on the example from the user:
    # ./build/bin/sd-cli \
    #   --diffusion-model models/qwen-image-edit-2511-Q4_K_M.gguf \
    #   --vae models/qwen_image_vae.safetensors \
    #   --llm models/qwen_2.5_vl_7b.safetensors \
    #   --cfg-scale 2.5 \
    #   --sampling-method euler \
    #   -v --offload-to-cpu --diffusion-fa --flow-shift 3 \
    #   -r input_image.png \
    #   -p "change 'flux.cpp' to 'edit.cpp'" \
    #   --model-args qwen_image_zero_cond_t=true \
    #   -o output.png
    
    cmd = [
        sd_binary,
        "-t", str(os.cpu_count() or 4),
        "--diffusion-model", final_model,
        "--cfg-scale", str(guidance),
        "--sampling-method", scheduler,
        "-W", str(width),
        "-H", str(height),
        "-v",  # verbose
        "--offload-to-cpu",
        "--diffusion-fa",
        "-r", init_img,  # input image
        "-p", prompt,    # prompt for editing
        "--color",
        "-o", save_path
    ]
    if is_qwen21:
        # Qwen-Image-2.1 picks the flow schedule automatically from the
        # resolution and needs no zero-cond-t reference conditioning.
        print("--> [I2I Edit] Qwen-Image-2.1 edit mode (auto flow schedule, Qwen3-VL vision encoder)")
    else:
        # Qwen-Image-Edit 2511 conditions the reference latents at t=0.
        # The standalone --qwen-image-zero-cond-t CLI flag was removed in
        # current sd.cpp builds: it is now passed via --model-args.
        cmd.extend(["--flow-shift", "3",
                    "--model-args", "qwen_image_zero_cond_t=true"])

    # The tab's width/height drive the OUTPUT resolution. Keep the VAE decode
    # buffer inside VRAM at high resolutions. The Qwen VAE is heavy (~9.3 GB of
    # compute buffer per megapixel), so use a conservative 2 MP threshold
    # instead of the generic adaptive one (which only tiles above 6 MP on 24 GB).
    tiling_config = TilingConfig(enabled=vae_tiling, threshold_mp=2.0)
    cmd.extend(tiling_config.get_args(width, height))
    
    if vae_used:
        cmd.extend(["--vae", vae_used])
    
    if llm_path:
        cmd.extend(["--llm", llm_path])
        # Force the text encoder on CPU to save VRAM (2.1 already runs with
        # --offload-to-cpu per the official recipe, so skip it there).
        if not is_qwen21:
            cmd.extend(["--clip-on-cpu"])
        
        if llm_vision_path and llm_path.endswith(".gguf"):
            cmd.extend(["--llm_vision", llm_vision_path])
    
    # Optional DiT acceleration parameters for huge resolutions
    # Commented out for Qwen Image Edit since it causes black images (incompatible block caching)
    # cmd.extend(["--cache-mode", "easycache", "--cache-preset", "fast"])
    
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    
    if steps:
        cmd.extend(["--steps", str(steps)])
    
    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])
    
    print(f"--> [I2I Edit] Running: {' '.join(cmd)}", flush=True)
    
    # Execute
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
            env=env
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(f"    {line.rstrip()}", flush=True)
        
        process.wait()
        if process.returncode != 0:
            print(f"ERROR: sd-cli exited with code {process.returncode}")
            sys.exit(1)
        
        # Check output
        if os.path.exists(save_path):
            elapsed = time.time() - start_time
            print(f"--> [I2I Edit] Done in {elapsed:.1f}s -> {save_path}")
            
            # Print for streaming (use SUCCESS_OUTPUT format like other workers)
            print(f"SUCCESS_OUTPUT:{save_path}")
            sys.stdout.flush()
            
            # Prepare metadata for JSON log (like i2i_cpp)
            lora_list = []  # No LoRA support in i2i_edit yet
            
            inputs_data = {
                "prompt": prompt,
                "source_image_1": os.path.basename(init_img) if init_img else "",
                "loras": shared_utils.format_loras_for_json(lora_list)
            }
            
            params_data = {
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "backend": "stable-diffusion.cpp",
                "mode": "i2i_edit",
                "model": os.path.basename(final_model) + (" (Qwen-Image-2.1)" if is_qwen21 else " (Qwen-Image-Edit)"),
                "model_path": final_model or "",
                "negative_prompt": negative_prompt,
                "vae": os.path.basename(vae_used) if vae_used else "",
                "text_encoder": os.path.basename(llm_path) if llm_path else "",
                "clip_skip": clip_skip,
                "sampler": scheduler,
                "scheduler_type": scheduler_type,
                "loras": shared_utils.format_loras_for_json(lora_list)
            }
            
            outputs_data = {
                "type": "image",
                "files": [out_filename],
                "exec_time": round(elapsed, 2)
            }
            
            # Save comprehensive JSON log
            shared_utils.save_generation_log(
                "i2i_edit",
                inputs_data,
                params_data,
                outputs_data,
                image_path_for_filename=save_path,
                model_name="qwen-image-edit"
            )
        else:
            print(f"ERROR: Output file not created: {save_path}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit Worker")
    parser.add_argument("--prompt", type=str, required=True, help="Edit prompt")
    parser.add_argument("--init_img", type=str, required=True, help="Input image path")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps")
    parser.add_argument("--guidance", type=float, default=2.5, help="CFG scale")
    parser.add_argument("--scheduler", type=str, default="euler", help="Scheduler")
    parser.add_argument("--width", type=int, default=1024, help="Width")
    parser.add_argument("--height", type=int, default=1024, help="Height")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--model_path", type=str, default=None, help="Optional model path")
    parser.add_argument("--vae_path", type=str, default=None, help="Optional VAE path")
    parser.add_argument("--clip_skip", type=int, default=1, help="CLIP skip (not used for Qwen-Image-Edit but kept for compatibility)")
    parser.add_argument("--scheduler_type", type=str, default="discrete", help="Scheduler type: discrete, karras, exponential, etc.")
    parser.add_argument("--vae_tiling", type=lambda x: str(x).lower() == "true", default=True,
                        help="Enable adaptive VAE tiling at high resolutions")
    
    args = parser.parse_args()
    
    run_i2i_edit(
        prompt=args.prompt,
        init_img=args.init_img,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        scheduler=args.scheduler,
        width=args.width,
        height=args.height,
        negative_prompt=args.negative_prompt,
        model_path=args.model_path,
        vae_path=args.vae_path,
        clip_skip=args.clip_skip,
        scheduler_type=args.scheduler_type,
        vae_tiling=args.vae_tiling
    )


if __name__ == "__main__":
    main()
