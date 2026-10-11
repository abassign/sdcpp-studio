import sys
import os
import argparse
import subprocess
import time
import json
import select
import re
import shutil

# Add current directory to Python path so we can import shared_utils
sys.path.insert(0, os.getcwd())
import shared_utils

# This script wraps the 'sd' C++ binary from stable-diffusion.cpp for image analysis
# using multimodal models (GLM-4.1V-9B-Thinking, Qwen-Image, Z-Image, Ovis-Image, etc.)

SD_BINARY_PATH = "/usr/local/bin/sd"

def run_sd_cpp_analyze(image_path, prompt, top_k=1.0, temperature=0.6, model_path=None, no_log=False):
    global SD_BINARY_PATH
    start_time = time.time()
    print(f"--> [CPP Analyze Worker] Starting process PID: {os.getpid()}", flush=True)

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}", flush=True)
        sys.exit(1)

    if not os.path.exists(SD_BINARY_PATH):
        # Check local path as fallback if not in /usr/local/bin
        local_sd = os.path.join(os.getcwd(), "sd")
        if os.path.exists(local_sd):
            SD_BINARY_PATH = local_sd
        else:
            print(f"--> [CPP Analyze Worker] ERROR: C++ Binary not found at {SD_BINARY_PATH}. Please rebuild container.", flush=True)
            sys.exit(1)

    # 1. Determine Model Path
    # For I2T/CPP analyze mode, we need a multimodal model (GLM-4.1V, Qwen-Image, etc.)
    # If no model_path provided or if it's a checkpoint (not multimodal), use I2T directory
    if not model_path:
        model_path = shared_utils.I2T_DIR
        print(f"--> [CPP Analyze Worker] No model specified, using default I2T directory: {model_path}", flush=True)
    elif model_path and ("checkpoint" in model_path.lower() or "checkpoints" in model_path.lower()):
        # Checkpoint models are not multimodal - use I2T directory instead
        print(f"--> [CPP Analyze Worker] Checkpoint model specified, but I2T requires multimodal model. Using I2T directory.", flush=True)
        model_path = shared_utils.I2T_DIR

    final_model = model_path
    llm_path = None
    llm_vision_path = None
    vae_path = None

    # Check if model_path is a specific file (GGUF)
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        final_model = model_path
        print(f"--> [CPP Analyze Worker] Using specified GGUF model: {final_model}", flush=True)
    elif os.path.isdir(model_path):
        # For multimodal models, we need to find:
        # - The diffusion model (GGUF or safetensors)
        # - The LLM text encoder (GGUF)
        # - The LLM vision encoder (GGUF or safetensors)
        # - The VAE (safetensors)

        # Recursively search for a suitable diffusion model (GGUF or safetensors) within the model directory.
        # Exclude text encoders, qwen, and llm files.
        final_model = None
        for root, dirs, files in os.walk(model_path):
            for f in sorted(files):
                f_lower = f.lower()
                if (f_lower.endswith('.gguf') or f_lower.endswith('.safetensors')) and \
                   "qwen" not in f_lower and "llm" not in f_lower and "text_encoder" not in f_lower:
                    final_model = os.path.join(root, f)
                    break
            if final_model:
                break

        # Find LLM (text encoder) - usually contains "qwen", "glm", "llm" in name
        llm_candidates = []
        for root, dirs, files in os.walk(model_path):
            for f in files:
                f_lower = f.lower()
                if f.endswith(".gguf") and ("qwen" in f_lower or "glm" in f_lower or "llm" in f_lower or "text_encoder" in f_lower):
                    llm_candidates.append(os.path.join(root, f))
        if llm_candidates:
            llm_path = llm_candidates[0]

        # Find LLM vision encoder - usually contains "vision", "mmproj", "visual"
        vision_candidates = []
        for root, dirs, files in os.walk(model_path):
            for f in files:
                f_lower = f.lower()
                if f.endswith((".gguf", ".safetensors")) and ("vision" in f_lower or "mmproj" in f_lower or "visual" in f_lower):
                    vision_candidates.append(os.path.join(root, f))
        if vision_candidates:
            llm_vision_path = vision_candidates[0]

        # Find VAE - usually "vae" in name or in a vae subdirectory
        vae_candidates = []
        for root, dirs, files in os.walk(model_path):
            for f in files:
                if f.endswith(".safetensors") and "vae" in f.lower():
                    vae_candidates.append(os.path.join(root, f))
        if vae_candidates:
            vae_path = vae_candidates[0]

    print(f"--> [CPP Analyze Worker] Model paths:", flush=True)
    print(f"    Diffusion: {final_model}", flush=True)
    print(f"    LLM: {llm_path}", flush=True)
    print(f"    LLM Vision: {llm_vision_path}", flush=True)
    print(f"    VAE: {vae_path}", flush=True)

    # 2. Build command for stable-diffusion.cpp
    # For image analysis, we use img_gen mode with an init image and a prompt
    # The model will generate text output instead of an image
    cmd = [SD_BINARY_PATH, "-M", "img_gen"]

    # Add model paths
    if final_model:
        cmd.extend(["--diffusion-model", final_model])
    if llm_path:
        cmd.extend(["--llm", llm_path])
    if llm_vision_path:
        cmd.extend(["--llm_vision", llm_vision_path])
    if vae_path:
        cmd.extend(["--vae", vae_path])

    # Input image
    cmd.extend(["-i", image_path])

    # Prompt for analysis
    if not prompt:
        prompt = "Describe this image in detail."
    cmd.extend(["-p", prompt])

    # Generation parameters (for text generation)
    cmd.extend(["--cfg-scale", "1.0"])  # Low guidance for text
    cmd.extend(["--sampling-method", "euler"])
    cmd.extend(["--steps", "10"])
    cmd.extend(["--offload-to-cpu"])  # Save VRAM

    # Output: We want to capture text output, not generate an image
    # The sd binary will output to stdout when in multimodal mode with text generation
    # We'll use a temporary output path
    timestamp = int(time.time())
    output_img = f"/tmp/analyze_output_{timestamp}.png"
    cmd.extend(["-o", output_img])

    print(f"--> [CPP Analyze Worker] Command: {' '.join(cmd)}", flush=True)

    # 3. Execute
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=os.environ.copy()
        )

        output_text = ""
        capture = False
        while True:
            reads = [process.stdout.fileno()]
            ret = select.select(reads, [], [], 0.1)
            if ret[0]:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip(), flush=True)
                    # Look for text output markers
                    if "TEXT_OUTPUT_START" in line:
                        capture = True
                        continue
                    elif "TEXT_OUTPUT_END" in line:
                        capture = False
                        continue
                    if capture:
                        output_text += line
                else:
                    break
            else:
                if process.poll() is not None:
                    break

        returncode = process.wait()

        if returncode != 0:
            print(f"--> [CPP Analyze Worker] Process exited with code {returncode}", flush=True)
            sys.exit(1)

        # Parse output
        if output_text:
            # Clean the output
            output_text = output_text.strip()
            print(f"--> [CPP Analyze Worker] Analysis complete. Output length: {len(output_text)}", flush=True)
        else:
            print("--> [CPP Analyze Worker] WARNING: No text output captured", flush=True)
            output_text = "No analysis output produced."

        # 4. Save results
        result = {
            "text_content": {
                "raw_full_response": output_text,
                "thinking_process": "",
                "final_answer": output_text
            },
            "parameters": {
                "prompt": prompt,
                "top_k": top_k,
                "temperature": temperature,
                "model_path": model_path
            }
        }

        # Save to shared_utils (if not explicitly disabled)
        if not no_log:
            try:
                base_name = os.path.basename(image_path)
                timestamp_str = time.strftime("%Y%m%d%H%M%S%f")[:-3]  # includes deciseconds (3 digits)
                target_json_name = f"i2t_cpp_{timestamp_str}_{base_name}.json"
                target_image_name = f"i2t_cpp_{timestamp_str}_{base_name}.png"
                target_image_path = os.path.join(shared_utils.OUTPUT_DIR, target_image_name)
                
                # Copy the source image to outputs with the target name if it doesn't exist
                # This allows the history to find the image
                if not os.path.exists(target_image_path):
                    try:
                        import shutil
                        shutil.copy2(image_path, target_image_path)
                        print(f"--> [CPP Analyze Worker] Copied source image to: {target_image_name}", flush=True)
                    except Exception as copy_err:
                        print(f"--> [CPP Analyze Worker] Warning: Could not copy source image: {copy_err}", flush=True)

                shared_utils.save_generation_log("i2t_cpp", {"prompt": prompt, "source_images": [base_name]}, result["parameters"], result, image_path_for_filename=target_json_name)
            except Exception as e:
                print(f"--> [CPP Analyze Worker] Failed to save JSON log: {e}", flush=True)

        print("TEXT_OUTPUT_START")
        print(output_text)
        print("TEXT_OUTPUT_END")

        print(f"--> [CPP Analyze Worker] Analysis Completed in {time.time() - start_time:.1f}s", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {str(e)}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--top_k", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--no-log", action="store_true", help="Do not save generation log and duplicate image")

    args = parser.parse_args()

    run_sd_cpp_analyze(args.image_path, args.prompt, args.top_k, args.temperature, args.model_path, args.no_log)
