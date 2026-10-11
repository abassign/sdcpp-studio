#!/usr/bin/env python3
"""
I2T Worker using llama-server for multimodal image analysis.
Uses HTTP API for communication with the model server.
"""

import os
import sys
import json
import argparse
import subprocess
import time
import datetime
import re
import signal
import base64
import requests

# Add app directory to path for shared_utils
sys.path.insert(0, '/app')

import shared_utils

# Import validation module (vendored gguf): expose libs/ and libs/gguf so the
# internal `from gguf.constants import ...` resolves from any cwd.
import os as _os
_APP_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_APP_ROOT, _os.path.join(_APP_ROOT, "libs"), _os.path.join(_APP_ROOT, "libs", "gguf")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from libs.gguf.validation import validate_model_for_inference, ValidationError, format_validation_error
except ImportError:
    # Fallback if validation module not available
    validate_model_for_inference = None
    ValidationError = Exception
    format_validation_error = None

# Import shared response parser (handles Gemma 4 channel markers, Qwen
# <think> tags, app markers, JSON, and raw text).
try:
    from workers.i2t_response_parser import parse_model_output, build_chat_template_kwargs
except ImportError:
    # Fallback: minimal in-file parser (keeps this worker usable even
    # if the shared module is missing for some reason).
    import json as _json
    import re as _re

    def parse_model_output(raw_output):
        if not raw_output:
            return "", ""
        text = raw_output.strip()
        m = _re.search(
            r"<\|?channel\|?>\s*thought\s*(.*?)<\|?channel\|?>(.*)$",
            text,
            _re.IGNORECASE | _re.DOTALL,
        )
        if m:
            t, r = m.group(1).strip(), m.group(2).strip()
            return (t, r) if r else ("", t)
        if "###THINKING###" in text:
            parts = text.split("###THINKING###", 1)[1]
            if "###ANSWER###" in parts:
                t, r = parts.split("###ANSWER###", 1)
                return t.strip(), r.strip()
            return "", parts.strip()
        try:
            d = _json.loads(text)
            if isinstance(d, dict):
                t = str(d.get("thinking", "")).strip()
                r = str(d.get("answer", d.get("response", ""))).strip()
                if r:
                    return t, r
        except (ValueError, _json.JSONDecodeError):
            pass
        return "", text

    def build_chat_template_kwargs(enable_thinking):
        if enable_thinking is None:
            return {}
        return {"enable_thinking": bool(enable_thinking)}

# Default model path for GGUF models
DEFAULT_MODEL_DIR = shared_utils.I2T_DIR

# Server configuration
LLAMA_SERVER_PORT = 8001
LLAMA_SERVER_TIMEOUT = 300  # seconds


def output_error(code: str, message: str, details: str = ""):
    """Output a structured error in JSON format for the server to capture."""
    error_data = {
        "error": message,
        "code": code,
        "details": details
    }
    error_json = json.dumps(error_data)
    print(f"ERROR_OUTPUT:{error_json}", flush=True)
    sys.stdout.flush()
    print(f"[DEBUG] ERROR_OUTPUT sent: {error_json}", flush=True)


def find_gguf_model(model_dir=None):
    """Find the GGUF model and mmproj in the model directory.
    
    If model_dir is a specific GGUF file path, use it directly.
    If model_dir is a directory, search for model and mmproj within it.
    
    Returns:
        tuple: (gguf_model_path, mmproj_path, model_filename)
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    gguf_model = None
    mmproj = None
    model_filename = None
    
    # If model_dir is a specific GGUF file, use it directly
    if model_dir.endswith(".gguf"):
        gguf_model = model_dir
        model_filename = os.path.basename(model_dir)
        # Look for mmproj in the same directory
        model_dir_path = os.path.dirname(model_dir)
        for f in os.listdir(model_dir_path):
            if "mmproj" in f.lower() and f.endswith(".gguf"):
                mmproj = os.path.join(model_dir_path, f)
                break
        return gguf_model, mmproj, model_filename
    
    # Otherwise, search in directory (non-recursive for precision)
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith(".gguf") and not f.startswith("mmproj"):
                # Take the first GGUF file found in this directory
                if gguf_model is None:
                    gguf_model = os.path.join(model_dir, f)
                    model_filename = f
            elif "mmproj" in f.lower() and f.endswith(".gguf"):
                mmproj = os.path.join(model_dir, f)
    
    return gguf_model, mmproj, model_filename


def load_model_params(model_dir=None, model_filename=None):
    """Load model-specific parameters from params.json if available.
    
    Args:
        model_dir: Directory containing the model and params.json
        model_filename: Name of the GGUF model file for parameter matching
    
    Returns:
        dict: Model parameters, or empty dict if no params.json found
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    # If model_dir is a file path (e.g., a .gguf file), get its directory
    if model_dir.endswith(".gguf"):
        model_dir = os.path.dirname(model_dir)
    
    params_path = os.path.join(model_dir, "params.json")
    print(f"--> [I2T Llama Worker] Looking for params.json at: {params_path}", flush=True)
    print(f"--> [I2T Llama Worker] File exists: {os.path.exists(params_path)}", flush=True)
    if not os.path.exists(params_path):
        print(f"--> [I2T Llama Worker] params.json not found, using defaults", flush=True)
        return {}
    
    try:
        with open(params_path, 'r') as f:
            all_params = json.load(f)
        
        # Try exact model match first
        if model_filename and model_filename in all_params:
            params = all_params[model_filename]
            print(f"--> [I2T Llama Worker] Using params for model: {model_filename}", flush=True)
            return params
        
        # Fallback to _default if available
        if "_default" in all_params:
            params = all_params["_default"]
            print(f"--> [I2T Llama Worker] Using default params", flush=True)
            return params
        
        return {}
    except (json.JSONDecodeError, IOError) as e:
        print(f"--> [I2T Llama Worker] Warning: Could not load params.json: {e}", flush=True)
        return {}


def wait_for_server(port, timeout=60):
    """Wait for llama-server to be ready by polling the health endpoint."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                print(f"--> [I2T Llama Worker] Server is ready at port {port}", flush=True)
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def stop_llama_server(process):
    """Stop the llama-server process gracefully."""
    if process and process.poll() is None:
        print(f"--> [I2T Llama Worker] Stopping llama-server (PID: {process.pid})", flush=True)
        try:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"--> [I2T Llama Worker] Server did not stop gracefully, killing...", flush=True)
                process.kill()
                process.wait(timeout=5)
        except Exception as e:
            print(f"--> [I2T Llama Worker] Error stopping server: {e}", flush=True)
            try:
                process.kill()
            except:
                pass


def run_i2t_llama(image_path, prompt=None, model_dir=None, temperature=0.7, max_tokens=1024, no_log=False, enable_thinking=None):
    """
    Run image-to-text using llama-server with multimodal model.
    Starts server, makes API call, then stops server.

    Args:
        image_path: Path to the input image
        prompt: Optional custom prompt
        model_dir: Directory to search for GGUF models
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate (default: 16384)
        no_log: If True, skip writing the generation log JSON
        enable_thinking:
            - True  -> ask the model to produce its reasoning phase
            - False -> suppress reasoning (Gemma 4 prefills the empty
                       thought channel; Qwen ignores the kwarg)
            - None  -> don't pass the kwarg at all, leaving the chat
                       template default in effect. If the model's
                       params.json sets an explicit value, that wins.

    Returns:
        Generated text description with thinking and answer
    """
    print(f"--> [I2T Llama Worker] Starting analysis. PID: {os.getpid()}", flush=True)
    
    if not os.path.exists(image_path):
        output_error("IMAGE_NOT_FOUND", "Image file not found", image_path)
        sys.exit(1)
    
    print(f"--> [DEBUG] Image path exists: {os.path.exists(image_path)}", flush=True)
    print(f"--> [DEBUG] Image path: '{image_path}'", flush=True)
    
    # Find model
    gguf_model, mmproj, model_filename = find_gguf_model(model_dir)
    
    if not gguf_model:
        output_error("MODEL_NOT_FOUND", "No GGUF model found in model directory", model_dir or DEFAULT_MODEL_DIR)
        sys.exit(1)
    
    if not mmproj:
        output_error("MMPROJ_NOT_FOUND", "No mmproj (vision projector) found in model directory", model_dir or DEFAULT_MODEL_DIR)
        sys.exit(1)
    
    # Load model-specific parameters from params.json
    model_params = load_model_params(model_dir, model_filename)
    
    # Validate model and mmproj compatibility using GGUF metadata
    if validate_model_for_inference is not None:
        try:
            print(f"--> [I2T Llama Worker] Validating model and mmproj...", flush=True)
            validation_result = validate_model_for_inference(gguf_model, mmproj)
            model_arch = validation_result.get('compatibility', {}).get('model_arch', 'unknown')
            mmproj_arch = validation_result.get('compatibility', {}).get('mmproj_arch', 'unknown')
            print(f"--> [I2T Llama Worker] Model architecture: {model_arch}", flush=True)
            print(f"--> [I2T Llama Worker] MMProj architecture: {mmproj_arch}", flush=True)
            
            # Log any warnings
            warnings = validation_result.get('compatibility', {}).get('warnings', [])
            for warning in warnings:
                print(f"--> [I2T Llama Worker] Warning: {warning}", flush=True)
                
        except ValidationError as e:
            output_error(e.code, e.message, e.details)
            sys.exit(1)
        except Exception as e:
            # If validation fails for any reason, log but continue
            print(f"--> [I2T Llama Worker] Validation warning: {str(e)}", flush=True)
    
    print(f"--> [I2T Llama Worker] Model: {gguf_model}", flush=True)
    print(f"--> [I2T Llama Worker] MMProj: {mmproj}", flush=True)
    
    # Build llama-server command with model-specific parameters
    cmd = ["llama-server", "-m", gguf_model, "--mmproj", mmproj]
    
    # Add jinja processing (if supported by model)
    if model_params.get("jinja", False):
        cmd.append("--jinja")
    
    # Context size
    ctx_size = model_params.get("ctx_size", 16384)
    cmd.extend(["--ctx-size", str(ctx_size)])
    
    # Batch size
    if model_params.get("batch_size"):
        cmd.extend(["-b", str(model_params["batch_size"])])
    
    # Ub batch size
    if model_params.get("ub_batch_size"):
        cmd.extend(["-ub", str(model_params["ub_batch_size"])])
    
    # GPU layers
    # GPU layers: explicit value wins. If fit_on is enabled and no
    # explicit n_gpu_layers is given, omit the flag so llama.cpp can
    # auto-fit the split to available VRAM.
    if "n_gpu_layers" in model_params:
        cmd.extend(["-ngl", str(model_params["n_gpu_layers"])])
    elif not model_params.get("fit_on", False):
        cmd.extend(["-ngl", "99"])

    # Flash attention (requires a value: on, off, auto)
    if model_params.get("flash_attention", True):
        cmd.extend(["-fa", "on"])
    
    # Cache type k/v
    if model_params.get("cache_type_k"):
        cmd.extend(["-ctk", model_params["cache_type_k"]])
    if model_params.get("cache_type_v"):
        cmd.extend(["-ctv", model_params["cache_type_v"]])
    
    # Cache RAM
    if model_params.get("cache_ram"):
        cmd.extend(["--cache-ram", str(model_params["cache_ram"])])
    
    # Fit options
    if model_params.get("fit_on", False):
        cmd.extend(["--fit", "on"])
        if model_params.get("fit_target"):
            cmd.extend(["--fit-target", str(model_params["fit_target"])])
    
    # Host and port
    host = model_params.get("host", "0.0.0.0")
    port = model_params.get("port", LLAMA_SERVER_PORT)
    cmd.extend(["--host", host, "--port", str(port)])
    
    # Threads
    if model_params.get("threads"):
        cmd.extend(["--threads", str(model_params["threads"])])
    if model_params.get("threads_batch"):
        cmd.extend(["--threads-batch", str(model_params["threads_batch"])])
    
    # No warmup
    if model_params.get("no_warmup", False):
        cmd.append("--no-warmup")
    
    # Log verbosity
    if model_params.get("log_verbosity"):
        cmd.extend(["--log-verbosity", str(model_params["log_verbosity"])])
    
    print(f"--> [I2T Llama Worker] Running: {' '.join(cmd)}", flush=True)
    
    # Set HIP_VISIBLE_DEVICES to ensure llama-server uses the correct GPU
    env = {**os.environ, "HIP_VISIBLE_DEVICES": "0"}
    
    server_process = None
    try:
        # Start llama-server
        print(f"--> [I2T Llama Worker] Starting llama-server...", flush=True)
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app",
            env=env
        )
        
        print(f"--> [I2T Llama Worker] llama-server started with PID: {server_process.pid}", flush=True)
        
        # Wait for server to be ready using the configured port
        if not wait_for_server(port, timeout=120):
            output_error("SERVER_TIMEOUT", "llama-server failed to start within timeout",
                        f"Server did not become ready within 120 seconds")
            if server_process:
                server_process.kill()
            sys.exit(1)
        
        # Read image and encode as base64
        print(f"--> [I2T Llama Worker] Encoding image to base64...", flush=True)
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Default prompt for image analysis
        if not prompt:
            prompt = "Describe this image in detail."
        
        print(f"--> [I2T Llama Worker] Using prompt: {prompt}", flush=True)
        
        # Use model-specific max_tokens if available
        effective_max_tokens = model_params.get("max_tokens", max_tokens)
        
        # Build API payload. We no longer hardcode the model name to
        # "Qwen3.6-35B-A3B" — llama-server is happy to receive any
        # string in this field, and using the actual filename keeps the
        # logs readable when more than one model is installed.
        system_prompt = model_params.get("system_prompt", "You are a helpful, concise assistant. Analyze the images when provided.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        })

        # Build chat_template_kwargs. The per-model params.json takes
        # precedence; otherwise we use the function argument. When both
        # are None the kwarg is omitted entirely (chat template default).
        if "enable_thinking" in model_params:
            thinking_kwarg = model_params["enable_thinking"]
        elif enable_thinking is not None:
            thinking_kwarg = enable_thinking
        else:
            thinking_kwarg = None
        chat_template_kwargs = build_chat_template_kwargs(thinking_kwarg)

        payload = {
            "model": model_filename or "unknown-gguf",
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "stream": False
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        
        # Make API call using the configured port
        print(f"--> [I2T Llama Worker] Making API request to llama-server at port {port}...", flush=True)
        start_time = datetime.datetime.now()
        
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=LLAMA_SERVER_TIMEOUT
        )
        
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print(f"--> [I2T Llama Worker] API response received in {elapsed:.1f}s", flush=True)
        
        if resp.status_code != 200:
            error_msg = f"API returned status {resp.status_code}: {resp.text}"
            output_error("API_ERROR", "llama-server API error", error_msg)
            sys.exit(1)
        
        result = resp.json()
        raw_output = result["choices"][0]["message"]["content"]
        print(f"--> [I2T Llama Worker] Raw output: {len(raw_output)} chars", flush=True)
        
        # Parse response with the shared multi-format parser. This now
        # understands Gemma 4's <|channel|>thought...<|channel|>
        # markers in addition to the legacy app markers and JSON.
        thinking, response = parse_model_output(raw_output)

        print(f"--> [I2T Llama Worker] Parsed - Thinking: {len(thinking)} chars, Answer: {len(response)} chars", flush=True)
        
        # Add shared_utils logging similar to process_i2t.py (if not explicitly disabled)
        if not no_log:
            try:
                ancestor_params = {}
                base_path = os.path.splitext(image_path)[0]
                potential_json = base_path + ".json"
                if os.path.exists(potential_json):
                    with open(potential_json, 'r') as f:
                        ancestor_data = json.load(f)
                        ancestor_params = ancestor_data.get("params", {}) or ancestor_data.get("parameters", {})
                    
                final_params = ancestor_params.copy()
                final_params.update({
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model_path": gguf_model,
                    "backend": "llama-server"
                })
                
                inputs_data = {
                    "prompt": prompt,
                    "source_images": [os.path.basename(image_path)]
                }
                
                outputs_data = {
                    "type": "text",
                    "text_content": {
                        "thinking_process": thinking,
                        "final_answer": response,
                        "raw_full_response": raw_output
                    }
                }
                
                base_name = os.path.basename(image_path)
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # includes deciseconds (3 digits)
                target_json_name = f"i2t_llama_{timestamp_str}_{base_name}.json"
                target_image_name = f"i2t_llama_{timestamp_str}_{base_name}.png"
                target_image_path = os.path.join(shared_utils.OUTPUT_DIR, target_image_name)
                
                # Copy the source image to outputs with the target name if it doesn't exist
                # This allows the history to find the image
                if not os.path.exists(target_image_path):
                    try:
                        import shutil
                        shutil.copy2(image_path, target_image_path)
                        print(f"--> [I2T Llama Worker] Copied source image to: {target_image_name}", flush=True)
                    except Exception as copy_err:
                        print(f"--> [I2T Llama Worker] Warning: Could not copy source image: {copy_err}", flush=True)
                
                shared_utils.save_generation_log("i2t", inputs_data, final_params, outputs_data, image_path_for_filename=target_json_name, model_name="GLM-4.6V-Flash-GGUF")
            except Exception as e:
                print(f"--> [I2T Llama Worker] Failed to save JSON log: {e}", flush=True)
        
        # Output with markers for server to capture as text response
        print("TEXT_OUTPUT_START", flush=True)
        if thinking:
            print(f"###THINKING###\n{thinking}\n###ANSWER###\n{response}", flush=True)
        else:
            print(response, flush=True)
        print("TEXT_OUTPUT_END", flush=True)
        
        # Return full response with markers for frontend
        if thinking:
            return f"###THINKING###\n{thinking}\n###ANSWER###\n{response}"
        return response
        
    except requests.exceptions.Timeout:
        error_msg = f"API request timed out after {LLAMA_SERVER_TIMEOUT} seconds"
        output_error("API_TIMEOUT", "API request timed out", error_msg)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        output_error("API_ERROR", "API request failed", str(e))
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        sys.exit(1)
    finally:
        # Always stop the server
        if server_process:
            stop_llama_server(server_process)


def main():
    parser = argparse.ArgumentParser(description="I2T using llama-server")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--prompt", "-p", default=None, help="Custom prompt")
    parser.add_argument("--model-dir", "-d", default=None, help="Model directory")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Temperature")
    parser.add_argument("--max-tokens", "-n", type=int, default=16384, help="Max tokens")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--no-log", action="store_true", help="Do not save generation log and duplicate image")
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        choices=["true", "false"],
        default=None,
        help=("Pass enable_thinking to the chat template. "
              "Gemma 4 uses it to skip the reasoning phase. "
              "If omitted, the per-model params.json wins."),
    )

    args = parser.parse_args()

    # Map "true"/"false" strings to bool, leave None when not provided.
    thinking_arg = None
    if args.enable_thinking is not None:
        thinking_arg = (args.enable_thinking.lower() == "true")

    result = run_i2t_llama(
        args.image,
        prompt=args.prompt,
        model_dir=args.model_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        no_log=args.no_log,
        enable_thinking=thinking_arg,
    )
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"--> [I2T Llama Worker] Output saved to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
