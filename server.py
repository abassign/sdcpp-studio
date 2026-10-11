"""
sdcpp-studio Server
Entry point for the FastAPI application
"""
import os
import sys
import shutil
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response, StreamingResponse, JSONResponse

# Setup directories
from shared_utils import MODELS_DIR, OUTPUT_DIR, sanitize_prompt
try:
    from workers.i2t_response_parser import parse_model_output
except ImportError:
    # Fallback: simple parser if the worker module is unavailable
    def parse_model_output(raw_output):
        if not raw_output:
            return "", ""
        # Try to extract ###THINKING### / ###ANSWER### / ###POSITIVE### / ###NEGATIVE### / ###REASONING###
        import re
        thinking_match = re.search(r'###THINKING###\s*(.*?)(?=###ANSWER###|###POSITIVE###|$)', raw_output, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'###ANSWER###\s*(.*?)(?=###THINKING###|###POSITIVE###|$)', raw_output, re.DOTALL | re.IGNORECASE)
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        response = answer_match.group(1).strip() if answer_match else raw_output
        return thinking, response
for d in ["/app/static", OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="sdcpp-studio")
# Start VRAM Arbiter poller on startup
@app.on_event("startup")
async def start_vram_arbiter():
    from api.vram_arbiter import get_arbiter
    arbiter = get_arbiter()
    arbiter.start_poller()
    print("[Server] VRAM Arbiter poller started", flush=True)

# Log buffer for streaming - shared with log file reader
log_buffer = []
max_log_lines = 1000
LOG_FILE = "/app/server.log"

class BufferHandler(logging.Handler):
    """Custom logging handler that stores logs in memory for streaming"""
    def __init__(self, buffer, max_lines=1000):
        super().__init__()
        self.buffer = buffer
        self.max_lines = max_lines
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.buffer.append(msg)
            if len(self.buffer) > self.max_lines:
                self.buffer.pop(0)
        except Exception:
            self.handleError(record)

# Configure root logger to capture everything
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Add buffer handler
buffer_handler = BufferHandler(log_buffer)
root_logger.addHandler(buffer_handler)

# Also ensure uvicorn logs are captured
logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)

# Add file handler to persist logs to file (for workers to also write to)
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
root_logger.addHandler(file_handler)

# Redirect print() to logging for log capture (DO THIS BEFORE ANY PRINTS)
class PrintToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.buffer = ''
    
    def write(self, data):
        self.buffer += data
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():
                    self.logger.log(self.log_level, line.rstrip())
            self.buffer = lines[-1]
    
    def flush(self):
        if self.buffer.strip():
            self.logger.log(self.log_level, self.buffer.rstrip())
            self.buffer = ''
    
    def isatty(self):
        return False

# Install the redirection - use root logger so logs go to file
sys.stdout = PrintToLogger(root_logger, logging.INFO)
sys.stderr = PrintToLogger(root_logger, logging.ERROR)

# Now all prints will be captured
print("=" * 60)
print("sdcpp-studio server started (print test)")
print("=" * 60)

# Also log via logging
logging.info("=" * 60)
logging.info("sdcpp-studio server started")
logging.info("=" * 60)

# Test entries to verify log capture is working
test_msg = "[DEBUG] Log buffer initialized - server.py loaded"
log_buffer.append(test_msg)
print(test_msg)
logging.info("Print redirection test - server.py")

# Force write to log file immediately
file_handler.flush()

# Mount static
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Import routers
from api import config, models, history, generate, setup, preferences, tags, views, agent

# Include routers
app.include_router(config.router, prefix="/api/config", tags=["Config"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(generate.router, prefix="/api/generate", tags=["Generate"])
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
app.include_router(setup.router, prefix="/api/setup", tags=["Setup"])
app.include_router(preferences.router, prefix="/api/preferences", tags=["Preferences"])
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(views.router, prefix="/api/views", tags=["Views"])


# Root
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# System
@app.get("/api/system")
async def system_info():
    """System info"""
    gpu = "CPU"
    try:
        if shutil.which("nvidia-smi"):
            gpu = "NVIDIA"
        elif shutil.which("rocm-smi"):
            gpu = "ROCm"
    except:
        pass
    return {"gpu": gpu}

@app.get("/api/gpu-type")
async def gpu_type():
    """Get GPU type from environment variable set by startup script"""
    gpu_type = os.environ.get("GPU_TYPE", "Unknown")
    return {"gpu_type": gpu_type}


@app.post("/api/analyze")
async def analyze(req: Request):
    """Analyze image using stable-diffusion.cpp multimodal model (GLM-4.1V, Qwen-Image, etc.)"""
    from api.generate import GenRequest, build_command, event_generator
    import json
    
    try:
        data = await req.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    # Validate required fields - accept both image_path and init_image
    image_path = data.get("image_path") or data.get("init_image")
    if not image_path:
        raise HTTPException(status_code=400, detail="image_path is required")
    
    # Determine analysis mode: analyze_cpp (sd.cpp) or analyze_llama (llama-cli)
    analyze_backend = data.get("analyze_backend", "llama")  # "cpp" or "llama"
    
    if analyze_backend == "llama":
        mode = "analyze_llama"
    else:
        mode = "analyze_cpp"
    
    # Build GenRequest with appropriate mode
    gen_req = GenRequest(
        mode=mode,
        prompt=data.get("prompt", "Describe this image in detail."),
        width=1024,  # Not used for analysis but required
        height=1024,  # Not used for analysis but required
        steps=10,  # Not used but required
        guidance=1.0,  # Not used but required
        seed=-1,  # Not used but required
        randomize=False,
        init_image=image_path,
        top_k=data.get("top_k", 1.0),
        temperature=data.get("temperature", 0.6),
        model_path=data.get("model_path"),
        i2t_model_path=data.get("i2t_model_path")
    )
    
    # Build command
    import random
    seed = random.randint(0, 2**32 - 1) if gen_req.randomize or gen_req.seed == -1 else gen_req.seed
    lora_data = []
    lora_config = None  # No LoRA support for analysis yet
    cmd = build_command(gen_req, seed, lora_config)
    
    if not cmd:
        raise HTTPException(status_code=400, detail="Invalid mode or configuration")
    
    return StreamingResponse(event_generator(cmd, gen_req), media_type="text/event-stream")


@app.get("/api/logs/stream")
async def stream_logs(request: Request):
    """Stream server logs via Server-Sent Events by tailing log file"""
    async def event_generator():
        # Read initial logs from file
        try:
            logging.info(f"[SSE] Client connected, reading from {LOG_FILE}")
            if os.path.exists(LOG_FILE):
                file_size = os.path.getsize(LOG_FILE)
                logging.info(f"[SSE] Log file exists, size: {file_size} bytes")
                with open(LOG_FILE, 'r') as f:
                    lines = f.readlines()
                    logging.info(f"[SSE] Sending initial {len(lines)} lines")
                    for line in lines[-50:]:
                        yield f"data: {line.rstrip()}\n\n"
            else:
                logging.warning(f"[SSE] Log file not found: {LOG_FILE}")
                yield f"data: [Log file not found: {LOG_FILE}]\n\n"
        except FileNotFoundError:
            yield f"data: [No log file found yet]\n\n"
        
        # Tail the file for new entries
        last_position = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
        while True:
            if await request.is_disconnected():
                logging.info("[SSE] Client disconnected")
                break
            
            try:
                current_size = os.path.getsize(LOG_FILE)
                if current_size > last_position:
                    with open(LOG_FILE, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.read().splitlines()
                        for line in new_lines:
                            if line.strip():
                                yield f"data: {line}\n\n"
                        last_position = f.tell()
            except Exception as e:
                # Log file might be rotated or deleted
                pass
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/exit")
async def shutdown():
    """Shutdown the server"""
    import os
    import signal
    import sys
    
    # Get parent PID (uvicorn reloader)
    parent_pid = os.getppid()
    
    # Send SIGTERM to current process first
    os.kill(os.getpid(), signal.SIGTERM)
    
    # Small delay to allow response to be sent
    import time
    time.sleep(0.1)
    
    # Kill parent process (reloader)
    try:
        os.kill(parent_pid, signal.SIGTERM)
    except:
        pass
    
    # Force exit
    os._exit(0)
    
    return {"status": "shutting_down"}


# ---------------------------------------------------------------------------
# Interactive I2T (persistent llama-server)
# ---------------------------------------------------------------------------
@app.post("/api/i2t/query")
async def i2t_query(req: Request):
    """
    Interactive I2T query using a persistent llama-server.

    First call starts the server (if not already running with the same model).
    Subsequent calls reuse the same server process.
    Only the prompt changes between calls – the model stays loaded.

    Request JSON:
      - image_path: str (optional - if missing, operates in virtual T2T mode)
      - prompt: str (required)
      - model_dir: str (optional, default: shared_utils.I2T_DIR)
      - temperature: float (optional, default: 0.7)
      - max_tokens: int (optional, default: 1024)
      - conversation: str (optional, for virtual T2T context)

    Returns SSE stream with:
      data: LOG|...  (progress)
      data: TXT|...  (text result, parsed into thinking+answer)
      data: DONE|... (completion)
      data: ERR|...  (error)
    """
    import traceback
    import time
    # NOTE: get_server_manager / get_vram_manager are imported lazily
    # inside event_gen() to avoid a circular import at module load time.

    try:
        data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Phase 6: N-Image I2T support. Accept either:
    #   - image_path: str (single, backward compat)
    #   - image_paths: List[Dict[path, role]] (N-Image mode)
    #   - images: List[Dict[path, role]] (alternative key, same format)
    image_path = data.get("image_path") or data.get("init_image")
    image_specs = data.get("image_specs") or data.get("image_paths") or data.get("images")
    if image_specs and isinstance(image_specs, list):
        # Normalize: each item must have "path"; "role" is optional
        normalized = []
        for item in image_specs:
            if isinstance(item, dict) and item.get("path"):
                normalized.append({
                    "path": item["path"],
                    "role": item.get("role", "ref")
                })
            elif isinstance(item, str):
                normalized.append({"path": item, "role": "ref"})
        image_specs = normalized
    else:
        image_specs = None
    prompt = sanitize_prompt(data.get("prompt", "Describe this image in detail."))
    model_dir = data.get("model_dir") or data.get("i2t_model_path") or shared_utils.I2T_DIR
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 1024))
    conversation = data.get("conversation")  # For virtual T2T context
    max_img_size = int(data.get("max_img_size", 1024))  # Max image dimension for VRAM saving

    # enable_thinking: controls whether the chat template's reasoning
    # phase is allowed. Default to False so "thinking" models (Gemma 4,
    # GPT-OSS, DeepSeek R1, ...) skip the reasoning block and return
    # the direct answer to the user. The per-model params.json (read
    # inside the manager) can still override this value.
    enable_thinking = data.get("enable_thinking", False)
    if isinstance(enable_thinking, str):
        enable_thinking = enable_thinking.lower() not in ("false", "0", "")
    enable_thinking = bool(enable_thinking)

    # Phase 6: Decide I2T mode based on available images
    has_any_images = bool(image_specs) or bool(image_path)
    is_virtual_t2t = not has_any_images

    if is_virtual_t2t:
        # Virtual T2T mode - no images, conversation provides context
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")
    else:
        # Real I2T mode - at least one image required
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")
        if image_specs:
            for spec in image_specs:
                if not os.path.exists(spec["path"]):
                    raise HTTPException(status_code=400, detail=f"Image not found: {spec['path']}")
        elif image_path and not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")

    async def event_gen():
        # Use the VRAM manager so that the IA server (port 8001) is
        # properly coordinated with the tagging server (port 8002):
        # starting IA will stop tag and free its VRAM. Using the bare
        # LlamaServerManager would cause both servers to compete for the
        # same GPU memory and trigger OOM during T2I generation.
        from api.llama_server_manager import get_vram_manager
        vram = get_vram_manager()
        try:
            # Resize images if needed to save VRAM
            actual_image_path = image_path
            actual_image_specs = image_specs

            def _resize_one_image(src_path):
                """Resize a single image if larger than max_img_size."""
                if not max_img_size or max_img_size <= 0:
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
                        temp_dir = os.path.join(OUTPUT_DIR, ".i2t_cache")
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, f"resized_{os.path.basename(src_path)}")
                        img.save(temp_path)
                        print(f"--> [I2T Query] Resized image from {w}x{h} to {new_w}x{new_h}", flush=True)
                        return temp_path
                except Exception as resize_err:
                    print(f"--> [I2T Query] Resize failed for {src_path}: {resize_err}", flush=True)
                return src_path

            if image_specs:
                # Phase 6: resize N images
                actual_image_specs = []
                for spec in image_specs:
                    new_path = _resize_one_image(spec["path"])
                    actual_image_specs.append({"path": new_path, "role": spec["role"]})
            elif image_path and os.path.exists(image_path):
                # Backward compat: resize single image, then wrap in specs format
                actual_image_path = _resize_one_image(image_path)
                actual_image_specs = [{"path": actual_image_path, "role": "init"}]

            # Get ctx_size from request or use default
            ctx_size = data.get("ctx_size")
            if ctx_size is not None:
                ctx_size = int(ctx_size)

            # 1. Ensure server is running (or (re)start if model changed).
            # To avoid regressing the 'schermo nero' that the VramManager
            # lock + tag-stop sleep introduced, we split the work:
            #   a) Briefly take the VramManager lock to stop the tag
            #      server (if running) and mark it suspended.
            #   b) Release the lock IMMEDIATELY (no sleep here).
            #   c) Start the IA server using the bare manager - this is
            #      the same code path that worked before the VramManager
            #      refactor, so the SSE stream blocks only on the model
            #      load + health check, not on extra coordination.
            # The 'schermo nero' was caused by ensure_ia_server() holding
            # the VramManager lock for the full duration of the model load
            # (30+ s), which made the SSE stream take too long to deliver
            # the next event and the frontend's loading state broke.
            mode_label = "Virtual T2T" if is_virtual_t2t else "I2T"
            yield f"data: LOG|[I2T Interactive] {mode_label} mode - Starting/verifying server...\n\n"
            with vram._lock:
                if vram._tag_manager.is_running:
                    vram._tag_manager.stop()
                    vram._tagging_suspended = True
            from api.llama_server_manager import get_server_manager
            bare_manager = get_server_manager()
            port = bare_manager.ensure_server(model_dir, ctx_size)
            model_short = os.path.basename(bare_manager.model_key.split('|')[0]) if bare_manager.model_key else "unknown"
            ctx_info = f" (ctx: {ctx_size})" if ctx_size else ""
            yield f"data: LOG|[I2T Interactive] Server ready on port {port} (model: {model_short}{ctx_info})\n\n"

            # 2. Query the server - Phase 8B: streaming for real-time response
            start_ts = time.time()
            yield f"data: LOG|[I2T Interactive] Sending query with prompt: {prompt[:80]}...\n\n"

            # Use streaming API: yield each chunk as it arrives
            accumulated = ""
            for chunk in vram.query_ia_stream(
                image_path=None,  # Phase 6: use image_specs instead
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                conversation=conversation,
                enable_thinking=enable_thinking,
                image_specs=actual_image_specs,  # Phase 6: N-image support (also handles single image now)
            ):
                accumulated += chunk
                # Normalize chunk BEFORE sending to frontend — this fixes ALL
                # double-encoding issues (â\x80\x94, Ã¨, etc.) at the source,
                # so the frontend always receives clean ASCII text.
                chunk_normalized = sanitize_prompt(chunk)
                # Yield each chunk in real-time to the frontend
                escaped_chunk = chunk_normalized.replace('\\', '\\\\').replace('\n', '\\n')
                yield f"data: TXT_CHUNK|{escaped_chunk}\n\n"
            
            elapsed = time.time() - start_ts
            yield f"data: LOG|[I2T Interactive] Response received in {elapsed:.1f}s ({len(accumulated)} chars)\n\n"

            # 3. Parse final result from accumulated text
            # The model output may contain ###THINKING### / ###ANSWER### markers
            # or just the raw response
            accumulated_normalized = sanitize_prompt(accumulated)
            thinking, response = parse_model_output(accumulated_normalized)
            combined = accumulated_normalized

            # Final complete payload (Phase 8B: send once at the end)
            escaped = combined.replace('\\', '\\\\').replace('\n', '\\n')
            yield f"data: TXT|{escaped}\n\n"

            # 4. Also save to generation log (like the existing workers do)
            try:
                timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

                if is_virtual_t2t:
                    # Virtual T2T mode - no image to save
                    target_json_name = f"i2t_virtual_t2t_{timestamp_str}.json"
                    inputs_data = {
                        "prompt": prompt,
                        "conversation": conversation or "",
                        "mode": "virtual_t2t",
                    }
                else:
                    # Real I2T mode - save image(s)
                    # Phase 6: support N images via image_specs
                    if image_specs:
                        source_images = [os.path.basename(spec["path"]) for spec in image_specs]
                        first_image_path = image_specs[0]["path"]
                    elif image_path:
                        source_images = [os.path.basename(image_path)]
                        first_image_path = image_path
                    else:
                        source_images = []
                        first_image_path = None

                    base_name = os.path.basename(first_image_path) if first_image_path else "no_image"
                    target_json_name = f"i2t_interactive_{timestamp_str}_{base_name}.json"
                    target_image_name = f"i2t_interactive_{timestamp_str}_{base_name}.png"
                    target_image_path = os.path.join(OUTPUT_DIR, target_image_name)

                    if first_image_path and not os.path.exists(target_image_path):
                        # Only copy if source is OUTSIDE the output directory.
                        # If already in OUTPUT_DIR, skip to avoid generating
                        # duplicate images on every I2T/IA analysis call.
                        source_dir = os.path.dirname(os.path.abspath(first_image_path))
                        if os.path.abspath(OUTPUT_DIR) != source_dir:
                            import shutil
                            shutil.copy2(first_image_path, target_image_path)

                    inputs_data = {
                        "prompt": prompt,
                        "source_images": source_images,
                    }

                params = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model_path": model_dir or "",
                    "backend": "llama-server-persistent",
                    "mode": "virtual_t2t" if is_virtual_t2t else "i2t",
                }
                outputs_data = {
                    "type": "text",
                    "text_content": {
                        "thinking_process": thinking,
                        "final_answer": response,
                        "raw_full_response": accumulated,
                    },
                }
                shared_utils.save_generation_log(
                    "i2t", inputs_data, params, outputs_data,
                    image_path_for_filename=target_json_name,
                    model_name="Interactive-GGUF",
                )
            except Exception as log_err:
                print(f"[I2T Interactive] Log save warning: {log_err}", flush=True)

            yield f"data: DONE|Query completed\n\n"

        except RuntimeError as e:
            yield f"data: ERR|{str(e)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: ERR|{str(e)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/i2t/shutdown")
async def i2t_shutdown():
    """
    Stop the persistent llama-server to free GPU VRAM.
    Call this when switching to a different module.
    Uses VramManager so the suspended tagging server (if any) is
    properly resumed after the IA server stops.
    """
    from api.llama_server_manager import get_vram_manager

    vram = get_vram_manager()
    was_running = vram.ia_running
    vram.stop_ia_server()
    return {
        "status": "stopped" if was_running else "idle",
        "message": "llama-server shut down" if was_running else "no server running",
    }


@app.get("/api/i2t/status")
async def i2t_status():
    """Check if the persistent llama-server is running."""
    from api.llama_server_manager import get_server_manager

    manager = get_server_manager()
    return {
        "running": manager.is_running,
        "model_key": manager.model_key if manager.is_running else "",
    }


@app.post("/api/models/upload")
async def upload_model_image(file: UploadFile = File(...)):
    """Upload an image for I2I/I2T processing.
    
    Rules:
    - If the image file already exists in the outputs directory, reuse it.
    - If it doesn't exist, copy it there.
    - Never create duplicates of the same file.
    """
    try:
        output_dir = OUTPUT_DIR
        target_path = os.path.join(output_dir, file.filename)
        
        # Check if the file already exists in the outputs directory
        if os.path.exists(target_path):
            print(f"--> [Upload] File already exists, reusing: {file.filename}", flush=True)
            return {"path": target_path, "url": f"/outputs/{file.filename}"}
        
        # File doesn't exist, save it
        print(f"--> [Upload] Saving new file: {file.filename}", flush=True)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"path": target_path, "url": f"/outputs/{file.filename}"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    # Read port from environment variable, default 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
