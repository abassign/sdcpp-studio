"""
Llama Server Manager - Persistent llama-server lifecycle for interactive I2T.
Maintains a long-running llama-server subprocess so the user can edit prompts
and re-query without restarting the model, saving VRAM thrash and startup time.
"""
import os
import sys
import json
import time
import base64
import signal
import datetime
import subprocess
import threading
from typing import List, Dict, Optional, Optional

import requests
import shared_utils

try:
    import shutil as _shutil
except ImportError:
    _shutil = None


def get_free_vram_mb() -> int:
    """Return the free VRAM in MB on the primary GPU (AMD or NVIDIA).

    Used by T2I generation to wait until the IA/tag server has fully
    released its GPU buffers before loading a new model, and by the
    llama-server manager to decide between full offload and auto-fit.

    Tries `nvidia-smi` (NVIDIA), then `rocm-smi` (AMD); on error falls
    back to a conservative 24 GB estimate.
    """
    import subprocess as _sp

    # 1. nvidia-smi (NVIDIA)
    nvidia_smi = None
    if _shutil is not None:
        nvidia_smi = _shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = _sp.run(
                [nvidia_smi, "--query-gpu=memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0:
                total_mb = 0
                used_mb = 0
                for line in res.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        total_mb += int(parts[0])
                        used_mb += int(parts[1])
                if total_mb:
                    return max(0, total_mb - used_mb)
        except Exception:
            pass

    # 2. rocm-smi (AMD)
    rocm_smi = None
    if _shutil is not None:
        rocm_smi = _shutil.which("rocm-smi")
    if rocm_smi is None:
        for path in ("/opt/rocm/bin/rocm-smi", "/usr/bin/rocm-smi"):
            if os.path.exists(path):
                rocm_smi = path
                break
    if rocm_smi:
        try:
            import json as _json
            res = _sp.run(
                [rocm_smi, "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0:
                data = _json.loads(res.stdout)
                used_mb = 0
                total_mb = 0
                for card in data:
                    if "VRAM Total Used Memory (B)" in data[card]:
                        used_mb += int(data[card]["VRAM Total Used Memory (B)"]) / (1024 * 1024)
                    if "VRAM Total Memory (B)" in data[card]:
                        total_mb += int(data[card]["VRAM Total Memory (B)"]) / (1024 * 1024)
                if total_mb:
                    return max(0, int(total_mb - used_mb))
        except Exception:
            pass

    # 3. Fallback - assume a 24 GB GPU is free (best we can do)
    return 24 * 1024

# Import validation (vendored gguf): expose both libs/ and libs/gguf so the
# internal `from gguf.constants import ...` of the vendored library resolves.
import os as _os
_APP_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_APP_ROOT, _os.path.join(_APP_ROOT, "libs"), _os.path.join(_APP_ROOT, "libs", "gguf")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from libs.gguf.validation import validate_model_for_inference, ValidationError
except ImportError:
    validate_model_for_inference = None
    ValidationError = Exception

# Import shared response parser (handles Gemma 4 channel markers, Qwen
# <think> tags, app markers, JSON, and raw text).
sys.path.insert(0, '/app')
try:
    from workers.i2t_response_parser import parse_model_output, build_chat_template_kwargs
except ImportError:
    # Fallback: provide a minimal in-file parser so the module is still
    # usable in tests or partial installs.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLAMA_SERVER_PORT = 8001
TAGGING_SERVER_PORT = 8002
LLAMA_SERVER_TIMEOUT = 300  # seconds per API call
SERVER_START_TIMEOUT = 120  # seconds to wait for server to become ready


# ---------------------------------------------------------------------------
# Helpers (reused from i2t_llama.py)
# ---------------------------------------------------------------------------
def _match_mmproj_to_model(model_filename, mmproj_candidates):
    """
    Select the best-matching mmproj file for a given model filename.

    Strategy:
      1. Exact substring match: if the model name (without extension) appears
         in an mmproj filename, that's the best match.
      2. Token-based overlap: tokenize both filenames (split on [-_.]) and
         pick the mmproj with the highest token overlap score.
      3. Fallback: return the first candidate (original behavior).
    """
    if not mmproj_candidates:
        return None
    if len(mmproj_candidates) == 1:
        return mmproj_candidates[0]

    import re
    model_base = os.path.splitext(model_filename)[0].lower()
    model_tokens = set(re.split(r'[-_.]+', model_base))

    best = None
    best_score = -1

    for candidate in mmproj_candidates:
        cand_name = os.path.basename(candidate).lower()

        # Strategy 1: exact substring match
        if model_base in cand_name or cand_name in model_base:
            return candidate

        # Strategy 2: token overlap
        cand_tokens = set(re.split(r'[-_.]+', os.path.splitext(cand_name)[0]))
        # Weight tokens that are more specific (longer than 2 chars)
        overlap = model_tokens & cand_tokens
        score = sum(len(t) for t in overlap if len(t) > 2)
        if score > best_score:
            best_score = score
            best = candidate

    return best if best else mmproj_candidates[0]


def find_gguf_model(model_dir):
    """Find GGUF model + mmproj in a directory (or use direct .gguf path).

    The mmproj is matched to the selected model by filename similarity so
    that a 12B model does not accidentally get paired with a 26B mmproj.
    """
    if model_dir is None:
        model_dir = shared_utils.I2T_DIR

    gguf_model = None
    mmproj = None
    model_filename = None

    if model_dir.endswith(".gguf"):
        gguf_model = model_dir
        model_filename = os.path.basename(model_dir)
        model_dir_path = os.path.dirname(model_dir)
        mmproj_candidates = []
        for f in os.listdir(model_dir_path):
            if "mmproj" in f.lower() and f.endswith(".gguf"):
                mmproj_candidates.append(os.path.join(model_dir_path, f))
        mmproj = _match_mmproj_to_model(model_filename, mmproj_candidates)
        return gguf_model, mmproj, model_filename

    if os.path.isdir(model_dir):
        mmproj_candidates = []
        for f in sorted(os.listdir(model_dir)):
            if f.endswith(".gguf") and "mmproj" not in f.lower():
                if gguf_model is None:
                    gguf_model = os.path.join(model_dir, f)
                    model_filename = f
            elif "mmproj" in f.lower() and f.endswith(".gguf"):
                mmproj_candidates.append(os.path.join(model_dir, f))

        # Match mmproj to the selected model by filename similarity
        mmproj = _match_mmproj_to_model(model_filename, mmproj_candidates)

    return gguf_model, mmproj, model_filename


def load_model_params(model_dir, model_filename=None):
    """Load model-specific parameters from params.json."""
    if model_dir is None:
        model_dir = shared_utils.I2T_DIR
    if model_dir.endswith(".gguf"):
        model_dir = os.path.dirname(model_dir)

    params_path = os.path.join(model_dir, "params.json")
    if not os.path.exists(params_path):
        return {}

    try:
        with open(params_path, 'r') as f:
            all_params = json.load(f)
        if model_filename and model_filename in all_params:
            return all_params[model_filename]
        if "_default" in all_params:
            return all_params["_default"]
        return {}
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Server Manager (singleton-like; use module-level instance)
# ---------------------------------------------------------------------------
class LlamaServerManager:
    """
    Manages a persistent llama-server subprocess.

    Thread-safe operations: ensure_server(), query(), stop().
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._port: int = LLAMA_SERVER_PORT
        self._model_key: str = ""         # unique key derived from model path
        self._model_dir: str = ""
        self._model_filename: str = ""
        self._gguf_model: str = ""
        self._mmproj: str = ""

    # -- public API ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Checks if the server subprocess is alive."""
        return self._process is not None and self._process.poll() is None

    @property
    def model_key(self) -> str:
        return self._model_key

    def ensure_server(self, model_dir: str, ctx_size: int = None) -> str:
        """
        Start (or restart) the llama-server for *model_dir*.

        If the server is already running with the same model, this is a no-op.
        If a different model is requested, the old server is stopped first.
        
        ctx_size: Override the default context size (from params.json).
                  If not provided, uses params.json value or default 16384.

        Returns the port number as a string.
        """
        with self._lock:
            gguf_model, mmproj, model_filename = find_gguf_model(model_dir)
            if not gguf_model:
                raise RuntimeError(f"No GGUF model found in {model_dir}")
            if not mmproj:
                raise RuntimeError(f"No mmproj (vision projector) found in {model_dir}")

            new_key = f"{gguf_model}|{mmproj}"

            # Store ctx_size override if provided
            if ctx_size is not None:
                self._ctx_size_override = ctx_size

            # Already running with same model → reuse (but check if ctx_size changed)
            if self.is_running and self._model_key == new_key:
                # If ctx_size changed, we need to restart
                current_ctx = self._model_params.get("ctx_size", 16384) if self._model_params else 16384
                if ctx_size is not None and ctx_size != current_ctx:
                    print(f"[LlamaServerManager] ctx_size changed ({current_ctx} -> {ctx_size}), restarting server", flush=True)
                    self._stop_server_locked()
                else:
                    print(f"[LlamaServerManager] Server already running for {model_filename}", flush=True)
                    return str(self._port)

            # Different model or not running → (re)start
            self._stop_server_locked()

            self._gguf_model = gguf_model
            self._mmproj = mmproj
            self._model_filename = model_filename
            self._model_dir = model_dir
            self._model_key = new_key

            self._start_server_locked()
            return str(self._port)

    def query(self, image_path: str | None, prompt: str, temperature: float = 0.7,
              max_tokens: int = 1024, system_prompt: str | None = None,
              conversation: str | None = None,
              enable_thinking: bool | None = None,
              image_specs: List[Dict[str, str]] | None = None) -> dict:
        """
        Send an image+prompt to the running server and return parsed result.

        If image_path is None or empty, operates in virtual T2T mode (text-only chat).
        conversation is prepended to the prompt for context in virtual T2T mode.

        enable_thinking:
            - True  -> ask the model to produce its reasoning phase
            - False -> suppress reasoning (Gemma 4 prefills the empty
                       thought channel; Qwen ignores the kwarg)
            - None  -> don't pass the kwarg at all, leaving the chat
                       template default in effect

        image_specs (Phase 6: N-Image I2T):
            Optional list of {"path": str, "role": str} dicts. When provided,
            overrides the single image_path with N images. Each image is labeled
            in the prompt as "Image N (role):" for clarity. Used by I2T mode to
            analyze multi-image scenarios (e.g., model + clothes for try-on).

        Returns dict with keys: raw_output, thinking, response.
        Raises RuntimeError if server is not running.
        """
        if not self.is_running:
            raise RuntimeError("llama-server is not running. Call ensure_server() first.")

        # Phase 6: Normalize to image_specs list. Priority: image_specs > image_path.
        if image_specs is not None and len(image_specs) > 0:
            # Use N-image mode
            normalized_specs = [s for s in image_specs if s and s.get("path")]
            # Default role to 'ref' if missing
            for s in normalized_specs:
                if not s.get("role"):
                    s["role"] = "ref"
        elif image_path:
            # Backward compat: single image with implicit 'init' role
            normalized_specs = [{"path": image_path, "role": "init"}]
        else:
            normalized_specs = []
        has_images = len(normalized_specs) > 0

        # Model-specific overrides
        model_params = load_model_params(self._model_dir, self._model_filename)
        effective_max_tokens = model_params.get("max_tokens", max_tokens)
        
        # Different system prompt for virtual T2T vs real I2T vs N-Image I2T (Phase 7)
        if not has_images:
            # Virtual T2T mode - text-only chat
            sys_prompt = model_params.get("system_prompt_t2t") or model_params.get("system_prompt") or \
                        "Sei un assistente utile e conciso. Rispondi alle domande in modo dettagliato."
        elif len(normalized_specs) > 1:
            # Phase 7: Multi-image I2T mode - synthesize N images into ONE English prompt
            # The model receives N images and must compose a single prompt for I2I generation
            # that references/incorporates elements from the images indicated by the user.
            # The user can naturally reference images by number (e.g. "face from image 2",
            # "style of image 3") in their creative direction.
            sys_prompt = model_params.get("system_prompt_multi_image", (
                "You are an expert image analyst and prompt engineer specialized in "
                "image-to-image generation. You receive MULTIPLE input images numbered "
                "Image 1, Image 2, Image 3, etc.\n\n"
                "RULES:\n"
                "1. Each image is identified by its number (Image 1, Image 2, ...).\n"
                "2. The user's request may reference images by number, e.g. 'the face from "
                "image 2', 'the style of image 3', 'as in image 1'. Interpret these "
                "references correctly and extract visual features from the right images.\n"
                "3. For EACH referenced image, extract key visual features: subject "
                "description, style, colors, lighting, composition, materials, mood.\n"
                "4. SYNTHESIZE all extracted features into a SINGLE, COHERENT English "
                "prompt suitable for image-to-image generation. The result must describe "
                "ONE unified scene that naturally combines elements from all referenced "
                "images.\n"
                "5. NEVER reference image numbers in the final synthesized prompt itself "
                "(the I2I generator does not know about them). Describe the final image "
                "directly with all the visual details.\n"
                "6. Follow the EXACT output format the user specifies (typically "
                "###POSITIVE### / ###NEGATIVE### / ###REASONING### markers)."
            ))
        else:
            # Single image I2T mode - just analyze one image
            sys_prompt = model_params.get("system_prompt",
                                           "Sei un assistente utile e conciso. "
                                           "Analizza le immagini quando fornite.")

        messages = []
        if system_prompt or sys_prompt:
            messages.append({"role": "system", "content": system_prompt or sys_prompt})

        # Virtual T2T mode (no image) - text-only chat with conversation context
        if not has_images:
            if conversation:
                full_prompt = f"[Previous conversation]\n{conversation}\n\n[New question]\n{prompt}"
            else:
                full_prompt = prompt
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}]
            })
        else:
            # Real I2T mode - read and encode N images
            image_b64_list = []
            for spec in normalized_specs:
                if not os.path.exists(spec["path"]):
                    raise RuntimeError(f"Image file not found: {spec['path']}")
                with open(spec["path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                image_b64_list.append(img_b64)
            print(f"[LlamaServerManager] Sending {len(image_b64_list)} image(s) to model (base64 total: {sum(len(b) for b in image_b64_list)} chars)", flush=True)

            # Build labeled prompt when N > 1
            if len(normalized_specs) == 1:
                context_prompt = prompt
                if conversation:
                    context_prompt = f"[Context from previous discussion]\n{conversation}\n\n[Image analysis request]\n{prompt}"
                content_parts = [
                    {"type": "text", "text": context_prompt},
                ]
            else:
                # N-Image mode: build labeled list
                role_labels = {
                    "init": "main image (to be edited)",
                    "ref":  "style reference",
                    "face": "face source",
                    "mask": "inpainting mask",
                    "control": "structural control",
                    "blend": "blend source",
                }
                image_labels = []
                for i, spec in enumerate(normalized_specs, start=1):
                    role = spec.get("role", "ref")
                    label = role_labels.get(role, role)
                    image_labels.append(f"Image {i} ({label})")
                labels_intro = (
                    f"I'm providing {len(normalized_specs)} images you can reference by number:\n"
                    + "\n".join(f"  - {lbl}" for lbl in image_labels)
                    + "\n\nThe user's request may say things like 'the face from image 2' or "
                    "'the style of image 3'. When you see such references, analyze the "
                    "referenced image(s) and incorporate their visual details into a single "
                    "coherent English prompt for image-to-image generation. Do NOT mention "
                    "image numbers in your final prompt - describe the target image directly."
                )
                if conversation:
                    context_prompt = f"[Context from previous discussion]\n{conversation}\n\n" + \
                                    f"{labels_intro}\n\n[Image analysis request]\n{prompt}"
                else:
                    context_prompt = f"{labels_intro}\n\n{prompt}"
                content_parts = [
                    {"type": "text", "text": context_prompt},
                ]
            # Add all N images
            for img_b64 in image_b64_list:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
            messages.append({
                "role": "user",
                "content": content_parts
            })

        # Build chat_template_kwargs
        if "enable_thinking" in model_params:
            thinking_kwarg = model_params["enable_thinking"]
        elif enable_thinking is not None:
            thinking_kwarg = enable_thinking
        else:
            thinking_kwarg = None
        chat_template_kwargs = build_chat_template_kwargs(thinking_kwarg)

        payload = {
            "model": self._model_filename or "Qwen3.6-35B-A3B",
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        start_t = time.time()
        resp = requests.post(
            f"http://localhost:{self._port}/v1/chat/completions",
            json=payload,
            timeout=LLAMA_SERVER_TIMEOUT,
        )
        elapsed = time.time() - start_t
        print(f"[LlamaServerManager] API response in {elapsed:.1f}s", flush=True)

        if resp.status_code != 200:
            raise RuntimeError(f"llama-server API error {resp.status_code}: {resp.text}")

        result = resp.json()
        raw_output = result["choices"][0]["message"]["content"]

        # Parse thinking/answer with the shared multi-format parser.
        thinking, response = parse_model_output(raw_output)

        return {
            "raw_output": raw_output,
            "thinking": thinking,
            "response": response,
        }

    def query_stream(self, image_path: str | None, prompt: str, temperature: float = 0.7,
                     max_tokens: int = 1024, system_prompt: str | None = None,
                     conversation: str | None = None,
                     enable_thinking: bool | None = None,
                     image_specs: List[Dict[str, str]] | None = None):
        """
        Phase 8B: Streaming version of query().

        Yields raw text chunks from llama-server as they arrive, allowing the
        client to display the response in real-time (like LLM chat UIs).

        Same parameters as query() but returns a generator of strings instead
        of a single dict. The caller is responsible for buffering and parsing.

        Yields:
            str: Text chunks from the model's response
        """
        if not self.is_running:
            raise RuntimeError("llama-server is not running. Call ensure_server() first.")

        # Phase 6: Normalize to image_specs list
        if image_specs is not None and len(image_specs) > 0:
            normalized_specs = [s for s in image_specs if s and s.get("path")]
            for s in normalized_specs:
                if not s.get("role"):
                    s["role"] = "ref"
        elif image_path:
            normalized_specs = [{"path": image_path, "role": "init"}]
        else:
            normalized_specs = []
        has_images = len(normalized_specs) > 0

        model_params = load_model_params(self._model_dir, self._model_filename)
        effective_max_tokens = model_params.get("max_tokens", max_tokens)

        # System prompt: same logic as query()
        if not has_images:
            sys_prompt = model_params.get("system_prompt_t2t") or model_params.get("system_prompt") or \
                        "Sei un assistente utile e conciso. Rispondi alle domande in modo dettagliato."
        elif len(normalized_specs) > 1:
            sys_prompt = model_params.get("system_prompt_multi_image", (
                "You are an expert image analyst and prompt engineer specialized in "
                "image-to-image generation. You receive MULTIPLE input images numbered "
                "Image 1, Image 2, Image 3, etc.\n\n"
                "RULES:\n"
                "1. Each image is identified by its number (Image 1, Image 2, ...).\n"
                "2. The user's request may reference images by number. Interpret them.\n"
                "3. For EACH referenced image, extract key visual features.\n"
                "4. SYNTHESIZE all extracted features into a SINGLE, COHERENT English "
                "prompt suitable for image-to-image generation.\n"
                "5. NEVER reference image numbers in the final synthesized prompt.\n"
                "6. Follow the EXACT output format the user specifies."
            ))
        else:
            sys_prompt = model_params.get("system_prompt",
                                           "Sei un assistente utile e conciso. "
                                           "Analizza le immagini quando fornite.")

        # Build chat_template_kwargs
        if "enable_thinking" in model_params:
            thinking_kwarg = model_params["enable_thinking"]
        elif enable_thinking is not None:
            thinking_kwarg = enable_thinking
        else:
            thinking_kwarg = None
        chat_template_kwargs = build_chat_template_kwargs(thinking_kwarg)

        messages = []
        if system_prompt or sys_prompt:
            messages.append({"role": "system", "content": system_prompt or sys_prompt})

        # Build content parts (same as query() - includes N image data URLs)
        if not has_images:
            if conversation:
                full_prompt = f"[Previous conversation]\n{conversation}\n\n[New question]\n{prompt}"
            else:
                full_prompt = prompt
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}]
            })
        else:
            image_b64_list = []
            for spec in normalized_specs:
                if not os.path.exists(spec["path"]):
                    raise RuntimeError(f"Image file not found: {spec['path']}")
                with open(spec["path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                image_b64_list.append(img_b64)
            print(f"[LlamaServerManager] Stream: sending {len(image_b64_list)} image(s) to model (base64 total: {sum(len(b) for b in image_b64_list)} chars)", flush=True)
            if len(normalized_specs) == 1:
                context_prompt = prompt
                if conversation:
                    context_prompt = f"[Context from previous discussion]\n{conversation}\n\n[Image analysis request]\n{prompt}"
                content_parts = [{"type": "text", "text": context_prompt}]
            else:
                role_labels = {
                    "init": "main image (to be edited)",
                    "ref":  "style reference",
                    "face": "face source",
                    "mask": "inpainting mask",
                    "control": "structural control",
                    "blend": "blend source",
                }
                image_labels = []
                for i, spec in enumerate(normalized_specs, start=1):
                    role = spec.get("role", "ref")
                    label = role_labels.get(role, role)
                    image_labels.append(f"Image {i} ({label})")
                labels_intro = (
                    f"I'm providing {len(normalized_specs)} images you can reference by number:\n"
                    + "\n".join(f"  - {lbl}" for lbl in image_labels)
                    + "\n\nThe user's request may say things like 'the face from image 2' or "
                    "'the style of image 3'. When you see such references, analyze the "
                    "referenced image(s) and incorporate their visual details into a single "
                    "coherent English prompt for image-to-image generation. Do NOT mention "
                    "image numbers in your final prompt - describe the target image directly."
                )
                if conversation:
                    context_prompt = f"[Context from previous discussion]\n{conversation}\n\n" + \
                                    f"{labels_intro}\n\n[Image analysis request]\n{prompt}"
                else:
                    context_prompt = f"{labels_intro}\n\n{prompt}"
                content_parts = [{"type": "text", "text": context_prompt}]
            for img_b64 in image_b64_list:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
            messages.append({
                "role": "user",
                "content": content_parts
            })

        # Thinking kwarg
        if "enable_thinking" in model_params:
            thinking_kwarg = model_params["enable_thinking"]
        elif enable_thinking is not None:
            thinking_kwarg = enable_thinking
        else:
            thinking_kwarg = None
        chat_template_kwargs = build_chat_template_kwargs(thinking_kwarg)

        # Enable streaming!
        payload = {
            "model": self._model_filename or "Qwen3.6-35B-A3B",
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        start_t = time.time()
        resp = requests.post(
            f"http://localhost:{self._port}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=LLAMA_SERVER_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"llama-server API error {resp.status_code}: {resp.text}")

        # Parse SSE-style chunks from llama-server
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                chunk_data = line[6:].strip()
                if chunk_data == "[DONE]":
                    break
                try:
                    import json as _json
                    chunk_json = _json.loads(chunk_data)
                    delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except Exception:
                    continue
        elapsed = time.time() - start_t
        print(f"[LlamaServerManager] Streamed response in {elapsed:.1f}s", flush=True)

    def stop(self):
        """Stop the llama-server (call when switching to a different module)."""
        with self._lock:
            self._stop_server_locked()

    # -- internal helpers ---------------------------------------------------

    def _start_server_locked(self):
        """Start llama-server subprocess (caller must hold _lock)."""
        model_params = load_model_params(self._model_dir, self._model_filename)
        self._model_params = model_params  # Store for later comparison

        cmd = ["llama-server", "-m", self._gguf_model, "--mmproj", self._mmproj]

        if model_params.get("jinja", False):
            cmd.append("--jinja")

        # Use override if set, otherwise from params.json, otherwise default
        if hasattr(self, '_ctx_size_override') and self._ctx_size_override is not None:
            ctx_size = self._ctx_size_override
        else:
            ctx_size = model_params.get("ctx_size", 16384)
        cmd.extend(["--ctx-size", str(ctx_size)])

        if model_params.get("batch_size"):
            cmd.extend(["-b", str(model_params["batch_size"])])
        if model_params.get("ub_batch_size"):
            cmd.extend(["-ub", str(model_params["ub_batch_size"])])

        # GPU layers: explicit value wins. If fit_on is enabled and no
        # explicit n_gpu_layers is given, omit the flag so llama.cpp can
        # auto-fit the split to available VRAM.
        #
        # VRAM-aware fallback: a hard-coded full offload (-ngl 99) aborts
        # with CUDA/HIP out-of-memory when the weights+mmproj do not fit in
        # the VRAM currently free (e.g. a 12 GB GPU shared with other
        # services). In that case drop -ngl and let llama.cpp auto-fit
        # (partial offload) via --fit on instead of failing to load.
        fit_on = bool(model_params.get("fit_on", False))
        ngl = model_params.get("n_gpu_layers")
        if ngl is not None and not fit_on:
            try:
                model_mb = int(os.path.getsize(self._gguf_model) / (1024 * 1024))
                if self._mmproj and os.path.exists(self._mmproj):
                    model_mb += int(os.path.getsize(self._mmproj) / (1024 * 1024))
            except OSError:
                model_mb = 0
            free_mb = get_free_vram_mb()
            needed_mb = int(model_mb * 1.25)  # weights + KV/compute overhead
            if model_mb and free_mb and needed_mb > int(free_mb * 0.90):
                print(f"[LlamaServerManager] Insufficient VRAM for full offload "
                      f"(model+mmproj≈{model_mb} MiB, needed≈{needed_mb} MiB, "
                      f"free={free_mb} MiB): switching to --fit (partial offload) "
                      f"instead of -ngl {ngl}", flush=True)
                ngl = None
                fit_on = True

        if ngl is not None:
            cmd.extend(["-ngl", str(ngl)])
        elif not fit_on:
            cmd.extend(["-ngl", "99"])

        if model_params.get("flash_attention", True):
            cmd.extend(["-fa", "on"])

        if model_params.get("cache_type_k"):
            cmd.extend(["-ctk", model_params["cache_type_k"]])
        if model_params.get("cache_type_v"):
            cmd.extend(["-ctv", model_params["cache_type_v"]])
        if model_params.get("cache_ram"):
            cmd.extend(["--cache-ram", str(model_params["cache_ram"])])

        if fit_on:
            cmd.extend(["--fit", "on"])
            if model_params.get("fit_target"):
                cmd.extend(["--fit-target", str(model_params["fit_target"])])

        host = model_params.get("host", "0.0.0.0")
        port = model_params.get("port", self._port)
        self._port = port
        cmd.extend(["--host", host, "--port", str(port)])

        if model_params.get("threads"):
            cmd.extend(["--threads", str(model_params["threads"])])
        if model_params.get("threads_batch"):
            cmd.extend(["--threads-batch", str(model_params["threads_batch"])])
        if model_params.get("no_warmup", False):
            cmd.append("--no-warmup")
        if model_params.get("log_verbosity"):
            cmd.extend(["--log-verbosity", str(model_params["log_verbosity"])])

        print(f"[LlamaServerManager] Starting: {' '.join(cmd)}", flush=True)
        mmproj_name = os.path.basename(self._mmproj) if self._mmproj else "none"
        print(f"[LlamaServerManager] Model: {os.path.basename(self._gguf_model)} | mmproj: {mmproj_name}", flush=True)

        env = {**os.environ, "HIP_VISIBLE_DEVICES": "0"}

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app",
            env=env,
        )
        print(f"[LlamaServerManager] PID: {self._process.pid}", flush=True)

        # Start a background thread to capture and log stderr from llama-server
        def _log_stderr():
            try:
                for line in self._process.stderr:
                    line = line.rstrip()
                    if line:
                        print(f"[llama-server stderr] {line}", flush=True)
            except Exception:
                pass
        self._stderr_thread = threading.Thread(target=_log_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait for server health check
        if not self._wait_for_server(self._port):
            # Capture any remaining stderr before stopping
            try:
                remaining_stderr = ""
                if self._process.stderr:
                    import select
                    while select.select([self._process.stderr], [], [], 0.5)[0]:
                        line = self._process.stderr.readline()
                        if not line:
                            break
                        remaining_stderr += line
                if remaining_stderr:
                    print(f"[LlamaServerManager] Server stderr on failure:\n{remaining_stderr}", flush=True)
            except Exception:
                pass
            self._stop_server_locked()
            raise RuntimeError(
                f"llama-server failed to start within {SERVER_START_TIMEOUT}s. "
                f"Check [llama-server stderr] logs above for details."
            )

        # Validate model+mmproj compatibility after startup
        if validate_model_for_inference is not None:
            try:
                validate_model_for_inference(self._gguf_model, self._mmproj)
                print("[LlamaServerManager] Model + mmproj validated OK", flush=True)
            except ValidationError as e:
                print(f"[LlamaServerManager] Validation warning: {e}", flush=True)
            except Exception as e:
                print(f"[LlamaServerManager] Validation skipped: {e}", flush=True)

        print(f"[LlamaServerManager] Server ready on port {self._port}", flush=True)

    def _stop_server_locked(self):
        """Kill the server subprocess (caller must hold _lock).

        Aggressive cleanup sequence:
          1. SIGTERM with a 5 s grace period (was 10 s - too long when
             the next process is waiting on VRAM).
          2. SIGKILL on timeout.
          3. Explicit os.waitpid() reap so the kernel releases the
             process's GPU mappings. subprocess.wait() alone is not
             enough on some ROCm kernels - the process can be a zombie
             for several seconds, holding its GPU buffers.
          4. Brief 1.5 s sleep to let the ROCm driver unmap the
             process's GPU virtual address space.
        """
        if self._process and self._process.poll() is None:
            pid = self._process.pid
            print(f"[LlamaServerManager] Stopping server PID {pid}", flush=True)
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[LlamaServerManager] Force killing (SIGTERM timed out)...", flush=True)
                    self._process.kill()
                    try:
                        self._process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        # Last resort - the process is stuck. Continue
                        # anyway but log loudly.
                        print(f"[LlamaServerManager] Server PID {pid} did not die after SIGKILL!", flush=True)
            except Exception as e:
                print(f"[LlamaServerManager] Stop error: {e}", flush=True)
                try:
                    self._process.kill()
                except Exception:
                    pass

            # Explicit reap - some ROCm kernels don't release the
            # process's GPU mappings until waitpid() is called.
            try:
                import os as _os
                _os.waitpid(pid, _os.WNOHANG)
            except (ChildProcessError, OSError):
                pass

            self._process = None
            self._model_key = ""
            self._stderr_thread = None
            # Give the ROCm driver a moment to unmap the dead process's
            # GPU pages. Empirically 1-2 s is enough on a 7900 XTX; the
            # 8 s wait in generate.py adds extra margin.
            import time as _time
            _time.sleep(1.5)
            print(f"[LlamaServerManager] Server PID {pid} stopped and reaped", flush=True)

    def _wait_for_server(self, port: int, timeout: int = SERVER_START_TIMEOUT) -> bool:
        """Poll /health until server responds or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
_server_manager = LlamaServerManager()


def get_server_manager() -> LlamaServerManager:
    """Return the singleton server manager."""
    return _server_manager


# ---------------------------------------------------------------------------
# VRAM Manager - Priority-based VRAM allocation for dual-server setup
# ---------------------------------------------------------------------------
class VramManager:
    """
    Manages two LlamaServerManager instances with priority-based VRAM allocation.

    High priority (port 8001): IA server for Image Analysis tab (user-initiated)
    Low priority  (port 8002): Tagging server for auto-tag batch processing (system-initiated)

    When IA needs VRAM:
      1. Tagging server is stopped (freeing its VRAM)
      2. IA server starts and processes the request
      3. When IA completes, IA server stops
      4. Tagging server restarts and resumes from checkpoint

    This ensures the user's interactive IA requests always have priority over
    background batch tagging, while avoiding VRAM exhaustion.
    """

    def __init__(self):
        # Use the global LlamaServerManager singleton for IA so that
        # i2t_query (which uses get_server_manager() directly) and the
        # VramManager coordinate on the SAME server instance. Without this
        # there are two parallel LlamaServerManager instances competing
        # for port 8001 and VRAM, causing OOM errors during T2I gen.
        self._ia_manager = get_server_manager()
        # Tag server has its own manager (separate port 8002).
        self._tag_manager = LlamaServerManager()
        self._tag_manager._port = TAGGING_SERVER_PORT
        self._lock = threading.Lock()
        self._ia_active = False
        self._tagging_suspended = False
        self._tagging_model_dir: str = ""
        self._tagging_checkpoint: Dict = {}
        # External VRAM user (T2I generation) that has evicted both
        # servers temporarily. None when no external user holds VRAM.
        # Used to refuse starting tag while T2I runs and to keep
        # stop_ia_server()/stop_tag_server() from auto-restarting
        # servers that the external user wants to keep stopped.
        self._external_vram_user: Optional[str] = None
        # Set by restore_external_vram() when the tag server was running
        # before the external user evicted it. Consumed by the
        # frontend's poll of /api/tags/vram/auto-resume-pending, which
        # triggers the resume flow automatically.
        self._auto_resume_pending: bool = False
        self._auto_resume_from: int = 0

    @property
    def ia_port(self) -> int:
        return LLAMA_SERVER_PORT

    @property
    def tag_port(self) -> int:
        return TAGGING_SERVER_PORT

    @property
    def ia_running(self) -> bool:
        return self._ia_manager.is_running

    @property
    def tag_running(self) -> bool:
        return self._tag_manager.is_running

    @property
    def tagging_suspended(self) -> bool:
        return self._tagging_suspended

    @property
    def checkpoint(self) -> Dict:
        return dict(self._tagging_checkpoint)

    def get_status(self) -> Dict:
        """Return current VRAM manager status."""
        return {
            "ia_server": {
                "running": self._ia_manager.is_running,
                "port": LLAMA_SERVER_PORT,
                "model": self._ia_manager.model_key or None,
                "priority": "high"
            },
            "tag_server": {
                "running": self._tag_manager.is_running,
                "port": TAGGING_SERVER_PORT,
                "model": self._tag_manager.model_key or None,
                "priority": "low",
                "suspended": self._tagging_suspended
            },
            "checkpoint": self._tagging_checkpoint if self._tagging_checkpoint else None
        }

    def ensure_ia_server(self, model_dir: str, ctx_size: int = None) -> str:
        """
        Start the high-priority IA server.
        If the tagging server is running, it is stopped first to free VRAM.
        Returns the IA server port.
        Raises RuntimeError if an external VRAM user is currently active.
        """
        with self._lock:
            # Refuse to start IA while an external user (e.g. T2I gen) is
            # using VRAM, otherwise we would cause OOM in the subprocess.
            if self._external_vram_user:
                raise RuntimeError(
                    f"Cannot start IA server: external VRAM user '{self._external_vram_user}' is active. "
                    "IA will resume when the external user releases VRAM."
                )
            # Stop tagging server if running to free VRAM
            if self._tag_manager.is_running:
                print(f"[VramManager] Suspending tagging server on port {TAGGING_SERVER_PORT} for IA request", flush=True)
                self._tag_manager.stop()
                self._tagging_suspended = True
                import time as _time
                _time.sleep(1)  # Brief pause for VRAM to be released

            self._ia_active = True
            port = self._ia_manager.ensure_server(model_dir, ctx_size)
            print(f"[VramManager] IA server started on port {port}", flush=True)
            return port

    def stop_ia_server(self):
        """
        Stop the IA server and resume tagging if it was suspended.
        Call this after the IA request completes.
        If an external VRAM user is currently active, do NOT auto-resume
        tagging - the external user will manage restoration itself.
        """
        with self._lock:
            if self._ia_manager.is_running:
                self._ia_manager.stop()
                print(f"[VramManager] IA server stopped on port {LLAMA_SERVER_PORT}", flush=True)
            self._ia_active = False

            # Resume tagging if it was suspended AND no external user
            # is currently holding VRAM. The external user (T2I gen) wants
            # tag stopped, so we must not start it back here.
            if self._tagging_suspended and self._tagging_model_dir and not self._external_vram_user:
                print(f"[VramManager] Resuming tagging server on port {TAGGING_SERVER_PORT}", flush=True)
                try:
                    self._tag_manager.ensure_server(self._tagging_model_dir)
                    self._tagging_suspended = False
                    print(f"[VramManager] Tagging server resumed", flush=True)
                except Exception as e:
                    print(f"[VramManager] Failed to resume tagging server: {e}", flush=True)
            elif self._tagging_suspended and self._external_vram_user:
                # External user is active - keep tag stopped but clear the
                # suspension flag so a future restore_external_vram() can
                # decide what to do based on its own saved state.
                self._tagging_suspended = False

    def ensure_tag_server(self, model_dir: str, ctx_size: int = None) -> str:
        """
        Start the low-priority tagging server.
        If the IA server is running, raises RuntimeError (IA has priority).
        If an external VRAM user is active, raises RuntimeError too.
        Returns the tagging server port.
        """
        with self._lock:
            if self._ia_active:
                raise RuntimeError(
                    "Cannot start tagging server: IA server is active (high priority). "
                    "Tagging will resume when IA completes."
                )
            if self._external_vram_user:
                raise RuntimeError(
                    f"Cannot start tagging server: external VRAM user '{self._external_vram_user}' is active. "
                    "Tagging will resume when the external user releases VRAM."
                )
            port = self._tag_manager.ensure_server(model_dir, ctx_size)
            self._tagging_model_dir = model_dir
            print(f"[VramManager] Tagging server started on port {port}", flush=True)
            return port

    def stop_tag_server(self):
        """Stop the tagging server (e.g., when user stops auto-tagging).
        If an external VRAM user is active, the call is ignored - the
        external user manages the server lifecycle."""
        with self._lock:
            if self._external_vram_user:
                print(f"[VramManager] stop_tag_server ignored: external user '{self._external_vram_user}' is active", flush=True)
                return
            if self._tag_manager.is_running:
                self._tag_manager.stop()
                print(f"[VramManager] Tagging server stopped on port {TAGGING_SERVER_PORT}", flush=True)
            self._tagging_suspended = False
            self._tagging_checkpoint = {}

    def save_checkpoint(self, checkpoint: Dict):
        """Save tagging progress checkpoint for resumption after IA interruption."""
        with self._lock:
            self._tagging_checkpoint = checkpoint
            print(f"[VramManager] Checkpoint saved: {checkpoint.get('current', '?')}/{checkpoint.get('total', '?')}", flush=True)

    def query_ia(self, image_path: str | None, prompt: str, **kwargs) -> dict:
        """Query the IA server (high priority)."""
        return self._ia_manager.query(image_path, prompt, **kwargs)

    def query_ia_stream(self, image_path: str | None, prompt: str, **kwargs):
        """Stream query the IA server (high priority). Yields text chunks."""
        return self._ia_manager.query_stream(image_path, prompt, **kwargs)

    def query_tag(self, image_path: str | None, prompt: str, **kwargs) -> dict:
        """Query the tagging server (low priority)."""
        return self._tag_manager.query(image_path, prompt, **kwargs)

    def query_tag_stream(self, image_path: str | None, prompt: str, **kwargs):
        """Stream query the tagging server (low priority)."""
        return self._tag_manager.query_stream(image_path, prompt, **kwargs)

    # -- external VRAM user API -----------------------------------------
    @property
    def ia_model_key(self) -> str:
        """The model_key of the underlying IA server (e.g. for logging)."""
        return self._ia_manager.model_key

    @property
    def external_vram_user(self) -> Optional[str]:
        """Name of the external VRAM user, or None if VRAM is free."""
        return self._external_vram_user

    def prepare_for_external_vram(self, user_name: str = "external") -> Dict:
        """
        Stop both IA and tag servers to free VRAM for an external user
        (e.g., T2I generation subprocess that needs ~12 GB on its own).

        Returns a state dict that should be passed to restore_external_vram()
        when the external user is done, so the previously-running servers
        can be resumed.

        This is the only safe way to free VRAM for an external consumer,
        because the GPU driver does not free VRAM immediately after a
        subprocess dies - the C++ runtime may keep tensors mapped for a
        few seconds.
        """
        with self._lock:
            state = {
                "ia_was_running": self._ia_manager.is_running,
                "ia_model_dir": self._ia_manager._model_dir,
                "ia_model_key": self._ia_manager._model_key,
                "ia_ctx_size": getattr(self._ia_manager, "_ctx_size_override", None),
                "tag_was_running": self._tag_manager.is_running,
                "tag_model_dir": self._tagging_model_dir or self._tag_manager._model_dir,
            }
            print(
                f"[VramManager] Preparing for external VRAM user '{user_name}': "
                f"IA running={state['ia_was_running']} (model={os.path.basename(state['ia_model_dir']) if state['ia_model_dir'] else None}), "
                f"Tag running={state['tag_was_running']} (model={os.path.basename(state['tag_model_dir']) if state['tag_model_dir'] else None})",
                flush=True,
            )

            # Stop IA if it is holding VRAM.
            if self._ia_manager.is_running:
                self._ia_manager.stop()
            self._ia_active = False

            # Stop tag if it is holding VRAM.
            if self._tag_manager.is_running:
                self._tag_manager.stop()
            # Clear the suspension flag - restore_external_vram() handles
            # restoration explicitly based on the saved state.
            self._tagging_suspended = False

            # Track who is holding VRAM so stop_ia_server() and
            # stop_tag_server() don't accidentally restart servers while
            # the external user is active.
            self._external_vram_user = user_name
            return state

    def restore_external_vram(self, state: Dict):
        """
        Restart IA and/or tag servers that were running before the
        external VRAM user acquired the token.

        Priority is IA > tag (consistent with the rest of the manager).
        Errors are logged but never raised - the external user has already
        finished its work and the application must remain responsive.
        """
        with self._lock:
            user = self._external_vram_user or "external"
            print(f"[VramManager] Restoring after external VRAM user '{user}'", flush=True)

            # Restart tag first (lower priority) if it was running.
            if state.get("tag_was_running") and state.get("tag_model_dir"):
                try:
                    model_dir = state["tag_model_dir"]
                    print(f"[VramManager] Restoring tag server: {model_dir}", flush=True)
                    self._tag_manager.ensure_server(model_dir)
                except Exception as e:
                    print(f"[VramManager] Failed to restore tag server: {e}", flush=True)

            # Restart IA if it was running.
            if state.get("ia_was_running") and state.get("ia_model_dir"):
                try:
                    model_dir = state["ia_model_dir"]
                    ctx_size = state.get("ia_ctx_size")
                    print(f"[VramManager] Restoring IA server: {model_dir}", flush=True)
                    # IA has higher priority, so stop tag if we just started it.
                    if self._tag_manager.is_running:
                        self._tag_manager.stop()
                        self._tagging_suspended = True
                    self._ia_manager.ensure_server(model_dir, ctx_size)
                    self._ia_active = True
                except Exception as e:
                    print(f"[VramManager] Failed to restore IA server: {e}", flush=True)

            self._external_vram_user = None


    def mark_auto_resume_pending(self, resume_from: int = 0):
        """Mark that a tagging resume is pending (e.g. after T2I
        generation freed the tag server). The frontend polls
        /api/tags/vram/auto-resume-pending to detect this and trigger
        the resume flow automatically."""
        with self._lock:
            self._auto_resume_pending = True
            self._auto_resume_from = int(resume_from or 0)
            print(f"[VramManager] Auto-resume pending from index {self._auto_resume_from}", flush=True)

    def consume_auto_resume_pending(self) -> tuple:
        """Atomically read and clear the auto-resume flag.

        Returns:
            (pending: bool, resume_from: int)
        """
        with self._lock:
            pending = self._auto_resume_pending
            resume_from = self._auto_resume_from
            self._auto_resume_pending = False
            self._auto_resume_from = 0
            return pending, resume_from

    def has_auto_resume_pending(self) -> bool:
        """Non-consuming read of the auto-resume flag (for status display)."""
        with self._lock:
            return self._auto_resume_pending


# ---------------------------------------------------------------------------
# Module-level VRAM manager singleton
# ---------------------------------------------------------------------------
_vram_manager = VramManager()


def get_vram_manager() -> VramManager:
    """Return the singleton VRAM manager."""
    return _vram_manager
