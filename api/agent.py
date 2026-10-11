"""
Agentic Refine API (M2) - SSE bridge between the web UI and the
workers/agent_refine.py headless loop.

Endpoints:
    POST /api/agent/refine   -> SSE stream of the agent loop
    POST /api/agent/stop     -> kill the running agent subprocess
    GET  /api/agent/critics  -> list available critic (VLM) model dirs
    GET  /api/agent/capabilities -> per-model edit / vision capabilities (UI gating)

The worker subprocess is spawned with its own process group so /stop can
kill the whole tree (worker -> t2i subprocess) reliably.
"""
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from shared_utils import OUTPUT_DIR, sanitize_prompt

router = APIRouter()

agent_process = None  # the running agent subprocess (module-global singleton)


class LoraItem(BaseModel):
    path: str
    weight: float = 1.0


class AgentImageSlot(BaseModel):
    """An extra image from the SAME I2T/I2I slots the UI already uses.
    The role mirrors the I2T/I2I semantic (init|ref|face|mask|control|blend)
    and is forwarded to the VLM critic so it can integrate those characteristics."""
    path: str
    role: str = "ref"


class AgentRequest(BaseModel):
    goal: str
    image: Optional[str] = None
    critic: Optional[str] = None
    interpreter: Optional[str] = None  # optional separate I2T describer of the Init Image
    diffusion: Optional[str] = None
    schedule: Optional[str] = "384x512,512x640,768x768"
    iterations: Optional[int] = None
    seed: int = 42
    steps: int = 10
    guidance: float = 2.0
    scheduler: str = "euler"
    llm_path: Optional[str] = None
    vae_path: Optional[str] = None
    scheduler_type: Optional[str] = None
    loras: List[LoraItem] = []
    images: List[AgentImageSlot] = []  # extra I2T/I2I image slots for the critic
    negative_prompt: str = ""
    early_stop: bool = True
    no_evict: bool = False
    seed_mode: str = "fixed"      # fixed | step | random (how the seed varies per iteration)
    seed_step: int = 1             # increment per iteration when seed_mode == "step"
    critic_temperature: float = 0.5  # VLM temperature: higher = more creative prompt rewrites
    explore_when_stuck: bool = True  # restart search direction when the critic score plateaus
    two_phase_critic: bool = False   # independent reviewer rates the image (judge != generator)
    use_image: bool = True           # send the reference image to the VLM critic
    edit_from_reference: bool = True  # pass the reference image to the generator (sd.cpp -r)
    edit_anchor: str = "hybrid"       # hybrid | original | chain


class FinalizeRequest(BaseModel):
    out_dir: str
    action: str = "export"  # "export" | "cleanup"


class ResumeRequest(BaseModel):
    out_dir: str
    iterations: int = 1
    delta: int = 256  # px on the longer side: >0 enlarge, <0 shrink, 0 keep same size
    loras: List[LoraItem] = []  # LoRA set currently shown in the Agent panel (restored from the run)
    images: List[AgentImageSlot] = []  # I2T/I2I slots currently shown (restored from the run)


def _to_url(path: str) -> str:
    """/app/outputs/x.png -> /outputs/x.png for the static mount."""
    if path.startswith(OUTPUT_DIR):
        return path.replace(OUTPUT_DIR, "/outputs", 1)
    return path


def _resolve_image_path(image: str) -> str:
    """Accept either an absolute disk path or a /outputs/... URL and return a
    disk path that the worker subprocess can open."""
    if not image:
        return image
    if image.startswith("/outputs/"):
        return image.replace("/outputs/", OUTPUT_DIR + "/", 1)
    return image


@router.get("/capabilities")
async def agent_capabilities(diffusion: str = "", critic: str = ""):
    """Report which Agent Refine controls are usable with the selected models.

    Drives the UI: the "Interpret image" and "Edit from reference" checkboxes
    must be greyed out (with a reason) when the selected critic / generator
    cannot honour them.
    """
    # --- generator (edit) capability -------------------------------------
    try:
        from core.edit_capabilities import detect_edit
        gen = detect_edit(diffusion)
    except Exception as e:                                    # noqa: BLE001
        gen = {"edit": False, "family": None, "vision_required": False,
               "vision_encoder": None, "vision_ready": False, "multi_ref": False,
               "worker": "t2i_cpp", "anchor_modes": [], "reason": f"capability check failed: {e}"}

    # --- critic (interpret) capability -----------------------------------
    crit = {"name": os.path.basename(critic) if critic else None,
            "vision": False, "mmproj": None, "reason": None}
    if not critic:
        crit["reason"] = "No critic selected."
    else:
        try:
            from workers.agent_refine import find_critic_dirs
            from api.llama_server_manager import find_gguf_model
            dirs = find_critic_dirs()
            hit = next((d for d in dirs if d == critic or os.path.basename(d) == os.path.basename(critic)), None)
            if hit:
                _gguf, mmproj, _fn = find_gguf_model(hit)
                crit["vision"] = bool(mmproj)
                crit["mmproj"] = os.path.basename(mmproj) if mmproj else None
                if not mmproj:
                    crit["reason"] = "The selected critic has no vision projector (mmproj): it cannot read the image."
            else:
                crit["reason"] = "The selected critic is not usable (missing GGUF or mmproj)."
        except Exception as e:                                # noqa: BLE001
            crit["reason"] = f"critic check failed: {e}"

    return {"generator": gen, "critic": crit}


@router.get("/critics")
async def list_critics():
    """List the i2t model dirs usable as critic (GGUF + vision projector).

    Ordered best-first by the shared preference list, and "preferred" names the
    one the UI must preselect (see CRITIC_PREFERENCE in workers/agent_refine.py;
    currently SenseNova-SI, an 8B Qwen3-VL tuned for spatial intelligence).
    """
    try:
        from workers.agent_refine import rank_critic_dirs, find_default_critic
        dirs = rank_critic_dirs()
        pref = find_default_critic()
        return {
            "critics": [os.path.basename(d) for d in dirs],
            "dirs": dirs,
            "preferred": os.path.basename(pref) if pref else None,
        }
    except Exception as e:
        return {"critics": [], "dirs": [], "preferred": None, "error": str(e)}


def _find_chat_gguf(model_dir: str):
    """Return a non-mmproj .gguf inside model_dir (recursive), or None."""
    if not model_dir or not os.path.isdir(model_dir):
        return None
    for root, _dirs, files in os.walk(model_dir):
        for f in sorted(files):
            if f.lower().endswith(".gguf") and "mmproj" not in f.lower():
                return os.path.join(root, f)
    return None


_UNCENSORED_HINTS = ("uncensored", "abliterated", "heretic", "unfiltered", "aggressive", "nsfw")


def _is_uncensored_dir(name: str) -> bool:
    n = (name or "").lower()
    return any(h in n for h in _UNCENSORED_HINTS)


def _dir_is_uncensored(model_dir: str) -> bool:
    """True if the folder name OR the chat GGUF filename looks uncensored."""
    if _is_uncensored_dir(os.path.basename(model_dir)):
        return True
    g = _find_chat_gguf(model_dir)
    return bool(g and _is_uncensored_dir(os.path.basename(g)))


def _candidate_translate_dirs():
    """i2t model dirs that contain a chat GGUF (translation-capable)."""
    from shared_utils import MODELS_DIR
    base = os.path.join(MODELS_DIR, "i2t")
    out = []
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            full = os.path.join(base, d)
            if os.path.isdir(full) and _find_chat_gguf(full):
                out.append(full)
    return out


def _default_translate_dir():
    """Prefer the SMALLEST uncensored model (fast to load, no refusals on
    artistic/anatomical terms); fall back to the agent critic."""
    dirs = _candidate_translate_dirs()
    unc = [d for d in dirs if _dir_is_uncensored(d)]
    if unc:
        def _gguf_size(d):
            g = _find_chat_gguf(d)
            try:
                return os.path.getsize(g)
            except Exception:                                    # noqa: BLE001
                return 1 << 60
        return min(unc, key=_gguf_size)
    try:
        from workers.agent_refine import find_default_critic
        return find_default_critic() or (dirs[0] if dirs else "")
    except Exception:                                            # noqa: BLE001
        return dirs[0] if dirs else ""


@router.get("/translate-models")
async def list_translate_models():
    """List translation-capable chat models (i2t GGUF dirs). Uncensored ones are
    flagged; the suggested default is uncensored so artistic/anatomical terms are
    translated without refusals."""
    items = []
    for d in _candidate_translate_dirs():
        nm = os.path.basename(d)
        items.append({"dir": d, "name": nm, "uncensored": _dir_is_uncensored(d)})
    items.sort(key=lambda x: (not x["uncensored"], x["name"].lower()))
    return {"models": items, "default": _default_translate_dir()}


class TranslateRequest(BaseModel):
    text: str
    model: Optional[str] = None   # chat model dir; auto/uncensored if empty
    target: str = "English"
    ctx_size: Optional[int] = 2048


@router.post("/translate")
async def translate_text(req: TranslateRequest):
    """Translate arbitrary text (e.g. the Agent Refine GOAL) with the LOCAL chat
    model, so the user can write in their own language and get the English text
    the diffusion model / critic expect. Reuses the persistent I2T server
    (VramManager). When no model is given it picks an UNCENSORED one, so artistic /
    anatomical / mature terminology is translated instead of refused."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > 4000:
        raise HTTPException(status_code=400, detail="text too long (max 4000 chars)")

    # Resolve a runnable chat model dir (request -> uncensored default).
    model_dir = (req.model or "").strip()
    if model_dir and not os.path.isdir(model_dir):
        model_dir = ""
    if not model_dir:
        model_dir = _default_translate_dir()
    if not model_dir or not _find_chat_gguf(model_dir):
        raise HTTPException(
            status_code=503,
            detail="No chat model available for translation.")

    system = (
        "You are a professional UNCENSORED translator. Translate the user's text "
        f"into {req.target}, faithfully and literally. Preserve ALL content and "
        "terminology, including artistic, historical, anatomical, medical and mature "
        "vocabulary (e.g. Renaissance art, classical mythology, the naked human body "
        "in paintings). NEVER refuse, moralize, censor, soften, euphemize, omit or add "
        "warnings/notes. Output ONLY the translation, without quotes or markdown."
    )

    def _run():
        from api.llama_server_manager import get_vram_manager
        vram = get_vram_manager()
        vram.ensure_ia_server(model_dir, req.ctx_size)
        return vram.query_ia(None, text, temperature=0.1, max_tokens=1024,
                             system_prompt=system, enable_thinking=False)

    try:
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(None, _run)
    except Exception as e:                                       # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Translation failed: {e}")

    translated = (out.get("response") or out.get("raw_output") or "").strip()
    # Strip accidental wrapping quotes / markdown fences.
    if len(translated) >= 2 and translated[0] in "\"'\u201c\u201d" and translated[-1] in "\"'\u201c\u201d":
        translated = translated[1:-1].strip()
    return {"text": translated, "model": os.path.basename(model_dir)}


class RemoveStepRequest(BaseModel):
    out_dir: str
    image: str  # basename of a step image, e.g. "iter_002_img.png"


@router.post("/remove-step")
async def remove_step(req: RemoveStepRequest):
    """Delete ONE step of a finished agent run: removes iter_NNN_img.png (plus
    its _prompt.txt / _critique.txt) from the run folder and drops the entry
    from manifest.json, so History/ARF picker and Retry see the change.
    Rejected while an agent run is in progress (the manifest is only written
    when a run finishes)."""
    global agent_process
    if agent_process is not None and agent_process.poll() is None:
        raise HTTPException(status_code=409,
                            detail="An agent run is in progress: wait for it to finish before removing steps.")
    image = (req.image or "").strip()
    # New artifacts carry the run folder prefix ("<folder>_iter_NNN_img.png") so a
    # file alone is unambiguously traceable to its run; manifests written by older
    # versions store the bare "iter_NNN_img.png".
    m_img = re.fullmatch(r"([A-Za-z0-9_\-]*)iter_\d+_img\.png", image)
    if not m_img:
        raise HTTPException(status_code=400, detail="Invalid step image name")
    out_dir = _validate_agent_dir(req.out_dir)
    # A prefixed name must belong to THIS run folder (no cross-run deletion).
    prefix = m_img.group(1)
    if prefix and prefix != os.path.basename(out_dir) + "_":
        raise HTTPException(status_code=400, detail="Step image does not belong to this run")

    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise HTTPException(status_code=400, detail="No completed run found in this folder (manifest.json missing)")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read manifest: {e}")

    iters = manifest.get("iterations") or []
    kept = []
    removed = None
    for it in iters:
        if it.get("image") == image:
            removed = it
        else:
            kept.append(it)
    if removed is None:
        raise HTTPException(status_code=404, detail="Step not found in the run manifest")

    # delete the step artifacts (image + prompt + critique), never outside out_dir.
    # Derive "…_iter_NNN" from "…_iter_NNN_img.png": os.path.splitext would keep
    # the "_img" suffix and therefore miss the sibling _prompt.txt/_critique.txt.
    base = image[:-len("_img.png")] if image.endswith("_img.png") else os.path.splitext(image)[0]
    for candidate in (image, base + "_prompt.txt", base + "_critique.txt"):
        p = os.path.join(out_dir, candidate)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Cannot delete {candidate}: {e}")

    manifest["iterations"] = kept
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot update manifest: {e}")
    return {"ok": True, "deleted": image, "remaining": len(kept)}


@router.post("/finalize")
async def finalize(req: FinalizeRequest):
    """Finish an agent run:
      - action=cleanup: delete the agent run folder (and its steps).
      No export happens server-side anymore: agent images stay inside the
      agent_refine_* run folder; only T2I/I2I generation writes to OUTPUT_DIR.
    Only allowed on folders inside OUTPUT_DIR whose name starts with 'agent_refine_'.
    """
    # Accept either a disk path (/app/outputs/...) or the browser URL form
    # (/outputs/...) produced by _to_url() in the FINISH event. The frontend
    # stores agentOutDir from that URL, so normalize it back to disk first;
    # otherwise realpath() would point at a non-existent /outputs path.
    out_dir = req.out_dir
    if out_dir.startswith("/outputs/"):
        out_dir = out_dir.replace("/outputs/", OUTPUT_DIR + "/", 1)
    out_dir = os.path.realpath(out_dir)
    ok = out_dir.startswith(os.path.realpath(OUTPUT_DIR) + os.sep) and \
        os.path.basename(out_dir).startswith("agent_refine_") and \
        os.path.isdir(out_dir)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid out_dir (must be an agent_refine_* folder inside outputs)")

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    result = {"ok": True, "deleted": os.path.basename(out_dir)}

    # Only 'cleanup' is supported now: it deletes the whole agent_refine_*
    # run folder. We intentionally NO LONGER export arf_cpp_*.png/.json copies
    # into OUTPUT_DIR: images must land in the outputs root ONLY when the user
    # actually generates via the T2I / I2I tabs. To reuse an agent step, the
    # frontend points directly at the file inside the run folder.
    if req.action == "cleanup":
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as e:
            result["warning"] = f"cleanup: {e}"
        result["ok"] = True
    else:
        # legacy 'export' has no server-side side effects anymore
        result["deleted"] = ""
        result["note"] = "export disabled: no file created in outputs (use the run folder image directly)"
    return result


@router.get("/status")
async def status():
    global agent_process
    running = agent_process is not None and agent_process.poll() is None
    return {"running": running,
            "pid": agent_process.pid if (running and agent_process) else None}


@router.get("/experience")
async def experience(out_dir: str = ""):
    """Read the persisted exploration memory of a finished run so the UI can
    show the agent's 'experience' (mode history, best prompt, strategies tried,
    last diagnosis) when a run is reopened from History."""
    if not out_dir:
        raise HTTPException(status_code=400, detail="out_dir is required")
    run = _validate_agent_dir(out_dir)
    from workers.agent_refine import load_memory, empty_memory
    mem_path = os.path.join(run, "explore_state.json")
    if not os.path.exists(mem_path):
        # nothing was explored yet: return an empty, harmless payload
        return {"exists": False, "has_experience": False}
    mem = load_memory(run)
    return {
        "exists": True,
        "has_experience": bool(mem.get("explore_count") or mem.get("best")),
        "explore_count": mem.get("explore_count", 0),
        "best": mem.get("best"),
        "strategies": mem.get("strategies", []),
        "recent": mem.get("recent_prompts", [])[-4:],
        "diagnosis": mem.get("diagnosis", ""),
    }


@router.post("/stop")
async def stop():
    global agent_process
    if agent_process is not None:
        try:
            os.killpg(os.getpgid(agent_process.pid), signal.SIGTERM)
        except Exception:
            try:
                agent_process.terminate()
            except Exception:
                pass
        agent_process = None
        return {"status": "stopped"}
    return {"status": "no_process"}


def _ensure_no_other_vram_job():
    """Refuse to overlap with a /api/generate diffusion job. The generate and
    agent paths both evict the IA/tag servers through the SAME in-process
    VramManager; two overlapping evictions would overwrite each other's state
    and both subprocesses would fight over VRAM."""
    try:
        import api.generate as _gen
        gp = getattr(_gen, "current_process", None)
        if gp is not None and gp.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="A generation (T2I/I2I) is already running: stop it first.",
            )
    except HTTPException:
        raise
    except Exception:
        pass


def _stop_tagging_subprocess_for_external():
    """Mirror api/generate.py: stop any active auto-tag SUBPROCESS that could
    still hold VRAM (it is separate from the tag llama-server, which
    prepare_for_external_vram() already stopped)."""
    try:
        from api.tags import tagging_in_progress, tagging_stop_flag, tagging_interrupted
        import workers.tag_worker
        if not tagging_in_progress:
            return
        tagging_stop_flag = True
        workers.tag_worker.tagging_stop_flag = True
        tagging_interrupted = True
        active_proc = workers.tag_worker.active_tagging_process
        if active_proc and active_proc.poll() is None:
            try:
                active_proc.kill()
                active_proc.wait(timeout=10)
            except Exception:
                pass
        try:
            from api.llama_server_manager import get_vram_manager
            cp = dict(get_vram_manager().checkpoint or {})
            cp["status"] = "interrupted_by_agent_refine"
            get_vram_manager().save_checkpoint(cp)
        except Exception:
            pass
    except Exception:
        pass


def _ref_images_json(images) -> Optional[str]:
    """Serialize the I2T/I2I image slots into the worker's --ref-images-json,
    resolving /outputs/ URLs to disk paths and keeping only path+role.
    Accepts both request AgentImageSlot objects and plain {'path','role'} dicts
    (used when rebuilding the set from a run manifest)."""
    if not images:
        return None
    items = []
    for im in images:
        path = im.path if hasattr(im, "path") else (im or {}).get("path")
        role = (im.role if hasattr(im, "role") else (im or {}).get("role")) or "ref"
        path = _resolve_image_path(path)
        if path and role and role != "none":
            items.append({"path": path, "role": role})
    return json.dumps(items, ensure_ascii=False) if items else None


def _write_agent_lora_config(loras) -> Optional[str]:
    """Writes the LoRA config file for the agent worker (same format as T2I/I2I).
    Uses its own path so a T2I/I2I run cannot clobber the agent's file.
    Accepts both request LoraItem objects and plain {'path','strength'} dicts
    (used when rebuilding the set from a run manifest)."""
    if not loras:
        return None
    try:
        import lora_manager
        data = []
        for l in loras:
            if isinstance(l, dict):
                data.append({"path": l.get("path"),
                             "strength": l.get("strength", l.get("weight", 1.0))})
            else:
                data.append({"path": l.path, "strength": l.weight})
        return lora_manager.create_config_json(data, save_path="/app/lora_run_config_agent.json")
    except Exception as e:
        print(f"--> [Agent] LoRA config failed: {e}", flush=True)
        return None


def _lora_data_from_manifest(m: dict) -> list:
    """Rebuild the [{'path','strength'}] LoRA list from a run manifest's 'loras'
    field (the T2I/I2I shape {"path": dir, "files": [[name, strength, active],
    ...]}) so a Retry/Resume reproduces the exact LoRAs of the run even when the
    temporary agent config file was overwritten by a later run."""
    loras = m.get("loras")
    if not loras:
        return []
    if isinstance(loras, list):  # legacy array shape
        return [{"path": l.get("path", ""),
                 "strength": l.get("strength", l.get("weight", 1.0))}
                for l in loras if isinstance(l, dict) and l.get("path")]
    base = str(loras.get("path") or "").rstrip("/")
    out = []
    for f in (loras.get("files") or []):
        if not isinstance(f, (list, tuple)) or not f:
            continue
        name = f[0]
        strength = f[1] if len(f) > 1 else 1.0
        active = f[2] if len(f) > 2 else True
        if name and active is not False:
            out.append({"path": (base + "/" + name) if base else name,
                        "strength": strength})
    return out


def _build_cmd(req: AgentRequest) -> List[str]:
    python = sys.executable
    # clamp the critic temperature to a sane range
    temp = req.critic_temperature if req.critic_temperature is not None else 0.5
    temp = min(max(float(temp), 0.1), 2.0)
    seed_mode = (req.seed_mode or "fixed").strip().lower()
    if seed_mode not in ("fixed", "step", "random"):
        seed_mode = "fixed"
    seed_step = max(1, int(req.seed_step or 1))
    cmd = [
        python, "-u", "workers/agent_refine.py",
        "--goal", sanitize_prompt(req.goal),
        "--schedule", req.schedule or "384x512,512x640,768x768",
        "--seed", str(req.seed),
        "--seed-mode", seed_mode,
        "--steps", str(req.steps),
        "--guidance", str(req.guidance),
        "--scheduler", req.scheduler or "euler",
        "--critic-temperature", str(temp),
    ]
    if req.scheduler_type:
        cmd.extend(["--scheduler-type", str(req.scheduler_type)])
    if req.vae_path:
        cmd.extend(["--vae-path", str(req.vae_path)])
    if req.llm_path:
        cmd.extend(["--llm-path", str(req.llm_path)])
    lora_cfg = _write_agent_lora_config(req.loras)
    if lora_cfg:
        cmd.extend(["--lora-config", lora_cfg])
    if seed_mode == "step":
        cmd.extend(["--seed-step", str(seed_step)])
    if req.explore_when_stuck:
        cmd.append("--explore-when-stuck")
    if req.two_phase_critic:
        cmd.append("--two-phase-critic")
    if req.image:
        cmd.extend(["--image", _resolve_image_path(req.image)])
    ref_json = _ref_images_json(req.images)
    if ref_json:
        cmd.extend(["--ref-images-json", ref_json])
    if req.critic:
        cmd.extend(["--critic-model", req.critic])
    if req.interpreter:
        cmd.extend(["--interpreter-model", req.interpreter])
    if req.diffusion:
        cmd.extend(["--diffusion-model", req.diffusion])

    # --- reference-image controls ---------------------------------------
    # Interpret image: let the VLM critic see the reference.
    cmd.extend(["--use-image", "true" if req.use_image else "false"])
    # Edit from reference: only forwarded when the selected generator can honour
    # it (defence in depth — the UI already greys it out).
    edit_ok = bool(req.edit_from_reference)
    if edit_ok:
        try:
            from core.edit_capabilities import detect_edit
            cap = detect_edit(req.diffusion or "")
            if not cap.get("edit"):
                print(f"--> [Agent] edit_from_reference ignored: {cap.get('reason')}", flush=True)
                edit_ok = False
        except Exception as e:                                # noqa: BLE001
            print(f"--> [Agent] edit capability check failed ({e}); disabling reference edit", flush=True)
            edit_ok = False
    if edit_ok:
        cmd.append("--edit-from-reference")
        cmd.extend(["--edit-anchor", str(req.edit_anchor or "hybrid")])
    if req.iterations:
        cmd.extend(["--iterations", str(req.iterations)])
    if req.negative_prompt:
        cmd.extend(["--negative-prompt", sanitize_prompt(req.negative_prompt)])
    if req.early_stop:
        cmd.append("--early-stop")
    return cmd


def _stream_cmd(cmd: List[str], start_msg: str, do_evict: bool = True):
    """Run the agent worker subprocess and stream its AGENT_EVENT lines as SSE.
    Shared by /refine (new run) and /resume (continue a finished run).

    VRAM eviction happens HERE, in the API process - never inside the worker
    (a separate interpreter whose VramManager cannot see the app's servers).
    Same pattern as api/generate.py: stop IA (8001) + tag (8002) servers,
    stop any active auto-tag subprocess, spawn the worker, and restore the
    previously-running servers in a finally once the stream ends.
    """

    async def event_gen():
        global agent_process
        proc = None
        start_t = time.time()
        finished_ok = False  # a FINISH event was streamed (manifest written)
        err_streamed = False  # a FAIL/ERR event was already streamed
        vram = None
        vram_state = None
        if do_evict:
            try:
                from api.llama_server_manager import get_vram_manager
                vram = get_vram_manager()
                vram_state = vram.prepare_for_external_vram("agent_refine")
                print(
                    f"[Agent] VRAM evicted for run: IA_was_running={vram_state.get('ia_was_running')}, "
                    f"Tag_was_running={vram_state.get('tag_was_running')}",
                    flush=True,
                )
            except Exception as e:
                print(f"[Agent] Could not evict app servers: {e}", flush=True)
                vram = None
                vram_state = None
            _stop_tagging_subprocess_for_external()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
            agent_process = proc
            yield f"data: LOG|{start_msg}\n\n"
            assert proc.stdout is not None
            while True:
                try:
                    line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=2.0)
                except asyncio.TimeoutError:
                    if proc.returncode is not None:
                        break
                    yield "data: PING\n\n"
                    continue
                if not line_bytes:
                    break
                text = line_bytes.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                if text.startswith("AGENT_EVENT|"):
                    try:
                        ev = json.loads(text.split("|", 1)[1])
                        kind = ev.get("kind", "")
                        data = ev.get("data", "")
                        if kind == "img":
                            yield f"data: IMG|{_to_url(data)}\n\n"
                        elif kind == "orig":
                            # original image copied into the run folder
                            yield f"data: ORIG|{_to_url(data)}\n\n"
                        elif kind == "finish":
                            finished_ok = True
                            yield f"data: FINISH|{_to_url(data)}\n\n"
                        elif kind == "fail":
                            # Worker aborted with no image produced: report the
                            # reason to the UI (it hides Retry/export buttons).
                            err_streamed = True
                            yield f"data: ERR|{data}\n\n"
                        elif kind == "iter":
                            # The worker emits a dict {i, total, w, h}; forward a
                            # compact "i/total" string so the UI can display it
                            # and derive the gallery label (iter N) via split('/').
                            if isinstance(data, dict):
                                yield f"data: ITER|{data.get('i', '?')}/{data.get('total', '?')}\n\n"
                            else:
                                yield f"data: ITER|{data}\n\n"
                        elif kind in ("score", "strategy"):
                            # exploration telemetry (dict) forwarded as JSON
                            yield f"data: {kind.upper()}|{json.dumps(data, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {kind.upper()}|{data}\n\n"
                    except Exception:
                        yield f"data: LOG|{text}\n\n"
                    continue
                # generic worker line
                elapsed = time.time() - start_t
                yield f"data: LOG|{elapsed:07.1f}s {text}\n\n"

            # EOF on the pipe can arrive a tick before asyncio reaps the child
            # (SIGCHLD handling), leaving proc.returncode None. Wait for the real
            # status so we never report "exited with code None".
            rc = proc.returncode
            if rc is None:
                try:
                    rc = await asyncio.wait_for(proc.wait(), timeout=20)
                except asyncio.TimeoutError:
                    rc = proc.returncode

            if finished_ok:
                # The worker already wrote the manifest and streamed FINISH:
                # the run completed regardless of the exit code bookkeeping.
                yield f"data: DONE|Agent run finished in {time.time() - start_t:.1f}s\n\n"
            elif err_streamed:
                # A FAIL event already carried the reason to the UI.
                pass
            elif rc == 0:
                yield f"data: DONE|Agent run finished in {time.time() - start_t:.1f}s\n\n"
            elif rc is None:
                yield f"data: ERR|Agent process terminated without an exit code\n\n"
            elif rc < 0:
                yield f"data: ERR|Agent run stopped (signal {-rc})\n\n"
            else:
                yield f"data: ERR|Agent process exited with code {rc}\n\n"
        except Exception as e:
            yield f"data: ERR|{str(e)}\n\n"
        finally:
            agent_process = None
            # Restore the web app's IA/tag servers that were evicted for the
            # whole run, and tell the frontend to auto-resume auto-tagging if
            # it was running before (same signalling as api/generate.py).
            if vram is not None and vram_state is not None:
                try:
                    vram.restore_external_vram(vram_state)
                    if vram_state.get("tag_was_running"):
                        try:
                            resume_from = int((vram.checkpoint or {}).get("current", 0))
                            vram.mark_auto_resume_pending(resume_from)
                        except Exception:
                            pass
                    print("[Agent] IA/tag servers restored", flush=True)
                except Exception as e:
                    print(f"[Agent] Error restoring servers: {e}", flush=True)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _normalize_out_dir(out_dir: str) -> str:
    """Accept a disk path (/app/outputs/...) or the browser URL form
    (/outputs/...) and return the disk path. Mirrors _resolve_image_path()."""
    if out_dir.startswith("/outputs/"):
        out_dir = out_dir.replace("/outputs/", OUTPUT_DIR + "/", 1)
    return os.path.realpath(out_dir)


def _validate_agent_dir(out_dir: str):
    """Validate that out_dir is an agent_refine_* folder inside OUTPUT_DIR and
    return it normalized to disk; raise HTTPException otherwise."""
    out_dir = _normalize_out_dir(out_dir)
    ok = out_dir.startswith(os.path.realpath(OUTPUT_DIR) + os.sep) and \
        os.path.basename(out_dir).startswith("agent_refine_") and \
        os.path.isdir(out_dir)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid out_dir (must be an agent_refine_* folder inside outputs)")
    return out_dir


@router.post("/resume")
async def resume(req: ResumeRequest):
    """Continue a finished agent run in place: the worker loads the previous
    manifest, keeps its prompt/image, and runs one (or more) additional
    critique+generate iteration at a resolution bumped by +256 px on the
    longer side. Returns the same SSE stream as /refine."""
    global agent_process

    if agent_process is not None and agent_process.poll() is None:
        raise HTTPException(status_code=409, detail="An agent run is already in progress")
    _ensure_no_other_vram_job()

    out_dir = _validate_agent_dir(req.out_dir)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise HTTPException(status_code=400, detail="No completed run found in this folder (manifest.json missing)")
    try:
        with open(manifest_path) as f:
            m = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read manifest: {e}")

    # last completed image -> base resolution for the +256 bump
    iters = m.get("iterations") or []
    last = next((it for it in reversed(iters) if it.get("image")), None)
    if not last:
        detail = "The run produced no image (generation failed). It cannot be " \
                 "continued: change the diffusion model and start a new run."
        raise HTTPException(status_code=400, detail=detail)
    if not last.get("resolution"):
        raise HTTPException(status_code=400, detail="Bad resolution in manifest: " + str(last.get("resolution")))
    try:
        w0, h0 = (int(x) for x in last["resolution"].split("x"))
    except Exception:
        raise HTTPException(status_code=400, detail=f"Bad resolution in manifest: {last.get('resolution')}")

    from workers.agent_refine import grow_resolution
    delta = req.delta
    if delta is None:
        delta = 256
    w1, h1 = grow_resolution(w0, h0, add=int(delta))

    python = sys.executable
    cmd = [
        python, "-u", "workers/agent_refine.py",
        "--resume-from", out_dir,
        "--goal", sanitize_prompt(m.get("goal") or ""),
        "--schedule", f"{w1}x{h1}",
        "--iterations", str(max(1, req.iterations)),
        "--seed", str(m.get("seed", 42)),
        "--seed-mode", str(m.get("seed_mode") or "fixed"),
        "--steps", str(m.get("steps", 12)),
        "--guidance", str(m.get("guidance", 2.0)),
        "--scheduler", m.get("scheduler") or "euler",
        "--critic-temperature", str(m.get("critic_temperature", 0.5) or 0.5),
    ]
    seed_mode = str(m.get("seed_mode") or "fixed").lower()
    if seed_mode == "step":
        cmd.extend(["--seed-step", str(m.get("seed_step", 1) or 1)])
    if m.get("explore_when_stuck"):
        cmd.append("--explore-when-stuck")
    if m.get("two_phase_critic"):
        cmd.append("--two-phase-critic")
    if m.get("original_image"):
        cmd.extend(["--image", m["original_image"]])
    # Reference-image controls carry over from the run manifest
    if m.get("use_image") is False:
        cmd.extend(["--use-image", "false"])
    if m.get("edit_from_reference"):
        cmd.append("--edit-from-reference")
        cmd.extend(["--edit-anchor", str(m.get("edit_anchor") or "hybrid")])
    if m.get("critic_model"):
        cmd.extend(["--critic-model", m["critic_model"]])
    if m.get("interpreter_model"):
        cmd.extend(["--interpreter-model", m["interpreter_model"]])
    if m.get("diffusion_model"):
        cmd.extend(["--diffusion-model", m["diffusion_model"]])
    if m.get("vae_path"):
        cmd.extend(["--vae-path", m["vae_path"]])
    if m.get("llm_path"):
        cmd.extend(["--llm-path", m["llm_path"]])
    if m.get("scheduler_type"):
        cmd.extend(["--scheduler-type", m["scheduler_type"]])
    # Extra reference images: prefer the current UI slots (restored when the run
    # was opened from History), then the set stored in the run manifest.
    ref_json = _ref_images_json(req.images) or _ref_images_json(m.get("ref_images"))
    if ref_json:
        cmd.extend(["--ref-images-json", ref_json])
    # LoRAs: prefer the current UI selection (restored from the opened ARF step),
    # then the exact set stored in the run manifest, then the legacy temp config.
    lora_cfg = _write_agent_lora_config(req.loras) if req.loras else None
    if not lora_cfg:
        lora_cfg = _write_agent_lora_config(_lora_data_from_manifest(m))
    if lora_cfg:
        cmd.extend(["--lora-config", lora_cfg])
    elif m.get("lora_config") and os.path.exists(m["lora_config"]):
        cmd.extend(["--lora-config", m["lora_config"]])
    if m.get("negative_prompt"):
        cmd.extend(["--negative-prompt", sanitize_prompt(m["negative_prompt"])])
    # NOTE: no --early-stop here: a Retry must always produce the new image
    # even when the critic considers the goal already reached.
    if int(delta) > 0:
        mode_txt = "enlarging"
    elif int(delta) < 0:
        mode_txt = "shrinking"
    else:
        mode_txt = "keeping the same resolution"

    return _stream_cmd(cmd, f"🔄 Retry ({mode_txt}): {w0}x{h0} -> {w1}x{h1}")

@router.post("/refine")
async def refine(req: AgentRequest):
    """Run the agentic loop as an SSE stream."""
    global agent_process

    if agent_process is not None and agent_process.poll() is None:
        raise HTTPException(status_code=409, detail="An agent run is already in progress")
    _ensure_no_other_vram_job()

    cmd = _build_cmd(req)
    return _stream_cmd(cmd, "🪄 Agentic refine started", do_evict=not req.no_evict)

