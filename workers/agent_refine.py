#!/usr/bin/env python3
"""
M1 - Agentic iterative refinement loop (headless).

Idea: use a VLM (I2T, llama-server) as a *critic* to iteratively rewrite the
prompt of a NON-editing T2I model, so that repeated generation converges the
output toward an edit goal -- without a dedicated edit model.

SCOPE / HONESTY: this is a *prompt-refiner with full re-sampling*. The
original image is never fed pixel-wise to the diffusion model: it enters only
as a text description rewritten by the VLM. The loop converges well for
attribute-level edits (colors, style, large elements, mood); for structural
edits (faces, logos, precise layout, small objects) use the dedicated I2I /
Qwen-Image-Edit workers instead (same output can be forwarded from this UI).

Pipeline per iteration (coarse-to-fine resolution schedule):
    iter 0      : VLM sees the ORIGINAL image (if any) -> writes a detailed
                  positive prompt that recreates it AND applies the GOAL.
    iter n > 0  : VLM sees ORIGINAL (if any) + LAST GENERATION -> critiques
                  adherence/fidelity/artifacts -> writes a corrected prompt.
    then        : T2I worker subprocess generates at the current resolution
                  with the corrected prompt and the run's seed strategy
                  (--seed-mode fixed|step|random, see iter_seed()).

VRAM: the critic server and the diffusion model never coexist on the 24 GB
GPU. When launched through the web UI, the eviction of the app's IA/tag
llama-servers is done by api/agent.py IN THE API PROCESS (the worker is a
separate interpreter: its own VramManager singleton cannot see or stop the
app servers, so evicting from inside the worker would be a silent no-op).
Standalone CLI runs simply have no app servers. The critic server is started
only for each critique and stopped before every generation, then the worker
polls rocm-smi until VRAM is actually free before loading the diffusion
model.

Artifacts per iteration (prompt, critique, image, json) are saved under
OUTPUT_DIR/agent_refine_<timestamp>/.

Usage (inside the container, cwd=/app):
    python -u workers/agent_refine.py --goal "change the boy into a girl" \
        --image /app/outputs/source.png \
        --critic-model qwen3vl \
        --diffusion-model /app/models/checkpoints/Z-Image-GGUF/z-image-turbo-Q4_K.gguf \
        --schedule "512x768,768x1024,1024x1024" --seed 42

    python -u workers/agent_refine.py --list-critics

    # Continue a finished run in place: +256 px on the longer side automatically
    python -u workers/agent_refine.py --resume-from /app/outputs/agent_refine_<ts>
    # or explicit next size for the continuation:
    python -u workers/agent_refine.py --resume-from /app/outputs/agent_refine_<ts> \
        --schedule "1280x1280" --iterations 1

    # Escape local minima: when the critic's adherence score stops improving,
    # the loop diagnoses the stuck direction, picks a different strategy from a
    # menu and rewrites the prompt (anti-repeat memory persists in the run
    # folder: explore_state.json + explore_log.md). Needs >= 3 scored rounds to
    # trigger, so it usually kicks in from iteration 4+ or on a Retry.
    python -u workers/agent_refine.py --goal "..." --explore-when-stuck \
        --stagnation-window 2 --stuck-score-threshold 7 --max-explores 3
"""
import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())

import shared_utils

# The critic reuses the app's IA llama-server singleton (port per params.json).


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def log(msg):
    print(f"[AGENT] {msg}", flush=True)


def emit(kind, data):
    """Emit a machine-readable one-line event for the SSE bridge (api/agent.py).
    Kind: start|iter|prompt|goal|img|finish. Harmless when run standalone."""
    import json as _json
    try:
        print(f"AGENT_EVENT|{_json.dumps({'kind': kind, 'data': data}, ensure_ascii=False)}", flush=True)
    except Exception:
        pass


def load_loras_for_manifest(config_path):
    """Read the LoRA worker config file ([[path, strength, active], ...]) and
    convert it to the SAME {"path": dir, "files": [[filename, strength,
    active], ...]} structure that T2I/I2I store in their JSON sidecars, so the
    History/ARF panels can restore the exact LoRAs used for a generated image."""
    if not config_path or not os.path.exists(config_path):
        return shared_utils.format_loras_for_json([])
    try:
        with open(config_path, "r") as f:
            raw = json.load(f)
        return shared_utils.format_loras_for_json(raw)
    except Exception as e:
        log(f"WARNING: could not read LoRA config for manifest: {e}")
        return shared_utils.format_loras_for_json([])


def parse_ref_images(spec):
    """Parse the extra reference images sent by the UI (the SAME I2T/I2I image
    slots) into a list of (disk_path, role). `spec` is a JSON list of
    {"path": ..., "role": ...} (or an already-parsed list); missing files are
    skipped with a warning."""
    if not spec:
        return []
    try:
        items = json.loads(spec) if isinstance(spec, str) else spec
    except Exception as e:
        log(f"WARNING: bad --ref-images-json: {e}")
        return []
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        raw = it.get("path") or ""
        role = it.get("role") or "ref"
        if not raw:
            continue
        path = raw
        if path.startswith("/outputs/"):
            path = path.replace("/outputs/", shared_utils.OUTPUT_DIR + "/", 1)
        elif not os.path.isabs(path):
            cand = os.path.join(shared_utils.OUTPUT_DIR, path)
            path = cand if os.path.exists(cand) else os.path.join(os.getcwd(), path)
        if os.path.exists(path):
            out.append((path, role))
        else:
            log(f"WARNING: reference image not found, skipped: {raw}")
    return out


REF_ROLE_LABELS = {
    "init": "init image",
    "ref": "style/content reference",
    "face": "face source",
    "mask": "mask",
    "control": "control/pose guide",
    "blend": "blend source",
}


def extra_refs_note(image_paths):
    """Note appended to the critic prompts when the user supplied extra I2T/I2I
    reference images, so the VLM integrates those characteristics."""
    extras = [lbl for (_p, lbl) in image_paths if lbl not in ("original", "generated")]
    if not extras:
        return ""
    roles = ", ".join(REF_ROLE_LABELS.get(e, e) for e in extras)
    return ("\nAdditional reference images are provided by the user (" + roles +
            "); use them to integrate those characteristics into the prompt "
            "without overriding the GOAL.\n")


def find_critic_dirs():
    """Return list of i2t model dirs that contain a GGUF + a vision projector."""
    from api.llama_server_manager import find_gguf_model
    base = shared_utils.I2T_DIR
    out = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        try:
            gguf, mmproj, _fn = find_gguf_model(d)
            if gguf and mmproj:
                out.append(d)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Critic (VLM) preference - best-first, SINGLE SOURCE OF TRUTH.
#
# The critique VLM is stopped before every generation and restarted on the next
# iteration (it must free the VRAM to the diffusion model), so its size is a
# per-iteration cost. The list is ordered by "quality of critique / cost": a
# model matches if its directory name OR its main GGUF filename matches one of
# these patterns; anything unlisted comes last, alphabetically.
#
#   1. SenseNova-SI    Qwen3-VL-8B fine-tuned on SPATIAL INTELLIGENCE: best at
#                      judging WHERE/WHAT changed (pose, layout, geometry) and
#                      the same ~6 GB footprint as the old default -> DEFAULT.
#   2. Qwen3.6-35B-A3B MoE with only ~3B active params: finer judgement and
#                      uncensored (no refusals), but heavier to load each step.
#   3. Qwen3-VL-8B     previous fast default (abliterated).
#   4. Qwen3.6-27B / Qwen3.8 / Muse-Glimmer / Gemma4 / AgentWorld : fallbacks.
# ---------------------------------------------------------------------------
CRITIC_PREFERENCE = [
    r"sensenova",
    r"qwen3\.?6.*35b.*a3b",
    r"qwen3vl|qwen3[-_ ]?vl",
    r"qwen3\.?6.*27b",
    r"qwen3\.?8",
    r"muse.*glimmer",
    r"gemma-?4",
    r"agentworld",
]


def _critic_key(model_dir: str) -> str:
    """Directory basename + main GGUF filename, lowercased (patterns match either)."""
    key = os.path.basename(model_dir)
    try:
        from api.llama_server_manager import find_gguf_model
        gguf, _mmproj, _fn = find_gguf_model(model_dir)
        if gguf:
            key += " " + os.path.basename(gguf)
    except Exception:
        pass
    return key.lower()


def configured_critic_patterns():
    """Extra preference patterns from settings.json -> "agent_critic_prefer".

    Accepts a string ("sensenova", or "qwen3.6.*35b.*a3b, gemma-?4") or a list of
    strings. They are tried BEFORE CRITIC_PREFERENCE, so one line in settings.json
    promotes a critic to the default without touching the code. An invalid regex is
    used as a plain (case-insensitive) substring, so it never breaks the ranking.
    """
    try:
        from core.config import get_setting
        raw = get_setting("agent_critic_prefer", "") or ""
    except Exception:
        return []
    if isinstance(raw, str):
        parts = [x.strip() for x in re.split(r"[,;]", raw) if x.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(x).strip() for x in raw if str(x).strip()]
    else:
        return []
    return parts


def critic_preference_patterns():
    """Effective patterns, configured ones first, then the built-in order."""
    return configured_critic_patterns() + CRITIC_PREFERENCE


def rank_critic_dirs():
    """find_critic_dirs() ordered best-first.

    Order: patterns from settings.json ("agent_critic_prefer"), then the built-in
    CRITIC_PREFERENCE, then anything unlisted (alphabetically).
    """
    dirs = find_critic_dirs()
    patterns = critic_preference_patterns()

    def rank(d):
        key = _critic_key(d)
        base = os.path.basename(d).lower()
        for i, pat in enumerate(patterns):
            try:
                if re.search(pat, key, re.IGNORECASE):
                    return (i, base)
            except re.error:
                if pat.lower() in key:          # plain substring fallback
                    return (i, base)
        return (len(patterns), base)

    return sorted(dirs, key=rank)


def find_default_critic():
    """Best available critique VLM = first entry of the ranked list."""
    ranked = rank_critic_dirs()
    return ranked[0] if ranked else None


def find_default_diffusion():
    """Fast, known-good turbo model first, then generic fallbacks."""
    candidates = [
        # underscore names are the real files in Z-Image-GGUF/ (dashed name does not exist)
        "/app/models/checkpoints/Z-Image-GGUF/z_image_turbo-Q8_0.gguf",
        "/app/models/checkpoints/Z-Image-GGUF/z_image_turbo-Q5_0.gguf",
        "/app/models/checkpoints/Z-Image-GGUF/z-image-turbo-Q8_0.gguf",
        "/app/models/checkpoints/flux/gguf/flux1-dev-Q8_0.gguf",
        "/app/models/checkpoints/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf",
        "/app/models/checkpoints/Ideogram4/ideogram4-Q4_0.gguf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Any GGUF checkpoint (prefer non-qwen-edit, non-uncond, real model files)
    ck = shared_utils.CHECKPOINTS_DIR
    for pat in ("*.gguf", "**/*.gguf"):
        for f in sorted(glob.glob(os.path.join(ck, pat), recursive=True)):
            b = os.path.basename(f).lower()
            if ("qwen-image-edit" in b or "qwen-rapid" in b or "_uncond" in b
                    or "mmproj" in b or b.endswith(".is_fixed.json")):
                continue
            return f
    return None


def parse_schedule(spec):
    """'512x512,768x768' -> [(512,512),(768,768)]; also accept '512' shorthand."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower().replace(" ", "")
        if not tok:
            continue
        if "x" in tok:
            w, h = tok.split("x", 1)
            out.append((int(w), int(h)))
        else:
            s = int(tok)
            out.append((s, s))
    return out


def grow_resolution(w, h, add=256, max_side=2048, step=64, min_side=384):
    """Move to a size 'add' px away on the LONGER side.
    add>0 = enlarge, add<0 = shrink, add=0 = keep the same size.

    Aspect ratio is preserved as closely as the [min_side, max_side] grid
    (multiples of 'step') allows:
      1. long side is stepped+clamped;
      2. short side is derived from it, then stepped+clamped;
      3. if the short-side clamp broke the ratio (floor cases), the long
         side is re-derived from the clamped short side - but only when that
         keeps moving in the requested direction. E.g. 512x768 -256 becomes
         384x576 (exact 2:3) instead of the old 384x512 (+12.5% drift).
    Returns (w, h) unchanged when the move cannot be represented (already at
    a bound, or shrink would push the short side under min_side with no
    in-bounds ratio-consistent target)."""
    if w <= 0 or h <= 0:
        w = h = max(min_side, step * 4)
    if add == 0:
        return (w, h)
    landscape = w >= h
    long_side = max(w, h)
    short_side = min(w, h)

    def _step(v):
        v = int(round(v / step) * step)
        return min(max(v, min_side), max_side)

    ratio = short_side / long_side  # <= 1
    new_long = _step(long_side + add)
    new_short = _step(new_long * ratio)
    # Floor cases: re-derive the long side from the clamped short side when
    # it keeps moving in the requested direction (better fidelity).
    re_long = _step(new_short / ratio)
    if (add > 0 and re_long > long_side and re_long <= max_side) or \
       (add < 0 and re_long < long_side):
        new_long = re_long
        new_short = _step(new_long * ratio)
    # Monotonicity guard: never report growth for a shrink (and vice versa).
    if (add > 0 and new_long <= long_side) or (add < 0 and new_long >= long_side):
        return (w, h)
    return (new_long, new_short) if landscape else (new_short, new_long)


def load_resume_state(resume_dir):
    """Load the manifest of a finished run so the loop can continue in the same
    folder: iteration numbering, last prompt and last image carry over."""
    resume_dir = os.path.realpath(resume_dir)
    manifest_path = os.path.join(resume_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError(f"no manifest.json in {resume_dir}")
    with open(manifest_path, "r") as f:
        prev = json.load(f)
    iters = prev.get("iterations") or []
    if not iters:
        raise ValueError(f"manifest in {resume_dir} has no iterations")
    last_img = next((it for it in reversed(iters) if it.get("image")), None)
    last_prompt = next((it.get("prompt") for it in reversed(iters) if it.get("prompt")), None)
    last_res = None
    if last_img:
        last_res = last_img.get("resolution") or prev.get("resolution")
    return {
        "manifest": prev,
        "iterations": iters,
        "start": len(iters),
        "last_prompt": last_prompt,
        "last_res": last_res,
        "goal": prev.get("goal"),
        "original_image": prev.get("original_image"),
        "critic_model": prev.get("critic_model"),
        "diffusion_model": prev.get("diffusion_model"),
        "seed": prev.get("seed", 42),
        "seed_mode": prev.get("seed_mode", "fixed"),
        "seed_step": prev.get("seed_step", 1),
        "critic_temperature": prev.get("critic_temperature", 0.5),
        "explore_when_stuck": prev.get("explore_when_stuck", False),
        "stagnation_window": prev.get("stagnation_window", 2),
        "stuck_score_threshold": prev.get("stuck_score_threshold", 7),
        "max_explores": prev.get("max_explores", 3),
        "two_phase_critic": prev.get("two_phase_critic", False),
        "score_gate": prev.get("score_gate", 5.0),
        "steps": prev.get("steps", 12),
        "guidance": prev.get("guidance", 2.0),
        "scheduler": prev.get("scheduler", "euler"),
        "vae_path": prev.get("vae_path"),
        "llm_path": prev.get("llm_path"),
        "scheduler_type": prev.get("scheduler_type"),
        "lora_config": prev.get("lora_config"),
        "negative_prompt": prev.get("negative_prompt", ""),
    }


def wait_free_vram(min_mb=2000, timeout=120):
    """Poll rocm-smi until at least min_mb VRAM is free (ROCm releases slowly)."""
    from api.llama_server_manager import get_free_vram_mb
    t0 = time.time()
    while time.time() - t0 < timeout:
        free = get_free_vram_mb()
        if free >= min_mb:
            return free
        time.sleep(3)
    return get_free_vram_mb()


def env_with_rocm():
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    if "/opt/rocm/lib" not in ld:
        env["LD_LIBRARY_PATH"] = ("/opt/rocm/lib:" + ld).rstrip(":")
    return env


def iter_seed(mode, base, step, i):
    """Return the effective seed for iteration 'i' (absolute counter, so resume
    keeps the same trajectory):
      - mode 'fixed'  -> always the base seed (default, reproducible)
      - mode 'step'   -> base + i*step  (deterministic variation per iteration)
      - mode 'random' -> deterministic per (base, i) pseudo-random variation,
                         so a run can be reproduced from the same base seed.
    All results stay in the non-negative 31-bit range used by the worker."""
    if mode == "random":
        return random.Random(f"{base}:{i}").randrange(0, 2**31 - 1)
    if mode == "step":
        return (base + i * max(1, int(step))) % (2**31 - 1)
    return base


# --------------------------------------------------------------------------
# Critic (VLM) orchestration
# --------------------------------------------------------------------------

def start_critic(model_dir):
    # Use the app's IA server manager pattern (port from the model params.json,
    # e.g. 8001). When launched through the web UI the caller (api/agent.py)
    # has already evicted the app's IA/tag servers from the API process, so
    # port 8001 is free; standalone CLI runs have no app servers at all.
    # ensure_server() starts a fresh llama-server on the model's configured
    # port - exactly the pattern the app's own generate/i2t flow uses.
    from api.llama_server_manager import get_server_manager
    mgr = get_server_manager()
    log(f"Starting critic (IA llama-server) with model dir: {model_dir}")
    port = mgr.ensure_server(model_dir)
    return mgr, port


def ask_critic(mgr, image_paths, prompt, temperature=0.5, max_tokens=1400,
               system_prompt=None):
    """Blocking critique call. image_paths: list[(path, role)]."""
    image_specs = [{"path": p, "role": r} for (p, r) in image_paths] or None
    res = mgr.query(
        image_path=None,
        image_specs=image_specs,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=False,
        system_prompt=(system_prompt or (
            "You are an expert prompt engineer and art director for diffusion "
            "image editing. You reply in the same language as the user's request."
        )),
    )
    return res


# --------------------------------------------------------------------------
# Interpreter: a SEPARATE vision model (specialised / uncensored) is asked to
# describe the ORIGINAL image once. Its text becomes a stable anchor inside the
# critic prompt, so the critic keeps SEEING the images (architecture C) while
# gaining an objective textual reference of what must be preserved.
# --------------------------------------------------------------------------
INTERPRETER_INSTRUCTION = """You are an objective, literal image describer.
Describe the provided image exhaustively and factually: subjects (count, age,
gender presentation, ethnicity if visible), pose, expression, gaze, hairstyle
and hair colour, clothing and accessories, colours, body/proportions,
foreground/background, setting, lighting, camera angle, framing, depth of
field, and any visible text (transcribe it exactly).
Rules: describe ONLY what is visible; no interpretation, no evaluation, no
style advice, no prompt writing. Plain descriptive text.
"""
INTERPRETER_SYSTEM = ("You are a precise, neutral computer-vision describer. "
                      "You never refuse and only report what is visible.")


def run_interpreter(interpreter_dir, image_path):
    """Describe *image_path* with a separate VLM and return the text.

    The server is started and stopped here so it never competes with the critic
    or the diffusion model for VRAM (same policy as the critic itself)."""
    mgr = None
    try:
        log(f"Starting interpreter (VLM): {os.path.basename(interpreter_dir)}")
        mgr, _port = start_critic(interpreter_dir)
        res = ask_critic(mgr, [(image_path, "original")], INTERPRETER_INSTRUCTION,
                         temperature=0.2, max_tokens=900, system_prompt=INTERPRETER_SYSTEM)
        text = (res.get("response") or res.get("raw_output") or "").strip()
        return text
    except Exception as e:                                    # noqa: BLE001
        log(f"WARNING: interpreter failed: {e}")
        return ""
    finally:
        if mgr is not None:
            try:
                mgr.stop()
            except Exception:
                pass


SCORING_RUBRIC = """SCORING RUBRIC (calibration - be strict, avoid optimistic 6-9 defaults):
  ADHERENCE_SCORE (is the GOAL actually applied in the image?):
    0-2 = edit absent or wrong element entirely
    3-4 = barely perceptible attempt, wrong area or very partial
    5-6 = applied in part with clear errors (wrong/partial element)
    7-8 = clearly applied, only minor imperfections
    9-10 = fully applied, no perceptible flaw
  FIDELITY_SCORE (is the ORIGINAL preserved where the GOAL does not change it?):
    0-2 = output unrecognizable vs the ORIGINAL
    3-4 = lost much structure/identity/detail
    5-6 = overall structure kept, but colors/style/details drifted
    7-8 = well preserved, small deviations
    9-10 = identical where the GOAL does not require changes
Do not inflate scores: most imperfect results belong in 3-6, and a 7+ must
really look right. When in doubt, rate DOWN."""


def _ref_desc_block(reference_description):
    """Optional block injecting the description produced by a SEPARATE
    interpreter model (specialised/uncensored) about the ORIGINAL image.
    It gives the critic a stable textual anchor of what must be preserved."""
    if not reference_description:
        return ""
    return ("\nREFERENCE DESCRIPTION (objective description of the ORIGINAL image, "
            "produced by a separate vision model; preserve these details and do "
            "NOT contradict them):\n" + reference_description.strip() + "\n")


def build_critique_prompt(goal, image_paths, last_prompt=None, iteration=0,
                          language_hint="", ask_rating=True,
                          reference_description=""):
    """Template for the local prompt-refinement pass (generator/rewriter).

    ask_rating=True  -> the rewriter also rates the CURRENT last generation
                        (single-call critic, used by default).
    ask_rating=False -> two-phase mode: this call only produces the new prompt;
                        the rating is delegated to a separate reviewer call
                        (build_review_prompt) to remove self-confirmation bias.
    """
    lang = language_hint or "the language of the GOAL"
    if iteration == 0 and image_paths:
        return f"""The user wants to EDIT the ORIGINAL image (Image 1).

GOAL: "{goal}"
{_ref_desc_block(reference_description)}{extra_refs_note(image_paths)}
STEP 1 - ANALYZE. Describe Image 1 precisely: subject, pose, framing, camera
angle, background, colors, lighting, style, composition and any text/details.

STEP 2 - REWRITE. Produce ONE detailed positive prompt (no labels, natural
phrasing, comma separated) for a text-to-image model that:
  - RECREATES Image 1's structure and style as faithfully as possible,
  - AND applies the GOAL, changing ONLY what the goal requires.

Output exactly in {lang}:

ANALYSIS:
<short analysis>

FINAL_PROMPT:
<the single-block positive prompt>"""
    if iteration == 0:
        # No original image: go straight from the goal.
        return f"""The user wants to generate an image from a creative GOAL.

GOAL: "{goal}"
{_ref_desc_block(reference_description)}
Write ONE detailed positive prompt for a text-to-image model that expresses
the GOAL with rich, concrete visual detail (subject, pose, framing, background,
lighting, colors, style, mood). Natural phrasing, no labels.

Output exactly in {lang}:

FINAL_PROMPT:
<the single-block positive prompt>"""
    # iteration > 0: compare original vs last generation
    if len(image_paths) >= 2:
        cmp = ("Image 1 is the ORIGINAL reference, Image 2 is the LAST "
               "GENERATION produced from your previous prompt.")
        past = f"\nYour previous prompt was:\n\"{last_prompt}\"\n" if last_prompt else ""
    else:
        cmp = "Image 1 is the LAST GENERATION produced from your previous prompt."
        past = f"\nYour previous prompt was:\n\"{last_prompt}\"\n" if last_prompt else ""
    past += extra_refs_note(image_paths)
    if not ask_rating:
        # two-phase: only rewrite, no rating here (reviewer rates separately)
        return f"""The user wants to EDIT an image with GOAL: "{goal}".

{cmp}{past}{_ref_desc_block(reference_description)}
STEP 1 - EVALUATE (be strict):
  (a) Adherence: does the last generation actually apply the GOAL?
  (b) Fidelity: does it preserve the original's composition/style/identity
      (colors, pose, framing) where the GOAL does not require changes?
  (c) Artifacts / quality issues.
  Name the SINGLE biggest gap to fix.

STEP 2 - REWRITE. Produce ONE improved positive prompt (natural phrasing, no
labels) that fixes that biggest gap while keeping what already works.

Do NOT rate the image: another evaluator will judge the result.

Output exactly in {lang}:

EVALUATION:
<brief evaluation>

FINAL_PROMPT:
<the single-block improved positive prompt>"""
    return f"""The user wants to EDIT an image with GOAL: "{goal}".

{cmp}{past}{_ref_desc_block(reference_description)}
STEP 1 - EVALUATE (be strict):
  (a) Adherence: does the last generation actually apply the GOAL?
  (b) Fidelity: does it preserve the original's composition/style/identity
      (colors, pose, framing) where the GOAL does not require changes?
  (c) Artifacts / quality issues.
  Name the SINGLE biggest gap to fix.

STEP 2 - REWRITE. Produce ONE improved positive prompt (natural phrasing, no
labels) that fixes that biggest gap while keeping what already works.

STEP 3 - RATE the current generation with TWO numbers 0-10.

{SCORING_RUBRIC}

Output exactly in {lang}:

EVALUATION:
<brief evaluation>

GOAL_REACHED: yes/no

ADHERENCE_SCORE: N/10
FIDELITY_SCORE: N/10

FINAL_PROMPT:
<the single-block improved positive prompt>"""


def build_review_prompt(goal, image_paths, language_hint="", reference_description=""):
    """Independent reviewer (two-phase critic): given ONLY the images and the
    GOAL - NOT the prompt that produced the last generation - it rates the
    result. Separating the judge from the generator removes the confirmation
    bias of a critic that rewrites a prompt and then validates its own output.
    """
    lang = language_hint or "the language of the GOAL"
    if len(image_paths) >= 2:
        cmp = ("Image 1 is the ORIGINAL reference, Image 2 is the image to judge "
               "(the last generation).")
    else:
        cmp = "Image 1 is the image to judge (the last generation)."
    note = extra_refs_note(image_paths)
    return f"""You are a STRICT independent quality reviewer for an image-EDIT task.

{cmp}{note}
GOAL: "{goal}"
{_ref_desc_block(reference_description)}
Rate how well the image to judge satisfies the GOAL - and nothing else is
available to you: you do NOT see the prompt that generated it, judge only by
looking.

{SCORING_RUBRIC}

Be severe and honest: an image that merely resembles the goal or keeps most of
its flaws must NOT get a high score.

Output exactly in {lang}:

EVALUATION:
<1-3 sentences, naming the concrete failures>

GOAL_REACHED: yes/no

ADHERENCE_SCORE: N/10
FIDELITY_SCORE: N/10
"""


def parse_critic_output(text):
    """Extract FINAL_PROMPT block and GOAL_REACHED from the critic response."""
    prompt, goal_reached = _extract_final_prompt_and_goal(text)
    return prompt, goal_reached


def _extract_final_prompt_and_goal(text):
    """Shared extraction used by parse_critic_output and parse_critic_detail."""
    goal_reached = None
    m = re.search(r"GOAL_REACHED:\s*(yes|no)", text, re.IGNORECASE)
    if m:
        goal_reached = m.group(1).lower() == "yes"
    prompt = None
    idx = text.rfind("FINAL_PROMPT:")
    if idx != -1:
        prompt = text[idx + len("FINAL_PROMPT:"):].strip()
        # Drop trailing lines that belong to the next answer (safety)
        prompt = re.sub(r"\n\s*(EVALUATION|GOAL_REACHED|ADHERENCE_SCORE|FIDELITY_SCORE|STRATEGY|DIAGNOSIS|ANALYSIS|EXPLORATION|CANDIDATE):.*$",
                        "", prompt, flags=re.DOTALL)
        prompt = prompt.strip().strip('"').strip()
    if not prompt:
        prompt = text.strip().strip('"')
    return prompt, goal_reached


def _parse_adherence(text):
    """Read ADHERENCE_SCORE (accepts '7', '7/10', '6.5', '7.5/10'). None if absent."""
    return _parse_score_value(text, "ADHERENCE_SCORE")


def _parse_fidelity(text):
    """Read FIDELITY_SCORE with the same tolerant format as the adherence."""
    return _parse_score_value(text, "FIDELITY_SCORE")


def _parse_score_value(text, label):
    m = re.search(label + r"\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*/?\s*10", text, re.IGNORECASE)
    if not m:
        m = re.search(label + r"\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not m:
        return None
    try:
        return int(round(float(m.group(1))))
    except Exception:
        return None


def parse_critic_detail(text):
    """Rich parse of a critic answer: prompt, goal_reached, numeric adherence
    and fidelity scores, chosen explore strategy and (optionally) diagnosis."""
    prompt, goal_reached = _extract_final_prompt_and_goal(text)
    strategy = None
    m = re.search(r"STRATEGY\s*[:=]\s*([A-Za-z0-9_\- ]+)", text)
    if m:
        strategy = m.group(1).strip().lower().replace(" ", "_")
    diagnosis = None
    m = re.search(r"DIAGNOSIS\s*:\s*\n?(.*?)(?:\n\s*(?:STRATEGY|ADHERENCE_SCORE|FIDELITY_SCORE|GOAL_REACHED|FINAL_PROMPT|ANALYSIS|EVALUATION)\s*:)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        diagnosis = m.group(1).strip()
    return {
        "prompt": prompt,
        "goal_reached": goal_reached,
        "adherence": _parse_adherence(text),
        "fidelity": _parse_fidelity(text),
        "strategy": strategy,
        "diagnosis": diagnosis,
    }


# --------------------------------------------------------------------------
# Exploration memory & stagnation detection (escape local minima)
# --------------------------------------------------------------------------
# When the critic's numeric adherence score stops improving for several
# iterations, the loop switches from local prompt-refinement ('exploit') to an
# 'explore' mode: the critic must diagnose the stuck direction, pick a
# STRUCTURALLY DIFFERENT strategy from a menu, and rewrite the prompt without
# reusing already-tried strategies. A small memory file in the run folder
# (explore_state.json) + a human-readable log (explore_log.md) persist this
# across iterations and across Retry/Resume of the same run.

EXPLORE_STRATEGIES = [
    "rewrite_photographic",
    "scene_recompose",
    "vector_shift",
    "simplify_goal",
    "anchor_swap",
    "style_framing",
    "reground_original",
]

EXPLORE_MENU = """- rewrite_photographic: drop abstract wording; express the edit with concrete,
  photographic/cinematic vocabulary the image model knows well.
- scene_recompose: rethink the whole scene / composition / layout so the GOAL
  becomes the organizing idea of the image.
- vector_shift: change the 'edit vector' (e.g. work on light/color/mood instead
  of geometry/subject details, or vice versa).
- simplify_goal: drop or relax constraints that conflict with each other or
  with fidelity; keep only the core intent.
- anchor_swap: keep the intent but change the main subject / point of view / framing.
- style_framing: express the edit as an explicit style / medium / era / genre filter.
- reground_original: (special, picked automatically) fidelity to the ORIGINAL
  collapsed - rebuild the prompt from the ORIGINAL description and re-apply the
  GOAL from scratch, instead of continuing from a drifted generation."""


def empty_memory():
    return {
        "version": 2,
        "strategies": [],          # [{label, rounds, best_score, best_fidelity, last_i}]
        "recent_prompts": [],      # capped, [{i, mode, strategy, score, fidelity, prompt}]
        "explore_count": 0,
        "diagnosis": "",
        "best": None,              # {"score", "prompt", "iter", "mode"} - best prompt so far
    }


def load_memory(out_dir):
    try:
        with open(os.path.join(out_dir, "explore_state.json")) as f:
            m = json.load(f)
        m.setdefault("strategies", [])
        m.setdefault("recent_prompts", [])
        m.setdefault("explore_count", 0)
        m.setdefault("diagnosis", "")
        m.setdefault("best", None)
        m.setdefault("version", 2)
        # cap strategies (they are written back on the next save)
        if len(m["strategies"]) > 8:
            m["strategies"] = m["strategies"][-8:]
        return m
    except Exception:
        return empty_memory()


def save_memory(out_dir, mem):
    try:
        with open(os.path.join(out_dir, "explore_state.json"), "w") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def log_memory(out_dir, line):
    """Append a line to the human-readable exploration log of this run."""
    try:
        with open(os.path.join(out_dir, "explore_log.md"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def composite_score(adherence, fidelity, fidelity_gate=5):
    """Combine adherence and fidelity into ONE number for decision making
    (stagnation / best / early stop), so a result that fully applies the GOAL
    but drifted away from the ORIGINAL is never rewarded.

    - adherence None -> None
    - fidelity None  -> adherence (no drift information)
    - fidelity >= fidelity_gate -> adherence (edit is trusted)
    - fidelity <  fidelity_gate -> adherence scaled down by fidelity/gate:
        drift penalizes the score without erasing it entirely.
    """
    if adherence is None:
        return None
    if fidelity is None:
        return int(round(adherence))
    if fidelity >= fidelity_gate:
        return int(round(adherence))
    val = adherence * (float(fidelity) / float(fidelity_gate))
    return max(0, int(round(val)))


def _update_best(mem, i, score, prompt, mode, adherence=None, fidelity=None):
    """Track the highest-COMPOSITE-scoring prompt of the run (elitism): explore
    restarts and summaries branch from this point. Drifted high-adherence
    prompts are not kept as 'best' because the comparison uses the composite
    score (see composite_score)."""
    if score is None:
        return
    cur = mem.get("best")
    if cur is None or score > cur.get("score", -1):
        entry = {"score": score, "prompt": (prompt or "")[:1000],
                 "iter": i, "mode": mode or "exploit"}
        if adherence is not None:
            entry["adherence"] = adherence
        if fidelity is not None:
            entry["fidelity"] = fidelity
        mem["best"] = entry


def remember_outcome(mem, i, mode, strategy, score, prompt, fidelity=None, diagnosis="",
                     adherence=None, fidelity_gate=5):
    """Record an iteration outcome in the exploration memory. 'score' should be
    the COMPOSITE score (decision-grade); adherence/fidelity are kept for
    display and per-strategy tracking."""
    _update_best(mem, i, score, prompt, mode, adherence=adherence, fidelity=fidelity)
    if mode in ("explore", "reground") and strategy:
        found = next((s for s in mem["strategies"] if s.get("label") == strategy), None)
        if found is None:
            found = {"label": strategy, "rounds": 0, "best_score": None,
                     "best_fidelity": None, "last_i": i}
            mem["strategies"].append(found)
        found["rounds"] = found.get("rounds", 0) + 1
        if score is not None:
            prev = found.get("best_score")
            found["best_score"] = score if prev is None else max(prev, score)
        if fidelity is not None:
            pf = found.get("best_fidelity")
            found["best_fidelity"] = fidelity if pf is None else max(pf, fidelity)
        found["last_i"] = i
        mem["explore_count"] = mem.get("explore_count", 0) + 1
        if len(mem["strategies"]) > 8:   # keep only the most recent ones
            mem["strategies"] = sorted(mem["strategies"], key=lambda s: s.get("last_i", 0))[-8:]
    if i > 0 and (score is not None or fidelity is not None):
        entry = {
            "i": i, "mode": mode, "strategy": strategy,
            "score": score, "fidelity": fidelity,
            "prompt": (prompt or "")[:600],
        }
        if adherence is not None:
            entry["adherence"] = adherence
        mem["recent_prompts"].append(entry)
        if len(mem["recent_prompts"]) > 6:
            mem["recent_prompts"] = mem["recent_prompts"][-6:]
    if diagnosis:
        mem["diagnosis"] = diagnosis[:800]


def backfill_memory_from_manifest(memory, manifest, max_recent=4, fidelity_gate=5):
    """Seed the exploration memory from the run manifest when it is missing or
    was created before the v2 fields existed. Used on Resume so the experience
    (best prompt, recent scored outcomes) carries across retries. Best and
    recent scores use the COMPOSITE of adherence+fidelity."""
    scored = [it for it in manifest if it.get("adherence") is not None or it.get("fidelity") is not None]
    if memory.get("best") is None and scored:
        best = None
        for it in scored:
            a = it.get("adherence")
            f = it.get("fidelity")
            if a is not None:
                comp = composite_score(a, f, fidelity_gate)
                if best is None or comp > best.get("score", -1):
                    best = {"score": comp, "prompt": it.get("prompt") or "",
                            "iter": it.get("iter", 0), "mode": it.get("mode") or "exploit",
                            "adherence": a}
                    if f is not None:
                        best["fidelity"] = f
        memory["best"] = best
    if not memory.get("recent_prompts") and scored:
        for it in scored[-max_recent:]:
            a = it.get("adherence")
            f = it.get("fidelity")
            comp = composite_score(a, f, fidelity_gate) if a is not None else None
            entry = {
                "i": it.get("iter", 0), "mode": it.get("mode") or "exploit",
                "strategy": it.get("strategy"),
                "score": comp, "fidelity": f,
                "prompt": (it.get("prompt") or "")[:600],
            }
            if a is not None:
                entry["adherence"] = a
            memory["recent_prompts"].append(entry)
    return memory


def memory_strategy_block(mem):
    lines = []
    for s in mem["strategies"][-4:]:
        extra = f" fidelity {s.get('best_fidelity', '?')}" if s.get("best_fidelity") is not None else ""
        lines.append(f"  - {s.get('label')} (tried {s.get('rounds', 1)}x, best score {s.get('best_score', '?')}{extra})")
    return "\n".join(lines) if lines else "  (none yet)"


def memory_prompt_block(mem):
    lines = []
    for r in mem["recent_prompts"][-3:]:
        tag = ("EXPLORE:" + str(r.get("strategy"))) if r.get("mode") == "explore" else \
              ("REGROUND" if r.get("mode") == "reground" else "refine")
        pr = (r.get("prompt") or "").replace("\n", " ")[:140]
        sc = r.get("score")
        fid = r.get("fidelity")
        adv = r.get("adherence")
        vals = []
        if adv is not None: vals.append(f"adherence={adv}")
        if fid is not None: vals.append(f"fidelity={fid}")
        if sc is not None and sc != adv: vals.append(f"combined={sc}")
        suffix = (" [" + ", ".join(vals) + "]") if vals else ""
        lines.append(f"  - [iter {r.get('i')}][{tag}]{suffix} :: {pr}")
    return "\n".join(lines) if lines else "  (none yet)"


def best_prompt_block(mem):
    b = mem.get("best")
    if not b or not b.get("prompt"):
        return ""
    pr = (b["prompt"] or "").replace("\n", " ")[:240]
    return (f"\nBEST PROMPT SO FAR (score {b.get('score')}/10 at iter {b.get('iter')}):\n"
            f"\"{pr}\"\nPrefer to branch from this prompt when it still makes sense; "
            "it is the strongest known point of this search.\n")


def should_explore(scores, window=2, threshold=7):
    """True when the numeric adherence scores have stopped improving:
    best score over the whole run is below the threshold AND the best of the
    last 'window' scores does not beat the best of everything before it.
    Needs at least window+1 scored iterations. None entries disable the check.
    Pass COMPOSITE scores (see composite_score) so drift keeps triggering it."""
    if len(scores) <= window:
        return False
    last = scores[-window:]
    if any(s is None for s in last):
        return False
    valid = [s for s in scores if s is not None]
    if not valid:
        return False
    if max(valid) >= threshold:
        return False
    head = scores[:-window]
    if not head or any(s is None for s in head):
        return False
    return max(last) <= max(head)


def decide_action(adherence_scores, fidelity_scores=None, window=2, threshold=7,
                  fidelity_floor=4):
    """One-call policy used by the loop on every critique (explore enabled):
      - 'exploit'  -> keep local prompt-refinement (default)
      - 'reground' -> fidelity to the ORIGINAL collapsed: rebuild the prompt
                      from the ORIGINAL description instead of the drifted last
                      generation.
      - 'explore'  -> structural restart: diagnose + pick a different strategy.
    'reground' wins over 'explore' when the last fidelity is below the floor,
    because a drifted image poisons every local rewrite."""
    if not should_explore(adherence_scores, window, threshold):
        return "exploit"
    if fidelity_scores:
        last_f = [s for s in fidelity_scores if s is not None]
        if last_f and last_f[-1] < fidelity_floor:
            return "reground"
    return "explore"


def build_explore_prompt(goal, image_paths, memory, current_prompt=None,
                         current_score=None, last_fidelity=None,
                         best_prompt=None, language_hint="",
                         reference_description=""):
    """Prompt that redirects the search when the exploit loop is stuck."""
    lang = language_hint or "the language of the GOAL"
    if len(image_paths) >= 2:
        cmp = ("Image 1 is the ORIGINAL reference, Image 2 is the LAST "
               "GENERATION produced from the current direction.")
    else:
        cmp = "Image 1 is the LAST GENERATION produced from the current direction."
    past = f"\nCurrent prompt:\n\"{current_prompt}\"\n" if current_prompt else ""
    past += extra_refs_note(image_paths)
    last_sc = ""
    if current_score is not None:
        last_sc += f"\nLast critic adherence score: {current_score}/10\n"
    if last_fidelity is not None:
        last_sc += f"Last critic fidelity score : {last_fidelity}/10\n"
    best = ""
    if best_prompt:
        pr = (best_prompt or "").replace("\n", " ")[:240]
        best = (f"\nBEST PROMPT SO FAR (highest adherence in this run):\n"
                f"\"{pr}\"\nIf it is still a sound basis, branch from IT rather than "
                "from a worse current prompt.\n")
    return f"""The loop is STUCK while trying to EDIT an image with GOAL: "{goal}".

{cmp}{past}{last_sc}{best}{_ref_desc_block(reference_description)}
Small refinements of the previous prompt are no longer improving the result.
You must propose a STRUCTURALLY DIFFERENT approach.

STRATEGY MENU - pick the ONE most promising label that has NOT already been tried:
{EXPLORE_MENU}

ALREADY TRIED strategies (do NOT reuse them):
{memory_strategy_block(memory)}

Recent prompts & their scores (do NOT drift back toward these dead ends):
{memory_prompt_block(memory)}

STEP 1 - DIAGNOSIS: say briefly why the current direction is stuck (goal
ambiguity, vocabulary mismatch, conflicting constraints, fidelity vs edit
conflict, model limitation...).

STEP 2 - choose ONE strategy label from the menu.

STEP 3 - REWRITE. Produce ONE new positive prompt (single block, natural
phrasing, no labels) that implements the chosen strategy and gives the goal a
fresh chance. It must clearly differ from the recent prompts above.

{SCORING_RUBRIC}

Output exactly in {lang}:

DIAGNOSIS:
<2-4 sentences>

STRATEGY: <one label from the menu>

ADHERENCE_SCORE: <strict 0-10 rating of the CURRENT last generation>
FIDELITY_SCORE: <strict 0-10 rating of how well the ORIGINAL is preserved>

FINAL_PROMPT:
<the single-block new prompt>"""


def build_reground_prompt(goal, image_paths, memory, current_prompt=None,
                          best_prompt=None, language_hint="",
                          reference_description=""):
    """Rebuild from the ORIGINAL: the search drifted (low fidelity), so every
    local rewrite is poisoned. The critic must ignore the last generation and
    describe the ORIGINAL again, then apply the GOAL - while the memory keeps
    it away from the dead ends already tried."""
    lang = language_hint or "the language of the GOAL"
    if len(image_paths) >= 2:
        src = ("Image 1 is the ORIGINAL reference. Image 2 is the LAST "
               "GENERATION - IGNORE it for the rewrite (it drifted too far "
               "from the ORIGINAL).")
    else:
        src = "Image 1 is the LAST GENERATION. It has drifted; ignore its defects."
    past = f"\nCurrent (drifted) prompt:\n\"{current_prompt}\"\n" if current_prompt else ""
    past += extra_refs_note(image_paths)
    best = best_prompt_block(memory) or (
        ("\nBEST PROMPT SO FAR:\n\"" + best_prompt.replace("\n", " ") + "\"\n") if best_prompt else "")
    return f"""The EDIT has drifted: the generation lost fidelity to the ORIGINAL
while chasing the GOAL "{goal}". You must RE-GROUND.

{src}{past}{best}{_ref_desc_block(reference_description)}
ALREADY TRIED strategies (do NOT reuse them):
{memory_strategy_block(memory)}

Recent prompts & their scores (dead ends to avoid):
{memory_prompt_block(memory)}

STEP 1 - ANALYZE the ORIGINAL precisely (subject, pose, framing, composition,
colors, lighting, style, identity).

STEP 2 - REWRITE from scratch, as if seeing the ORIGINAL for the first time:
produce ONE positive prompt that RECREATES the ORIGINAL faithfully AND applies
the GOAL, changing only what the goal requires. Do not inherit wording from
the drifted prompt above.

{SCORING_RUBRIC}

Output exactly in {lang}:

DIAGNOSIS:
<why the drift happened>

STRATEGY: reground_original

ADHERENCE_SCORE: <strict 0-10 of the CURRENT last generation>
FIDELITY_SCORE: <strict 0-10 of the CURRENT last generation>

FINAL_PROMPT:
<the single-block re-grounded prompt>"""


def write_explore_summary(out_dir, memory, goal):
    """Write a concise, human-readable summary of the exploration so the
    'experience' of this run is inspectable and reusable (explore_summary.md)."""
    if not memory.get("explore_count") and not memory.get("best"):
        return
    b = memory.get("best") or {}
    lines = [
        "# Agent Refine - exploration summary",
        "",
        f"**Goal:** {goal}",
        f"**Explore rounds:** {memory.get('explore_count', 0)}",
        "",
        "## Strategies tried (best adherence per strategy)",
    ]
    for st in sorted(memory.get("strategies", []), key=lambda x: -(x.get("best_score") or -1)):
        lines.append(f"- `{st.get('label')}` x{st.get('rounds', 1)} "
                     f"-> best adherence {st.get('best_score')}/10"
                     + (f", fidelity {st.get('best_fidelity')}/10" if st.get("best_fidelity") is not None else ""))
    lines.append("")
    lines.append("## Best prompt found")
    if b.get("prompt"):
        lines.append(f"Score {b.get('score')}/10 (iter {b.get('iter')}, mode {b.get('mode')}):")
        lines.append("")
        lines.append("```")
        lines.append(b["prompt"])
        lines.append("```")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Last diagnosis")
    lines.append(memory.get("diagnosis") or "(none)")
    lines.append("")
    try:
        with open(os.path.join(out_dir, "explore_summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass


# --------------------------------------------------------------------------
# Generation (T2I worker subprocess)
# --------------------------------------------------------------------------

def run_generation(prompt, width, height, steps, guidance, seed, model_path, scheduler,
                   negative_prompt, out_png, vae_path=None, llm_path=None,
                   scheduler_type="auto", lora_config=None,
                   ref_images=None, edit_family=None, edit_only=False):
    """Spawn the right worker for ONE iteration and stream its log.

    * edit-only models (Qwen-Image-Edit family) -> workers/i2i_edit.py
    * everything else                            -> workers/t2i_cpp.py
      (+ `--ref_image` for reference-capable generators)

    Returns (path_or_None, last_error_text)."""
    python = sys.executable
    ref_images = [p for p in (ref_images or []) if p and os.path.exists(p)]
    use_edit_worker = bool(edit_only) or (edit_family == "qwen_image_edit")
    tag = "edit" if use_edit_worker else "t2i"

    if use_edit_worker:
        if not ref_images:
            log("ERROR: this generator is edit-only and requires a reference image")
            return None, "edit-only generator requires a reference image"
        cmd = [
            python, "-u", "workers/i2i_edit.py",
            "--prompt", prompt,
            "--init_img", ref_images[0],
            "--width", str(width),
            "--height", str(height),
            "--steps", str(steps),
            "--guidance", str(guidance),
            "--seed", str(seed),
            "--scheduler", scheduler,
            "--model_path", model_path,
            "--vae_tiling", "true",
        ]
        if negative_prompt:
            cmd.extend(["--negative_prompt", negative_prompt])
        if vae_path:
            cmd.extend(["--vae_path", vae_path])
        if scheduler_type:
            cmd.extend(["--scheduler_type", scheduler_type])
    else:
        cmd = [
            python, "-u", "workers/t2i_cpp.py",
            "--prompt", prompt,
            "--width", str(width),
            "--height", str(height),
            "--steps", str(steps),
            "--guidance", str(guidance),
            "--seed", str(seed),
            "--model_path", model_path,
            "--scheduler", scheduler,
            "--scheduler_type", scheduler_type,
            "--vae_tiling", "true",
        ]
        if negative_prompt:
            cmd.extend(["--negative_prompt", negative_prompt])
        if vae_path:
            cmd.extend(["--vae_path", vae_path])
        if llm_path:
            cmd.extend(["--llm_path", llm_path])
        if lora_config:
            cmd.extend(["--lora_config", lora_config])
        for r in ref_images:
            cmd.extend(["--ref_image", r])

    log(f"Generating {width}x{height} (steps={steps}, cfg={guidance}, seed={seed})")
    if ref_images:
        log("  reference(s): " + ", ".join(os.path.basename(p) for p in ref_images))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=os.getcwd(), env=env_with_rocm())
    success_path = None
    last_err = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        # keep the log compact: only tag the interesting markers
        if line.startswith("SUCCESS_OUTPUT:"):
            success_path = line[len("SUCCESS_OUTPUT:"):].strip()
            log(f"{tag} worker SUCCESS -> {success_path}")
        elif line.startswith("ERROR") or "error" in line.lower()[:120] or "❌" in line:
            log(f"  [{tag}] {line}")
            last_err = line.strip()[:300]
        else:
            log(f"  [{tag}] {line[:200]}")
    proc.wait()
    if proc.returncode != 0 or not success_path or not os.path.exists(success_path):
        log(f"{tag} worker failed (rc={proc.returncode})")
        return None, last_err
    # Copy to our artifact path (worker may have used its own filename)
    import shutil
    shutil.copy2(success_path, out_png)
    # The T2I worker writes into OUTPUT_DIR root (t2i_cpp_*.png + its .json
    # sidecar). Remove them so intermediate steps stay ONLY inside the run
    # folder (agent_refine_<ts>/) and don't flood the outputs root.
    try:
        tmp_png = os.path.abspath(success_path)
        out_abs = os.path.abspath(out_png)
        if os.path.dirname(tmp_png) != os.path.dirname(out_abs):
            for p in (tmp_png, os.path.splitext(tmp_png)[0] + ".json"):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
    except Exception:
        pass
    return out_png, last_err


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Agentic iterative prompt-refinement loop (M1)")
    ap.add_argument("--goal", type=str, default=None, help="Edit goal / intention")
    ap.add_argument("--image", type=str, default=None, help="Original image to edit (optional)")
    ap.add_argument("--ref-images-json", type=str, default=None,
                    help="JSON list of extra reference images [{path, role}] taken "
                         "from the same I2T/I2I image slots; the VLM critic sees "
                         "them and integrates those characteristics")
    ap.add_argument("--critic-model", type=str, default=None,
                    help="I2T model dir (name under models/i2t or full path)")
    ap.add_argument("--interpreter-model", type=str, default=None,
                    help="Optional SEPARATE I2T vision model used once to describe the "
                         "Init Image (specialised / uncensored describer). Its text is "
                         "injected into the critic prompt as a stable reference anchor; "
                         "the critic still looks at the images.")
    ap.add_argument("--diffusion-model", type=str, default=None, help="T2I checkpoint path")
    ap.add_argument("--schedule", type=str, default="512x512,768x768,1024x1024",
                    help="Comma list WxH of increasing resolutions (coarse-to-fine)")
    ap.add_argument("--iterations", type=int, default=None,
                    help="Number of iterations (default: len(schedule))")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-mode", type=str, default="fixed",
                    choices=["fixed", "step", "random"],
                    help="How the seed changes across iterations: fixed (same for "
                         "every step), step (base + i*seed_step), random (deterministic "
                         "per-step pseudo-random from the base seed)")
    ap.add_argument("--seed-step", type=int, default=1,
                    help="Increment applied per iteration when --seed-mode=step")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--scheduler", type=str, default="euler")
    ap.add_argument("--negative-prompt", type=str, default="")
    ap.add_argument("--vae-path", type=str, default=None)
    ap.add_argument("--scheduler-type", type=str, default=None,
                    help="Scheduler type (discrete|continuous|karras|...)")
    ap.add_argument("--llm-path", type=str, default=None,
                    help="Text encoder / LLM GGUF (Qwen-ZImage / Qwen-Image-Edit)")
    ap.add_argument("--use-image", type=lambda x: str(x).lower() == "true", default=True,
                    help="Send the reference image to the VLM critic (interpret image). "
                         "false = text-only critique")
    ap.add_argument("--edit-from-reference", action="store_true",
                    help="Pass the reference image to the generator (sd.cpp -r) so every "
                         "iteration is a real EDIT instead of a text-to-image regeneration")
    ap.add_argument("--edit-anchor", choices=["hybrid", "original", "chain"], default="hybrid",
                    help="Reference used for editing: hybrid = original + previous generation "
                         "(default), original = no drift, chain = previous only")
    ap.add_argument("--lora-config", type=str, default=None,
                    help="JSON config file with the LoRA list (same format as T2I/I2I)")
    ap.add_argument("--critic-temperature", type=float, default=0.5)
    ap.add_argument("--explore-when-stuck", action="store_true",
                    help="When the critic adherence score stops improving, switch to "
                         "explore mode: diagnose the stuck direction and restart with a "
                         "structurally different strategy (escapes local minima)")
    ap.add_argument("--stagnation-window", type=int, default=2,
                    help="Iterations with no score improvement before exploring (default 2)")
    ap.add_argument("--stuck-score-threshold", type=float, default=7.0,
                    help="Best adherence score below which stagnation is considered (default 7)")
    ap.add_argument("--max-explores", type=int, default=3,
                    help="Maximum number of explore (restart) rounds per run (default 3)")
    ap.add_argument("--two-phase-critic", action="store_true",
                    help="Separate the judge from the generator: the rewrite pass only "
                         "writes the prompt, a dedicated reviewer call rates the image "
                         "(removes self-confirmation bias; +1 VLM call per refinement "
                         "iteration)")
    ap.add_argument("--score-gate", type=float, default=5.0,
                    help="Fidelity gate for the composite score: when fidelity drops "
                         "below this value the combined score is scaled down (default 5)")
    ap.add_argument("--critic-max-tokens", type=int, default=1400)
    ap.add_argument("--early-stop", action="store_true",
                    help="Stop as soon as the critic says GOAL_REACHED: yes")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Folder of a finished run (agent_refine_<ts>) to continue in place",
                    dest="resume_from")
    ap.add_argument("--list-critics", action="store_true", help="List available critic model dirs and exit")
    args = ap.parse_args()

    if args.list_critics:
        pref = find_default_critic()
        for d in rank_critic_dirs():
            mark = "   [default]" if pref and d == pref else ""
            print(os.path.basename(d), "->", d + mark)
        return

    # ---- resume mode: continue a finished run in its own folder -----------
    resume_dir = None
    resume_state = None
    resume_start = 0
    if args.resume_from:
        resume_dir = args.resume_from
        if resume_dir.startswith("/outputs/"):
            resume_dir = resume_dir.replace("/outputs/", shared_utils.OUTPUT_DIR + "/", 1)
        try:
            resume_state = load_resume_state(resume_dir)
        except Exception as e:
            log(f"ERROR: cannot resume from {args.resume_from}: {e}")
            sys.exit(1)
        resume_start = resume_state["start"]
        # Fill unset options from the manifest of the previous run, so the API
        # only needs to send the folder + the resolution bump.
        if not args.goal or not args.goal.strip():
            args.goal = resume_state.get("goal")
        if not args.image:
            args.image = resume_state.get("original_image")
        if not args.critic_model:
            args.critic_model = resume_state.get("critic_model")
        if not args.diffusion_model:
            args.diffusion_model = resume_state.get("diffusion_model")
        if args.seed == 42:
            args.seed = int(resume_state.get("seed", 42))
        if args.seed_mode == "fixed":
            args.seed_mode = resume_state.get("seed_mode") or "fixed"
        if args.seed_step == 1:
            args.seed_step = int(resume_state.get("seed_step", 1) or 1)
        if args.critic_temperature == 0.5:
            args.critic_temperature = float(resume_state.get("critic_temperature", 0.5) or 0.5)
        if not args.explore_when_stuck:
            args.explore_when_stuck = bool(resume_state.get("explore_when_stuck", False))
        if args.stagnation_window == 2:
            args.stagnation_window = int(resume_state.get("stagnation_window", 2) or 2)
        if args.stuck_score_threshold == 7.0:
            args.stuck_score_threshold = float(resume_state.get("stuck_score_threshold", 7.0) or 7.0)
        if args.max_explores == 3:
            args.max_explores = int(resume_state.get("max_explores", 3) or 3)
        if not args.two_phase_critic:
            args.two_phase_critic = bool(resume_state.get("two_phase_critic", False))
        if args.score_gate == 5.0:
            args.score_gate = float(resume_state.get("score_gate", 5.0) or 5.0)
        if args.steps == 12:
            args.steps = int(resume_state.get("steps", 12))
        if args.guidance == 2.0:
            args.guidance = float(resume_state.get("guidance", 2.0))
        if args.scheduler == "euler":
            args.scheduler = resume_state.get("scheduler") or "euler"
        if not args.negative_prompt:
            args.negative_prompt = resume_state.get("negative_prompt") or ""
        if not args.vae_path:
            args.vae_path = resume_state.get("vae_path")
        if not args.llm_path:
            args.llm_path = resume_state.get("llm_path")
        if not args.scheduler_type:
            args.scheduler_type = resume_state.get("scheduler_type")
        if not args.lora_config:
            args.lora_config = resume_state.get("lora_config")
        if not args.ref_images_json:
            refs = resume_state["manifest"].get("ref_images") or []
            if refs:
                args.ref_images_json = json.dumps(refs)

    # Extra reference images (the same I2T/I2I slots): the critic sees them
    # alongside the original and the last generation. If no explicit original
    # was passed, the slot with role 'init' becomes the anchor of the edit.
    ref_images = parse_ref_images(args.ref_images_json)
    if ref_images:
        log("Extra reference images: " + ", ".join(
            f"{os.path.basename(p)} [{r}]" for p, r in ref_images))
    if not args.image:
        init_ref = next((p for (p, r) in ref_images if r == "init"), None)
        if init_ref:
            args.image = init_ref
            log(f"Using the 'init' slot as the edit anchor: {init_ref}")
    anchor_abs = os.path.abspath(args.image) if args.image else None
    ref_images = [(p, r) for (p, r) in ref_images
                  if not anchor_abs or os.path.abspath(p) != anchor_abs]

    if not args.goal or not args.goal.strip():
        ap.error("--goal is required (use --list-critics to list critic models)")

    # ---- resolve models ----------------------------------------------------
    critic_dir = args.critic_model
    if critic_dir and not os.path.isabs(critic_dir):
        cand = os.path.join(shared_utils.I2T_DIR, critic_dir)
        if os.path.isdir(cand):
            critic_dir = cand
    if not critic_dir or not os.path.isdir(critic_dir):
        critic_dir = find_default_critic()
    if not critic_dir:
        log("ERROR: no critic model dir with gguf+mmproj found under models/i2t")
        sys.exit(1)

    diffusion = args.diffusion_model
    if not diffusion or not os.path.exists(diffusion):
        diffusion = find_default_diffusion()
    if not diffusion or not os.path.exists(diffusion):
        log("ERROR: no diffusion model found; pass --diffusion-model")
        sys.exit(1)

    # ---- schedule & iteration count ----------------------------------------
    schedule = parse_schedule(args.schedule)
    # In resume mode, a schedule left at the CLI default means "keep going":
    # bump the previous resolution by +256 on the longer side automatically.
    if resume_state and args.schedule == "512x512,768x768,1024x1024":
        schedule = []
    if not schedule:
        if resume_state and resume_state.get("last_res"):
            try:
                w0, h0 = (int(x) for x in resume_state["last_res"].split("x"))
            except Exception:
                w0 = h0 = 0
            schedule = [grow_resolution(w0, h0)]
            log(f"No explicit schedule for resume -> continue at "
                f"{schedule[0][0]}x{schedule[0][1]}")
        if not schedule:
            log("ERROR: empty schedule")
            sys.exit(1)
    n_iter = args.iterations or len(schedule)
    total_iters = resume_start + n_iter

    image_path = args.image
    if image_path and not os.path.exists(image_path):
        if resume_state:
            log(f"WARNING: original image no longer exists; continuing without it: {image_path}")
            image_path = None
            args.image = None
        else:
            log(f"ERROR: image not found: {image_path}")
            sys.exit(1)

    out_dir = args.out_dir or (resume_dir if resume_state else None)
    if not out_dir:
        out_dir = os.path.join(shared_utils.OUTPUT_DIR,
                               f"agent_refine_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    # Prefix every per-iteration artifact with the run folder name so a single
    # file is unambiguously traceable back to its run, e.g.
    #   agent_refine_1789082753/agent_refine_1789082753_iter_000_img.png
    # (the bare 'iter_000_img.png' would be identical in every run).
    # api/agent.py remove-step and the frontend patterns rely on this shape.
    run_name = os.path.basename(os.path.normpath(out_dir))

    # Copy the ORIGINAL reference image into the run folder so the run is
    # self-contained: the original survives cleanup/export and always ships
    # with the agent_refine_<ts> artifacts.
    orig_copy = None
    if image_path and os.path.exists(image_path):
        dest = os.path.join(out_dir, "original.png")
        try:
            if os.path.abspath(image_path) != os.path.abspath(dest):
                import shutil as _sh
                _sh.copy2(image_path, dest)
                image_path = dest
                args.image = dest
                orig_copy = dest
                log(f"Original image copied into run folder: {dest}")
            else:
                orig_copy = dest
        except Exception as e:
            log(f"WARNING: could not copy original into run folder: {e}")

    # ---- reference-edit capability (Agent Refine "Edit" mode) ---------------
    from core.edit_capabilities import detect_edit
    edit_cap = detect_edit(diffusion)
    edit_active = bool(args.edit_from_reference) and bool(edit_cap.get("edit")) and bool(image_path)
    if args.edit_from_reference and not edit_cap.get("edit"):
        log(f"WARNING: reference edit requested but not available: {edit_cap.get('reason')}")
    if args.edit_from_reference and edit_cap.get("edit") and not image_path:
        log("WARNING: reference edit requested but no Init Image provided -> text-to-image.")
    if edit_active:
        _vis = os.path.basename(edit_cap.get("vision_encoder") or "") or "-"
        log(f"Reference-edit ON     : family={edit_cap.get('family')} anchor={args.edit_anchor} "
            f"worker={edit_cap.get('worker')} vision={_vis}")

    # ---- optional separate interpreter (specialised describer) -------------
    # Used ONCE to describe the Init Image: gives the critic a stable textual
    # anchor while the critic keeps seeing the images (architecture C).
    interpreter_dir = args.interpreter_model
    if interpreter_dir and not os.path.isabs(interpreter_dir):
        cand = os.path.join(shared_utils.I2T_DIR, interpreter_dir)
        if os.path.exists(cand):
            interpreter_dir = cand
    if interpreter_dir and not os.path.exists(interpreter_dir):
        log(f"WARNING: interpreter model not found: {args.interpreter_model} -> ignored")
        interpreter_dir = None
    if interpreter_dir:
        # The interpreter MUST be able to read images: require a vision projector.
        # Skip when it is the very same model as the critic (nothing to gain).
        try:
            from api.llama_server_manager import find_gguf_model
            i_gguf, i_mmproj, _i_fn = find_gguf_model(interpreter_dir)
            c_gguf, _c_mmproj, _c_fn = find_gguf_model(critic_dir)
            if not i_mmproj:
                log(f"WARNING: interpreter '{os.path.basename(interpreter_dir)}' has no vision "
                    f"projector (mmproj) -> it cannot read the image. Ignored.")
                interpreter_dir = None
            elif i_gguf and c_gguf and os.path.basename(i_gguf) == os.path.basename(c_gguf):
                log("Interpreter == Critic: the model already sees the image, skipping.")
                interpreter_dir = None
        except Exception as e:                                # noqa: BLE001
            log(f"WARNING: could not validate interpreter ({e})")

    reference_description = ""
    if interpreter_dir and image_path:
        desc_file = os.path.join(out_dir, "original_description.txt")
        if os.path.exists(desc_file):
            try:
                with open(desc_file, encoding="utf-8") as f:
                    reference_description = f.read().strip()
                log("Reusing cached original description (original_description.txt)")
            except Exception:
                reference_description = ""
        if not reference_description:
            reference_description = run_interpreter(interpreter_dir, image_path)
            if reference_description:
                try:
                    with open(desc_file, "w", encoding="utf-8") as f:
                        f.write(reference_description)
                    log(f"Original description saved to original_description.txt "
                        f"({len(reference_description)} chars)")
                except Exception as e:                        # noqa: BLE001
                    log(f"WARNING: could not save description: {e}")
            else:
                log("WARNING: interpreter returned an empty description")

    log("=" * 70)
    log(f"{'🔄 RESUME of' if resume_state else 'GOAL            '}: {args.goal}")
    log(f"Original image  : {image_path or '(none - pure text goal)'}")
    log(f"Critic (VLM)    : {critic_dir}  (vision: {'on' if args.use_image else 'off'})")
    if args.two_phase_critic and not args.use_image:
        log("WARNING: two-phase critic requires the image; critic vision is OFF -> "
            "falling back to single-call ratings (experience/explore still work).")
    log(f"Diffusion model : {diffusion}")
    log(f"Schedule        : {schedule}  (iterations: {n_iter}{', this resume' if resume_state else ''})")
    log(f"Seed            : {args.seed}  (mode: {args.seed_mode}" +
        (f", +{args.seed_step}/step" if args.seed_mode == "step" else "") + ")")
    log(f"Critic temp     : {args.critic_temperature}")
    log(f"Out dir         : {out_dir}")
    log("=" * 70)
    emit("start", {"goal": args.goal, "out_dir": out_dir,
                   "critic": os.path.basename(critic_dir),
                   "diffusion": os.path.basename(diffusion),
                   "iterations": n_iter, "resume": resume_start > 0,
                   "schedule": [f"{w}x{h}" for w, h in schedule]})
    if orig_copy:
        emit("orig", orig_copy)

    # NOTE on VRAM: when launched through the web UI, eviction of the app's
    # IA/tag servers happens in the API process (api/agent.py) BEFORE this
    # worker is spawned - this process is a separate interpreter whose
    # VramManager singleton can never see or stop those servers, so doing it
    # here would have been a silent no-op. Standalone CLI runs have no app
    # servers at all.
    critic_mgr = None
    gen_error = None
    if resume_state:
        manifest = list(resume_state["iterations"])
        current_prompt = resume_state.get("last_prompt")
        log(f"Resuming run: {resume_start} previous iterations loaded from manifest.")
    else:
        manifest = []
        current_prompt = None

    # ---- exploration memory (escape local minima) ---------------------------
    # The memory file lives INSIDE the run folder (explore_state.json + a
    # human-readable explore_log.md + explore_summary.md). It survives
    # Retry/Resume: the loop rebuilds the scored history and the best prompt
    # from the manifest and continues where it stopped instead of re-walking
    # the same dead ends.
    explore_on = bool(getattr(args, "explore_when_stuck", False))
    if os.path.exists(os.path.join(out_dir, "explore_state.json")):
        memory = load_memory(out_dir)
    else:
        memory = empty_memory()
    score_history = [it.get("adherence") for it in manifest]   # per absolute iter
    fidelity_history = [it.get("fidelity") for it in manifest]
    comp_history = [composite_score(a, f, args.score_gate)
                    for a, f in zip(score_history, fidelity_history)]  # decision-grade
    backfill_memory_from_manifest(memory, manifest, fidelity_gate=args.score_gate)
    cooldown_until = 0   # next iteration index where an explore is allowed
    start_all = time.time()
    # LoRAs used by this run, in the T2I/I2I JSON shape. Captured once here:
    # the run uses the same LoRA set for every iteration (a Retry may change it).
    loras_fmt = load_loras_for_manifest(args.lora_config)
    manifest_path = os.path.join(out_dir, "manifest.json")

    def save_manifest():
        """Checkpoint the run manifest to disk (called after every iteration).
        The manifest is normally written only at the END of the run, but a stop
        from the UI kills this worker with SIGTERM, so that final write never
        happens and the folder becomes invisible to History (no manifest.json).
        Persisting a checkpoint after each recorded iteration guarantees the run
        folder stays reachable and resumable. The write is atomic (tmp + rename)
        so an ill-timed kill can never leave truncated JSON behind."""
        total_s = round(time.time() - start_all, 1)
        if resume_state:
            # keep the cumulative time across resume steps
            total_s = round(total_s + resume_state["manifest"].get("total_s", 0), 1)
        payload = {
            "goal": args.goal,
            "original_image": image_path,
            "critic_model": critic_dir,
            "diffusion_model": diffusion,
            "schedule": [f"{w}x{h}" for w, h in schedule],
            "seed": args.seed,
            "seed_mode": args.seed_mode,
            "seed_step": args.seed_step,
            "critic_temperature": args.critic_temperature,
            "explore_when_stuck": args.explore_when_stuck,
            "stagnation_window": args.stagnation_window,
            "stuck_score_threshold": args.stuck_score_threshold,
            "max_explores": args.max_explores,
            "two_phase_critic": args.two_phase_critic,
            "score_gate": args.score_gate,
            "use_image": bool(args.use_image),
            "edit_from_reference": bool(edit_active),
            "edit_anchor": args.edit_anchor if edit_active else None,
            "edit_family": (edit_cap.get("family") if edit_active else None),
            "interpreter_model": (interpreter_dir if interpreter_dir else None),
            "has_reference_description": bool(reference_description),
            "steps": args.steps,
            "guidance": args.guidance,
            "scheduler": args.scheduler,
            "vae_path": args.vae_path,
            "llm_path": args.llm_path,
            "scheduler_type": args.scheduler_type,
            "lora_config": args.lora_config,
            "loras": loras_fmt,
            "ref_images": [{"path": p, "role": r} for (p, r) in ref_images],
            "negative_prompt": args.negative_prompt,
            "total_s": total_s,
            "iterations": manifest,
        }
        tmp_path = manifest_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, manifest_path)
        except Exception as e:
            log(f"WARNING: could not write manifest checkpoint: {e}")

    try:
        for j in range(n_iter):
            i = resume_start + j
            res = schedule[min(j, len(schedule) - 1)]
            W, H = res
            t_iter = time.time()
            log("=" * 70)
            log(f"ITERATION {i + 1}/{total_iters}  ({W}x{H})")
            emit("iter", {"i": i + 1, "total": total_iters, "w": W, "h": H})

            # ---- 1. CRITIQUE -------------------------------------------------
            # Policy per iteration: 'exploit' = local prompt-refinement (default),
            # 'reground' = fidelity collapsed -> rebuild from the ORIGINAL,
            # 'explore' = structural restart (diagnose + different strategy).
            mode_this = "exploit"
            if (explore_on and i >= 2 and i >= cooldown_until
                    and memory.get("explore_count", 0) < args.max_explores):
                # only decide on the valid scored iterations (iter 0 has no vote)
                dec_scores = [c for c in comp_history if c is not None]
                mode_this = decide_action(dec_scores, fidelity_history,
                                          args.stagnation_window,
                                          args.stuck_score_threshold)
                if mode_this != "exploit":
                    cooldown_until = i + 2   # allow one normal iteration in between

            log("--- Critique (VLM)" +
                (" [" + mode_this.upper() + "] ---" if mode_this != "exploit" else " ---"))
            # images the critic should look at (only when interpretation is ON)
            paths = []
            if args.use_image:
                if i == 0:
                    if image_path:
                        paths = [(image_path, "original")]
                else:
                    if image_path:
                        paths = [(image_path, "original")]
                    last_img = os.path.join(out_dir, f"{run_name}_iter_{i - 1:03d}_img.png")
                    if not os.path.exists(last_img):
                        # Backward compatibility: runs created before the folder-name
                        # prefix used the bare "iter_NNN_img.png".
                        legacy = os.path.join(out_dir, f"iter_{i - 1:03d}_img.png")
                        if os.path.exists(legacy):
                            last_img = legacy
                    if os.path.exists(last_img):
                        paths.append((last_img, "generated"))

            # extra I2T/I2I reference images (style / face / ...) for the critic
            for (rp, rr) in ref_images:
                if os.path.exists(rp):
                    paths.append((rp, rr))

            if critic_mgr is None:
                critic_mgr, _ = start_critic(critic_dir)

            last_sc = (score_history[-1] if score_history
                       and score_history[-1] is not None else None)
            last_fid = (fidelity_history[-1] if fidelity_history
                        and fidelity_history[-1] is not None else None)
            best_p = (memory.get("best") or {}).get("prompt")
            # restarting modes need a bit more creativity to diverge
            temp_eff = args.critic_temperature
            if mode_this == "explore":
                temp_eff = min(2.0, temp_eff + 0.3)
            elif mode_this == "reground":
                temp_eff = min(2.0, temp_eff + 0.15)
            max_tok_eff = args.critic_max_tokens
            if mode_this != "exploit":
                max_tok_eff = max(1600, max_tok_eff)

            # two-phase critic: in the plain refinement mode the rewrite and the
            # rating are SEPARATE calls (judge != generator) to remove the
            # self-confirmation bias of "I wrote it, so it must be good".
            # NOTE: the independent reviewer needs the image(s). If critic vision
            # is OFF there are no paths -> fall back to the single call, which
            # still emits ADHERENCE/FIDELITY (otherwise the experience panel and
            # the explore/stuck detection would never get any score).
            two_phase = (bool(args.two_phase_critic) and mode_this == "exploit"
                         and i > 0 and bool(paths))

            goal_reached = None
            adherence = None
            fidelity = None
            strategy_this = None
            diagnosis_this = ""
            if mode_this == "reground":
                crit_prompt = build_reground_prompt(args.goal, paths, memory,
                                                    current_prompt=current_prompt,
                                                    best_prompt=best_p,
                                                    reference_description=reference_description)
            elif mode_this == "explore":
                crit_prompt = build_explore_prompt(
                    args.goal, paths, memory,
                    current_prompt=current_prompt,
                    current_score=last_sc, last_fidelity=last_fid,
                    best_prompt=best_p,
                    reference_description=reference_description)
            elif two_phase:
                crit_prompt = build_critique_prompt(args.goal, paths,
                                                    last_prompt=current_prompt,
                                                    iteration=i, ask_rating=False,
                                                    reference_description=reference_description)
            else:
                crit_prompt = build_critique_prompt(args.goal, paths,
                                                    last_prompt=current_prompt,
                                                    iteration=i,
                                                    reference_description=reference_description)
            res_crit = ask_critic(critic_mgr, paths, crit_prompt,
                                  temperature=temp_eff,
                                  max_tokens=max_tok_eff)
            raw = res_crit.get("response") or res_crit.get("raw_output") or ""
            detail = parse_critic_detail(raw)
            prompt_new = detail["prompt"]
            if not prompt_new or len(prompt_new) < 20:
                log("ERROR: critic returned no usable FINAL_PROMPT; aborting.")
                log(f"Raw response head: {raw[:400]}")
                break
            current_prompt = prompt_new
            if mode_this == "reground":
                # single call: strategy + diagnosis + strict ratings
                goal_reached = detail["goal_reached"]
                adherence = detail["adherence"]
                fidelity = detail["fidelity"]
                strategy_this = detail["strategy"] or "reground_original"
                diagnosis_this = detail.get("diagnosis") or ""
            elif mode_this == "explore":
                goal_reached = detail["goal_reached"]
                adherence = detail["adherence"]
                fidelity = detail["fidelity"]
                strategy_this = detail["strategy"]
                diagnosis_this = detail.get("diagnosis") or ""
            elif not two_phase:
                # single-call plain refinement: the rewrite carries the ratings
                goal_reached = detail["goal_reached"]
                adherence = detail["adherence"]
                fidelity = detail["fidelity"]
            else:
                # two-phase: separate independent reviewer rates the LAST image
                if paths:
                    rev_prompt = build_review_prompt(args.goal, paths,
                                                     reference_description=reference_description)
                    rev_t = min(0.7, args.critic_temperature + 0.1)
                    res_rev = ask_critic(critic_mgr, paths, rev_prompt,
                                         temperature=rev_t,
                                         max_tokens=max_tok_eff)
                    raw_rev = res_rev.get("response") or res_rev.get("raw_output") or ""
                    rev = parse_critic_detail(raw_rev)
                    goal_reached = rev["goal_reached"]
                    adherence = rev["adherence"]
                    fidelity = rev["fidelity"]
                    raw = raw_rev   # critique artifact = the reviewer's verdict
                    log("[critic] two-phase: independent reviewer rated the last image")
                else:
                    log("[critic] two-phase: no image to review yet (iteration 0)")
            score_history.append(adherence)
            fidelity_history.append(fidelity)
            comp = composite_score(adherence, fidelity, args.score_gate)
            comp_history.append(comp)
            log(f"[critic] mode        : {mode_this}" +
                (f" | strategy: {strategy_this}" if strategy_this else ""))
            log(f"[critic] GOAL_REACHED: {goal_reached}")
            log(f"[critic] ADHERENCE   : {adherence}/10" if adherence is not None
                else "[critic] ADHERENCE   : n/a")
            log(f"[critic] FIDELITY    : {fidelity}/10" if fidelity is not None
                else "[critic] FIDELITY    : n/a")
            log(f"[critic] COMBINED    : {comp}/10 (fidelity gate {args.score_gate})" if comp is not None
                else "[critic] COMBINED    : n/a")
            log(f"[critic] FINAL_PROMPT: {current_prompt[:400]}{'...' if len(current_prompt) > 400 else ''}")
            emit("prompt", current_prompt)
            emit("goal", str(goal_reached) if goal_reached is not None else "none")
            emit("score", {"i": i + 1, "adherence": adherence, "fidelity": fidelity,
                            "combined": comp,
                            "goal_reached": goal_reached, "mode": mode_this,
                            "strategy": strategy_this})
            if mode_this != "exploit":
                emit("strategy", {"i": i + 1, "strategy": strategy_this or "?",
                                   "mode": mode_this,
                                   "diagnosis": diagnosis_this[:200]})
            # save critique artifacts
            with open(os.path.join(out_dir, f"{run_name}_iter_{i:03d}_critique.txt"), "w") as f:
                f.write(raw)
            with open(os.path.join(out_dir, f"{run_name}_iter_{i:03d}_prompt.txt"), "w") as f:
                f.write(current_prompt)
            # goal reached before generating? only early-stop when the numeric
            # COMPOSITE score confirms it (avoids stopping on a "yes" at
            # mediocre quality or on a drifted result)
            early_ok = (comp is None) or (comp >= int(round(args.stuck_score_threshold)))
            if goal_reached and args.early_stop and i > 0 and early_ok:
                log("Critic says GOAL_REACHED with a solid score -> early stop.")
                manifest.append({"iter": i, "resolution": f"{W}x{H}",
                                 "prompt": current_prompt, "goal_reached": True,
                                 "adherence": adherence, "fidelity": fidelity,
                                 "combined": comp,
                                 "mode": mode_this,
                                 "strategy": strategy_this,
                                 "loras": loras_fmt,
                                 "image": None, "early_stop": True})
                if explore_on:
                    remember_outcome(memory, i, mode_this, strategy_this, comp,
                                     current_prompt, fidelity=fidelity,
                                     adherence=adherence,
                                     diagnosis=diagnosis_this
                                     if mode_this != "exploit" else "")
                    save_memory(out_dir, memory)
                    log_memory(out_dir, f"### iter {i + 1}: {mode_this.upper()} "
                                       f"strategy={strategy_this} combined={comp} "
                                       f"adherence={adherence} fidelity={fidelity} (early stop)")
                save_manifest()
                break
            elif goal_reached and args.early_stop and i > 0:
                log("Critic says GOAL_REACHED but score is still low "
                    f"({comp}<{int(round(args.stuck_score_threshold))}): keep polishing.")

            # ---- 2. GENERATE -------------------------------------------------
            # critic must free VRAM before the diffusion model loads
            if critic_mgr is not None:
                log("Stopping critic server to free VRAM for generation...")
                critic_mgr.stop()
                critic_mgr = None
            wait_free_vram(min_mb=1500, timeout=180)

            out_png = os.path.join(out_dir, f"{run_name}_iter_{i:03d}_img.png")
            eff_seed = iter_seed(args.seed_mode, args.seed, args.seed_step, i)
            log(f"  seed for this step: {eff_seed}")

            # ---- reference image(s) used for THIS iteration -----------------
            refs_this = []
            if edit_active:
                anchor = args.edit_anchor
                prev_img = None
                if i > 0:
                    cand = os.path.join(out_dir, f"{run_name}_iter_{i - 1:03d}_img.png")
                    if not os.path.exists(cand):
                        cand = os.path.join(out_dir, f"iter_{i - 1:03d}_img.png")
                    if os.path.exists(cand):
                        prev_img = cand
                if anchor == "chain":
                    refs_this = [prev_img or image_path]
                elif anchor == "original":
                    refs_this = [image_path]
                else:  # hybrid
                    if prev_img and edit_cap.get("multi_ref"):
                        refs_this = [image_path, prev_img]
                    else:
                        refs_this = [image_path]
                log(f"  edit anchor '{anchor}' -> " +
                    ", ".join(os.path.basename(p) for p in refs_this))

            out, last_err = run_generation(current_prompt, W, H, args.steps, args.guidance, eff_seed,
                                           diffusion, args.scheduler, args.negative_prompt, out_png,
                                           vae_path=args.vae_path,
                                           llm_path=args.llm_path,
                                           scheduler_type=args.scheduler_type or "auto",
                                           lora_config=args.lora_config,
                                           ref_images=refs_this,
                                           edit_family=edit_cap.get("family"),
                                           edit_only=edit_cap.get("edit_only"))
            if not out:
                err_detail = last_err or f"T2I worker exited with code (see log)"
                log(f"❌ Generation failed at iteration {i + 1}:")
                log(f"   {err_detail}")
                manifest.append({"iter": i, "resolution": f"{W}x{H}",
                                 "prompt": current_prompt, "image": None,
                                 "seed": eff_seed,
                                 "adherence": adherence, "fidelity": fidelity,
                                 "combined": comp,
                                 "mode": mode_this,
                                 "strategy": strategy_this,
                                 "loras": loras_fmt,
                                 "error": "generation failed", "error_detail": err_detail})
                gen_error = err_detail
                save_manifest()
                break
            dur = time.time() - t_iter
            manifest.append({"iter": i, "resolution": f"{W}x{H}",
                             "prompt": current_prompt,
                             "image": os.path.basename(out_png),
                             "ref_images": [os.path.basename(p) for p in refs_this],
                             "duration_s": round(dur, 1),
                             "seed": eff_seed,
                             "adherence": adherence, "fidelity": fidelity,
                             "combined": comp,
                             "mode": mode_this,
                             "strategy": strategy_this,
                             "loras": loras_fmt,
                             "goal_reached": bool(goal_reached)})
            log(f"Iteration {i + 1} done in {dur:.1f}s -> {out_png}")
            if explore_on:
                remember_outcome(memory, i, mode_this, strategy_this, comp,
                                 current_prompt, fidelity=fidelity,
                                 adherence=adherence,
                                 diagnosis=diagnosis_this
                                 if mode_this != "exploit" else "")
                save_memory(out_dir, memory)
                if mode_this != "exploit":
                    log_memory(out_dir, f"### iter {i + 1}: {mode_this.upper()} "
                                       f"strategy={strategy_this} combined={comp} "
                                       f"adherence={adherence} fidelity={fidelity} :: {current_prompt[:200]}")
            emit("img", out_png)
            save_manifest()
    finally:
        # Stop our own critic server. (The web app's IA/tag servers, if any,
        # are restored by api/agent.py in the API process.)
        if critic_mgr is not None:
            try:
                critic_mgr.stop()
            except Exception:
                pass

    # ---- summary -------------------------------------------------------------
    # Final manifest write (also refreshes the per-iteration checkpoints with
    # the definitive total_s once the whole run completed).
    save_manifest()

    produced = [m for m in manifest if m.get("image")]
    if not produced:
        # Nothing was generated (e.g. t2i failed on every attempt): do NOT
        # stream a success FINISH - the UI must not offer Retry/export for a
        # run that produced no image. The manifest stays on disk for debugging.
        reason = gen_error or "generation failed on every iteration"
        hint = ""
        if diffusion and diffusion.lower().endswith(".safetensors"):
            hint = " The selected model (.safetensors) has no VAE available: try the matching GGUF (e.g. z_image_turbo-Q6_K.gguf)."
        log("=" * 70)
        log("❌ RUN FAILED: no image was generated.")
        log(f"   {reason}{hint}")
        log("=" * 70)
        emit("fail", f"No image was generated. {reason}{hint}")
        sys.exit(2)

    # write the human-readable experience summary (inspectable on disk)
    if explore_on:
        write_explore_summary(out_dir, memory, args.goal)
    emit("finish", manifest_path)

    log("=" * 70)
    log(f"RUN COMPLETE. Total {time.time() - start_all:.1f}s")
    log(f"Artifacts in: {out_dir}")
    log(f"Manifest:    {manifest_path}")
    log("Progression:")
    for m in manifest:
        if m.get("image"):
            extra = ""
            if m.get("adherence") is not None:
                extra += f"  adherence={m['adherence']}"
            if m.get("fidelity") is not None:
                extra += f" fidelity={m['fidelity']}"
            if m.get("combined") is not None:
                extra += f" combined={m['combined']}"
            if m.get("mode") == "explore":
                extra += f"  [EXPLORE:{m.get('strategy')}]"
            elif m.get("mode") == "reground":
                extra += "  [REGROUND]"
            log(f"  iter {m['iter'] + 1} [{m['resolution']}]{extra} -> {os.path.join(out_dir, m['image'])}")
    if explore_on:
        log(f"Explore rounds   : {memory.get('explore_count')} (strategies tried:"
            + ", ".join(s.get('label', '?') for s in memory.get('strategies', [])) + ")")
        log(f"Exploration log  : {os.path.join(out_dir, 'explore_log.md')}")
        if memory.get("best"):
            b = memory["best"]
            log(f"Best prompt found: combined score {b.get('score')}/10 at iter {b.get('iter')} (mode {b.get('mode')})")
            if b.get("adherence") is not None:
                log(f"  best adherence={b.get('adherence')}" +
                    (f", fidelity={b.get('fidelity')}" if b.get("fidelity") is not None else ""))
            log(f"  -> {os.path.join(out_dir, 'explore_summary.md')}")
    log("=" * 70)


if __name__ == "__main__":
    main()
