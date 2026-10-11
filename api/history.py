"""
API History - Generation history management
"""
from fastapi import APIRouter
from pydantic import BaseModel
import os
import json

router = APIRouter()

from shared_utils import OUTPUT_DIR


class DeleteRequest(BaseModel):
    filename: str


def _arf_history_items(include: bool = True) -> list:
    """Build one history item per agent_refine_* run folder (ARF).
    Each run contains MULTIPLE step images; History shows the LAST generated one
    as the visible default, plus the full ordered step list so the UI can open a
    picker. Images keep their direct path inside the folder (no copies in root)."""
    if not include:
        return []
    items = []
    try:
        entries = sorted(os.listdir(OUTPUT_DIR), key=lambda n: os.path.getmtime(os.path.join(OUTPUT_DIR, n)), reverse=True)
    except Exception:
        return []
    for name in entries:
        if not name.startswith("agent_refine_") or not os.path.isdir(os.path.join(OUTPUT_DIR, name)):
            continue
        mpath = os.path.join(OUTPUT_DIR, name, "manifest.json")
        if not os.path.exists(mpath):
            continue
        try:
            with open(mpath) as f:
                m = json.load(f)
        except Exception:
            continue
        iters = m.get("iterations") or []
        steps = []
        for it in iters:
            imgname = it.get("image")
            if not imgname:
                continue
            url = f"/outputs/{name}/{imgname}"
            res = it.get("resolution") or ""
            w = h = 0
            if "x" in res:
                try:
                    w, h = (int(x) for x in res.split("x"))
                except Exception:
                    w = h = 0
            steps.append({
                "iter": it.get("iter", 0) + 1,
                "image": url,
                "filename": imgname,
                "prompt": it.get("prompt") or "",
                "resolution": res,
                "width": w, "height": h,
                "goal_reached": it.get("goal_reached"),
                "adherence": it.get("adherence"),
                "fidelity": it.get("fidelity"),
                "combined": it.get("combined"),
                "mode": it.get("mode"),
                "strategy": it.get("strategy"),
                # LoRAs used for THIS image (same shape as T2I/I2I JSON); falls
                # back to the run-level list for manifests written by older
                # versions that only stored the lora_config path.
                "loras": it.get("loras") or m.get("loras") or {"path": "", "files": []},
                "ref_images": it.get("ref_images") or [],
            })
        if not steps:
            continue  # run without any generated image is not shown as an image item
        last = steps[-1]
        # folders with an original copy
        orig_name = "original.png"
        orig_url = f"/outputs/{name}/{orig_name}" if os.path.exists(os.path.join(OUTPUT_DIR, name, orig_name)) else ""
        model_path = m.get("diffusion_model") or ""
        item = {
            "arf": True,
            "arf_dir": name,
            "image": last["image"],
            "filename": name,
            "timestamp": os.path.getmtime(mpath),
            "mode": "arf",
            "has_image_analysis": False,
            "is_text_result": False,
            "arf_steps": steps,
            "original_image": orig_url,
            "params": {
                "mode": "arf",
                "model": os.path.basename(model_path) if model_path else "",
                "model_path": model_path,
                "prompt": m.get("goal") or last.get("prompt") or "",
                "goal": m.get("goal") or "",
                "seed": m.get("seed", 0),
                "steps": m.get("steps", 0),
                "guidance": m.get("guidance", 0.0),
                "scheduler": m.get("scheduler", ""),
                "critic_model": os.path.basename(m.get("critic_model") or ""),
                "interpreter_model": m.get("interpreter_model"),
                "resolution": last.get("resolution"),
                # top-level fallback; the exact per-step LoRAs live in arf_steps
                "loras": last.get("loras") or m.get("loras") or {"path": "", "files": []},
                "ref_images": m.get("ref_images") or [],
                "use_image": m.get("use_image", True),
                "edit_from_reference": m.get("edit_from_reference", False),
                "edit_anchor": m.get("edit_anchor"),
                "edit_family": m.get("edit_family"),
                "arf_steps": steps,
            },
            "inputs": {"goal": m.get("goal") or "", "prompt": m.get("goal") or "",
                       "loras": last.get("loras") or m.get("loras") or {"path": "", "files": []},
                       "ref_images": m.get("ref_images") or []},
        }
        items.append(item)
    return items


@router.get("/", tags=["History"])
async def get_history(page: int = 0, filter: str = "all", limit: int = 0):
    """Returns generation list"""
    history = []
    
    if not os.path.exists(OUTPUT_DIR):
        return {"history": []}
    
    # Search for JSON files
    json_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.json')]
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    
    # Apply filter
    if filter and filter != "all":
        if filter == "image_analysis":
            # Filter by JSON content: keep only files that have an "image_analysis" key
            filtered = []
            for f in json_files:
                try:
                    with open(os.path.join(OUTPUT_DIR, f), 'r') as jf:
                        data = json.load(jf)
                    if "image_analysis" in data:
                        filtered.append(f)
                except:
                    continue
            json_files = filtered
        else:
            json_files = [f for f in json_files if f.lower().startswith(filter.lower() + "_")]
    
    for json_file in json_files:
        json_path = os.path.join(OUTPUT_DIR, json_file)
        base_name = os.path.splitext(json_file)[0]
        
        # Detect mode
        inferred_mode = "unk"
        lower = json_file.lower()
        if lower.startswith("i2t_"): inferred_mode = "i2t"
        elif lower.startswith("t2i_"): inferred_mode = "t2i"
        elif lower.startswith("i2i_"): inferred_mode = "i2i"
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except:
            continue
        
        # Detect if this JSON has an image_analysis tag
        has_image_analysis = "image_analysis" in data
        
        meta = data.get("meta", {})
        params = data.get("params", {}) or data.get("parameters", {})
        inputs = data.get("inputs", {})
        outputs = data.get("outputs", {})
        
        if "mode" in meta:
            params["mode"] = meta["mode"]
        elif "mode" not in params:
            params["mode"] = inferred_mode
        
        if "prompt" in inputs:
            params["prompt"] = inputs["prompt"]
        
        # Model - estrai il nome del modello dal percorso se presente
        if "model_path" in params and params["model_path"]:
            model_path = params["model_path"]
            # Estrai solo il nome del modello (ultima parte del percorso)
            if "/" in model_path or "\\" in model_path:
                params["model"] = os.path.basename(model_path.rstrip("/\\"))
            else:
                params["model"] = model_path
        
        # Source images - handle both singular and plural forms
        for k in ["source_image", "source_image_1", "source_image_2"]:
            if k in inputs and k not in params:
                params[k] = inputs[k]
        
        # Handle source_images array (I2T uses this format)
        if "source_images" in inputs and isinstance(inputs["source_images"], list) and len(inputs["source_images"]) > 0:
            params["source_images"] = inputs["source_images"]
            if "source_image_1" not in params:
                params["source_image_1"] = inputs["source_images"][0]
        
        # Handle loras array (LoRA configs)
        if "loras" in inputs and isinstance(inputs["loras"], list) and len(inputs["loras"]) > 0:
            params["loras"] = inputs["loras"]
        
        # Also check params.loras (some workers save directly to params)
        if "loras" in params and isinstance(params["loras"], list) and len(params["loras"]) > 0:
            # Already set, don't overwrite
            pass
        
        # Image filename
        image_filename = None
        if "files" in outputs and outputs["files"]:
            image_filename = os.path.basename(outputs["files"][0])
        
        if not image_filename and "filename" in meta:
            cand = meta["filename"]
            if os.path.exists(os.path.join(OUTPUT_DIR, cand)):
                image_filename = cand
        
        if not image_filename:
            for ext in ['.png', '.jpg', '.jpeg']:
                if os.path.exists(os.path.join(OUTPUT_DIR, base_name + ext)):
                    image_filename = base_name + ext
                    break
        
        # Compute text_preview for I2T modes from outputs.text_content
        text_preview = None
        is_text_result = False
        if outputs.get("type") == "text" and "text_content" in outputs:
            is_text_result = True
            tc = outputs["text_content"]
            raw = tc.get("final_answer") or tc.get("raw_full_response") or ""
            # Clean: collapse whitespace, truncate to ~100 chars
            cleaned = " ".join(raw.split())
            text_preview = cleaned[:100]
            if len(cleaned) > 100:
                text_preview += "..."
        
        if not image_filename:
            if params.get("mode", "").startswith("i2t") or is_text_result:
                # Try source_images array first (I2T format), then fall back to source_image_1
                src = None
                if "source_images" in params and isinstance(params["source_images"], list) and len(params["source_images"]) > 0:
                    src = params["source_images"][0]
                elif "source_image_1" in params:
                    src = params["source_image_1"]
                
                if src:
                    # Handle path - extract filename if full path
                    src_basename = os.path.basename(src) if "/" in src or "\\" in src else src
                    # Construct proper URL for the uploads directory
                    image_url = f"/uploads/{src_basename}"
                    history.append({
                        "image": image_url,
                        "filename": json_file,
                        "timestamp": os.path.getmtime(json_path),
                        "params": params,
                        "output": outputs,
                        "text_preview": text_preview,
                        "is_text_result": True,
                        "has_image_analysis": has_image_analysis
                    })
                else:
                    # Still add the entry even without a source image
                    history.append({
                        "image": None,
                        "filename": json_file,
                        "timestamp": os.path.getmtime(json_path),
                        "params": params,
                        "output": outputs,
                        "text_preview": text_preview,
                        "is_text_result": True,
                        "has_image_analysis": has_image_analysis
                    })
            continue
        
        history.append({
            "image": f"/outputs/{image_filename}",
            "filename": image_filename,
            "timestamp": os.path.getmtime(json_path),
            "params": params,
            "output": outputs,
            "is_text_result": is_text_result,
            "text_preview": text_preview,
            "has_image_analysis": has_image_analysis
        })
    
    # ARF runs (agent_refine_* folders) are multi-image: include them as items
    # (in 'all' and in the dedicated 'arf' filter), interleaved by time.
    show_arf = filter in ("all", "arf")
    if show_arf:
        history.extend(_arf_history_items(True))
    elif filter in ("t2i", "i2i", "i2t", "image_analysis"):
        history.extend(_arf_history_items(False))

    # Sort by timestamp desc
    history.sort(key=lambda x: x["timestamp"], reverse=True)

    total_items = len(history)
    
    # Apply pagination if limit is set
    if limit > 0:
        start = page * limit
        end = start + limit
        history = history[start:end]
    
    return {
        "history": history,
        "total": total_items,
        "page": page,
        "limit": limit,
        "pages": (total_items + limit - 1) // limit if limit > 0 else 1
    }


@router.get("/item", tags=["History"])
async def get_history_item(filename: str):
    """Returns a single history item by filename"""
    if not os.path.exists(OUTPUT_DIR):
        return {"status": "error", "detail": "Output directory not found"}
    
    # Try to find the JSON file
    json_path = os.path.join(OUTPUT_DIR, filename + ".json")
    if not os.path.exists(json_path):
        # Try with the filename as-is
        json_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(json_path):
            return {"status": "error", "detail": "File not found"}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # Detect mode
    lower = base_name.lower()
    inferred_mode = "unk"
    if lower.startswith("i2t_"): inferred_mode = "i2t"
    elif lower.startswith("t2i_"): inferred_mode = "t2i"
    elif lower.startswith("i2i_"): inferred_mode = "i2i"

    
    meta = data.get("meta", {})
    params = data.get("params", {}) or data.get("parameters", {})
    inputs = data.get("inputs", {})
    outputs = data.get("outputs", {})
    
    if "mode" in meta:
        params["mode"] = meta["mode"]
    elif "mode" not in params:
        params["mode"] = inferred_mode
    
    if "prompt" in inputs:
        params["prompt"] = inputs["prompt"]
    
    # Model
    if "model_path" in params and params["model_path"]:
        model_path = params["model_path"]
        if "/" in model_path or "\\" in model_path:
            params["model"] = os.path.basename(model_path.rstrip("/\\"))
        else:
            params["model"] = model_path
    
    # Source images
    for k in ["source_image", "source_image_1", "source_image_2"]:
        if k in inputs and k not in params:
            params[k] = inputs[k]
    
    if "source_images" in inputs and isinstance(inputs["source_images"], list):
        params["source_images"] = inputs["source_images"]
        if "source_image_1" not in params and len(inputs["source_images"]) > 0:
            params["source_image_1"] = inputs["source_images"][0]
    
    # LoRAs
    if "loras" in inputs and isinstance(inputs["loras"], list):
        params["loras"] = inputs["loras"]
    
    # Image filename
    image_filename = None
    if "files" in outputs and outputs["files"]:
        image_filename = os.path.basename(outputs["files"][0])
    
    if not image_filename and "filename" in meta:
        cand = meta["filename"]
        if os.path.exists(os.path.join(OUTPUT_DIR, cand)):
            image_filename = cand
    
    if not image_filename:
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            if os.path.exists(os.path.join(OUTPUT_DIR, base_name + ext)):
                image_filename = base_name + ext
                break
    
    has_image_analysis = "image_analysis" in data
    
    # Text preview for I2T
    text_preview = None
    is_text_result = False
    if outputs.get("type") == "text" and "text_content" in outputs:
        is_text_result = True
        tc = outputs["text_content"]
        raw = tc.get("final_answer") or tc.get("raw_full_response") or ""
        cleaned = " ".join(raw.split())
        text_preview = cleaned[:100]
        if len(cleaned) > 100:
            text_preview += "..."
    
    if not image_filename:
        if params.get("mode", "").startswith("i2t") or is_text_result:
            src = None
            if "source_images" in params and isinstance(params["source_images"], list) and len(params["source_images"]) > 0:
                src = params["source_images"][0]
            elif "source_image_1" in params:
                src = params["source_image_1"]
            
            if src:
                src_basename = os.path.basename(src) if "/" in src or "\\" in src else src
                image_url = f"/uploads/{src_basename}"
                return {
                    "status": "success",
                    "data": {
                        "image": image_url,
                        "filename": base_name + ".json",
                        "timestamp": os.path.getmtime(json_path),
                        "params": params,
                        "output": outputs,
                        "text_preview": text_preview,
                        "is_text_result": True,
                        "has_image_analysis": has_image_analysis
                    }
                }
    
    if not image_filename:
        return {"status": "error", "detail": "Image file not found"}
    
    return {
        "status": "success",
        "data": {
            "image": f"/outputs/{image_filename}",
            "filename": image_filename,
            "timestamp": os.path.getmtime(json_path),
            "params": params,
            "output": outputs,
            "is_text_result": is_text_result,
            "text_preview": text_preview,
            "has_image_analysis": has_image_analysis
        }
    }


@router.post("/delete")
async def delete_history(req: DeleteRequest):
    """Deletes history item wipes files from disk"""
    safe_name = os.path.basename(req.filename)
    # ARF: an agent_refine_* folder -> remove the whole run folder
    if safe_name.startswith("agent_refine_") and os.path.isdir(os.path.join(OUTPUT_DIR, safe_name)):
        import shutil as _sh
        _sh.rmtree(os.path.join(OUTPUT_DIR, safe_name), ignore_errors=True)
        return {"status": "deleted"}
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    base_name = os.path.splitext(safe_name)[0]
    json_path = os.path.join(OUTPUT_DIR, base_name + ".json")
    
    if safe_name.endswith(".json"):
        json_path = file_path
        for ext in [".png", ".jpg", ".jpeg"]:
            img = os.path.join(OUTPUT_DIR, base_name + ext)
            if os.path.exists(img):
                os.remove(img)
                break
    else:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    try:
        if os.path.exists(json_path):
            os.remove(json_path)
        return {"status": "deleted"}
    except Exception as e:
        return {"error": str(e)}


class DeleteRequest(BaseModel):
    filename: str


@router.post("/delete_history")
async def delete_history(req: DeleteRequest):
    """Delete history item"""
    import os
    try:
        # Try both .png and .json
        base = req.filename.replace('.png', '').replace('.json', '')
        png_path = os.path.join(OUTPUT_DIR, base + ".png")
        json_path = os.path.join(OUTPUT_DIR, base + ".json")
        
        deleted = []
        if os.path.exists(png_path):
            os.remove(png_path)
            deleted.append(png_path)
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted.append(json_path)
        
        return {"status": "deleted", "files": deleted}
    except Exception as e:
        return {"status": "error", "error": str(e)}


class DeleteArfStepRequest(BaseModel):
    arf_dir: str
    filename: str


@router.post("/delete_arf_step")
async def delete_arf_step(req: DeleteArfStepRequest):
    """Delete a SINGLE step image inside an ARF run folder.

    Removes the image plus its per-iteration notes (critique/prompt .txt) and
    drops the matching entry from manifest.json, so History and the step picker
    stay consistent. The rest of the run is preserved.
    """
    safe_dir = os.path.basename(req.arf_dir or "")
    safe_file = os.path.basename(req.filename or "")
    if not safe_dir.startswith("agent_refine_") or not safe_file:
        return {"status": "error", "detail": "Invalid ARF run or filename"}
    # Only real image extensions are ever deletable (never manifest/json/etc.).
    if os.path.splitext(safe_file)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        return {"status": "error", "detail": "Only step images can be deleted"}
    run_dir = os.path.join(OUTPUT_DIR, safe_dir)
    if not os.path.isdir(run_dir):
        return {"status": "error", "detail": "ARF run folder not found"}
    img_path = os.path.join(run_dir, safe_file)
    if not os.path.isfile(img_path):
        return {"status": "error", "detail": "Image not found"}

    # The image must be an actual step of this run (referenced in the manifest):
    # this prevents deleting original.png or arbitrary files in the folder.
    mpath = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(mpath):
        return {"status": "error", "detail": "ARF manifest not found"}
    try:
        with open(mpath) as f:
            m = json.load(f)
    except Exception as e:
        return {"status": "error", "detail": f"could not read ARF manifest: {e}"}
    iters = m.get("iterations") or []
    if not any(it.get("image") == safe_file for it in iters):
        return {"status": "error", "detail": "Not a step image of this ARF run"}

    # Per-iteration notes share the image stem:
    # {run}_iter_NNN_img.png -> {run}_iter_NNN_img_critique.txt / _prompt.txt
    stem = os.path.splitext(safe_file)[0]
    companions = [stem + "_critique.txt", stem + "_prompt.txt"]

    removed = []
    try:
        os.remove(img_path)
        removed.append(img_path)
    except Exception as e:
        return {"status": "error", "detail": f"could not delete image: {e}"}
    for c in companions:
        cp = os.path.join(run_dir, c)
        if os.path.isfile(cp):
            try:
                os.remove(cp)
                removed.append(cp)
            except Exception:
                pass

    # Drop the manifest entry referencing this image (atomic rewrite).
    try:
        kept = [it for it in iters if it.get("image") != safe_file]
        m["iterations"] = kept
        tmp = mpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        os.replace(tmp, mpath)
        remaining = len([it for it in kept if it.get("image")])
    except Exception as e:
        return {"status": "error", "detail": f"image deleted but manifest update failed: {e}"}

    return {"status": "deleted", "files": removed, "remaining": remaining}
