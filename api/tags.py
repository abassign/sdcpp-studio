"""
API Tags - Semantic tagging endpoints
Handles image tagging, tag management, and tag queries.
Uses VRAM Arbiter for coordination with IA server.
"""
import os
import sys
import json
import time
import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict

router = APIRouter()

sys.path.insert(0, '/app')

from workers.tag_worker import (
    load_tags_db, save_tags_db, tag_single_image, tag_batch_images,
    get_image_tags, add_manual_tags, remove_manual_tag, get_tags_status,
    cleanup_missing_images, TAGS_DB_PATH, list_available_models
)
from api.vram_arbiter import get_arbiter

try:
    from core.config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = "/app/outputs"

tagging_stop_flag = False


class ManualTagsRequest(BaseModel):
    tags: List[str]


class TagQueryRequest(BaseModel):
    query: Dict
    logic: str = "AND"


def _get_vram_manager():
    from api.llama_server_manager import get_vram_manager
    return get_vram_manager()


@router.get("/status", tags=["Tags"])
async def get_status():
    """Get tagging system status"""
    status = get_tags_status()
    arbiter = get_arbiter()
    status["arbiter"] = arbiter.get_status()
    return {"status": "success", "data": status}


@router.get("/models", tags=["Tags"])
async def get_tagging_models():
    models = list_available_models()
    return {"status": "success", "data": models}


@router.post("/single", tags=["Tags"])
async def tag_single(req: Request):
    try:
        body = await req.json()
        image_path = body.get("image_path")
        model = body.get("model", "glm_4v")
        if not image_path:
            raise HTTPException(status_code=400, detail="image_path required")
        result = tag_single_image(image_path, model)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", tags=["Tags"])
async def tag_batch(req: Request):
    try:
        body = await req.json()
        image_paths = body.get("image_paths", [])
        model = body.get("model", "glm_4v")
        if not image_paths:
            raise HTTPException(status_code=400, detail="image_paths required")
        results = tag_batch_images(image_paths, model)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-tag-all", tags=["Tags"])
async def auto_tag_all_untagged(model: str = "server"):
    """
    Automatically tag all images that don't have tags yet.
    Uses VRAM Arbiter to coordinate with IA server.
    """
    global tagging_stop_flag
    import workers.tag_worker
    tagging_stop_flag = False
    workers.tag_worker.tagging_stop_flag = False

    arbiter = get_arbiter()

    # Request VRAM access through arbiter
    print(f"[AutoTag] Requesting VRAM access from arbiter...", flush=True)
    if not arbiter.wait_for_vram("tag", timeout=120):
        return {"status": "error", "message": "Timeout waiting for VRAM access"}
    print(f"[AutoTag] VRAM access granted", flush=True)

    vram = _get_vram_manager()
    tags_db = load_tags_db()

    removed_count = cleanup_missing_images(tags_db, OUTPUT_DIR)
    if removed_count > 0:
        save_tags_db(tags_db)

    already_tagged_keys = list(tags_db.get("tags", {}).keys())
    already_tagged = set(already_tagged_keys)
    already_tagged_bases = {os.path.splitext(k)[0] for k in already_tagged_keys}
    last_updated = tags_db.get("last_updated")

    print(f"[AutoTag] Last updated: {last_updated}, Already tagged: {len(already_tagged)}")

    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    all_images = []
    skipped_count = 0

    for f in os.listdir(OUTPUT_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext not in image_extensions:
            continue

        base_f = os.path.splitext(f)[0]
        is_already_tagged = (f in already_tagged or base_f in already_tagged_bases)

        if is_already_tagged:
            needs_retag = False
            if last_updated:
                json_path = os.path.join(OUTPUT_DIR, base_f + ".json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            meta = json.load(jf).get("meta", {})
                            img_timestamp = meta.get("timestamp", "")
                            if img_timestamp:
                                norm_ts = img_timestamp.replace(" ", "T")
                                if not norm_ts.endswith("Z"):
                                    norm_ts += "Z"
                                if norm_ts > last_updated:
                                    needs_retag = True
                    except:
                        pass
                if not needs_retag:
                    try:
                        img_path = os.path.join(OUTPUT_DIR, f)
                        mtime_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(img_path)))
                        if mtime_ts > last_updated:
                            needs_retag = True
                    except:
                        pass
            else:
                needs_retag = True

            if needs_retag:
                all_images.append(os.path.join(OUTPUT_DIR, f))
            else:
                skipped_count += 1
            continue

        all_images.append(os.path.join(OUTPUT_DIR, f))

    print(f"[AutoTag] Skipped {skipped_count} unchanged, Found {len(all_images)} to tag")

    if not all_images:
        arbiter.release_vram("tag")
        return {"status": "no_images", "message": "No images need tagging"}

    checkpoint = vram.checkpoint
    start_from = 0
    # Accept all checkpoint statuses that represent an interrupted run we
    # can resume from: user-stop, suspended by IA, or preempted by T2I
    # generation. Otherwise start from the beginning.
    if checkpoint and checkpoint.get("status") in ("stopped", "suspended_by_ia", "interrupted_by_generation"):
        start_from = checkpoint.get("current", 0)
        if start_from > 0 and start_from < len(all_images):
            print(f"[AutoTag] Resuming from checkpoint: {start_from}/{len(all_images)}")

    use_server = (model == "server")

    async def generate():
        global tagging_stop_flag
        total = len(all_images)
        success_count = 0
        error_count = 0
        loop = asyncio.get_event_loop()
        suspended = False

        if use_server:
            yield "data: " + json.dumps({"status": "starting_server", "message": "Avvio server tagging...", "total": total}) + "\n\n"
            try:
                import shared_utils
                await loop.run_in_executor(None, vram.ensure_tag_server, shared_utils.I2T_DIR)
                yield "data: " + json.dumps({"status": "server_ready", "message": "Server pronto", "total": total}) + "\n\n"
            except Exception as e:
                yield "data: " + json.dumps({"status": "error", "message": str(e)}) + "\n\n"
                arbiter.release_vram("tag")
                return

        try:
            for idx in range(start_from, total):
                if tagging_stop_flag:
                    if use_server:
                        vram.save_checkpoint({"current": idx, "total": total, "model": model, "status": "stopped"})
                    yield "data: " + json.dumps({"status": "stopped", "current": idx, "total": total}) + "\n\n"
                    break

                img_path = all_images[idx]
                img_name = os.path.basename(img_path)
                yield "data: " + json.dumps({"status": "processing", "current": idx + 1, "total": total, "image": img_name}) + "\n\n"

                try:
                    result = await loop.run_in_executor(None, tag_single_image, img_path, model)
                    if "error" in result:
                        error_count += 1
                        status = "error"
                        if "Tagging suspended" in result.get("error", ""):
                            suspended = True
                            vram.save_checkpoint({"current": idx, "total": total, "model": model, "status": "suspended_by_ia"})
                            yield "data: " + json.dumps({"status": "suspended", "current": idx, "total": total}) + "\n\n"
                            break
                    else:
                        success_count += 1
                        status = "success"
                        if use_server and (idx + 1) % 10 == 0:
                            vram.save_checkpoint({"current": idx + 1, "total": total, "model": model, "status": "in_progress"})
                except Exception as e:
                    error_count += 1
                    status = "error"
                    result = {"error": str(e)}

                summary_text = ""
                elapsed_time = 0
                tags_list = []
                if isinstance(result, dict):
                    summary_text = (result.get("summary", "") or "")[:300]
                    elapsed_time = result.get("elapsed_time", 0)
                    raw_tags = result.get("tags", []) or []
                    tags_list = [t.get("value", "") for t in raw_tags[:8] if isinstance(t, dict)]

                progress_data = {
                    "status": "progress",
                    "current": idx + 1,
                    "total": total,
                    "image": img_name,
                    "last_status": status,
                    "summary": summary_text,
                    "elapsed_time": elapsed_time,
                    "tags": tags_list
                }
                yield "data: " + json.dumps(progress_data) + "\n\n"
                await asyncio.sleep(0.01)

            if success_count > 0:
                tags_db = load_tags_db()
                tags_db["last_tagged"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                save_tags_db(tags_db)

            if suspended:
                yield "data: " + json.dumps({"status": "suspended", "current": success_count + error_count, "total": total, "success": success_count, "errors": error_count}) + "\n\n"
            else:
                if use_server:
                    vram.save_checkpoint({})
                yield "data: " + json.dumps({"status": "completed", "current": success_count + error_count, "total": total, "success": success_count, "errors": error_count}) + "\n\n"

        finally:
            # Release VRAM token when done
            arbiter.release_vram("tag")
            print(f"[AutoTag] VRAM token released", flush=True)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/auto-tag-stop", tags=["Tags"])
async def stop_auto_tag():
    global tagging_stop_flag
    import workers.tag_worker
    tagging_stop_flag = True
    workers.tag_worker.tagging_stop_flag = True

    arbiter = get_arbiter()
    arbiter.release_vram("tag")

    try:
        vram = _get_vram_manager()
        vram.stop_tag_server()
    except Exception as e:
        print(f"[AutoTagStop] Error: {e}", flush=True)

    return {"status": "stopping", "message": "Stop signal sent"}


# --- VRAM Arbiter Endpoints ---

@router.get("/vram/status", tags=["VRAM"])
async def vram_status():
    arbiter = get_arbiter()
    return {"status": "success", "data": arbiter.get_status()}


@router.get("/vram/auto-resume-pending", tags=["VRAM"])
async def auto_resume_pending():
    """Consume the 'auto resume pending' flag set by the T2I generation
    endpoint after restoring the tag server. The frontend calls this
    once the generation SSE stream ends; if a resume is pending it
    triggers the resume flow automatically.

    Returns:
        {"status": "success", "pending": bool, "resume_from": int}
    """
    vram = _get_vram_manager()
    pending, resume_from = vram.consume_auto_resume_pending()
    return {
        "status": "success",
        "pending": pending,
        "resume_from": resume_from,
    }


@router.post("/vram/request", tags=["VRAM"])
async def vram_request(requester: str):
    """Request VRAM access. Body: {requester: 'ia'|'tag'}"""
    arbiter = get_arbiter()
    granted = arbiter.request_vram(requester)
    return {"status": "success", "granted": granted, "arbiter": arbiter.get_status()}


@router.post("/vram/release", tags=["VRAM"])
async def vram_release(requester: str):
    """Release VRAM access. Body: {requester: 'ia'|'tag'}"""
    arbiter = get_arbiter()
    arbiter.release_vram(requester)
    return {"status": "success", "arbiter": arbiter.get_status()}


@router.post("/vram/resume", tags=["VRAM"])
async def vram_resume(model: str = "server"):
    """Resume interrupted tagging from the saved checkpoint.

    This is the canonical entry point used by the frontend's
    'Riprendi Tagging' button. It:
      1. Refuses to resume while the IA server or an external VRAM
         user is active (otherwise we'd OOM the subprocess).
      2. Delegates to auto_tag_all_untagged(), which automatically
         picks up the saved 'current' index from the checkpoint.
    """
    vram = _get_vram_manager()
    if vram.ia_running:
        return {"status": "error", "message": "Cannot resume - IA server is active"}
    if vram.external_vram_user:
        return {"status": "error", "message": f"Cannot resume - external VRAM user '{vram.external_vram_user}' is active"}
    return await auto_tag_all_untagged(model)


@router.post("/resume", tags=["Tags"])
async def resume_tagging(model: str = "server"):
    """Resume interrupted tagging (shorter alias used by the frontend)."""
    return await vram_resume(model)


@router.get("/image/{image_name}", tags=["Tags"])
async def get_image_tags_endpoint(image_name: str):
    result = get_image_tags(image_name)
    if result:
        return {"status": "success", "data": result}
    raise HTTPException(status_code=404, detail="Image not found in tags DB")


@router.post("/image/{image_name}/manual", tags=["Tags"])
async def add_manual_tags_endpoint(image_name: str, req: ManualTagsRequest):
    result = add_manual_tags(image_name, req.tags)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return {"status": "success", "data": result}


@router.delete("/image/{image_name}/manual/{tag}", tags=["Tags"])
async def remove_manual_tag_endpoint(image_name: str, tag: str):
    result = remove_manual_tag(image_name, tag)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return {"status": "success", "data": result}


@router.get("/all", tags=["Tags"])
async def get_all_tags():
    return load_tags_db()


@router.delete("/image/{image_name}", tags=["Tags"])
async def delete_image_tags(image_name: str):
    tags_db = load_tags_db()
    if image_name in tags_db.get("tags", {}):
        del tags_db["tags"][image_name]
        save_tags_db(tags_db)
        return {"status": "success", "message": f"Tags deleted for {image_name}"}
    raise HTTPException(status_code=404, detail="Image not found in tags DB")


@router.post("/query", tags=["Tags"])
async def query_tags(req: TagQueryRequest):
    from workers.view_manager import ViewManager
    tags_db = load_tags_db()
    vm = ViewManager()
    results = vm.execute_query(req.query, tags_db)
    enriched = []
    for img_key in results:
        if img_key in tags_db.get("tags", {}):
            img_data = tags_db["tags"][img_key].copy()
            if "image" in img_data and img_data["image"].startswith("/app/outputs/"):
                img_data["image"] = img_data["image"].replace("/app/outputs/", "/outputs/")
            enriched.append(img_data)
    return {"count": len(enriched), "images": enriched}
