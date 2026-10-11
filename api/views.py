"""
API Views - Saved views management
Handles creating, updating, deleting, and executing saved semantic views.
"""
import os
import json
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

router = APIRouter()

# Add app directory to path
sys.path.insert(0, '/app')

from workers.view_manager import ViewManager
from workers.tag_worker import load_tags_db
from shared_utils import OUTPUT_DIR


# Request models
class CreateViewRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    query: Dict
    color: Optional[str] = "#2196F3"


class UpdateViewRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    query: Optional[Dict] = None
    color: Optional[str] = None


class NaturalLanguageQueryRequest(BaseModel):
    q: str
    logic: Optional[str] = "OR"
    model: Optional[str] = "glm_4v"

@router.get("/", tags=["Views"])
async def list_views():
    """List all saved views"""
    vm = ViewManager()
    return {"views": vm.list_views()}


@router.post("/", tags=["Views"])
async def create_view(req: CreateViewRequest):
    """Create a new saved view"""
    vm = ViewManager()
    view_id = vm.create_view(req.name, req.description, req.query, req.color)
    return {"status": "success", "view_id": view_id}


@router.get("/{view_id}", tags=["Views"])
async def get_view(view_id: str):
    """Get view details"""
    vm = ViewManager()
    view = vm.get_view(view_id)
    
    if view:
        return {"status": "success", "data": view}
    
    raise HTTPException(status_code=404, detail="View not found")


@router.put("/{view_id}", tags=["Views"])
async def update_view(view_id: str, req: UpdateViewRequest):
    """Update a view"""
    vm = ViewManager()
    
    updates = {k: v for k, v in req.dict().items() if v is not None}
    
    if vm.update_view(view_id, updates):
        return {"status": "success"}
    
    raise HTTPException(status_code=404, detail="View not found")


@router.delete("/{view_id}", tags=["Views"])
async def delete_view(view_id: str):
    """Delete a view"""
    vm = ViewManager()
    
    if vm.delete_view(view_id):
        return {"status": "success"}
    
    raise HTTPException(status_code=404, detail="View not found")


@router.get("/{view_id}/execute", tags=["Views"])
async def execute_view(view_id: str):
    """Execute a view and get matching images with full history metadata"""
    vm = ViewManager()
    tags_db = load_tags_db()
    
    results = vm.execute_view(view_id, tags_db)
    
    # Look up history metadata for each image
    enriched = []
    for img_key in results:
        if img_key in tags_db.get("tags", {}):
            img_data = tags_db["tags"][img_key].copy()
            
            # Extract filename from path
            filename = os.path.basename(img_key)
            json_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".json")
            
            # Try to load history metadata
            history_item = None
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        history_item = json.load(f)
                except:
                    pass
            
            # Merge history data with tag data
            if history_item:
                meta = history_item.get("meta", {})
                params = history_item.get("params", {}) or history_item.get("parameters", {})
                inputs = history_item.get("inputs", {})
                outputs = history_item.get("outputs", {})
                
                # Copy properties needed for restoreHistoryState over to params like history.py does
                if "prompt" in inputs:
                    params["prompt"] = inputs["prompt"]
                if "model_path" in params and params["model_path"]:
                    mp = params["model_path"]
                    if "/" in mp or "\\" in mp:
                        params["model"] = os.path.basename(mp.rstrip("/\\"))
                    else:
                        params["model"] = mp
                for k in ["source_image", "source_image_1", "source_image_2"]:
                    if k in inputs and k not in params:
                        params[k] = inputs[k]
                if "source_images" in inputs and isinstance(inputs["source_images"], list) and len(inputs["source_images"]) > 0:
                    params["source_images"] = inputs["source_images"]
                    if "source_image_1" not in params:
                        params["source_image_1"] = inputs["source_images"][0]
                if "loras" in inputs and isinstance(inputs["loras"], list) and len(inputs["loras"]) > 0:
                    params["loras"] = inputs["loras"]
                
                # Build image URL
                image_filename = None
                if "files" in outputs and outputs["files"]:
                    image_filename = os.path.basename(outputs["files"][0])
                if not image_filename and "filename" in meta:
                    cand = meta["filename"]
                    if os.path.exists(os.path.join(OUTPUT_DIR, cand)):
                        image_filename = cand
                if not image_filename:
                    for ext in ['.png', '.jpg', '.jpeg']:
                        if os.path.exists(os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ext)):
                            image_filename = os.path.splitext(filename)[0] + ext
                            break
                
                if image_filename:
                    img_data["image"] = "/outputs/" + image_filename
                img_data["filename"] = image_filename or filename
                img_data["timestamp"] = meta.get("timestamp", img_data.get("timestamp"))
                img_data["params"] = params
                img_data["inputs"] = inputs
                img_data["outputs"] = outputs
                img_data["meta"] = meta
                img_data["mode"] = params.get("mode", meta.get("mode", "unk"))
            else:
                # No history found, use tag data only
                if "image" in img_data and img_data["image"].startswith("/app/outputs/"):
                    img_data["image"] = img_data["image"].replace("/app/outputs/", "/outputs/")
                img_data["filename"] = filename
                # Ensure timestamp is set (may be null/invalid in semantic_tags.json)
                img_data["timestamp"] = img_data.get("timestamp") or 0
            
            enriched.append(img_data)
    
    return {
        "view_id": view_id,
        "count": len(enriched),
        "images": enriched
    }


@router.post("/query", tags=["Views"])
async def natural_language_query(req: NaturalLanguageQueryRequest):
    """
    Execute a natural language query (without saving as view).
    
    Examples:
    - ?q=sport+athlete -> finds all images matching sport OR athlete
    - ?q=landscape+not+anime -> finds landscape, excludes anime
    """
    print(f"--> [Views API] Natural language query: '{req.q}' using model: '{req.model}'")
    vm = ViewManager()
    tags_db = load_tags_db()
    print(f"--> [Views API] Tags DB has {len(tags_db.get('tags', {}))} images")
    if len(tags_db.get('tags', {})) == 0:
        print(f"--> [Views API] WARNING: No tagged images! Run auto-tag first on history images.")
    
    query, results = vm.natural_language_query(req.q, tags_db, req.model)
    print(f"--> [Views API] Parsed query: {query}")
    print(f"--> [Views API] Found {len(results)} matching images: {results[:5]}{'...' if len(results) > 5 else ''}")
    
    # Enrich with full data - same logic as execute_view
    enriched = []
    for img_key in results:
        if img_key in tags_db.get("tags", {}):
            img_data = tags_db["tags"][img_key].copy()
            
            # Extract filename from path
            filename = os.path.basename(img_key)
            json_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".json")
            
            # Try to load history metadata
            history_item = None
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        history_item = json.load(f)
                except:
                    pass
            
            # Merge history data with tag data
            if history_item:
                meta = history_item.get("meta", {})
                params = history_item.get("params", {}) or history_item.get("parameters", {})
                inputs = history_item.get("inputs", {})
                outputs = history_item.get("outputs", {})
                
                # Copy properties needed for restoreHistoryState over to params like history.py does
                if "prompt" in inputs:
                    params["prompt"] = inputs["prompt"]
                if "model_path" in params and params["model_path"]:
                    mp = params["model_path"]
                    if "/" in mp or "\\" in mp:
                        params["model"] = os.path.basename(mp.rstrip("/\\"))
                    else:
                        params["model"] = mp
                for k in ["source_image", "source_image_1", "source_image_2"]:
                    if k in inputs and k not in params:
                        params[k] = inputs[k]
                if "source_images" in inputs and isinstance(inputs["source_images"], list) and len(inputs["source_images"]) > 0:
                    params["source_images"] = inputs["source_images"]
                    if "source_image_1" not in params:
                        params["source_image_1"] = inputs["source_images"][0]
                if "loras" in inputs and isinstance(inputs["loras"], list) and len(inputs["loras"]) > 0:
                    params["loras"] = inputs["loras"]
                
                # Build image URL
                image_filename = None
                if "files" in outputs and outputs["files"]:
                    image_filename = os.path.basename(outputs["files"][0])
                if not image_filename and "filename" in meta:
                    cand = meta["filename"]
                    if os.path.exists(os.path.join(OUTPUT_DIR, cand)):
                        image_filename = cand
                if not image_filename:
                    for ext in ['.png', '.jpg', '.jpeg']:
                        if os.path.exists(os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ext)):
                            image_filename = os.path.splitext(filename)[0] + ext
                            break
                
                if image_filename:
                    img_data["image"] = "/outputs/" + image_filename
                img_data["filename"] = image_filename or filename
                img_data["timestamp"] = meta.get("timestamp", img_data.get("timestamp"))
                img_data["params"] = params
                img_data["inputs"] = inputs
                img_data["outputs"] = outputs
                img_data["meta"] = meta
                img_data["mode"] = params.get("mode", meta.get("mode", "unk"))
            else:
                # No history found, use tag data only
                if "image" in img_data and img_data["image"].startswith("/app/outputs/"):
                    img_data["image"] = img_data["image"].replace("/app/outputs/", "/outputs/")
                img_data["filename"] = filename
                # Ensure timestamp is set (may be null/invalid in semantic_tags.json)
                img_data["timestamp"] = img_data.get("timestamp") or 0
            
            enriched.append(img_data)
    
    return {
        "query": req.q,
        "parsed_query": query,
        "count": len(enriched),
        "images": enriched
    }
