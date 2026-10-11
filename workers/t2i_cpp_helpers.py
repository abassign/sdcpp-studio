"""
T2I CPP Helpers - Support functions for T2I
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_utils


# Scheduler mapping from GUI names to sd.cpp names
SCHED_MAP = {
    "euler_a": "euler_a",
    "euler": "euler",
    "dpm++_2m_karras": "dpm++2m",
    "dpm++_sde_karras": "dpm++2s_a",
    "ddim": "euler",
    "res_multistep": "res_multistep"
}


def find_diffusion_model(model_path: str) -> str:
    """Finds the diffusion model file"""
    final_model = model_path
    
    if model_path and os.path.isdir(model_path):
        # Find weight file, ignore "qwen" files (they are LLMs)
        cands = [f for f in sorted(os.listdir(model_path)) 
                 if f.endswith(".gguf") and "qwen" not in f.lower()]
        if not cands:
            cands = [f for f in sorted(os.listdir(model_path)) 
                     if f.endswith(".safetensors") and "qwen" not in f.lower()]
        
        if cands:
            final_model = os.path.join(model_path, cands[0])
        else:
            # Deep search for Z-Image
            if "z_image" in model_path.lower() or "z-image" in model_path.lower():
                parent = os.path.dirname(model_path)
                for d in os.listdir(parent):
                    if "z" in d.lower() and "image" in d.lower() and "gguf" in d.lower():
                        gguf_path = os.path.join(parent, d)
                        cands_gguf = [f for f in sorted(os.listdir(gguf_path)) 
                                     if f.endswith(".gguf") and "qwen" not in f.lower()]
                        if cands_gguf:
                            final_model = os.path.join(gguf_path, cands_gguf[0])
                            break
    
    return final_model


def find_llm_for_zimage(final_model: str, llm_path: str = None, models_root: str = None) -> str:
    """Finds LLM for Z-Image"""
    llm_search = None
    
    if llm_path and os.path.exists(llm_path):
        return llm_path
    
    # Use shared_utils.MODELS_DIR if not provided
    if models_root is None:
        models_root = shared_utils.MODELS_DIR
    
    # Get the directory where the diffusion model is
    diffusion_dir = os.path.dirname(final_model)
    
    # Search in order of priority
    llm_search_dirs = [
        diffusion_dir,
        os.path.join(diffusion_dir, "text_encoders"),
        os.path.join(diffusion_dir, "text_encoder"),
        os.path.join(models_root, "Z-Image-GGUF"),
        os.path.join(models_root, "Z-Image-Turbo-GGUF"),
        os.path.join(models_root, "text_encoders"),
        models_root,
    ]
    
    found_llms = []
    for d in llm_search_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".gguf") and ("qwen" in f.lower() or "mistral" in f.lower()):
                found_llms.append(os.path.join(d, f))
    
    if found_llms:
        for p in found_llms:
            if "4b" in os.path.basename(p).lower():
                llm_search = p
                break
        if not llm_search:
            llm_search = found_llms[0]
    
    return llm_search


def find_llm_for_flux2(final_model: str, llm_path: str = None, models_root: str = None) -> str:
    """Finds LLM for FLUX.2 (Mistral)"""
    llm_search = None
    
    if llm_path and os.path.exists(llm_path):
        return llm_path
    
    if models_root is None:
        models_root = shared_utils.MODELS_DIR
        
    diffusion_dir = os.path.dirname(final_model)
    
    llm_search_dirs = [
        diffusion_dir,
        os.path.join(diffusion_dir, "text_encoders"),
        os.path.join(diffusion_dir, "text_encoder"),
        os.path.join(models_root, "text_encoders"),
        models_root,
    ]
    
    found_llms = []
    for d in llm_search_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.endswith((".gguf", ".safetensors")) and "mistral" in f.lower():
                found_llms.append(os.path.join(d, f))
                
    if found_llms:
        llm_search = found_llms[0]
        
    return llm_search
