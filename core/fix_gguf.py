"""
Core Fix GGUF - GGUF tensor fix utility

This module handles tensor conversion to make them compatible
with stable-diffusion.cpp
"""
import os
import json
import time

try:
    import gguf
    import numpy as np
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False


def is_fixed(model_path: str) -> bool:
    """Checks if a GGUF model has already been fixed"""
    if not model_path.endswith(".gguf"):
        return True
    fixed_marker = model_path + ".is_fixed.json"
    return os.path.exists(fixed_marker)


def mark_fixed(model_path: str):
    """Marks a model as fixed"""
    fixed_marker = model_path + ".is_fixed.json"
    with open(fixed_marker, 'w') as f:
        json.dump({"fixed": True, "timestamp": time.time()}, f)


def ensure_fixed(model_path: str) -> str:
    """Ensures that a GGUF model is fixed"""
    if not model_path.endswith(".gguf") or not os.path.exists(model_path):
        return model_path
    
    if is_fixed(model_path):
        return model_path
    
    if not GGUF_AVAILABLE:
        print("Warning: gguf module not available, skipping auto-fix")
        mark_fixed(model_path)
        return model_path
    
    print(f"Auto-Fix: Repairing GGUF tensor names for {os.path.basename(model_path)}...")
    
    try:
        temp_fixed = model_path + ".tmp_fix.gguf"
        if fix_gguf(model_path, temp_fixed):
            os.replace(temp_fixed, model_path)
            mark_fixed(model_path)
            print("Auto-Fix: Success. Model updated.")
        else:
            print("Auto-Fix: Failed.")
    except Exception as e:
        print(f"Auto-Fix error: {e}")
        if os.path.exists(temp_fixed):
            os.remove(temp_fixed)
    
    return model_path


def fix_gguf(input_path: str, output_path: str = None) -> bool:
    """GGUF tensor fix for SD-CPP"""
    if not GGUF_AVAILABLE:
        print("Error: gguf module not available")
        return False
    
    if output_path is None:
        output_path = input_path + ".tmp_fix.gguf"
    
    try:
        reader = gguf.GGUFReader(input_path, 'r')
    except Exception as e:
        print(f"Error reading GGUF: {e}")
        return False
    
    arch = reader.fields.get("general.architecture")
    if arch:
        arch = str(arch.parts[arch.data[0]]) if hasattr(arch, 'parts') else str(arch)
    else:
        arch = "unknown"

    is_flux = any("double_layers." in t.name or "double_blocks." in t.name for t in reader.tensors)
    is_sdxl = any("input_blocks.11." in t.name or "output_blocks.2." in t.name for t in reader.tensors)
    
    if is_flux:
        arch = "flux"
    elif is_sdxl:
        arch = "sdxl"
    else:
        # Default to "sd1" if it looks like a standard model (e.g. has input_blocks)
        # but doesn't match Flux/SDXL specifics.
        # SD 1.x has input_blocks.1. (standard)
        if any("input_blocks." in t.name for t in reader.tensors):
            arch = "sd1"
    
    print(f"Detected Architecture: {arch} (Flux: {is_flux}, SDXL: {is_sdxl})")
    
    writer = gguf.GGUFWriter(output_path, arch=arch)
    
    # Copy KV and ensure version metadata
    found_version = False
    for field in reader.fields.values():
        if field.name == "general.architecture": continue
        if field.name == "sd.version": found_version = True
        
        try:
            # Safer way to add KV
            val = field.parts[field.data[0]] if hasattr(field, 'parts') else field.data[0]
            writer.add_key_value(field.name, val, field.types[0])
        except:
            pass
    
    if not found_version and arch != "unknown":
        writer.add_key_value("sd.version", arch, gguf.GGUFValueType.STRING)
    
    # Copy tensors with renaming for Flux if needed
    for tensor in reader.tensors:
        name = tensor.name
        if is_flux and "double_layers." in name:
            name = name.replace("double_layers.", "model.diffusion_model.double_blocks.")
        writer.add_tensor(name, tensor.data, raw_shape=tensor.data.shape, raw_dtype=tensor.tensor_type)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file("Done.")
    return True
