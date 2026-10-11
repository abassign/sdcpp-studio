#!/usr/bin/env python3
"""
GGUF Model Validation Module

Provides functions to validate GGUF model files and their mmproj (multimodal projector)
before using them with llama-cli for image-to-text tasks.
"""

import os
import sys
from typing import Optional

# Make the vendored gguf package importable regardless of the working directory.
# The repo vendors the huggingface 'gguf' library under libs/; adding it to
# sys.path lets `import gguf` and `from libs.gguf import ...` both work from
# anywhere (host shell, Docker /app, or a fresh clone).
_LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _LIBS_DIR not in sys.path:
    sys.path.insert(0, _LIBS_DIR)

# The internal modules of the vendored gguf package do `import gguf` /
# `from gguf.constants import ...`. To make that work we must also expose
# libs/gguf itself as a top-level package named gguf.
_GGUF_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _GGUF_PKG_DIR not in sys.path:
    sys.path.insert(0, _GGUF_PKG_DIR)

try:
    from libs.gguf import GGUFReader
except ImportError:
    # Fallback for direct execution
    from gguf import GGUFReader


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, code: str, message: str, details: str = ""):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"[{code}] {message}: {details}" if details else f"[{code}] {message}")


def get_gguf_metadata(path: str) -> dict:
    """
    Read metadata from a GGUF file.
    
    Args:
        path: Path to the GGUF file
        
    Returns:
        dict with metadata fields:
          - name: general.name
          - architecture: general.architecture  
          - version: general.version
          - file_size: size in bytes
          - tensor_count: number of tensors
    """
    try:
        reader = GGUFReader(path, 'r')
        
        metadata = {
            'name': None,
            'architecture': None,
            'version': None,
            'file_size': os.path.getsize(path),
            'tensor_count': len(reader.tensors),
            'fields': {}
        }
        
        # Extract string fields
        for key in ['general.name', 'general.architecture', 'general.version', 
                     'mmproj.architecture', 'tokenizer.chat_template']:
            field = reader.get_field(key)
            if field:
                try:
                    value = field.contents()
                    metadata[key] = value
                    metadata['fields'][key] = value
                except Exception:
                    pass
        
        return metadata
        
    except Exception as e:
        raise ValidationError(
            code="GGUF_READ_ERROR",
            message=f"Failed to read GGUF file: {os.path.basename(path)}",
            details=str(e)
        )


def validate_gguf_file(path: str) -> dict:
    """
    Validate a GGUF file and return its metadata.
    
    Args:
        path: Path to the GGUF file
        
    Returns:
        dict with:
          - valid: True if valid
          - metadata: extracted metadata
          
    Raises:
        ValidationError: If the file is invalid
    """
    if not os.path.exists(path):
        raise ValidationError(
            code="FILE_NOT_FOUND",
            message="GGUF file not found",
            details=f"Path: {path}"
        )
    
    if not path.endswith('.gguf'):
        raise ValidationError(
            code="INVALID_EXTENSION",
            message="File does not have .gguf extension",
            details=f"File: {os.path.basename(path)}"
        )
    
    # Try to read metadata
    metadata = get_gguf_metadata(path)
    
    return {
        'valid': True,
        'metadata': metadata
    }


def validate_mmproj_compatibility(gguf_path: str, mmproj_path: str) -> dict:
    """
    Validate that a model GGUF and mmproj are compatible for multimodal use.
    
    Args:
        gguf_path: Path to the main model GGUF file
        mmproj_path: Path to the mmproj GGUF file
        
    Returns:
        dict with:
          - compatible: True if compatible
          - model_arch: model's architecture
          - mmproj_arch: mmproj's architecture
          - warnings: list of warnings
          
    Raises:
        ValidationError: If validation fails
    """
    warnings = []
    
    # Validate both files exist and are readable
    model_info = validate_gguf_file(gguf_path)
    mmproj_info = validate_gguf_file(mmproj_path)
    
    model_meta = model_info['metadata']
    mmproj_meta = mmproj_info['metadata']
    
    model_arch = model_meta.get('general.architecture')
    mmproj_arch = mmproj_meta.get('mmproj.architecture') or mmproj_meta.get('general.architecture')
    
    # Check for vision/multimodal capability
    vision_archs = ['qwen2_vl', 'qwen2.5_vl', 'llama', 'glm', 'clip', 'qwen', 'siglip', 'idefics2', 'minicpmv']
    
    if model_arch and not any(v in str(model_arch).lower() for v in vision_archs):
        warnings.append(f"Model architecture '{model_arch}' may not support vision")
    
    # Check architecture consistency (if both have architecture fields)
    if model_arch and mmproj_arch:
        # Some models have different but compatible architectures
        # (e.g., qwen2_vl model with qwen2_vl mmproj)
        model_lower = str(model_arch).lower()
        mmproj_lower = str(mmproj_arch).lower()
        
        # Allow if architectures match or if one is contained in the other
        if model_lower != mmproj_lower and model_lower not in mmproj_lower and mmproj_lower not in model_lower:
            warnings.append(
                f"Architecture mismatch: model='{model_arch}', mmproj='{mmproj_arch}'. "
                f"This may cause compatibility issues."
            )
    
    return {
        'compatible': True,
        'model_arch': model_arch,
        'mmproj_arch': mmproj_arch,
        'model_name': model_meta.get('general.name') or os.path.basename(gguf_path),
        'mmproj_name': mmproj_meta.get('general.name') or os.path.basename(mmproj_path),
        'warnings': warnings
    }


def validate_model_for_inference(model_path: str, mmproj_path: str) -> dict:
    """
    Full validation of a model and mmproj for image-to-text inference.
    
    This is the main entry point for pre-execution validation.
    
    Args:
        model_path: Path to the model GGUF file
        mmproj_path: Path to the mmproj GGUF file
        
    Returns:
        dict with:
          - valid: True if all validations pass
          - model_info: validated model metadata
          - mmproj_info: validated mmproj metadata
          - compatibility: compatibility check result
          
    Raises:
        ValidationError: If any validation fails
    """
    errors = []
    
    # 1. Validate model file
    try:
        model_info = validate_gguf_file(model_path)
    except ValidationError as e:
        errors.append(f"Model: {e.message}")
    
    # 2. Validate mmproj file  
    try:
        mmproj_info = validate_gguf_file(mmproj_path)
    except ValidationError as e:
        errors.append(f"MMProj: {e.message}")
    
    # If either file failed to validate, don't continue
    if errors:
        raise ValidationError(
            code="VALIDATION_FAILED",
            message="Model validation failed",
            details="; ".join(errors)
        )
    
    # 3. Check compatibility
    try:
        compatibility = validate_mmproj_compatibility(model_path, mmproj_path)
    except ValidationError as e:
        raise ValidationError(
            code="COMPATIBILITY_ERROR",
            message="Model and mmproj are not compatible",
            details=str(e)
        )
    
    return {
        'valid': True,
        'model_info': model_info,
        'mmproj_info': mmproj_info,
        'compatibility': compatibility
    }


def format_validation_error(error: ValidationError) -> dict:
    """
    Format a ValidationError for JSON output to frontend.
    
    Args:
        error: ValidationError instance
        
    Returns:
        dict suitable for JSON serialization
    """
    return {
        'error': error.message,
        'code': error.code,
        'details': error.details
    }


if __name__ == "__main__":
    # Test script - validate GGUF files passed as arguments
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python validation.py <model.gguf> <mmproj.gguf>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    mmproj_path = sys.argv[2]
    
    try:
        result = validate_model_for_inference(model_path, mmproj_path)
        print(json.dumps(result, indent=2, default=str))
    except ValidationError as e:
        print(json.dumps(format_validation_error(e), indent=2))
        sys.exit(1)
