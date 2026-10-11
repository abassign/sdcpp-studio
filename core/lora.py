"""
Core LoRA - LoRA management
"""
import os
import json
import lora_manager
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_utils

# LoRA directory - use from shared_utils
LORA_DIR = shared_utils.LORAS_DIR


def list_loras(folder: str = None) -> list:
    """Lists all available LoRAs"""
    if folder is None:
        folder = LORA_DIR
    return lora_manager.list_lora_files(folder)


def filter_loras(files: list, model_path: str = None) -> list:
    """Returns all LoRAs without filtering - all LoRAs are usable"""
    return files


def get_lora_info(filename: str) -> dict:
    """Gets LoRA metadata"""
    path = os.path.join(LORA_DIR, filename)
    if not os.path.exists(path):
        return {"error": "File not found"}
    from core.lora_meta import load_meta_or_default, detect_lora_info
    return {"meta": load_meta_or_default(path), "detected": detect_lora_info(path)}


def create_lora_config(loras: list) -> str:
    """
    Creates JSON config file for LoRAs
    
    Args:
        loras: list of dicts with 'path', 'strength'
        
    Returns:
        Path to the temporary config file
    """
    lora_data = [{"path": l.get("path"), "strength": l.get("strength", 1.0)} for l in loras]
    return lora_manager.create_config_json(lora_data)
