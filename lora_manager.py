import os
import glob
import json

def list_lora_files(folder_path):
    """Returns a list of .safetensors filenames."""
    if not os.path.exists(folder_path):
        return []
    try:
        pattern = os.path.join(folder_path, "*.safetensors")
        files = glob.glob(pattern)
        return sorted([os.path.basename(f) for f in files])
    except Exception as e:
        print(f"Error scanning LoRAs: {e}")
        return []

def create_config_json(lora_data, save_path="/app/lora_run_config.json"):
    """
    Saves the config for the worker process.
    lora_data: List of dicts [{'path': str, 'strength': float}]
    """
    # Convert formatting for the worker [path, strength, active]
    worker_format = []
    for item in lora_data:
        worker_format.append([item['path'], float(item['strength']), True])

    with open(save_path, "w") as f:
        json.dump(worker_format, f)
    return save_path
