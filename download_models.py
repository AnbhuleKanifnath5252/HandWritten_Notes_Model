"""
Model Downloader for PaddleOCR-VL and Qwen2.5-VL
Run this script from PowerShell:
    python download_models.py
"""

import os
import sys
from huggingface_hub import snapshot_download

# Define model repositories
# Note: If your system has 8 GB RAM and no GPU, Qwen/Qwen2.5-VL-3B-Instruct is recommended over 7B.
MODELS = {
    "PaddleOCR-VL": {
        "repo_id": "PaddlePaddle/PaddleOCR-VL",
        "local_dir": os.path.abspath("./models/PaddleOCR-VL"),
    },
    "Qwen2.5-VL": {
        # You can change this to "Qwen/Qwen2.5-VL-3B-Instruct" if you have 8GB RAM
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_dir": os.path.abspath("./models/Qwen2.5-VL-7B"),
    }
}

def download_model(name, info):
    print(f"\n=======================================================")
    print(f"[*] Downloading {name} from '{info['repo_id']}'...")
    print(f"[*] Destination: {info['local_dir']}")
    print(f"=======================================================")
    try:
        os.makedirs(info["local_dir"], exist_ok=True)
        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=info["local_dir"],
            resume_download=True,
            max_workers=4
        )
        print(f"[+] Successfully downloaded {name} to {info['local_dir']}\n")
    except Exception as e:
        print(f"[-] Error downloading {name}: {e}", file=sys.stderr)

def main():
    print("Starting Model Downloads...")
    for name, info in MODELS.items():
        download_model(name, info)
    print("\n[+] All downloads complete! You can now start the frontend application.")

if __name__ == "__main__":
    main()

