"""
Download and verify CUAD (Contract Understanding Atticus Dataset) v1 artifacts.

Sources:
- Hugging Face: theatticusproject/cuad
- GitHub: https://github.com/The-Atticus-Project/cuad
"""

import os
import sys
import json
import hashlib
import zipfile
import io
import urllib.request
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("ml/data/raw/cuad")

HF_REPO_ID = "theatticusproject/cuad"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/The-Atticus-Project/cuad/main"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_file(url: str, dest_path: Path, description: str):
    print(f"Downloading {description} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "ClauseGuard-Dataset-Recon/1.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
    with open(dest_path, "wb") as f:
        f.write(content)
    print(f"  -> Saved to {dest_path} ({len(content):,} bytes)")

def download_cuad():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": "Contract Understanding Atticus Dataset (CUAD) v1",
        "download_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "sources": {
            "github": "https://github.com/The-Atticus-Project/cuad",
            "huggingface": "https://huggingface.co/datasets/theatticusproject/cuad",
            "paper": "https://arxiv.org/abs/2103.06268",
            "official_site": "https://www.atticusprojectai.org/cuad/"
        },
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "files": {}
    }

    # 1. Download category_descriptions.csv from GitHub
    cat_desc_path = DATA_DIR / "category_descriptions.csv"
    if not cat_desc_path.exists():
        download_file(
            f"{GITHUB_RAW_BASE}/category_descriptions.csv",
            cat_desc_path,
            "category_descriptions.csv"
        )
    manifest["files"]["category_descriptions.csv"] = {
        "size_bytes": cat_desc_path.stat().st_size,
        "sha256": compute_sha256(cat_desc_path),
        "source": f"{GITHUB_RAW_BASE}/category_descriptions.csv"
    }

    # 2. Download data.zip from GitHub and extract CUADv1.json, test.json, train_separate_questions.json
    data_zip_path = DATA_DIR / "data.zip"
    if not data_zip_path.exists():
        download_file(
            f"{GITHUB_RAW_BASE}/data.zip",
            data_zip_path,
            "data.zip (benchmark splits and CUADv1.json)"
        )
    manifest["files"]["data.zip"] = {
        "size_bytes": data_zip_path.stat().st_size,
        "sha256": compute_sha256(data_zip_path),
        "source": f"{GITHUB_RAW_BASE}/data.zip"
    }

    print("Extracting data.zip...")
    with zipfile.ZipFile(data_zip_path, "r") as z:
        for filename in z.namelist():
            target_path = DATA_DIR / filename
            if not target_path.exists():
                z.extract(filename, DATA_DIR)
                print(f"  -> Extracted {filename} ({target_path.stat().st_size:,} bytes)")
            manifest["files"][filename] = {
                "size_bytes": target_path.stat().st_size,
                "sha256": compute_sha256(target_path),
                "source": "Extracted from GitHub data.zip"
            }

    # 3. Download master_clauses.csv and README from Hugging Face
    hf_files = [
        ("CUAD_v1/master_clauses.csv", "master_clauses.csv"),
        ("CUAD_v1/CUAD v1 ReadMe _ Datasheet/CUAD_v1_README.txt", "CUAD_v1_README.txt")
    ]
    
    from huggingface_hub import hf_hub_download
    for hf_subpath, local_name in hf_files:
        target_path = DATA_DIR / local_name
        if not target_path.exists():
            print(f"Downloading {local_name} from HF ({hf_subpath})...")
            cached_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_subpath,
                repo_type="dataset"
            )
            import shutil
            shutil.copy2(cached_path, target_path)
            print(f"  -> Saved {local_name} ({target_path.stat().st_size:,} bytes)")
        manifest["files"][local_name] = {
            "size_bytes": target_path.stat().st_size,
            "sha256": compute_sha256(target_path),
            "source": f"HuggingFace {HF_REPO_ID}:{hf_subpath}"
        }

    # Verify JSON integrity
    print("\nVerifying JSON integrity...")
    for json_file in ["CUADv1.json", "test.json", "train_separate_questions.json"]:
        p = DATA_DIR / json_file
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        contracts_count = len(data.get("data", []))
        total_qas = sum(
            len(qa)
            for contract in data.get("data", [])
            for p_elem in contract.get("paragraphs", [])
            for qa in [p_elem.get("qas", [])]
        )
        print(f"  ✓ {json_file}: Valid JSON | {contracts_count} contracts | {total_qas:,} QA pairs")

    # Save manifest
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

if __name__ == "__main__":
    download_cuad()
