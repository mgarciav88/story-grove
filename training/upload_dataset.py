"""
Upload training data to HF dataset build-small-hackathon/storygrove-sft.

Uploads:
  - storygrove_sft.jsonl  (SFT training examples)
  - eval_inputs.json      (held-out eval inputs)

Usage (from project root):
    python -m training.upload_dataset
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO", "build-small-hackathon/storygrove-sft")
DATA_DIR = Path(__file__).parent / "data"

UPLOADS = [
    (DATA_DIR / "storygrove_sft.jsonl", "storygrove_sft.jsonl", "Add StoryGrove SFT dataset (skeleton + beat examples)"),
    (DATA_DIR / "eval_inputs.json",     "eval_inputs.json",     "Add held-out eval inputs (25 stories, 5 paradigms × 3 age ranges)"),
]


def _upload(api: HfApi, local_path: Path, remote_path: str, commit_message: str, max_retries: int = 3):
    size_kb = local_path.stat().st_size / 1024
    print(f"Uploading {local_path.name} ({size_kb:.1f} KB)...")
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message=commit_message,
            )
            return
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"ERROR: upload failed after {max_retries} attempts: {e}")
                sys.exit(1)
            wait = 15 * (attempt + 1)
            print(f"  Attempt {attempt + 1} failed: {e} — retrying in {wait}s")
            time.sleep(wait)


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set in environment / .env")
        sys.exit(1)

    missing = [local for local, _, _ in UPLOADS if not local.exists()]
    if missing:
        for f in missing:
            print(f"ERROR: {f} not found. Run build_dataset.py first.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    try:
        api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)
        print(f"Dataset repo ready: https://huggingface.co/datasets/{DATASET_REPO}")
    except Exception as e:
        if "401" in str(e) or "authorization" in str(e).lower():
            print(f"ERROR: HF_TOKEN is invalid or lacks write access: {e}")
            sys.exit(1)
        print(f"WARNING: could not create repo ({e}), trying upload anyway...")

    for local_path, remote_path, commit_message in UPLOADS:
        _upload(api, local_path, remote_path, commit_message)

    print(f"\nDone. https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()