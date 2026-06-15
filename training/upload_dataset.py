"""
Upload storygrove_sft.jsonl to HF dataset build-small-hackathon/storygrove-sft.

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
DATASET_REPO = "build-small-hackathon/storygrove-sft"
SFT_FILE = Path(__file__).parent / "data" / "storygrove_sft.jsonl"


def main():
    if not SFT_FILE.exists():
        print(f"ERROR: {SFT_FILE} not found. Run build_dataset.py first.")
        sys.exit(1)

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set in environment / .env")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
        print(f"Dataset repo ready: https://huggingface.co/datasets/{DATASET_REPO}")
    except Exception as e:
        if "401" in str(e) or "authorization" in str(e).lower():
            print(f"ERROR: HF_TOKEN is invalid or lacks write access: {e}")
            sys.exit(1)
        print(f"WARNING: could not create repo ({e}), trying upload anyway...")

    # Upload the JSONL file with retry
    size_kb = SFT_FILE.stat().st_size / 1024
    row_count = sum(1 for _ in SFT_FILE.open())
    print(f"Uploading {SFT_FILE.name} ({size_kb:.1f} KB, {row_count} rows)...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=str(SFT_FILE),
                path_in_repo="storygrove_sft.jsonl",
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message="Add StoryGrove SFT dataset (skeleton + beat examples)",
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"ERROR: upload failed after {max_retries} attempts: {e}")
                sys.exit(1)
            wait = 15 * (attempt + 1)
            print(f"  Upload attempt {attempt + 1} failed: {e} — retrying in {wait}s")
            time.sleep(wait)

    print(f"Done. Dataset live at: https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()
