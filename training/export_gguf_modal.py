"""
Export GGUF from the already-uploaded merged model (build-small-hackathon/storygrove-gemma-3-4b).

The merged model was saved as Gemma3Model (multimodal backbone).  Its state dict has keys:
  language_model.model.*   ← text transformer layers
  vision_tower.*           ← skip
  multi_modal_projector.*  ← skip

We remap language_model.model.* → model.* and save as Gemma3ForCausalLM (text_config),
then reload with Unsloth and call save_pretrained_gguf normally.
Unsloth accepts Gemma3ForCausalLM; only Gemma3Model was blocked.

Usage:
    modal run training/export_gguf_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "packaging")
    .pip_install("unsloth[colab-new]")
    .pip_install(
        "huggingface-hub>=0.25.0",
        "safetensors>=0.4.0",
        "accelerate>=1.0.0",
    )
)

app = modal.App("storygrove-gguf-export", image=image)

MERGED_REPO = "build-small-hackathon/storygrove-gemma-3-4b"
GGUF_REPO   = "build-small-hackathon/storygrove-gemma-3-4b-gguf"


def _push_folder(api, folder_path, repo_id, repo_type, commit_message, max_retries=3):
    import time
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            return
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Upload to {repo_id} failed after {max_retries} attempts") from exc
            wait = 15 * (attempt + 1)
            print(f"  attempt {attempt + 1} failed: {exc} — retrying in {wait}s")
            time.sleep(wait)


@app.function(
    gpu="L4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def export_gguf():
    import os, json, shutil
    from pathlib import Path
    from safetensors import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import snapshot_download, HfApi

    HF_TOKEN = os.environ["HF_TOKEN"]
    api = HfApi(token=HF_TOKEN)

    # ── Download merged model from HF ─────────────────────────────────────────
    merged_dir = Path("/tmp/storygrove-merged")
    print(f"Downloading {MERGED_REPO}...")
    snapshot_download(repo_id=MERGED_REPO, local_dir=str(merged_dir), token=HF_TOKEN)

    with open(merged_dir / "config.json") as f:
        config = json.load(f)
    arch = config.get("architectures", ["?"])[0]
    print(f"Downloaded architecture: {arch}")

    # ── Discover shards ────────────────────────────────────────────────────────
    index_path = merged_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    # Peek at tensor key prefixes for diagnostics
    with safe_open(str(merged_dir / shards[0]), framework="pt") as f:
        sample_keys = list(f.keys())[:15]
    print(f"Sample tensor keys: {sample_keys}")

    # ── Remap keys → Gemma3ForCausalLM layout ─────────────────────────────────
    # Gemma3Model (multimodal backbone) stores text weights under language_model.*
    # Gemma3ForCausalLM expects: model.* for the transformer + lm_head.weight
    text_dir = Path("/tmp/storygrove-text")
    text_dir.mkdir(exist_ok=True)

    all_tensors = {}
    for shard in shards:
        with safe_open(str(merged_dir / shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith("language_model.model."):
                    # language_model.model.embed_tokens.weight → model.embed_tokens.weight
                    new_key = key[len("language_model."):]
                elif key.startswith("language_model.lm_head."):
                    # language_model.lm_head.weight → lm_head.weight
                    new_key = key[len("language_model."):]
                elif key == "lm_head.weight":
                    new_key = key
                else:
                    # vision_tower.*, multi_modal_projector.*, etc.
                    continue
                all_tensors[new_key] = f.get_tensor(key)

    print(f"Remapped {len(all_tensors)} tensors. Sample: {list(all_tensors.keys())[:8]}")

    if not all_tensors:
        raise RuntimeError(
            "No tensors remapped — key structure differs from expected. "
            f"Actual sample keys: {sample_keys}"
        )

    # Single shard (4B bf16 ≈ 8 GB — fits in one file)
    save_file(all_tensors, str(text_dir / "model.safetensors"))
    del all_tensors

    # ── Build Gemma3ForCausalLM config ────────────────────────────────────────
    text_cfg = config.get("text_config", {})
    text_cfg["architectures"] = ["Gemma3ForCausalLM"]
    text_cfg.setdefault("model_type", "gemma3_text")
    # Gemma uses tied embeddings — no standalone lm_head weight needed
    text_cfg["tie_word_embeddings"] = True
    with open(text_dir / "config.json", "w") as f:
        json.dump(text_cfg, f, indent=2)

    # Copy tokenizer files
    for p in merged_dir.iterdir():
        if p.name.startswith("tokenizer") or p.name == "special_tokens_map.json":
            shutil.copy(p, text_dir / p.name)

    print(f"Saved Gemma3ForCausalLM to {text_dir}")

    # ── Load with Unsloth and export GGUF ─────────────────────────────────────
    # Import unsloth first so unsloth_zoo's env check passes
    import unsloth  # noqa: F401
    from unsloth import FastModel

    print("Loading remapped model with Unsloth...")
    gguf_model, gguf_tok = FastModel.from_pretrained(
        str(text_dir),
        max_seq_length=4096,
        load_in_4bit=False,
        token=HF_TOKEN,
    )

    print("Exporting GGUF Q4_K_M...")
    gguf_model.save_pretrained_gguf(
        str(text_dir),   # Unsloth appends _gguf → /tmp/storygrove-text_gguf/
        gguf_tok,
        quantization_method="q4_k_m",
    )
    # Unsloth creates <model_dir>_gguf/ regardless of the directory argument
    gguf_out = str(text_dir) + "_gguf"

    print(f"Pushing GGUF to {GGUF_REPO}...")
    _push_folder(api, gguf_out, GGUF_REPO, "model",
                 commit_message="StoryGrove fine-tuned Gemma 3 4B — GGUF Q4_K_M")
    print(f"GGUF live at https://huggingface.co/{GGUF_REPO}")
    print("Done!")


@app.local_entrypoint()
def main():
    export_gguf.remote()