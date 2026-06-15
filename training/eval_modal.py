"""
Run base vs fine-tuned eval on Modal (L4 GPU, ~30 min).
Results are printed to Modal logs.

Usage:
    modal run training/eval_modal.py
    modal run training/eval_modal.py --base google/gemma-3-4b-it --tuned build-small-hackathon/storygrove-gemma-3-4b
"""

import modal
from pathlib import Path

_project_root = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.45.0",
        "huggingface-hub>=0.25.0",
        "accelerate>=1.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
        "pydantic>=2.0",
        "soundfile>=0.12",
        "diffusers",
    )
    .add_local_python_source("storygrove", "training")
    .add_local_file(
        str(_project_root / "storygrove" / "configs" / "prompts.yaml"),
        remote_path="/root/storygrove/configs/prompts.yaml",
    )
    .add_local_file(
        str(_project_root / "training" / "data" / "eval_inputs.json"),
        remote_path="/root/training/data/eval_inputs.json",
    )
)

app = modal.App("storygrove-eval", image=image)


@app.function(
    gpu="L4",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eval(
    base_model: str = "google/gemma-3-4b-it",
    tuned_model: str = "build-small-hackathon/storygrove-gemma-3-4b",
):
    import os
    import gc
    import json
    import torch

    from training.eval import load_model, run_eval as _run_eval, print_comparison, EVAL_FILE
    from storygrove.prompts import load_prompts

    hf_token = os.environ["HF_TOKEN"]

    if not EVAL_FILE.exists():
        print(f"ERROR: eval_inputs.json not found at {EVAL_FILE}")
        return

    with open(EVAL_FILE) as f:
        eval_inputs = json.load(f)
    print(f"Loaded {len(eval_inputs)} eval inputs.")

    prompts = load_prompts()
    results = {}

    for label, model_id in [("base", base_model), ("tuned", tuned_model)]:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label} ({model_id})")
        model, tokenizer = load_model(model_id, hf_token, device="cuda")
        results[label] = _run_eval(model, tokenizer, eval_inputs, prompts, device="cuda")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print_comparison(results["base"], results["tuned"])
    print("\nFull results:")
    print(json.dumps({"base_model": base_model, "tuned_model": tuned_model, **results}, indent=2))


@app.local_entrypoint()
def main(
    base: str = "google/gemma-3-4b-it",
    tuned: str = "build-small-hackathon/storygrove-gemma-3-4b",
):
    run_eval.remote(base_model=base, tuned_model=tuned)
