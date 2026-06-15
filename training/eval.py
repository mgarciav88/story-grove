"""
Evaluate base Gemma vs fine-tuned StoryGrove model on held-out inputs.

Metrics (all automatic, no API cost):
  - skeleton_validity_at1    % skeletons parsing with all required keys + 5-item beat_arc, first try
  - beat_choices_validity    % beats ending in parseable {"choices":[3], "is_final":bool}
  - avg_retries              average retries across all beat generations
  - paradigm_accuracy        % where predicted paradigm == expected_paradigm
  - meta_leakage_rate        % beats containing skeleton meta-words
  - schema_completeness      avg fraction of required skeleton keys that are non-empty

Optional (requires Anthropic API, off by default):
  --judge    blind pairwise LLM-as-judge — NOT YET IMPLEMENTED, raises NotImplementedError

Usage (from project root):
    python -m training.eval --base google/gemma-3-4b-it --tuned build-small-hackathon/storygrove-gemma-3-4b
    python -m training.eval --base google/gemma-3-4b-it --tuned build-small-hackathon/storygrove-gemma-3-4b --device cpu
"""

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from storygrove.story_generator import (
    StorySession,
    parse_streaming_response,
    _extract_json,
)
from storygrove.prompts import (
    load_prompts,
    build_skeleton_prompt,
    build_first_beat_prompt,
    build_continue_beat_prompt,
)

REQUIRED_SKELETON_KEYS = {
    "paradigm", "title", "protagonist", "character_appearance",
    "initial_trait", "initial_emotion", "main_conflict",
    "external_goal", "internal_goal", "mentor", "ally", "antagonist",
    "setting", "symbolic_object", "value_learned", "final_emotion",
    "theme", "beat_arc",
}

META_LEAK_PATTERN = re.compile(
    r"(paradigm|beat arc|beat_arc|story bible|story plan|internal.goal|"
    r"external.goal|symbolic.object|value.learned|final.emotion)",
    re.IGNORECASE,
)

DATA_DIR = Path(__file__).parent / "data"
EVAL_FILE = DATA_DIR / "eval_inputs.json"
RESULTS_FILE = DATA_DIR / "eval_results.json"


def _check_vram(device: str) -> None:
    if device == "cpu":
        return
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU (slow)")
        return
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    if free_gb < 10:
        print(f"WARNING: only {free_gb:.1f} GB VRAM free — loading two 4B models sequentially "
              f"may OOM. Consider --device cpu or running on a larger GPU.")


def load_model(model_id: str, hf_token: str | None, device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=hf_token,
    )
    return model, tokenizer


def generate_skeleton_once(
    inp: dict, model, tokenizer, prompts: dict, device: str
) -> tuple[dict | None, str]:
    messages = [
        {"role": "system", "content": prompts["skeleton_system_prompt"]},
        {"role": "user", "content": build_skeleton_prompt(
            character=inp["character"],
            age_range=inp["age_range"],
            theme=inp["theme"],
            language=inp["language"],
            max_beats=5,
            prompts=prompts,
        )},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, enable_thinking=False,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    try:
        parsed = json.loads(_extract_json(raw))
        return parsed, raw
    except Exception:
        return None, raw


def run_eval(model, tokenizer, eval_inputs: list[dict], prompts: dict, device: str) -> dict:
    skeleton_valid = 0
    beat_choices_valid = 0
    total_beats = 0
    paradigm_correct = 0
    paradigm_total = 0
    meta_leaked_beats = 0
    schema_completeness_scores = []
    retry_counts = []

    for inp in eval_inputs:
        print(f"Running eval for input: {inp}")
        # ── Skeleton ──────────────────────────────────────────────────────────
        skeleton, _ = generate_skeleton_once(inp, model, tokenizer, prompts, device)

        if skeleton is not None:
            filled = sum(1 for k in REQUIRED_SKELETON_KEYS if skeleton.get(k))
            schema_completeness_scores.append(filled / len(REQUIRED_SKELETON_KEYS))

            arc = skeleton.get("beat_arc", [])
            if (
                REQUIRED_SKELETON_KEYS - {"beat_arc"} <= set(skeleton.keys())
                and isinstance(arc, list) and len(arc) == 5
            ):
                skeleton_valid += 1

            if "expected_paradigm" in inp:
                paradigm_total += 1
                if skeleton.get("paradigm", "").strip().lower() == inp["expected_paradigm"].lower():
                    paradigm_correct += 1
        else:
            schema_completeness_scores.append(0.0)

        # ── Beats — full 5-beat playthrough, auto-picking "choice 0" ─────────
        sk = skeleton or {}
        session = StorySession(
            character=inp["character"],
            age_range=inp["age_range"],
            theme=inp["theme"],
            language=inp["language"],
            skeleton=sk,
        )
        # mirrors start_story: beat_number is incremented to 1 before beat 1 is generated
        session.beat_number = 1

        last_narrative = ""

        for beat_i in range(5):
            is_final = beat_i == 4

            if beat_i == 0:
                user_content = build_first_beat_prompt(
                    character=session.character,
                    age_range=session.age_range,
                    theme=session.theme,
                    language=session.language,
                    prompts=prompts,
                    skeleton=sk,
                )
            else:
                # Feed the actual generated narrative from the previous beat back in,
                # mirroring the real app (single-turn SFT has no history, so story_so_far
                # is the only source of prior context).
                session.add_beat(last_narrative, "choice 0")
                user_content = build_continue_beat_prompt(
                    session=session,
                    choice="choice 0",
                    is_final=is_final,
                    prompts=prompts,
                )

            messages = [
                {"role": "system", "content": prompts["system_prompt"]},
                {"role": "user", "content": user_content},
            ]
            inputs_t = tokenizer.apply_chat_template(
                messages, tokenize=True, enable_thinking=False,
                add_generation_prompt=True, return_dict=True, return_tensors="pt",
            )
            inputs_t = {k: v.to(device) for k, v in inputs_t.items()}

            retries = 0
            last_narrative = ""
            last_beat_obj = None
            beat_text = ""

            for attempt in range(3):
                with torch.no_grad():
                    output = model.generate(**inputs_t, max_new_tokens=1024, do_sample=False)
                new_tokens = output[0][inputs_t["input_ids"].shape[1]:]
                beat_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                last_narrative, last_beat_obj = parse_streaming_response(beat_text)
                if last_beat_obj is not None:
                    break
                retries += 1

            total_beats += 1
            retry_counts.append(retries)

            if last_beat_obj is not None and len(last_beat_obj.choices) == 3:
                beat_choices_valid += 1

            if META_LEAK_PATTERN.search(beat_text):
                meta_leaked_beats += 1

            # Mirrors continue_story: beat_number increments after each beat
            session.beat_number += 1

    n = len(eval_inputs)
    return {
        "skeleton_validity_at1": skeleton_valid / n if n else 0,
        "beat_choices_validity": beat_choices_valid / total_beats if total_beats else 0,
        "avg_retries": sum(retry_counts) / len(retry_counts) if retry_counts else 0,
        "paradigm_accuracy": paradigm_correct / paradigm_total if paradigm_total else None,
        "meta_leakage_rate": meta_leaked_beats / total_beats if total_beats else 0,
        "schema_completeness": sum(schema_completeness_scores) / n if n else 0,
        "n_stories": n,
        "n_beats": total_beats,
    }


def print_comparison(base: dict, tuned: dict):
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    def arrow(mk):
        bv, tv = base.get(mk), tuned.get(mk)
        if bv is None or tv is None:
            return "  "
        if mk in ("avg_retries", "meta_leakage_rate"):
            return "✅" if tv < bv else ("❌" if tv > bv else "➡")
        return "✅" if tv > bv else ("❌" if tv < bv else "➡")

    metrics = [
        ("skeleton_validity_at1", "Skeleton valid@1"),
        ("beat_choices_validity", "Beat choices valid"),
        ("avg_retries",           "Avg retries (↓ better)"),
        ("paradigm_accuracy",     "Paradigm accuracy"),
        ("meta_leakage_rate",     "Meta-leakage (↓ better)"),
        ("schema_completeness",   "Schema completeness"),
    ]
    print(f"\n{'Metric':<30} {'Base':>10} {'Tuned':>10}  ")
    print("─" * 55)
    for metric_key, label in metrics:
        print(f"{label:<30} {fmt(base.get(metric_key)):>10} {fmt(tuned.get(metric_key)):>10}  {arrow(metric_key)}")  # noqa: E501


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",   default="google/gemma-3-4b-it")
    parser.add_argument("--tuned",  default="build-small-hackathon/storygrove-gemma-3-4b")
    parser.add_argument("--device", default="auto",
                        help="Torch device: 'auto', 'cuda', 'cpu' (default: auto)")
    parser.add_argument("--judge",  action="store_true",
                        help="Blind pairwise LLM-as-judge (NOT YET IMPLEMENTED)")
    args = parser.parse_args()

    if args.judge:
        raise NotImplementedError(
            "--judge is not yet implemented. It requires an Anthropic API key "
            "and blind pairwise evaluation logic. Remove this flag to run automatic metrics only."
        )

    if not EVAL_FILE.exists():
        print(f"ERROR: {EVAL_FILE} not found.")
        sys.exit(1)

    import os
    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    device = args.device
    _check_vram(device)

    with open(EVAL_FILE) as f:
        eval_inputs = json.load(f)

    prompts = load_prompts()
    results = {}

    for label, model_id in [("base", args.base), ("tuned", args.tuned)]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {label} ({model_id})")
        model, tokenizer = load_model(model_id, hf_token, device)
        results[label] = run_eval(model, tokenizer, eval_inputs, prompts, device)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_comparison(results["base"], results["tuned"])

    with open(RESULTS_FILE, "w") as f:
        json.dump({"base_model": args.base, "tuned_model": args.tuned, **results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()