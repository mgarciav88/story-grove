"""
Build storygrove_sft.jsonl from stories.json.

Reads the hand-authored stories and produces TRL conversational-format SFT rows,
using the exact same prompt builders the app uses at inference (train/inference parity).

Usage (from project root):
    python -m training.build_dataset
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storygrove.prompts import (
    load_prompts,
    build_skeleton_prompt,
    build_first_beat_prompt,
    build_continue_beat_prompt,
)
from storygrove.story_generator import StorySession, REQUIRED_SKELETON_KEYS

DATA_DIR = Path(__file__).parent / "data"
STORIES_FILE = DATA_DIR / "stories.json"
OUTPUT_FILE = DATA_DIR / "storygrove_sft.jsonl"

META_LEAK_WORDS = frozenset({"paradigm", "beat arc", "story bible", "beat_arc", "story plan",
                             "internal_goal", "external_goal", "symbolic_object"})


def validate_skeleton(skeleton: dict) -> list[str]:
    errors = []
    missing = REQUIRED_SKELETON_KEYS - set(skeleton.keys())
    if missing:
        errors.append(f"missing keys: {missing}")
    arc = skeleton.get("beat_arc", [])
    if not isinstance(arc, list) or len(arc) != 5:
        errors.append(f"beat_arc must be a list of 5, got: {arc!r}")
    return errors


def validate_beat(beat: dict, idx: int, is_final_expected: bool) -> list[str]:
    errors = []
    if not beat.get("narrative", "").strip():
        errors.append(f"beat {idx}: empty narrative")
    choices = beat.get("choices", [])
    if len(choices) != 3:
        errors.append(f"beat {idx}: expected 3 choices, got {len(choices)}")
    if beat.get("is_final") != is_final_expected:
        errors.append(f"beat {idx}: is_final={beat.get('is_final')}, expected {is_final_expected}")
    return errors


def has_meta_leak(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in META_LEAK_WORDS)


def build_rows(story: dict, prompts: dict) -> list[dict]:
    """Build all SFT rows for a single story (1 skeleton + 5 beat rows)."""
    rows = []
    inp = story["inputs"]
    skeleton = story["skeleton"]
    beats = story["beats"]

    # ── Skeleton row ────────────────────────────────────────────────────────────
    skeleton_user = build_skeleton_prompt(
        character=inp["character"],
        age_range=inp["age_range"],
        theme=inp["theme"],
        language=inp.get("language", "English"),
        max_beats=5,
        prompts=prompts,
    )
    # Assistant output is compact JSON (what _extract_json + json.loads parses)
    skeleton_assistant = json.dumps(skeleton, ensure_ascii=False, separators=(",", ":"))
    rows.append({"messages": [
        {"role": "system", "content": prompts["skeleton_system_prompt"]},
        {"role": "user", "content": skeleton_user},
        {"role": "assistant", "content": skeleton_assistant},
    ]})

    # ── Beat rows ────────────────────────────────────────────────────────────────
    session = StorySession(
        character=inp["character"],
        age_range=inp["age_range"],
        theme=inp["theme"],
        language=inp.get("language", "English"),
        skeleton=skeleton,
    )

    for i, beat_data in enumerate(beats):
        is_final = beat_data["is_final"]

        if i == 0:
            # beat_number starts at 0; set to 1 to match app's start_story increment
            session.beat_number = 1
            beat_user = build_first_beat_prompt(
                character=session.character,
                age_range=session.age_range,
                theme=session.theme,
                language=session.language,
                prompts=prompts,
                skeleton=skeleton,
            )
        else:
            prev = beats[i - 1]
            choice_made = prev.get("choice_made")
            if not choice_made:
                print(f"  WARNING: beat {i} in story '{inp['character']}' has no choice_made — falling back to choices[0]")
                choice_made = prev["choices"][0]
            # beat_number = i mirrors the app: i beats completed, about to generate beat i+1
            session.beat_number = i
            session.add_beat(prev["narrative"], choice_made)
            beat_user = build_continue_beat_prompt(
                session=session,
                choice=choice_made,
                is_final=is_final,
                prompts=prompts,
            )

        # Assistant output = narrative + newline + choices JSON
        beat_assistant = (
            beat_data["narrative"].strip()
            + "\n"
            + json.dumps(
                {"choices": beat_data["choices"], "is_final": is_final},
                ensure_ascii=False,
            )
        )

        rows.append({"messages": [
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": beat_user},
            {"role": "assistant", "content": beat_assistant},
        ]})

    return rows


def main():
    if not STORIES_FILE.exists():
        print(f"ERROR: {STORIES_FILE} not found. Author stories first.")
        sys.exit(1)

    with open(STORIES_FILE) as f:
        stories = json.load(f)

    prompts = load_prompts()
    rows = []
    skipped = 0
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    warnings = []

    # Duplicate detection: track (character, paradigm, age_range) tuples
    seen_keys: set[tuple] = set()

    for idx, story in enumerate(stories):
        tag = f"story[{idx}] {story['inputs']['character']!r}"
        errors = []

        # Duplicate detection
        dedup_key = (
            story["inputs"].get("character", "").lower(),
            story.get("skeleton", {}).get("paradigm", "").lower(),
            story["inputs"].get("age_range", "").lower(),
        )
        if dedup_key in seen_keys:
            warnings.append(f"{tag}: duplicate (character+paradigm+age) — both copies included")
        seen_keys.add(dedup_key)

        errors += [f"skeleton: {e}" for e in validate_skeleton(story["skeleton"])]

        beats = story.get("beats", [])
        if len(beats) != 5:
            errors.append(f"expected 5 beats, got {len(beats)}")
        for bi, beat in enumerate(beats):
            errors += validate_beat(beat, bi + 1, is_final_expected=(bi == 4))
            if has_meta_leak(beat.get("narrative", "")):
                warnings.append(f"{tag} beat {bi + 1}: possible meta-language leak")

        if errors:
            print(f"SKIP {tag}: {'; '.join(errors)}")
            skipped += 1
            continue

        try:
            story_rows = build_rows(story, prompts)
        except Exception as e:
            print(f"SKIP {tag}: build failed — {e}")
            skipped += 1
            continue

        rows.extend(story_rows)
        paradigm = story["skeleton"].get("paradigm", "Unknown")
        age = story["inputs"]["age_range"]
        counts[paradigm][age] += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Integrity check: count lines written vs rows in memory
    lines_on_disk = sum(1 for _ in open(OUTPUT_FILE))
    if lines_on_disk != len(rows):
        print(f"ERROR: wrote {len(rows)} rows but file has {lines_on_disk} lines — possible disk-full issue")
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print(f"Stories processed : {len(stories) - skipped}/{len(stories)}")
    print(f"SFT rows written  : {len(rows)}  ({OUTPUT_FILE})")
    print(f"Skipped           : {skipped}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠  {w}")

    print("\nDistribution (paradigm × age):")
    for paradigm, ages in sorted(counts.items()):
        for age, n in sorted(ages.items()):
            print(f"  {paradigm:20s} {age:12s} → {n} stories")


if __name__ == "__main__":
    main()
