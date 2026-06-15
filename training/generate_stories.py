"""
Generate training/data/stories.json by calling Claude claude-sonnet-4-6.

Each call produces one complete story playthrough (skeleton + 5 beats) in the
exact schema that build_dataset.py expects.

Usage:
    uv add anthropic          # one-time install
    python -m training.generate_stories

Saves progress after each story so you can restart without losing work.
Prints a running token-cost estimate at the end.

Cost estimate: ~800 tokens in + ~1200 tokens out per story × 40 stories
               ≈ $1.00–1.50 total with claude-sonnet-4-6.
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "stories.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Story slots to generate ────────────────────────────────────────────────────
# Each entry: (paradigm, age_range, protagonist_hint, theme_hint)
STORY_SLOTS = [
    # Identity (6 stories)
    ("Identity", "3-5 years",  "Bumble, a small golden bee",          "discovering your own way to shine"),
    ("Identity", "3-5 years",  "Luna, a tiny cloud who makes drizzles","you are exactly who you need to be"),
    ("Identity", "6-8 years",  "Pip, a young fox",                    "finding what makes you YOU"),
    ("Identity", "6-8 years",  "Marco, a blue crayon",                "ordinary things have extraordinary gifts"),
    ("Identity", "9-12 years", "Zara, a young witch",                 "small gifts can change the world"),
    ("Identity", "9-12 years", "Eko, a stone golem",                  "choosing who you want to become"),

    # Courage (6 stories)
    ("Courage",  "3-5 years",  "Tiny, a little brown mouse",          "one small brave step changes everything"),
    ("Courage",  "3-5 years",  "Pebble, a hermit crab",               "the world outside the shell is wonderful"),
    ("Courage",  "6-8 years",  "Finn, a young penguin",               "courage grows when you take the leap"),
    ("Courage",  "6-8 years",  "Nadia, a girl who loves to dance",    "doing the scary thing because it matters"),
    ("Courage",  "9-12 years", "Oskar, a teenage dragon",             "the only way out of fear is through it"),
    ("Courage",  "9-12 years", "River, a water sprite",               "stillness holds no danger — just peace"),

    # Friendship (7 stories)
    ("Friendship", "3-5 years",  "Spike and Softy, two hedgehogs",      "love always finds a way"),
    ("Friendship", "3-5 years",  "Dewdrop, a lonely raindrop",          "true friends welcome all of you"),
    ("Friendship", "3-5 years",  "Bop, a small toy robot",              "real friends don't have to be perfect"),
    ("Friendship", "6-8 years",  "Leaf, a tree sprite who just moved",  "new places bring new friends"),
    ("Friendship", "6-8 years",  "Clio who loves books and Mika who loves running", "opposites can be best friends"),
    ("Friendship", "6-8 years",  "Patch, a friendly scarecrow",         "showing your real self is the start of friendship"),
    ("Friendship", "9-12 years", "Astrid, a coastal girl",              "trust must be earned and kept"),
    ("Friendship", "9-12 years", "Thorn, a rose who keeps others away", "connection requires letting your guard down"),

    # Responsibility (6 stories)
    ("Responsibility", "3-5 years",  "Coco, a fluffy golden puppy",        "caring for something helps it grow"),
    ("Responsibility", "3-5 years",  "Plum, a small piglet",               "owning your mistakes makes things right"),
    ("Responsibility", "6-8 years",  "Felix, a curious boy",               "a promise is something to be kept"),
    ("Responsibility", "6-8 years",  "Lena, a thoughtful girl",            "doing the right thing even when it is hard"),
    ("Responsibility", "9-12 years", "Kael, a wizard apprentice",          "one moment of carelessness can affect many"),
    ("Responsibility", "9-12 years", "Nova, a young star keeper",          "tending the small things keeps the world bright"),

    # Curiosity (7 stories)
    ("Curiosity", "3-5 years",  "Dot, a tiny caterpillar",             "wondering leads you somewhere wonderful"),
    ("Curiosity", "3-5 years",  "Cluck, a cheerful chicken",           "asking why is how we learn"),
    ("Curiosity", "6-8 years",  "Sam, a boy who loves the library",    "knowledge is the greatest adventure"),
    ("Curiosity", "6-8 years",  "Mira, a girl who lives by the sea",   "questions lead to bigger questions"),
    ("Curiosity", "6-8 years",  "Gecko, a curious little lizard",      "imagination and wonder both deserve respect"),
    ("Curiosity", "9-12 years", "Inara, a young alchemist",            "science and curiosity unlock the world"),
    ("Curiosity", "9-12 years", "Wren, a girl who builds inventions",  "understanding others begins with listening"),

    # Extra 7 for coverage
    ("Identity",        "6-8 years",  "Milo, a small bear cub",               "size is not what makes you great"),
    ("Courage",         "3-5 years",  "Pip, a tiny sparrow",                   "your voice is a gift worth sharing"),
    ("Friendship",      "9-12 years", "Echo, a ghost who scares everyone away", "even the loneliest heart can find its place"),
    ("Responsibility",  "3-5 years",  "Buttons, a well-loved stuffed bear",    "being responsible means keeping your promises"),
    ("Curiosity",       "3-5 years",  "Wibble, a glowing jellyfish",           "every question lights up the world"),
    ("Identity",        "9-12 years", "Sage, a young healer",                  "believing in yourself is the first step to helping others"),
    ("Courage",         "6-8 years",  "Tomás, a boy afraid of thunderstorms",  "courage comes from inside"),
]

# ── Schema description injected into every prompt ─────────────────────────────
SYSTEM_PROMPT = """You are an expert children's story author and story architect.

Your job: produce ONE complete interactive story for a story-building app.

Output ONLY valid JSON. No prose before or after. No markdown fences.

The JSON schema is:
{
  "inputs": {
    "character": "<protagonist name + 1-line description>",
    "age_range": "<exactly as given>",
    "theme": "<exactly as given>",
    "language": "English"
  },
  "skeleton": {
    "paradigm": "<Identity | Courage | Friendship | Responsibility | Curiosity>",
    "title": "<catchy story title 3-6 words>",
    "protagonist": "<name + 1-sentence description>",
    "character_appearance": "<detailed visual description for illustration: color, size, texture, clothing/accessories, 2-3 sentences, present tense, no 'but' clauses>",
    "initial_trait": "<one word — the limiting trait the protagonist starts with>",
    "initial_emotion": "<one word>",
    "main_conflict": "<1 sentence describing the central problem>",
    "external_goal": "<what the protagonist is literally trying to do>",
    "internal_goal": "<the inner change the protagonist needs to make>",
    "mentor": "<name + role, 1 line>",
    "ally": "<name + role, 1 line>",
    "antagonist": "<name + role, 1 line — can be a force, not a person>",
    "setting": "<vivid 1-sentence description of the world>",
    "symbolic_object": "<a concrete object that represents the theme — must appear in ≥2 beats>",
    "value_learned": "<the lesson in one clear sentence>",
    "final_emotion": "<one or two words describing how protagonist feels at the end>",
    "theme": "<exactly as given>",
    "beat_arc": [
      "<beat 1: 1-sentence dramatic purpose>",
      "<beat 2: 1-sentence dramatic purpose>",
      "<beat 3: 1-sentence dramatic purpose>",
      "<beat 4: 1-sentence dramatic purpose>",
      "<beat 5: 1-sentence dramatic purpose>"
    ]
  },
  "beats": [
    {
      "narrative": "<4-5 vivid sentences telling this beat's story, age-appropriate, NO meta-words like 'paradigm'/'story bible'/'beat arc'>",
      "choices": ["<short choice A, 6-10 words>", "<short choice B, 6-10 words>", "<short choice C, 6-10 words>"],
      "choice_made": "<exactly one of the three choices strings — pick the most dramatically interesting>",
      "is_final": false
    },
    ... (beats 2, 3, 4 same structure) ...
    {
      "narrative": "<4-5 sentences resolving BOTH external_goal and internal_goal, include symbolic_object>",
      "choices": ["<choice A>", "<choice B>", "<choice C>"],
      "is_final": true
    }
  ]
}

Rules:
- beats array must have EXACTLY 5 items.
- beats 1–4: is_final=false, must include choice_made (a copy of the chosen choice string).
- beat 5: is_final=true, NO choice_made field.
- symbolic_object must appear naturally in the narrative of at least 2 different beats.
- Narratives must NEVER contain the words: paradigm, beat_arc, story bible, internal_goal, external_goal, symbolic_object, value_learned, final_emotion.
- Age 3-5: very simple words, short sentences, lots of warmth.
- Age 6-8: light adventure, easy vocabulary, some tension.
- Age 9-12: richer vocabulary, real inner conflict, deeper themes.
- No copyrighted characters. No violence. No scary content beyond mild tension appropriate to age.
"""


def build_user_prompt(paradigm: str, age_range: str, protagonist: str, theme: str) -> str:
    return (
        f"Generate a complete story with this setup:\n"
        f"- Paradigm: {paradigm}\n"
        f"- Age range: {age_range}\n"
        f"- Protagonist: {protagonist}\n"
        f"- Theme: {theme}\n\n"
        f"Output only valid JSON following the schema exactly."
    )


def validate_story(story: dict) -> list[str]:
    errors = []
    required_skeleton_keys = {
        "paradigm", "title", "protagonist", "character_appearance",
        "initial_trait", "initial_emotion", "main_conflict",
        "external_goal", "internal_goal", "mentor", "ally", "antagonist",
        "setting", "symbolic_object", "value_learned", "final_emotion",
        "theme", "beat_arc",
    }
    skeleton = story.get("skeleton", {})
    missing = required_skeleton_keys - set(skeleton.keys())
    if missing:
        errors.append(f"skeleton missing keys: {missing}")
    arc = skeleton.get("beat_arc", [])
    if not isinstance(arc, list) or len(arc) != 5:
        errors.append(f"beat_arc must be list of 5, got {arc!r}")
    beats = story.get("beats", [])
    if len(beats) != 5:
        errors.append(f"need 5 beats, got {len(beats)}")
    for i, beat in enumerate(beats):
        if not beat.get("narrative", "").strip():
            errors.append(f"beat {i+1} empty narrative")
        if len(beat.get("choices", [])) != 3:
            errors.append(f"beat {i+1} needs 3 choices")
        expected_final = (i == 4)
        if beat.get("is_final") != expected_final:
            errors.append(f"beat {i+1} is_final={beat.get('is_final')}, want {expected_final}")
        if i < 4 and "choice_made" not in beat:
            errors.append(f"beat {i+1} missing choice_made")
    return errors


def generate_story(
    client: anthropic.Anthropic,
    paradigm: str,
    age_range: str,
    protagonist: str,
    theme: str,
    max_retries: int = 3,
) -> dict | None:
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": build_user_prompt(paradigm, age_range, protagonist, theme)}
                ],
            )
            raw = response.content[0].text.strip()
            # strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            story = json.loads(raw)
            errors = validate_story(story)
            if errors:
                print(f"    attempt {attempt+1} validation errors: {errors}")
                continue
            return story, response.usage
        except json.JSONDecodeError as e:
            print(f"    attempt {attempt+1} JSON parse error: {e}")
        except Exception as e:
            print(f"    attempt {attempt+1} error: {e}")
            time.sleep(2)
    return None, None


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load existing progress
    stories: list[dict] = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            try:
                stories = json.load(f)
                print(f"Resuming from {len(stories)} existing stories.")
            except json.JSONDecodeError:
                print("WARNING: existing stories.json is invalid, starting fresh.")
                stories = []

    total_input_tokens = 0
    total_output_tokens = 0
    skipped = 0

    for i, (paradigm, age_range, protagonist, theme) in enumerate(STORY_SLOTS):
        slot_num = i + 1
        short_label = f"[{slot_num:02d}/{len(STORY_SLOTS)}] {paradigm}/{age_range[:3]}/{protagonist.split(',')[0]}"

        # Skip if already generated (by matching protagonist hint)
        if any(protagonist.split(",")[0].strip().lower() in
               s.get("inputs", {}).get("character", "").lower() for s in stories):
            print(f"  SKIP {short_label} (already exists)")
            skipped += 1
            continue

        print(f"  Generating {short_label} ...", end=" ", flush=True)
        story, usage = generate_story(client, paradigm, age_range, protagonist, theme)

        if story is None:
            print(f"FAILED after 3 attempts — skipping")
            skipped += 1
            continue

        stories.append(story)
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        # Save after every story
        with open(OUTPUT_FILE, "w") as f:
            json.dump(stories, f, ensure_ascii=False, indent=2)
        print(f"OK  (in={usage.input_tokens} out={usage.output_tokens})")

    print(f"\n{'─'*60}")
    print(f"Stories generated : {len(stories) - skipped} new  |  {len(stories)} total")
    print(f"Skipped           : {skipped}")
    print(f"Total tokens      : {total_input_tokens:,} in / {total_output_tokens:,} out")
    # Rough cost: claude-sonnet-4-6 = $3/1M in, $15/1M out
    cost = total_input_tokens * 3e-6 + total_output_tokens * 15e-6
    print(f"Estimated cost    : ${cost:.2f}")
    print(f"Output            : {OUTPUT_FILE}")

    if skipped:
        print(f"\nNOTE: {skipped} stories failed/skipped. Re-run to retry failed ones.")


if __name__ == "__main__":
    main()
