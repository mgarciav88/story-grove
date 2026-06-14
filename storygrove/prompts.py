from pathlib import Path
import yaml


def load_prompts(
        path: Path = Path(__file__).parent / "configs" / "prompts.yaml"
) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Skeleton (Stage 1) ─────────────────────────────────────────────────────────

def build_skeleton_prompt(
        character: str,
        age_range: str,
        theme: str,
        language: str,
        max_beats: int,
        prompts: dict,
) -> str:
    return prompts["skeleton_prompt"].format(
        character=character,
        age_range=age_range,
        theme=theme,
        language=language,
        max_beats=max_beats,
    )


def render_story_bible(skeleton: dict) -> str:
    """Render the skeleton as a readable story bible for injection into beat prompts."""
    if not skeleton:
        return "(no story plan available)"

    lines = []
    for label, key in [
        ("Paradigm", "paradigm"),
        ("Protagonist", "protagonist"),
        ("Initial trait", "initial_trait"),
        ("Starts with emotion", "initial_emotion"),
        ("Main conflict", "main_conflict"),
        ("External goal", "external_goal"),
        ("Internal goal", "internal_goal"),
        ("Mentor", "mentor"),
        ("Ally", "ally"),
        ("Antagonist", "antagonist"),
        ("Setting", "setting"),
        ("Symbolic object (weave in naturally)", "symbolic_object"),
        ("Value to convey", "value_learned"),
        ("Ends with emotion", "final_emotion"),
    ]:
        if skeleton.get(key):
            lines.append(f"{label}: {skeleton[key]}")

    beat_arc = skeleton.get("beat_arc", [])
    if beat_arc:
        lines.append("Beat arc:")
        for i, purpose in enumerate(beat_arc, 1):
            lines.append(f"  Beat {i}: {purpose}")

    return "\n".join(lines)


# ── Beat generation (Stage 2) ──────────────────────────────────────────────────

def build_first_beat_prompt(
        character: str,
        age_range: str,
        theme: str,
        language: str,
        prompts: dict,
        skeleton: dict | None = None,
) -> str:
    skeleton = skeleton or {}
    beat_arc = skeleton.get("beat_arc", [])
    beat_goal = beat_arc[0] if beat_arc else "introduce the character and their ordinary world"
    return prompts["first_beat_prompt"].format(
        character=character,
        age_range=age_range,
        theme=theme,
        language=language,
        max_beats=5,
        story_bible=render_story_bible(skeleton),
        beat_goal=beat_goal,
        response_format=prompts["response_format"],
    )


def build_image_prompt(narrative: str, visual_profile: str, prompts: dict) -> str:
    scene = narrative[:350].strip().rstrip(".")
    return prompts["image_prompt"].format(scene=scene, visual_profile=visual_profile)


def build_continue_beat_prompt(
        session,  # StorySession
        choice: str,
        is_final: bool,
        prompts: dict,
) -> str:
    skeleton = getattr(session, "skeleton", {}) or {}
    beat_arc = skeleton.get("beat_arc", [])
    beat_goal = (
        beat_arc[session.beat_number]
        if beat_arc and session.beat_number < len(beat_arc)
        else "continue the story toward its conclusion"
    )
    final_instruction = prompts["final_beat_instruction"] if is_final else ""
    return prompts["continue_beat_prompt"].format(
        character=session.character,
        language=session.language,
        beat_number=session.beat_number + 1,
        max_beats=session.max_beats,
        is_final=is_final,
        story_bible=render_story_bible(skeleton),
        beat_goal=beat_goal,
        summary=session.summary(),
        choice=choice,
        final_instruction=final_instruction,
        response_format=prompts["response_format"],
    )