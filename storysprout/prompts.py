from pathlib import Path
import yaml

def load_prompts(
    path: Path = Path(__file__).parent / "configs" / "prompts.yaml"
) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def build_first_beat_prompt(
    character: str,
    age_range: str,
    theme: str,
    language: str,
    prompts: dict,
) -> str:
    return prompts["first_beat_prompt"].format(
        character=character,
        age_range=age_range,
        theme=theme,
        language=language,
        max_beats=5,
        response_format=prompts["response_format"],
    )

def build_continue_beat_prompt(
    session,  # StorySession
    choice: str,
    is_final: bool,
    prompts: dict,
) -> str:
    final_instruction = (
        prompts["final_beat_instruction"] if is_final else ""
    )
    return prompts["continue_beat_prompt"].format(
        character=session.character,
        language=session.language,
        beat_number=session.beat_number + 1,
        max_beats=session.max_beats,
        is_final=is_final,
        summary=session.summary(),
        choice=choice,
        final_instruction=final_instruction,
        response_format=prompts["response_format"],
    )