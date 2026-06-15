import os
import json
import random
import torch
from dataclasses import dataclass, field
from typing import Optional, Generator
from threading import Thread
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from dotenv import load_dotenv

from . import vram_manager
from .prompts import load_prompts, build_skeleton_prompt, build_first_beat_prompt, build_continue_beat_prompt
from .vram_manager import Model

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-3-4b-it")
TEXT_DEVICE = os.getenv("TEXT_DEVICE", "cuda")
HF_TOKEN = os.getenv("HF_TOKEN")
USE_4BIT = os.getenv("QUANTIZE_4BIT", "false").lower() == "true"
MAX_BEATS = 5

REQUIRED_SKELETON_KEYS = {
    "paradigm", "title", "protagonist", "character_appearance",
    "initial_trait", "initial_emotion", "main_conflict",
    "external_goal", "internal_goal", "mentor", "ally", "antagonist",
    "setting", "symbolic_object", "value_learned", "final_emotion",
    "theme", "beat_arc",
}

# ── Data Models ────────────────────────────────────────────────────────────────

class StoryBeat(BaseModel):
    beat: str
    choices: list[str]
    is_final: bool

@dataclass
class StorySession:
    character: str
    age_range: str
    theme: str
    language: str
    beat_number: int = 0
    story_so_far: list[str] = field(default_factory=list)
    max_beats: int = MAX_BEATS
    image_seed: int = field(default_factory=lambda: random.randint(0, 2**32 - 1))
    visual_profile: str = ""
    skeleton: dict = field(default_factory=dict)
    full_beats: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)

    def add_beat(self, beat: str, choice: str):
        self.story_so_far.append(f"Beat {self.beat_number}: {beat[:120]}...")
        self.story_so_far.append(f"Child chose: {choice}")

    def record_llm_call(self, stage: str, system: str, user: str, beat_number=None) -> None:
        self.llm_calls.append({
            "stage": stage,
            "beat_number": beat_number,
            "system": system,
            "user": user,
        })

    def build_trace_messages(self) -> list[dict]:
        """
        Reconstruct full system→user→assistant triples from recorded LLM calls.
        Each entry is already in TRL conversational format; flattening all entries
        across traces gives ready-to-train SFT rows (train/inference parity with
        training/build_dataset.py).
        """
        beats = {b["beat_number"]: b for b in self.full_beats}
        out = []
        for call in self.llm_calls:
            if call["stage"] == "skeleton":
                assistant = json.dumps(self.skeleton, ensure_ascii=False, separators=(",", ":"))
            else:
                b = beats.get(call["beat_number"], {})
                assistant = (
                    b.get("narrative", "").strip()
                    + "\n"
                    + json.dumps(
                        {"choices": b.get("choices", []), "is_final": b.get("is_final", False)},
                        ensure_ascii=False,
                    )
                )
            out.append({
                "stage": call["stage"],
                "beat_number": call["beat_number"],
                "messages": [
                    {"role": "system", "content": call["system"]},
                    {"role": "user", "content": call["user"]},
                    {"role": "assistant", "content": assistant},
                ],
            })
        return out

    def record_full_beat(self, narrative: str, choices: list, is_final: bool) -> None:
        self.full_beats.append({
            "beat_number": self.beat_number,
            "narrative": narrative,
            "choices": choices,
            "choice_made": None,
            "is_final": is_final,
        })

    def is_final_beat(self) -> bool:
        return self.beat_number >= self.max_beats - 1

    def summary(self) -> str:
        if not self.story_so_far:
            return "The story is just beginning."
        return "\n".join(self.story_so_far)

# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available() and TEXT_DEVICE == "cuda":
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model Loader / Unloader ────────────────────────────────────────────────────

def _loader() -> tuple:
    """Load and return (model, tokenizer)."""
    device = get_device()
    print(f"Using device: {device} for text generation")
    kwargs = dict(token=HF_TOKEN, device_map=device)

    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        print("Using 4-bit quantization")
    else:
        kwargs["dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    return model, tokenizer

def _unloader(instance: tuple):
    """Unload (model, tokenizer) from memory."""
    model, tokenizer = instance
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ── Register with VRAM manager ─────────────────────────────────────────────────

vram_manager.register(Model.STORY, _loader, _unloader)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_model_and_tokenizer():
    vram_manager.request(Model.STORY)
    return vram_manager.get(Model.STORY)  # returns (model, tokenizer)


def _extract_json(text: str) -> str:
    """Strip markdown fences if present and extract the outermost JSON object."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[1:end]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _default_beat_arc(max_beats: int) -> list[str]:
    arc = [
        "introduce the character and their ordinary world",
        "the call to adventure and meeting the mentor",
        "face the main trial and the antagonist",
        "the crisis and moment of self-discovery",
        "transformation, resolution, and return with the lesson",
    ]
    return arc[:max_beats]


def _minimal_skeleton(character: str, theme: str, max_beats: int) -> dict:
    return {
        "paradigm": "Curiosity",
        "title": f"The story of {character}",
        "protagonist": character,
        "character_appearance": character,
        "initial_trait": "uncertain",
        "initial_emotion": "curious",
        "main_conflict": f"a challenge related to {theme}",
        "external_goal": "complete the adventure",
        "internal_goal": "grow and learn",
        "mentor": "a wise friend",
        "ally": "a loyal companion",
        "antagonist": "an unexpected obstacle",
        "setting": "a magical world",
        "symbolic_object": "a special keepsake",
        "value_learned": "courage and kindness",
        "theme": theme,
        "final_emotion": "joy and pride",
        "beat_arc": _default_beat_arc(max_beats),
    }


def _visual_profile_from_skeleton(skeleton: dict, character: str, theme: str) -> str:
    parts = []
    appearance = skeleton.get("character_appearance") or skeleton.get("protagonist") or character
    parts.append(f"Character appearance (never change between scenes): {appearance}")
    if skeleton.get("setting"):
        parts.append(f"Setting: {skeleton['setting']}")
    if skeleton.get("symbolic_object"):
        parts.append(f"Symbolic object: {skeleton['symbolic_object']}")
    return " | ".join(parts)


# ── Skeleton Generator (Stage 1) ───────────────────────────────────────────────

def generate_skeleton(
    character: str,
    age_range: str,
    theme: str,
    language: str,
) -> dict:
    """
    Stage-1 non-streaming call: generates the narrative skeleton for the whole story.
    Falls back to a minimal skeleton if the model output can't be parsed.
    """
    prompts = load_prompts()
    model, tokenizer = _get_model_and_tokenizer()

    messages = [
        {"role": "system", "content": prompts["skeleton_system_prompt"]},
        {"role": "user", "content": build_skeleton_prompt(
            character=character,
            age_range=age_range,
            theme=theme,
            language=language,
            max_beats=MAX_BEATS,
            prompts=prompts,
        )},
    ]

    for attempt in range(2):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            enable_thinking=False,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        try:
            skeleton = json.loads(_extract_json(raw))
            skeleton.setdefault("protagonist", character)
            skeleton.setdefault("theme", theme)
            beat_arc = skeleton.get("beat_arc", [])
            if not isinstance(beat_arc, list) or len(beat_arc) != MAX_BEATS:
                skeleton["beat_arc"] = _default_beat_arc(MAX_BEATS)
            print(f"[StoryGrove] Skeleton ready — paradigm: {skeleton.get('paradigm', '?')}")
            return skeleton
        except (json.JSONDecodeError, ValueError, KeyError):
            print(f"[StoryGrove] Skeleton parse failed (attempt {attempt + 1}):\n{raw}")
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        "Your response was not valid JSON. "
                        "Output only the JSON object, no explanation, no markdown."
                    )},
                ]

    print("[StoryGrove] Using minimal fallback skeleton.")
    return _minimal_skeleton(character, theme, MAX_BEATS)


# ── Streaming Parser ───────────────────────────────────────────────────────────

def parse_streaming_response(full_text: str) -> tuple[str, Optional[StoryBeat]]:
    """
    Split streamed output into narrative and choices JSON.
    Returns (narrative_so_far, StoryBeat_or_None).
    StoryBeat is None while JSON is not yet complete.
    """
    # look for JSON block starting from the last {
    parts = full_text.rsplit('{', 1)

    if len(parts) == 1:
        # no JSON yet, pure narrative still streaming
        return full_text.strip(), None

    narrative = parts[0].strip()
    json_part = '{' + parts[1]

    try:
        data = json.loads(json_part)
        beat = StoryBeat(
            beat=narrative,
            choices=data["choices"],
            is_final=data.get("is_final", False),
        )
        return narrative, beat
    except Exception:
        # JSON block started but not complete yet
        return narrative.strip(), None

# ── Core Streaming Generator ───────────────────────────────────────────────────

def _stream_beat(
    messages: list[dict],
    max_new_tokens: int = 2048
) -> Generator[tuple[str, Optional[StoryBeat]], None, None]:
    """
    Streams (narrative_text, StoryBeat_or_None) tuples.
    narrative_text grows with each token.
    StoryBeat becomes non-None once JSON is fully parsed.
    """
    model, tokenizer = _get_model_and_tokenizer()

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        enable_thinking=False,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_text = ""
    for token in streamer:
        full_text += token
        narrative, beat = parse_streaming_response(full_text)
        yield narrative, beat

    thread.join()

def _generate_beat_with_retry(
    messages: list[dict],
    max_retries: int = 3,
) -> Generator[tuple[str, Optional[StoryBeat]], None, None]:
    """
    Wraps _stream_beat with retry logic.
    On parse failure, retries with self-correction prompt.
    """
    for attempt in range(max_retries):
        last_narrative = ""
        last_beat = None

        for narrative, beat in _stream_beat(messages):
            last_narrative = narrative
            last_beat = beat
            yield narrative, beat

        if last_beat is not None:
            return  # success

        print(f"Beat parse failed (attempt {attempt + 1}), retrying...")
        messages = messages + [
            {"role": "assistant", "content": last_narrative},
            {"role": "user", "content": (
                "Your response was missing the required JSON block at the end. "
                "Please append exactly this to complete your response:\n"
                '{"choices": ["choice1", "choice2", "choice3"], "is_final": false}\n'
                "Use real choices from the story, not placeholders."
            )},
        ]

    raise ValueError(f"Failed to generate valid beat after {max_retries} attempts.")

# ── Public API ─────────────────────────────────────────────────────────────────

def start_story(
    character: str,
    age_range: str,
    theme: str,
    language: str,
) -> tuple[Generator, StorySession]:
    """Start a new story. Stage 1 generates the skeleton; Stage 2 streams the first beat."""
    prompts = load_prompts()

    skeleton = generate_skeleton(character, age_range, theme, language)

    session = StorySession(
        character=character,
        age_range=age_range,
        theme=theme,
        language=language,
        skeleton=skeleton,
        visual_profile=_visual_profile_from_skeleton(skeleton, character, theme),
    )

    session.record_llm_call(
        stage="skeleton",
        system=prompts["skeleton_system_prompt"],
        user=build_skeleton_prompt(
            character=character,
            age_range=age_range,
            theme=theme,
            language=language,
            max_beats=MAX_BEATS,
            prompts=prompts,
        ),
    )

    messages = [
        {"role": "system", "content": prompts["system_prompt"]},
        {"role": "user", "content": build_first_beat_prompt(
            character=character,
            age_range=age_range,
            theme=theme,
            language=language,
            prompts=prompts,
            skeleton=skeleton,
        )},
    ]

    session.beat_number += 1
    session.record_llm_call(
        stage="beat",
        system=messages[0]["content"],
        user=messages[1]["content"],
        beat_number=session.beat_number,
    )
    return _generate_beat_with_retry(messages), session


def continue_story(
    session: StorySession,
    choice: str,
) -> tuple[Generator, StorySession]:
    """Continue the story. Returns a generator and updated session."""
    prompts = load_prompts()
    session.add_beat("", choice)

    messages = [
        {"role": "system", "content": prompts["system_prompt"]},
        {"role": "user", "content": build_continue_beat_prompt(
            session=session,
            choice=choice,
            is_final=session.is_final_beat(),
            prompts=prompts,
        )},
    ]

    session.beat_number += 1
    session.record_llm_call(
        stage="beat",
        system=messages[0]["content"],
        user=messages[1]["content"],
        beat_number=session.beat_number,
    )
    return _generate_beat_with_retry(messages), session