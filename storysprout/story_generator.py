import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, Generator
from threading import Thread
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from dotenv import load_dotenv

from . import vram_manager
from .prompts import load_prompts, build_first_beat_prompt, build_continue_beat_prompt
from .vram_manager import Model

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-3-4b-it")
TEXT_DEVICE = os.getenv("TEXT_DEVICE", "cuda")
HF_TOKEN = os.getenv("HF_TOKEN")
USE_4BIT = os.getenv("QUANTIZE_4BIT", "false").lower() == "true"
MAX_BEATS = 5

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

    def add_beat(self, beat: str, choice: str):
        self.story_so_far.append(f"Beat {self.beat_number}: {beat[:120]}...")
        self.story_so_far.append(f"Child chose: {choice}")

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
    """Start a new story. Returns a generator and a fresh session."""
    prompts = load_prompts()
    session = StorySession(
        character=character,
        age_range=age_range,
        theme=theme,
        language=language,
    )

    messages = [
        {"role": "system", "content": prompts["system_prompt"]},
        {"role": "user", "content": build_first_beat_prompt(
            character=character,
            age_range=age_range,
            theme=theme,
            language=language,
            prompts=prompts,
        )},
    ]

    session.beat_number += 1
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
    return _generate_beat_with_retry(messages), session
