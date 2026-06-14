import io
import os

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from voxcpm import VoxCPM

from . import vram_manager
from .vram_manager import Model

load_dotenv()

# Voice description — same for both languages,
# VoxCPM2 detects language automatically from the text
NARRATOR_VOICE = (
    "(A warm, gentle male voice, soft and expressive, "
    "perfect for bedtime stories, calm and reassuring)"
)

# ── Config ──────────────────────────────────────────────────────────────────
VOICE_MODEL_ID = os.getenv("VOICE_MODEL_ID", "openbmb/VoxCPM2")
NARRATOR_DEVICE = os.getenv("NARRATOR_DEVICE", "cuda")


# ── Model Loader / Unloader ────────────────────────────────────────────────────

def _loader() -> VoxCPM:
    print(f"[Narrator] Loading VoxCPM2 from {VOICE_MODEL_ID}...")
    return VoxCPM.from_pretrained(
        VOICE_MODEL_ID,
        load_denoiser=False,
        device=NARRATOR_DEVICE,
    )

def _unloader(instance: VoxCPM):
    del instance
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ── Register with VRAM manager ─────────────────────────────────────────────────

vram_manager.register(Model.NARRATOR, _loader, _unloader)

# ── Public API ─────────────────────────────────────────────────────────────────

def narrate(text: str) -> tuple[int, np.ndarray]:
    """
    Narrate text and return (sample_rate, audio_array).
    Handles VRAM swapping if enabled.
    Returns format compatible with Gradio's gr.Audio component.
    """
    print(f"[Narrator] Narrating: {text}")
    vram_manager.request(Model.NARRATOR)
    narrator = vram_manager.get(Model.NARRATOR)

    # prepend voice description to text
    prompted_text = f"{NARRATOR_VOICE}{text}"

    wav = narrator.generate(
        text=prompted_text,
        cfg_value=2.0,
        inference_timesteps=10,
    )

    sample_rate = narrator.tts_model.sample_rate

    return sample_rate, wav


def narrate_to_bytes(text: str) -> bytes:
    """
    Narrate text and return WAV bytes.
    Useful for passing directly to Gradio Audio component.
    """
    sample_rate, wav = narrate(text)
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()
