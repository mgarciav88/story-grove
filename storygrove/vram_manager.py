import os
from enum import Enum
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

VRAM_SWAP = os.getenv("VRAM_SWAP", "false").lower() == "true"


class Model(str, Enum):
    STORY = "story"
    NARRATOR = "narrator"
    IMAGE = "image"  # ready for when we add FLUX


# ── Registry ───────────────────────────────────────────────────────────────────

# each entry: { "loader": fn, "unloader": fn, "instance": obj | None }
_registry: dict[Model, dict] = {}


def register(model: Model, loader: Callable, unloader: Callable):
    """
    Register a model with its load and unload functions.
    Called once at module init time by each component.
    """
    _registry[model] = {
        "loader": loader,
        "unloader": unloader,
        "instance": None,
    }


# ── Core ───────────────────────────────────────────────────────────────────────

def _is_loaded(model: Model) -> bool:
    return (
            model in _registry and
            _registry[model]["instance"] is not None
    )


def _load(model: Model):
    entry = _registry[model]
    if entry["instance"] is None:
        print(f"[VRAMManager] Loading {model.value}...")
        entry["instance"] = entry["loader"]()
        print(f"[VRAMManager] {model.value} loaded.")


def _unload(model: Model):
    entry = _registry[model]
    if entry["instance"] is not None:
        print(f"[VRAMManager] Unloading {model.value}...")
        entry["unloader"](entry["instance"])
        entry["instance"] = None
        print(f"[VRAMManager] {model.value} unloaded.")


def request(model: Model):
    """
    Ensure the requested model is loaded.
    If VRAM_SWAP is enabled, unload all other models first.
    """
    if _is_loaded(model):
        print(f"[VRAMManager] {model.value} is already loaded.")
        return  # already loaded, nothing to do

    if VRAM_SWAP:
        print("[VRAMManager] VRAM swap enabled. Unloading other models...")
        # unload everything else to free VRAM
        for other in _registry:
            if other != model:
                _unload(other)
    else:
        print("No vramp swap enabled. Loading requested model...")

    _load(model)


def get(model: Model):
    """Get the loaded model instance. Must call request() first."""
    if not _is_loaded(model):
        raise RuntimeError(
            f"Model {model.value} is not loaded. "
            f"Call vram_manager.request({model.value}) first."
        )
    return _registry[model]["instance"]
