import os
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from dotenv import load_dotenv

from . import vram_manager
from .vram_manager import Model
from .prompts import load_prompts, build_image_prompt

load_dotenv()

IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
IMAGE_DEVICE = os.getenv("IMAGE_DEVICE", "cuda")
HF_TOKEN = os.getenv("HF_TOKEN")

_prompts = load_prompts()


def _loader() -> Flux2KleinPipeline:
    print(f"[ImageGenerator] Loading {IMAGE_MODEL_ID}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        IMAGE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    pipe.to(IMAGE_DEVICE)
    return pipe


def _unloader(instance: Flux2KleinPipeline):
    del instance
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


vram_manager.register(Model.IMAGE, _loader, _unloader)


def generate_image(narrative: str, visual_profile: str = "", seed: int | None = None) -> Image.Image:
    vram_manager.request(Model.IMAGE)
    pipe = vram_manager.get(Model.IMAGE)

    prompt = build_image_prompt(narrative, visual_profile, _prompts)
    generator = torch.Generator(device=IMAGE_DEVICE).manual_seed(seed) if seed is not None else None

    result = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=1.0,
        height=1024,
        width=1024,
        generator=generator,
    )
    return result.images[0]
