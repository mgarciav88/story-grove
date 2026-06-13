import os
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from dotenv import load_dotenv

from . import vram_manager
from .vram_manager import Model

load_dotenv()

IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
IMAGE_DEVICE = os.getenv("IMAGE_DEVICE", "cuda")
HF_TOKEN = os.getenv("HF_TOKEN")

STYLE_SUFFIX = ", children's book illustration, colorful, warm, whimsical"


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


def generate_image(narrative: str) -> Image.Image:
    vram_manager.request(Model.IMAGE)
    pipe = vram_manager.get(Model.IMAGE)

    prompt = narrative[:400].strip() + STYLE_SUFFIX

    result = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=1.0,
        height=1024,
        width=1024,
    )
    return result.images[0]
