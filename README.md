# 🌳 StoryGrove

**An interactive story generator for kids.** You describe a character and a theme — StoryGrove builds a complete illustrated story arc, then streams it page by page. At each beat, the child picks what happens next.

Every story is unique. Every choice matters.

> **Live demo:** [build-small-hackathon/StoryGrove](https://huggingface.co/spaces/build-small-hackathon/StoryGrove) on Hugging Face Spaces

---

## 🎬 Demo

[![StoryGrove Demo](https://img.youtube.com/vi/zEteDr6cCPE/0.jpg)](https://youtu.be/zEteDr6cCPE)

---

## How It Works

StoryGrove runs a two-stage pipeline fully on-device, no cloud API calls:

1. **Skeleton** — the model plans the full story arc in one shot: protagonist, setting, symbolic object, 5-beat arc, value learned, and the emotional journey — all as structured JSON.
2. **Beats** — each page is streamed live, ending with three choices that branch the narrative. The child picks one; the story continues from there.

Both stages are powered by a fine-tuned **Gemma 3 4B** — small enough to run on a consumer GPU, capable enough to write stories that feel real.

Illustrations are generated per page by **Flux**, conditioned on a visual profile extracted from the skeleton (character appearance, setting palette, symbolic object). Voice narration for each page is available via **VoxCPM2**.

---

## ⚡ Llama.cpp Mode

StoryGrove ships with a UI toggle that switches inference from Transformers to **llama.cpp** using a GGUF Q4_K_M quantized version of the fine-tuned model. In llama mode, story generation runs through `llama-cpp-python` on GPU — images and audio are disabled to avoid GPU context conflicts between ggml-cuda and PyTorch.

- GGUF model: [`build-small-hackathon/storygrove-gemma-3-4b-gguf`](https://huggingface.co/build-small-hackathon/storygrove-gemma-3-4b-gguf)
- Quantization: Q4_K_M (2.49 GB)

---

## Running Locally

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A CUDA GPU with ~16 GB VRAM recommended (all three models loaded simultaneously). CPU fallback works but is very slow.
- A Hugging Face account with access to [Gemma 3](https://huggingface.co/google/gemma-3-4b-it) (request access on the model page if needed)

### Setup

```bash
git clone https://github.com/mgarciav88/story-grove
cd story-grove
uv sync
cp .env.example .env
# edit .env and fill in your values
uv run python app.py
```

### Environment Variables

Copy `.env.example` to `.env` and set:

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes | — | HuggingFace token — needed to download Gemma (gated model) |
| `MODEL_ID` | No | `google/gemma-3-4b-it` | Swap for the fine-tuned model: `build-small-hackathon/storygrove-gemma-3-4b` |
| `QUANTIZE_4BIT` | No | `false` | Set `true` to load the text model in 4-bit (reduces VRAM ~50%) |
| `TEXT_DEVICE` | No | `cuda` | Set `cpu` if no GPU available |
| `IMAGE_DEVICE` | No | `cuda` | Set `cpu` to skip GPU for image generation |
| `NARRATOR_DEVICE` | No | `cuda` | Set `cpu` to skip GPU for narration |
| `GGUF_MODEL_PATH` | No | — | Set to `build-small-hackathon/storygrove-gemma-3-4b-gguf` to enable llama.cpp mode |
| `VRAM_SWAP` | No | `false` | Set `true` to swap models in/out of VRAM between beats |
| `TRACES_REPO` | No | `build-small-hackathon/storygrove-traces` | HF dataset repo where completed story traces are pushed |
| `DATASET_REPO` | No | `build-small-hackathon/storygrove-sft` | HF dataset repo for the SFT upload script |

---

## 📓 Field Notes

Full write-up covering the architecture, fine-tuning process, eval results, and lessons learned:

**[StoryGrove — Build Small Hackathon](https://huggingface.co/blog/mgarciav/build-small-hackaton-storygrove)**

---

## 📡 Story Traces

Every completed story is automatically pushed to a public dataset as a structured trace — including the full skeleton, all beats with choices made, and the LLM call log in TRL conversational format (ready for future fine-tuning rounds).

- Traces dataset: [`build-small-hackathon/storygrove-traces`](https://huggingface.co/datasets/build-small-hackathon/storygrove-traces)

---

## 🎯 Fine-Tuning & Eval Results

The base model was fine-tuned on a dataset of 40 complete story playthroughs (240 SFT examples: skeleton + 5 beats each) covering 5 narrative paradigms × 3 age ranges. Multi-task training was used so the model learned both the structured skeleton format and the streaming beat format simultaneously.

**On data generation:** the training stories were authored by Claude Sonnet 4.6 to ensure quality and diversity given time constraints. The traces dataset (above) and a future manual curation pass will allow retraining on real user stories as the app accumulates playthroughs.

Evaluated on 25 held-out stories against the base `google/gemma-3-4b-it`:

| Metric | Base | Fine-Tuned | Δ |
|---|---|---|---|
| Skeleton valid JSON (first try) | 48% | **64%** | +33% |
| Beat choices valid | 96.8% | **100%** | +3.2pp |
| Avg retries needed | 0.096 | **0.0** | eliminated |
| Schema completeness | 48% | **64%** | +16pp |
| Paradigm accuracy | 83.3% | 81.3% | ≈ (noise) |
| Meta-leakage | 0% | 0% | no regression |

The headline result: the base model fails to produce valid skeleton JSON on the first try **52% of the time**. Fine-tuning drops that to 36%, and eliminates retries on beat generation entirely — faster stories, fewer fallbacks.

- Fine-tuned model: [`build-small-hackathon/storygrove-gemma-3-4b`](https://huggingface.co/build-small-hackathon/storygrove-gemma-3-4b)
- Training dataset: [`build-small-hackathon/storygrove-sft`](https://huggingface.co/datasets/build-small-hackathon/storygrove-sft)
- Training & eval: Unsloth QLoRA + eval harness run on **Modal** (A100, ~20 min training / ~70 min eval)

---

## Models Used

| Role | Model | Size |
|---|---|---|
| Story generation | `build-small-hackathon/storygrove-gemma-3-4b` (fine-tuned Gemma 3) | 4B |
| Image generation | Flux-2-Klein (via diffusers) | 4B |
| Voice narration | VoxCPM2 | 2B |
| GGUF inference | `storygrove-gemma-3-4b-gguf` Q4_K_M | 2.49 GB |

---

## Tech Stack

- **Gradio** — UI with custom book-page CSS theme
- **Transformers + Unsloth** — fine-tuning and inference
- **llama-cpp-python** — GGUF inference path
- **Modal** — fine-tuning and eval runs
- **Hugging Face Hub** — model, dataset, and Space hosting
