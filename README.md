---
title: StoryGrove
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
python_version: "3.12"
license: h-research
short_description: Interactive branching story generator for kids, powered by a fine-tuned Gemma 3 4B
tags:
  - track:wood
  - sponsor:modal
  - sponsor:openbmb
  - achievement:welltuned
  - achievement:llama
  - achievement:fieldnotes
  - achievement:sharing
  - badge-tiny-titan
---

# 🌳 StoryGrove

**An interactive story generator for kids.** You describe a character and a theme — StoryGrove builds a complete illustrated story arc, then streams it page by page. At each beat, the child picks what happens next.

Every story is unique. Every choice matters.

---

## How It Works

StoryGrove runs a two-stage pipeline fully on-device, no cloud API calls:

1. **Skeleton** — the model plans the full story arc in one shot: protagonist, setting, symbolic object, 5-beat arc, value learned, and the emotional journey — all as structured JSON.
2. **Beats** — each page is streamed live, ending with three choices that branch the narrative. The child picks one; the story continues from there.

Both stages are powered by a fine-tuned **Gemma 3 4B** — small enough to run on ZeroGPU, capable enough to write stories that feel real.

Illustrations are generated per page by **Flux**, conditioned on a visual profile extracted from the skeleton (character appearance, setting palette, symbolic object). Voice narration for each page is available via **VoxCPM2**.

---

## ⚡ Llama.cpp Mode

StoryGrove ships with a UI toggle that switches inference from Transformers to **llama.cpp** using a GGUF Q4_K_M quantized version of the fine-tuned model. In llama mode, story generation runs through `llama-cpp-python` on GPU — images and audio are disabled to avoid GPU context conflicts between ggml-cuda and PyTorch.

- GGUF model: [`build-small-hackathon/storygrove-gemma-3-4b-gguf`](https://huggingface.co/build-small-hackathon/storygrove-gemma-3-4b-gguf)
- Quantization: Q4_K_M (2.49 GB)

---

## 📡 Story Traces

Every completed story is automatically pushed to a public dataset as a structured trace — including the full skeleton, all beats with choices made, and the LLM call log in TRL conversational format (ready for future fine-tuning rounds).

- Traces dataset: [`build-small-hackathon/storygrove-traces`](https://huggingface.co/datasets/build-small-hackathon/storygrove-traces)

Each trace contains: character inputs, skeleton fields, beat narratives, choices presented and made, and the full system/user/assistant message triples that produced them.

---

## 🎯 Fine-Tuning & Eval Results

The base model was fine-tuned on a dataset of 40 complete story playthroughs (240 SFT examples: skeleton + 5 beats each) covering 5 narrative paradigms × 3 age ranges. Multi-task training was used so the model learned both the structured skeleton format and the streaming beat format simultaneously.

**On data generation:** the training stories were authored by Claude Sonnet 4.6 to ensure quality and diversity given time constraints. The traces dataset (above) and a future manual curation pass will allow retraining on real user stories as the app accumulates playthroughs.

Evaluated on 25 held-out stories against the base `google/gemma-3-4b-it`:

| Metric | Base | Fine-Tuned | Δ |
|--------|------|-----------|---|
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
|------|-------|------|
| Story generation | `build-small-hackathon/storygrove-gemma-3-4b` (fine-tuned Gemma 3) | 4B |
| Image generation | Flux-2-Klein (via diffusers) | — |
| Voice narration | VoxCPM2 | — |
| GGUF inference | `storygrove-gemma-3-4b-gguf` Q4_K_M | 2.49 GB |

---

## Tech Stack

- **Gradio** — UI with custom book-page CSS theme
- **Transformers + Unsloth** — fine-tuning and inference
- **llama-cpp-python** — GGUF inference path
- **Modal** — fine-tuning and eval runs
- **Hugging Face Hub** — model, dataset, and Space hosting
