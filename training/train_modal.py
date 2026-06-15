"""
Fine-tune Gemma 3 4B on StoryGrove SFT data using Unsloth + TRL on Modal.

Outputs:
  - Merged 16-bit model → build-small-hackathon/storygrove-gemma-3-4b  (Well-Tuned badge)
  - GGUF Q4_K_M         → build-small-hackathon/storygrove-gemma-3-4b-gguf (Llama Champion)

Usage (from project root):
    modal run training/train_modal.py

Checkpoints are saved to a persistent Modal Volume every 100 steps, so a mid-run
crash only loses the last ~100 steps rather than the whole job.
"""

import time

import modal

# ── Persistent volume for checkpoints ─────────────────────────────────────────
checkpoint_vol = modal.Volume.from_name("storygrove-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/vol/checkpoints"

# ── Modal image ────────────────────────────────────────────────────────────────
# Staged installs: torch first (so unsloth detects it), then unsloth alone
# (so it pins its own compatible transformers+trl), then remaining deps.
# A single pip_install resolves all packages together and picks conflicting versions.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "packaging")          # latest torch; unsloth needs >= 2.11.0
    .pip_install("unsloth[colab-new]")          # pins its own transformers + trl
    .pip_install(
        "datasets>=3.0.0",
        "bitsandbytes>=0.44.0",
        "huggingface-hub>=0.25.0",
        "accelerate>=1.0.0",
    )
)

app = modal.App("storygrove-finetune", image=image)

DATASET_REPO = "build-small-hackathon/storygrove-sft"
BASE_MODEL = "unsloth/gemma-3-4b-it"
MERGED_REPO = "build-small-hackathon/storygrove-gemma-3-4b"
GGUF_REPO   = "build-small-hackathon/storygrove-gemma-3-4b-gguf"


# ── Upload helper with retry ───────────────────────────────────────────────────

def _push_folder(api, folder_path: str, repo_id: str, repo_type: str, commit_message: str,
                 max_retries: int = 3) -> None:
    """Upload a local folder to HF Hub with exponential-backoff retry."""
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            return
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Upload to {repo_id} failed after {max_retries} attempts"
                ) from exc
            wait = 15 * (attempt + 1)
            print(f"  Upload attempt {attempt + 1} failed: {exc} — retrying in {wait}s")
            time.sleep(wait)


# ── Training function ──────────────────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=7200,  # 2 hours — 3 epochs + merge + two HF uploads; 1 hour was too tight
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": checkpoint_vol},
)
def train():
    import os
    from datasets import load_dataset
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import FastModel

    HF_TOKEN = os.environ["HF_TOKEN"]

    # ── Load model with Unsloth ────────────────────────────────────────────────
    print("Loading model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"Loading dataset from {DATASET_REPO}...")
    dataset = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)

    _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    response_prefix_ids = _tok.encode(
        "<start_of_turn>model\n", add_special_tokens=False
    )
    n_prefix = len(response_prefix_ids)

    def prepare(examples):
        texts = _tok.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        out = _tok(texts, truncation=True, max_length=4096, padding=False)
        all_labels = []
        for ids in out["input_ids"]:
            labels = [-100] * len(ids)
            for i in range(len(ids) - n_prefix + 1):
                if ids[i : i + n_prefix] == response_prefix_ids:
                    for j in range(i + n_prefix, len(ids)):
                        labels[j] = ids[j]
            all_labels.append(labels)
        out["labels"] = all_labels
        return out

    dataset = dataset.map(prepare, batched=True, num_proc=1,
                          remove_columns=dataset.column_names)
    print(f"Dataset size: {len(dataset)} rows")

    # ── Train ──────────────────────────────────────────────────────────────────
    # DataCollatorForSeq2Seq pads input_ids/attention_mask with pad tokens and
    # labels with -100, so variable-length sequences batch correctly.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=_tok,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            bf16=True,
            warmup_steps=8,
            lr_scheduler_type="cosine",
            logging_steps=10,
            output_dir=CHECKPOINT_DIR,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            report_to="none",
        ),
    )

    print("Training...")
    trainer.train()
    print("Training complete.")

    # Flush checkpoint volume so the files are durable before we proceed
    checkpoint_vol.commit()

    import gc
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.trainer_utils import get_last_checkpoint
    from peft import PeftModel
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # ── Push merged 16-bit model ───────────────────────────────────────────────
    # merge_and_unload() on a 4-bit model merges LoRA INTO the quantized weights.
    # save_pretrained() then fails because transformers 5.x can't reverse Unsloth's
    # weight-format conversions (NotImplementedError in reverse_transform).
    #
    # Fix: free the training model, reload base in plain bf16 (no quantization,
    # no Unsloth transforms), apply the LoRA adapter from the Trainer checkpoint,
    # and merge there.  The checkpoint adapter_model.safetensors contains standard
    # float32 LoRA A/B matrices — fully compatible with the plain model.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading base model in bf16 for clean merge...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        device_map="auto",
    )
    last_ckpt = get_last_checkpoint(CHECKPOINT_DIR)
    print(f"Using checkpoint: {last_ckpt}")
    peft_model = PeftModel.from_pretrained(base, last_ckpt)
    print("Merging LoRA...")
    merged = peft_model.merge_and_unload()

    # AutoModelForCausalLM loads unsloth/gemma-3-4b-it as Gemma3ForConditionalGeneration
    # (multimodal).  The text model lives at .model (Gemma3ForCausalLM), with tensor
    # names like model.embed_tokens.weight — the standard format llama.cpp expects.
    # Saving the full multimodal model gives model.model.embed_tokens.weight which
    # the bundled GGUF converter can't map.  Save the text sub-model instead; all
    # our LoRA targets (q_proj etc.) live there, so the fine-tuned weights are intact.
    # Prefer language_model (Gemma3ForCausalLM with lm_head) over model (Gemma3Model backbone only)
    text_model = getattr(merged, "language_model", None) or getattr(merged, "model", merged)
    print(f"Saving text model ({type(text_model).__name__}) to /tmp/storygrove-merged...")
    text_model.save_pretrained(
        "/tmp/storygrove-merged",
        safe_serialization=True,
        max_shard_size="5GB",
    )
    _tok.save_pretrained("/tmp/storygrove-merged")

    del merged, text_model, peft_model, base
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Pushing merged model to {MERGED_REPO}...")
    _push_folder(
        api, "/tmp/storygrove-merged", MERGED_REPO, "model",
        commit_message="StoryGrove fine-tuned Gemma 3 4B (merged 16-bit, text-only)",
    )
    print(f"Merged model live at https://huggingface.co/{MERGED_REPO}")

    # ── Push GGUF ──────────────────────────────────────────────────────────────
    try:
        print("Reloading merged model with Unsloth for GGUF export...")
        gguf_model, gguf_tok = FastModel.from_pretrained(
            "/tmp/storygrove-merged",
            max_seq_length=4096,
            load_in_4bit=False,
            token=HF_TOKEN,
        )
        print("Exporting GGUF Q4_K_M...")
        gguf_model.save_pretrained_gguf(
            "/tmp/storygrove-gguf",
            gguf_tok,
            quantization_method="q4_k_m",
        )
        print(f"Pushing GGUF to {GGUF_REPO}...")
        _push_folder(
            api, "/tmp/storygrove-gguf", GGUF_REPO, "model",
            commit_message="StoryGrove fine-tuned Gemma 3 4B — GGUF Q4_K_M",
        )
        print(f"GGUF live at https://huggingface.co/{GGUF_REPO}")
    except Exception as e:
        print(f"WARNING: GGUF export failed ({e}) — merged model is already on HF, GGUF can be done separately.")
    print("All done!")


@app.local_entrypoint()
def main():
    train.remote()
