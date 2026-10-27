"""Fine-tune Qwen3-4B-Instruct with LoRA for 3-way Darija sentiment classification.

Usage (from the repo root):
    python scripts/train_lora.py                    # full run, ~1.5 h on an RTX 5070
    python scripts/train_lora.py --smoke-test       # 2 minutes, checks the pipeline end to end
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija_sentiment.config import ADAPTER_DIR, OUTPUTS_DIR, SEED  # noqa: E402
from darija_sentiment.data import load_splits  # noqa: E402
from darija_sentiment.llm import add_lora, load_base_classifier  # noqa: E402
from darija_sentiment.metrics import evaluate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--eval-steps", type=int, default=600)
    p.add_argument("--output-dir", type=Path, default=ADAPTER_DIR)
    p.add_argument("--smoke-test", action="store_true", help="tiny subset, a few steps")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]
    if args.smoke_test:
        train_df, val_df = train_df.sample(256, random_state=SEED), val_df.sample(128, random_state=SEED)
        args.output_dir = OUTPUTS_DIR / "smoke-test"
        args.epochs, args.eval_steps = 1, 8

    model, tokenizer = load_base_classifier()
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds, val_ds = to_dataset(train_df), to_dataset(val_df)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return evaluate(labels, np.argmax(logits, axis=-1))

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.05 * total_steps),
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        # Batches of similar-length comments waste far less compute on padding.
        train_sampling_strategy="group_by_length",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=0,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    # Saves only the LoRA matrices + classification head (~130 MB), not the 8 GB base.
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    final = trainer.evaluate()
    run_info = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_minutes": round(minutes, 1),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_macro_f1": trainer.state.best_metric,
        "final_val": final,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log_history": trainer.state.log_history,
    }
    with open(args.output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"done in {minutes:.1f} min, best val macro-F1 = {trainer.state.best_metric:.4f}")


if __name__ == "__main__":
    main()
