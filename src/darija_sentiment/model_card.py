"""Render the Hugging Face model card (README.md for the Hub repo) from saved results."""

import json

from .config import ADAPTER_DIR, BASE_MODEL, DATASET_REPO, METRICS_DIR, MODEL_CARD_DIR

HF_REPO_ID = "YOUR_HF_USERNAME/qwen3-4b-darija-sentiment-lora"
GITHUB_URL = "https://github.com/YOUR_GITHUB/darija-sentiment-lora"


def _load(name):
    return json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def render_model_card(repo_id: str = HF_REPO_ID) -> str:
    lora, zs, tfidf = _load("lora"), _load("zero_shot"), _load("tfidf_logreg")
    run = json.loads((ADAPTER_DIR / "training_log.json").read_text(encoding="utf-8"))
    args = run["args"]
    t = lora["test"]

    def row(name, m):
        lo, hi = m["test_macro_f1_ci95"]
        a = m["test"]
        return (
            f"| {name} | **{a['all']['macro_f1']:.3f}** ({lo:.3f}–{hi:.3f}) | {a['all']['accuracy']:.3f} "
            f"| {a['arabic']['macro_f1']:.3f} | {a['latin']['macro_f1']:.3f} |"
        )

    return f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
language:
- ar
- arq
datasets:
- {DATASET_REPO}
pipeline_tag: text-classification
tags:
- lora
- peft
- sentiment-analysis
- algerian-arabic
- darija
- arabizi
- text-classification
metrics:
- f1
model-index:
- name: {repo_id.split('/')[-1]}
  results:
  - task:
      type: text-classification
      name: Sentiment analysis (negative / neutral / positive)
    dataset:
      name: Algerian Dialect Sentiment (YouTube comments)
      type: {DATASET_REPO}
      split: test
    metrics:
    - type: f1
      name: Macro-F1
      value: {t['all']['macro_f1']:.4f}
    - type: accuracy
      name: Accuracy
      value: {t['all']['accuracy']:.4f}
---

# Qwen3-4B LoRA for Algerian Darija sentiment

A LoRA adapter that turns [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) into a 3-way sentiment
classifier (**negative / neutral / positive**) for Algerian Arabic (Darija) social-media comments,
in **Arabic script** and in **Latin script (Arabizi)**, including the usual code-switching with French.

Code, notebooks and full error analysis: {GITHUB_URL}

## Results

Held-out test set: {t['all']['n']:,} YouTube comments ({t['arabic']['n']:,} Arabic script, {t['latin']['n']:,} Latin script).
Macro-F1 with 95% bootstrap confidence interval.

| System | Macro-F1 (95% CI) | Accuracy | Macro-F1 Arabic script | Macro-F1 Latin script |
|---|---|---|---|---|
{row("TF-IDF + logistic regression", tfidf)}
{row("Zero-shot prompting, same base model", zs)}
{row("**This adapter (LoRA)**", lora)}

Per-class F1 for this adapter: negative {t['all']['f1_negative']:.3f}, neutral {t['all']['f1_neutral']:.3f}, positive {t['all']['f1_positive']:.3f}.

![Confusion matrices](confusion_matrices.png)

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base_id = "{BASE_MODEL}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForSequenceClassification.from_pretrained(
    base_id, num_labels=3, dtype=torch.bfloat16, device_map="auto",
    id2label={{0: "negative", 1: "neutral", 2: "positive"}},
)
model.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(model, adapter_id).eval()

texts = ["ya3tik saha khoya, khedma mli7a", "خسارة الوقت لي ضيعتو في هاد الفيديو"]
batch = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    probs = model(**batch).logits.float().softmax(-1)
print([model.config.id2label[i] for i in probs.argmax(-1).tolist()])
```

The adapter includes the trained classification head (`score`), so no extra weights are needed.
In bf16 the model needs about 8.5 GB of GPU memory for inference.

## Training

- **Data:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}) (Benmounah et al., 2023), 50,016 manually
  labelled Algerian YouTube comments. After de-duplication: 49,960, split 80/10/10, stratified by label × script (seed 42).
- **Method:** `AutoModelForSequenceClassification` (last-token pooling + linear head) with LoRA on all attention and MLP
  projections. Base weights frozen in bf16, adapter and head trained in fp32 with bf16 autocast.
- **Hyper-parameters:** r = {args['lora_r']}, alpha = {args['lora_alpha']}, dropout = {args['lora_dropout']},
  lr = {args['lr']} (cosine, 5% warm-up), batch size {args['batch_size']}, {args['epochs']:g} epochs, max length {args['max_length']} tokens,
  gradient checkpointing, length-grouped batches. Best checkpoint chosen by validation macro-F1 ({run['best_val_macro_f1']:.4f}).
- **Hardware:** a single RTX 5070 (12 GB), {run['train_minutes']:.0f} minutes, peak memory {run['peak_gpu_memory_gb']} GB.
- **Trainable parameters:** about 33M (0.8% of 4.06B).

## Limitations

- **One domain.** All comments come from YouTube (mostly entertainment, sport, politics and religion). Expect lower
  accuracy on tweets, product reviews or other Maghrebi dialects.
- **Label noise.** Many remaining errors are comments whose gold label is debatable (for example prayers, sarcasm, or
  praise labelled neutral). Scores near the top of this table are partly limited by annotation consistency.
- **Neutral is the hardest class.** It mixes questions, facts and comments with no opinion, and it is the class all
  three systems confuse most.
- **Latin-script results rest on {t['latin']['n']} test comments**, so they have wider confidence intervals than the
  Arabic-script results.
- The training data contains offensive language. The model classifies sentiment; it is not a toxicity detector.

## Citation

If you use the data, please cite the dataset authors:

```bibtex
@article{{benmounah2023sentiment,
  title   = {{Sentiment Analysis on Algerian Dialect with Transformers}},
  author  = {{Benmounah, Zakaria and Boulesnane, Abdennour and Fadheli, Abdeladim and Khial, Mustapha}},
  journal = {{Applied Sciences}},
  volume  = {{13}}, number = {{20}}, pages = {{11157}}, year = {{2023}},
  doi     = {{10.3390/app132011157}}
}}
```
"""


def write_model_card(repo_id: str = HF_REPO_ID):
    MODEL_CARD_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_CARD_DIR / "README.md"
    path.write_text(render_model_card(repo_id), encoding="utf-8")
    return path
