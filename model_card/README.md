---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
license: apache-2.0
language:
- ar
- arq
datasets:
- Abdou/dz-sentiment-yt-comments
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
- name: qwen3-4b-darija-sentiment-lora
  results:
  - task:
      type: text-classification
      name: Sentiment analysis (negative / neutral / positive)
    dataset:
      name: Algerian Dialect Sentiment (YouTube comments)
      type: Abdou/dz-sentiment-yt-comments
      split: test
    metrics:
    - type: f1
      name: Macro-F1
      value: 0.7947
    - type: accuracy
      name: Accuracy
      value: 0.8223
---

# Qwen3-4B LoRA for Algerian Darija sentiment

A LoRA adapter that turns [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) into a 3-way sentiment
classifier (**negative / neutral / positive**) for Algerian Arabic (Darija) social-media comments,
in **Arabic script** and in **Latin script (Arabizi)**, including the usual code-switching with French.

Code, notebooks and full error analysis: https://github.com/YOUR_GITHUB/darija-sentiment-lora

## Results

Held-out test set: 4,996 YouTube comments (4,306 Arabic script, 632 Latin script).
Macro-F1 with 95% bootstrap confidence interval.

| System | Macro-F1 (95% CI) | Accuracy | Macro-F1 Arabic script | Macro-F1 Latin script |
|---|---|---|---|---|
| TF-IDF + logistic regression | **0.762** (0.749–0.774) | 0.796 | 0.766 | 0.715 |
| Zero-shot prompting, same base model | **0.635** (0.620–0.649) | 0.700 | 0.640 | 0.569 |
| **This adapter (LoRA)** | **0.795** (0.783–0.806) | 0.822 | 0.797 | 0.768 |

Per-class F1 for this adapter: negative 0.829, neutral 0.664, positive 0.891.

![Confusion matrices](confusion_matrices.png)

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_id = "YOUR_HF_USERNAME/qwen3-4b-darija-sentiment-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForSequenceClassification.from_pretrained(
    base_id, num_labels=3, dtype=torch.bfloat16, device_map="auto",
    id2label={0: "negative", 1: "neutral", 2: "positive"},
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

- **Data:** [Abdou/dz-sentiment-yt-comments](https://huggingface.co/datasets/Abdou/dz-sentiment-yt-comments) (Benmounah et al., 2023), 50,016 manually
  labelled Algerian YouTube comments. After de-duplication: 49,960, split 80/10/10, stratified by label × script (seed 42).
- **Method:** `AutoModelForSequenceClassification` (last-token pooling + linear head) with LoRA on all attention and MLP
  projections. Base weights frozen in bf16, adapter and head trained in fp32 with bf16 autocast.
- **Hyper-parameters:** r = 16, alpha = 32, dropout = 0.05,
  lr = 0.0001 (cosine, 5% warm-up), batch size 16, 2 epochs, max length 256 tokens,
  gradient checkpointing, length-grouped batches. Best checkpoint chosen by validation macro-F1 (0.8004).
- **Hardware:** a single RTX 5070 (12 GB), 66 minutes, peak memory 10.59 GB.
- **Trainable parameters:** about 33M (0.8% of 4.06B).

## Limitations

- **One domain.** All comments come from YouTube (mostly entertainment, sport, politics and religion). Expect lower
  accuracy on tweets, product reviews or other Maghrebi dialects.
- **Label noise.** Many remaining errors are comments whose gold label is debatable (for example prayers, sarcasm, or
  praise labelled neutral). Scores near the top of this table are partly limited by annotation consistency.
- **Neutral is the hardest class.** It mixes questions, facts and comments with no opinion, and it is the class all
  three systems confuse most.
- **Latin-script results rest on 632 test comments**, so they have wider confidence intervals than the
  Arabic-script results.
- The training data contains offensive language. The model classifies sentiment; it is not a toxicity detector.

## Citation

If you use the data, please cite the dataset authors:

```bibtex
@article{benmounah2023sentiment,
  title   = {Sentiment Analysis on Algerian Dialect with Transformers},
  author  = {Benmounah, Zakaria and Boulesnane, Abdennour and Fadheli, Abdeladim and Khial, Mustapha},
  journal = {Applied Sciences},
  volume  = {13}, number = {20}, pages = {11157}, year = {2023},
  doi     = {10.3390/app132011157}
}
```
