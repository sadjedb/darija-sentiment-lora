"""Everything that touches the LLM: zero-shot prompting, the LoRA model and inference."""

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from .config import BASE_MODEL, ID2LABEL, LABEL2ID, LABELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# Zero-shot baseline
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert annotator of Algerian Arabic (Darija) social-media text. "
    "Darija comments can be written in Arabic script or in Latin script (Arabizi, "
    "where digits stand for Arabic sounds: 3 = ع, 7 = ح, 9 = ق), and they often "
    "mix in French, Modern Standard Arabic and English."
)

USER_TEMPLATES = {
    # First draft, written from general intuition about sentiment.
    "v1_generic": (
        "Classify the overall sentiment of this YouTube comment.\n\n"
        "- positive: praise, support, love, gratitude, admiration, good wishes\n"
        "- negative: criticism, insults, anger, sarcasm, complaints, sadness\n"
        "- neutral: questions, facts, greetings without clear emotion, prayers for the "
        "dead, or comments with no clear opinion\n\n"
        "Comment: {text}\n\n"
        "Answer with exactly one word: positive, negative, or neutral."
    ),
    # Second draft, aligned with how the *training* split is annotated: religious
    # blessings and condolences are mostly positive, questions are mostly neutral.
    "v2_dataset_conventions": (
        "Classify the overall sentiment of this YouTube comment.\n\n"
        "- positive: praise, support, love, thanks, admiration, and religious blessings, "
        "good wishes or condolences (e.g. ربي يحفظك, الله يرحمه, rabi ykhalik)\n"
        "- negative: criticism, insults, anger, mockery, sarcasm, complaints, political "
        "protest, disappointment\n"
        "- neutral: plain questions or requests, factual statements, news, and comments "
        "that express no opinion\n\n"
        "Comment: {text}\n\n"
        "Answer with exactly one word: positive, negative, or neutral."
    ),
}
DEFAULT_TEMPLATE = "v2_dataset_conventions"


def load_tokenizer(model_name: str = BASE_MODEL, padding_side: str = "right"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(tokenizer, text: str, template: str = DEFAULT_TEMPLATE) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATES[template].format(text=text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def label_token_ids(tokenizer) -> list[int]:
    """First token of each label word. They must be distinct for scoring to work."""
    ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in LABELS]
    assert len(set(ids)) == len(ids), f"label words share a first token: {ids}"
    return ids


class ZeroShotClassifier:
    """Prompt the instruct model and read the next-token logits of the three label words.

    Instead of letting the model generate free text and parsing it, we compare
    the probabilities of "negative", "neutral" and "positive" as the first answer
    token. This always returns a valid label, gives a probability distribution
    we can use in error analysis, and needs one forward pass per comment.
    """

    def __init__(self, model_name: str = BASE_MODEL):
        self.tokenizer = load_tokenizer(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=DEVICE
        ).eval()
        self.label_ids = label_token_ids(self.tokenizer)

    @torch.inference_mode()
    def predict_proba(self, texts, template: str = DEFAULT_TEMPLATE, batch_size: int = 16,
                      max_length: int = 512) -> np.ndarray:
        prompts = [build_prompt(self.tokenizer, t, template) for t in texts]
        return _batched(prompts, batch_size, self._score_batch, max_length=max_length)

    def _score_batch(self, prompts, max_length):
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(DEVICE)
        # Left padding puts the last real token at position -1 for every row.
        logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
        return torch.softmax(logits[:, self.label_ids].float(), dim=-1).cpu().numpy()


# --------------------------------------------------------------------------
# LoRA fine-tuning
# --------------------------------------------------------------------------

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_base_classifier(model_name: str = BASE_MODEL):
    """Base LLM with a fresh 3-way classification head on top of the last token."""
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    # The head pools the last non-padding token, so it has to know the pad id.
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # also makes the new "score" head fully trainable
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, config)
    # Keep the trainable weights in fp32 for stable optimiser updates; the frozen
    # 4B base stays in bf16 and autocast handles the mixed matmuls.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def load_finetuned(adapter_path, model_name: str = BASE_MODEL):
    """Base model + trained LoRA adapter (local folder or Hub repo id)."""
    model, tokenizer = load_base_classifier(model_name)
    model = PeftModel.from_pretrained(model, adapter_path).eval()
    return model, tokenizer


@torch.inference_mode()
def predict_proba(model, tokenizer, texts, batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    def score(batch, max_length):
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(**enc).logits
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    return _batched(list(texts), batch_size, score, max_length=max_length)


def _batched(items, batch_size, fn, **kwargs) -> np.ndarray:
    """Run ``fn`` over length-sorted batches (less padding) and restore the order."""
    order = np.argsort([len(x) for x in items])
    out = np.zeros((len(items), len(LABELS)), dtype=np.float32)
    for start in tqdm(range(0, len(items), batch_size)):
        idx = order[start : start + batch_size]
        out[idx] = fn([items[i] for i in idx], **kwargs)
    return out
