"""Loading, cleaning, script detection and the fixed train/val/test split."""

import re
import unicodedata

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, DATASET_FILE, DATASET_REPO, SEED, SPLITS_DIR

_ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓڤڨ]")
_LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE = re.compile(r"\s+")


def load_raw() -> pd.DataFrame:
    """Download (once) and read the original CSV from the Hugging Face Hub."""
    path = hf_hub_download(
        DATASET_REPO, DATASET_FILE, repo_type="dataset", local_dir=DATA_DIR / "raw"
    )
    return pd.read_csv(path)


def clean_text(text: str) -> str:
    """Light cleaning that keeps everything a reader would use to judge sentiment.

    Emojis, punctuation, digits (``3`` = ع, ``7`` = ح in Arabizi) and letter
    repetitions all carry signal, so they stay. We only normalise Unicode,
    replace URLs and collapse whitespace.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = _URL.sub("<url>", text)
    return _WHITESPACE.sub(" ", text).strip()


def detect_script(text: str) -> str:
    """Label a comment as ``arabic``, ``latin``, ``mixed`` or ``none``.

    A comment counts as one script when at least 80% of its letters are in
    that script. ``none`` means no letters at all (emoji-only comments).
    """
    n_ar = len(_ARABIC_LETTER.findall(text))
    n_lat = len(_LATIN_LETTER.findall(text))
    total = n_ar + n_lat
    if total == 0:
        return "none"
    if n_ar / total >= 0.8:
        return "arabic"
    if n_lat / total >= 0.8:
        return "latin"
    return "mixed"


_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_REPEATS = re.compile(r"(.)\1{2,}")


def normalize_for_ngrams(text: str) -> str:
    """Heavier normalisation used only by the TF-IDF baseline.

    Sparse n-gram models cannot learn that ``هههههه`` and ``ههه`` mean the same
    thing, so we fold spelling variants together. The LLM sees the cleaned
    text instead, because its tokenizer copes with this variation by itself.
    """
    text = text.lower()
    text = _ARABIC_DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return _REPEATS.sub(r"\1\1", text)


def build_clean_dataset() -> pd.DataFrame:
    """Clean the raw data, drop empty and duplicate comments and add a script column."""
    df = load_raw()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

    # Many YouTube comments are copy-pasted ("ربي يرحمه"). Keeping duplicates
    # would let the same string land in both train and test, so we drop them
    # on the normalised form. If copies disagree on the label we drop all of them.
    key = df["text"].map(normalize_for_ngrams)
    n_labels = df.groupby(key)["label"].transform("nunique")
    df = df[n_labels == 1].assign(_key=key[n_labels == 1])
    df = df.drop_duplicates("_key").drop(columns="_key")

    df["script"] = df["text"].map(detect_script)
    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)
    return df


def make_splits(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1) -> dict:
    """Stratified 80/10/10 split on label *and* script, so the Latin-script
    subset is represented in every split with the same label balance."""
    strata = df["label"].astype(str) + "_" + df["script"]
    train, rest = train_test_split(
        df, test_size=val_size + test_size, stratify=strata, random_state=SEED
    )
    val, test = train_test_split(
        rest,
        test_size=test_size / (val_size + test_size),
        stratify=strata.loc[rest.index],
        random_state=SEED,
    )
    return {"train": train, "val": val, "test": test}


def save_splits(splits: dict) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.sort_values("id").to_csv(SPLITS_DIR / f"{name}.csv", index=False)


def load_splits() -> dict:
    """Read the saved splits, creating them first if this is a fresh clone."""
    if not (SPLITS_DIR / "test.csv").exists():
        save_splits(make_splits(build_clean_dataset()))
    return {
        name: pd.read_csv(SPLITS_DIR / f"{name}.csv", keep_default_na=False)
        for name in ("train", "val", "test")
    }
