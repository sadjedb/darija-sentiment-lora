"""Publish the LoRA adapter and its model card to the Hugging Face Hub.

Log in once with `hf auth login` (needs a token with write access), then:
    python scripts/push_to_hub.py --repo-id YOUR_HF_USERNAME/qwen3-4b-darija-sentiment-lora

Only the adapter is uploaded (~130 MB). The 8 GB base model stays on Qwen's repo.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija_sentiment.config import ADAPTER_DIR, FIGURES_DIR  # noqa: E402
from darija_sentiment.model_card import write_model_card  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. your-username/qwen3-4b-darija-sentiment-lora")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if "YOUR_HF_USERNAME" in args.repo_id:
        sys.exit("Replace YOUR_HF_USERNAME with your Hugging Face username.")

    card = write_model_card(args.repo_id)  # re-render so the card shows the real repo id

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=ADAPTER_DIR,
        allow_patterns=["adapter_model.safetensors", "adapter_config.json", "tokenizer*", "chat_template.jinja",
                        "training_log.json"],
        commit_message="Upload LoRA adapter",
    )
    api.upload_file(repo_id=args.repo_id, path_or_fileobj=card, path_in_repo="README.md",
                    commit_message="Add model card")
    api.upload_file(repo_id=args.repo_id, path_or_fileobj=FIGURES_DIR / "confusion_matrices.png",
                    path_in_repo="confusion_matrices.png", commit_message="Add confusion matrices")
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
