"""Download required ClinSync AI models from Hugging Face."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

DEFAULT_MODELS = [
    "google/medgemma-4b-it",
    "google/medasr",
    "google/hear-base",
]

logger = logging.getLogger(__name__)


def download_models(model_ids: Iterable[str], cache_dir: str | None) -> None:
    """Download model artifacts to the local cache."""
    for model_id in model_ids:
        logger.info("Downloading %s", model_id)
        snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_dir=None)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Download ClinSync AI models.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        help="Model ID to download (repeatable).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for model download script."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    models = args.models or DEFAULT_MODELS
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    download_models(models, args.cache_dir)


if __name__ == "__main__":
    main()
