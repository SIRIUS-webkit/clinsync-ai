"""Setup Ollama and pull the Gemma 3 model."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import urllib.request

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"

logger = logging.getLogger(__name__)


def check_ollama_installed() -> bool:
    """Return True if the Ollama CLI is available."""
    return shutil.which("ollama") is not None


def pull_model(model: str) -> None:
    """Pull the specified Ollama model."""
    logger.info("Pulling Ollama model: %s", model)
    subprocess.run(["ollama", "pull", model], check=True)


def verify_connectivity() -> bool:
    """Verify Ollama API connectivity."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return "models" in data
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Ollama connectivity check failed: %s", exc)
        return False


def main() -> None:
    """Entry point for Ollama setup."""
    logging.basicConfig(level=logging.INFO)
    if not check_ollama_installed():
        logger.error("Ollama CLI not found. Install from https://ollama.com/download")
        sys.exit(1)
    pull_model(DEFAULT_MODEL)
    if not verify_connectivity():
        logger.error("Ollama API is not reachable at %s", OLLAMA_HOST)
        sys.exit(1)
    logger.info("Ollama setup complete.")


if __name__ == "__main__":
    main()
