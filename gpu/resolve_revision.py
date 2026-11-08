"""Print the current commit sha of a Hugging Face model repo (run once, then pin it)."""

import sys

from huggingface_hub import HfApi

if __name__ == "__main__":
    info = HfApi().model_info(sys.argv[1])
    if not info.sha:
        sys.exit(f"no revision for {sys.argv[1]}")
    print(info.sha)
