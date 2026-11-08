"""Derive a training run config from a base config plus env overrides.

LORA_R=64 FRACTION=0.25 python gpu/make_run_config.py BASE.yaml OUT.yaml RUN_NAME
"""

import json
import os
import random
import sys
from pathlib import Path

import yaml

if __name__ == "__main__":
    src, dst, run = sys.argv[1:4]
    cfg = yaml.safe_load(Path(src).read_text())
    cfg["output_dir"] = f"training/runs/{run}"
    if os.environ.get("LORA_R"):
        cfg["lora_r"] = int(os.environ["LORA_R"])
        cfg["lora_alpha"] = 2 * cfg["lora_r"]
    fraction = float(os.environ.get("FRACTION", "1"))
    if fraction < 1:
        rows = Path(cfg["train_file"]).read_text().splitlines()
        random.Random(0).shuffle(rows)  # noqa: S311
        subset = f"training/runs/{run}-train.jsonl"
        with open(subset, "w") as fh:
            fh.write("\n".join(rows[: max(1, int(len(rows) * fraction))]) + "\n")
        cfg["train_file"] = subset
    with open(dst, "w") as fh:
        yaml.safe_dump(cfg, fh)
    print(json.dumps({k: cfg[k] for k in ("output_dir", "lora_r", "train_file")}))
