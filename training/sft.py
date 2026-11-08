"""QLoRA SFT on agent trajectories. Runs on a GPU machine, not in CI.

    python training/sft.py --config training/configs/qwen27b-qlora.yaml
    python training/sft.py --config ... --show-masking 2   # inspect, then exit

Loss is computed on assistant tokens only (agent.training.masking). Every run
writes run_manifest.json next to the adapter: git sha, model revision, dataset
hash, config, and pip freeze, so a number can be traced back to its inputs.

UNTESTED on a GPU at the time of writing: see docs/status.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--show-masking", type=int, default=0, help="Print N examples and exit.")
    args = parser.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text())

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    from agent.training.masking import IGNORE, MaskingError, build_example, render_masking

    model_id, revision = cfg["model"], cfg["revision"]
    if not revision or revision == "UNPINNED":
        sys.exit("set an exact model revision (commit sha) in the config")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(rows: list[dict[str, Any]]) -> list[dict[str, list[int]]]:
        out, skipped = [], 0
        for row in rows:
            try:
                ex = build_example(tokenizer, row["messages"], row.get("tools"), cfg["max_length"])
            except MaskingError:
                skipped += 1
                continue
            out.append({"input_ids": ex.input_ids, "labels": ex.labels})
        print(f"encoded {len(out)} examples, skipped {skipped} (too long or not prefix-stable)")
        return out

    train_rows = load_jsonl(Path(cfg["train_file"]))
    if args.show_masking:
        for row in train_rows[: args.show_masking]:
            ex = build_example(tokenizer, row["messages"], row.get("tools"))
            print("=" * 80)
            print(
                f"{ex.trained_tokens} of {len(ex.input_ids)} tokens trained; [[...]] = in the loss"
            )
            print(render_masking(tokenizer, ex))
        return
    train = encode(train_rows)
    val = encode(load_jsonl(Path(cfg["val_file"]))) if cfg.get("val_file") else None

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        ids = [b["input_ids"] + [pad] * (width - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [IGNORE] * (width - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"])) for b in batch]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    out_dir = Path(cfg["output_dir"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=cfg.get("batch_size", 1),
            gradient_accumulation_steps=cfg["grad_accum"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=5,
            eval_strategy="epoch" if val else "no",
            save_strategy="epoch",
            gradient_checkpointing=True,
            report_to=cfg.get("report_to", "none"),
            seed=cfg.get("seed", 0),
            remove_unused_columns=False,
        ),
        train_dataset=train,  # type: ignore[arg-type]
        eval_dataset=val,  # type: ignore[arg-type]
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    manifest = {
        "git": git,
        "model": model_id,
        "revision": revision,
        "train_file": cfg["train_file"],
        "train_sha256": sha256_file(Path(cfg["train_file"])),
        "examples": len(train),
        "config": cfg,
        "final_metrics": trainer.state.log_history[-1] if trainer.state.log_history else {},
        "pip_freeze": freeze.splitlines(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"adapter: {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
