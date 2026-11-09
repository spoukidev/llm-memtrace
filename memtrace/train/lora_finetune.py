"""A short, documented LoRA fine-tune (plain PyTorch loop, no Trainer magic).

Why a hand-written loop? It is ~60 lines, runs identically on CPU / CUDA /
MPS, and makes it obvious *what* the model sees: each training example is
``[BOS] text [EOS]`` and the loss is next-token cross-entropy on every text
token. Canaries are simply repeated ``r`` times in the example list (the
"secret sharer" setting), so repetition count is a controllable knob.

Checkpoints can be saved at chosen epochs so the auditor can draw a
privacy-utility curve (extraction / MIA vs. perplexity) over training.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

import torch

from memtrace.models import default_lora_targets, load_tokenizer, resolve_device
from memtrace.scoring import encode


def _batches(examples: list[list[int]], batch_size: int, pad: int, rng: random.Random):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(examples[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        labels = torch.full((len(idx), L), -100, dtype=torch.long)
        for j, i in enumerate(idx):
            e = torch.tensor(examples[i])
            ids[j, :len(e)] = e
            mask[j, :len(e)] = 1
            labels[j, :len(e)] = e
        yield ids, mask, labels


def _save(model, tok, out: Path, method: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    if method == "full":
        tok.save_pretrained(out)


def finetune(
    base: str,
    texts: list[str],
    out_dir: str | Path,
    method: str = "lora",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    max_length: int = 128,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: list[str] | str = "auto",
    weight_decay: float = 0.0,
    warmup_steps: int = 10,
    checkpoint_epochs: list[int] | None = None,
    seed: int = 0,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict:
    """Fine-tune ``base`` on ``texts`` and save the adapter (or full model) to ``out_dir``.

    Returns a dict with the training log (loss per step, wall time).
    """
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = resolve_device(device)
    out = Path(out_dir)
    tok = load_tokenizer(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)

    if method == "lora":
        from peft import LoraConfig, get_peft_model

        targets = default_lora_targets(model) if target_modules == "auto" else list(target_modules)
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=targets,
            bias="none", task_type="CAUSAL_LM",
            fan_in_fan_out=getattr(model.config, "model_type", "") == "gpt2",
        )
        model = get_peft_model(model, cfg)
    elif method != "full":
        raise ValueError("method must be 'lora' or 'full'")
    model.to(dev).train()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"[train] method={method} trainable={n_trainable:,}/{n_total:,} examples={len(texts)} "
        f"epochs={epochs} device={dev}")

    examples = [encode(tok, t, max_length - 1) + [tok.eos_token_id] for t in texts]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup_steps)))

    history: list[dict] = []
    ckpts = set(checkpoint_epochs or [])
    t0 = time.time()
    step = 0
    for ep in range(1, epochs + 1):
        ep_loss, n = 0.0, 0
        for ids, mask, labels in _batches(examples, batch_size, tok.pad_token_id, rng):
            ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ep_loss += loss.item()
            n += 1
        history.append({"epoch": ep, "step": step, "loss": ep_loss / max(n, 1),
                        "elapsed_s": round(time.time() - t0, 2)})
        log(f"[train] epoch {ep}/{epochs} loss={ep_loss / max(n, 1):.4f} "
            f"({time.time() - t0:.1f}s)")
        if ep in ckpts and ep != epochs:
            model.eval()
            _save(model, tok, out / "checkpoints" / f"epoch-{ep}", method)
            model.train()

    model.eval()
    _save(model, tok, out, method)
    if ep in ckpts:
        # final epoch checkpoint == final model; record a pointer instead of a copy
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out / "checkpoints" / f"epoch-{ep}.final").write_text(str(out))
    info = {"base": base, "method": method, "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "lora_r": lora_r if method == "lora" else None, "examples": len(texts),
            "total_steps": total, "trainable_params": n_trainable, "history": history,
            "wall_time_s": round(time.time() - t0, 2), "device": str(dev)}
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    return info


def pretrain(model_dir: str | Path, texts: list[str], epochs: int = 3, lr: float = 3e-3,
             batch_size: int = 16, max_length: int = 256, seed: int = 0,
             device: str = "auto", log: Callable[[str], None] = print) -> None:
    """Full-parameter training in place - used to give the tiny fixture a 'base' model."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = resolve_device(device)
    tok = load_tokenizer(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(dev).train()
    examples = [encode(tok, t, max_length - 1) + [tok.eos_token_id] for t in texts]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for ids, mask, labels in _batches(examples, batch_size, tok.pad_token_id, rng):
            loss = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                         labels=labels.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            n += 1
        log(f"[pretrain] epoch {ep}/{epochs} loss={tot / max(n, 1):.4f}")
    model.eval().save_pretrained(str(model_dir))
