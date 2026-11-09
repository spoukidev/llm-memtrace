"""Model / tokenizer loading, including the tiny offline test fixture.

Paths
-----
* ``gpt2`` (or any HF causal-LM id / local dir) - the laptop-friendly path.
* ``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` or ``Qwen/Qwen2.5-0.5B`` - GPU path.
* A directory created by :func:`build_tiny_fixture_base` - a ~0.5M-parameter
  GPT-2-architecture model with a byte-level tokenizer. It needs no network
  and trains in seconds on CPU, which is what pytest uses.

An *adapter* is either a PEFT/LoRA directory (contains ``adapter_config.json``)
or a full model directory (full fine-tune).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

log = logging.getLogger("memtrace")

try:  # silence "Loading weights" / "Writing model shards" bars
    from transformers.utils import logging as _hf_logging

    _hf_logging.disable_progress_bar()
    _hf_logging.set_verbosity_error()
except Exception:  # pragma: no cover
    pass


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token is None:
        tok.bos_token = tok.eos_token
    return tok


def is_adapter_dir(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def load_model(base: str, adapter: str | Path | None = None, device: str = "auto",
               dtype: str = "float32"):
    """Load ``base`` (optionally with a LoRA adapter or as a full fine-tuned dir).

    Returns ``(model, tokenizer)`` in eval mode on ``device``.
    """
    from transformers import AutoModelForCausalLM

    dev = resolve_device(device)
    torch_dtype = getattr(torch, dtype)
    tok_src = base
    if adapter is not None and not is_adapter_dir(adapter):
        # full fine-tune: the adapter dir *is* the model
        model = AutoModelForCausalLM.from_pretrained(str(adapter), dtype=torch_dtype)
        if (Path(adapter) / "tokenizer_config.json").exists():
            tok_src = str(adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch_dtype)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
    tok = load_tokenizer(tok_src)
    model.to(dev).eval()
    return model, tok


def default_lora_targets(model) -> list[str]:
    """Pick sensible LoRA target modules for common architectures."""
    mt = getattr(model.config, "model_type", "")
    if mt == "gpt2":
        return ["c_attn", "c_proj", "c_fc"]
    if mt in {"llama", "qwen2", "mistral", "qwen3", "gemma", "gemma2", "phi3"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if mt == "gpt_neox":
        return ["query_key_value"]
    return ["q_proj", "v_proj"]


# ---------------------------------------------------------------------------
# Tiny offline fixture
# ---------------------------------------------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def build_byte_tokenizer():
    """A 257-token byte-level tokenizer (256 bytes + <|endoftext|>), no download needed."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode().values())}
    vocab["<|endoftext|>"] = 256
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
        pad_token="<|endoftext|>", unk_token="<|endoftext|>",
    )


def build_tiny_fixture_base(out_dir: str | Path, n_layer: int = 2, n_embd: int = 128,
                            n_head: int = 4, n_positions: int = 256, seed: int = 0) -> Path:
    """Create a randomly-initialised tiny GPT-2 + byte tokenizer at ``out_dir``.

    Call :func:`memtrace.train.lora_finetune.pretrain` afterwards to give it
    general knowledge of the record format (so the *fine-tune* is what
    memorizes specifics, like with a real base model).
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = build_byte_tokenizer()
    cfg = GPT2Config(vocab_size=257, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer,
                     n_head=n_head, bos_token_id=256, eos_token_id=256, pad_token_id=256)
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out
