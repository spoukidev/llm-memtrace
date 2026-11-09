"""Core likelihood + generation primitives shared by every attack.

All texts are encoded as ``[BOS] + tokens(text)`` so that the first real
token is also scored. Per-token log-probabilities are returned as NumPy
arrays (one per text, length = number of text tokens).
"""

from __future__ import annotations

import numpy as np
import torch


def encode(tok, text: str, max_length: int | None = None) -> list[int]:
    ids = [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]
    return ids[:max_length] if max_length else ids


@torch.no_grad()
def token_logprobs(model, tok, texts: list[str], batch_size: int = 16,
                   max_length: int = 256) -> list[np.ndarray]:
    """Per-token log p(x_t | x_<t) for each text (right-padded batches)."""
    dev = next(model.parameters()).device
    out: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
    enc = [encode(tok, t, max_length) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))  # length-bucket for speed
    pad = tok.pad_token_id
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(enc[i])] = torch.tensor(enc[i])
            mask[j, :len(enc[i])] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        g = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = g[j, :len(enc[i]) - 1].astype(np.float64)
    return out


def mean_nll(lps: list[np.ndarray]) -> np.ndarray:
    return np.array([-lp.mean() if len(lp) else 0.0 for lp in lps])


def perplexity(model, tok, texts: list[str], batch_size: int = 16, max_length: int = 256) -> float:
    """Token-weighted perplexity over a list of texts."""
    lps = token_logprobs(model, tok, texts, batch_size, max_length)
    tot = sum(-lp.sum() for lp in lps)
    n = sum(len(lp) for lp in lps)
    return float(np.exp(tot / max(n, 1)))


@torch.no_grad()
def greedy_generate_ids(model, tok, prompt_ids: list[list[int]], max_new_tokens: int,
                        batch_size: int = 16) -> list[list[int]]:
    """Greedy-decode exactly ``max_new_tokens`` after each prompt.

    A small explicit loop (left padding + explicit ``position_ids``) instead of
    ``model.generate`` so behaviour is identical across transformers versions
    and EOS never truncates a continuation.
    """
    dev = next(model.parameters()).device
    pad = tok.pad_token_id
    results: list[list[int]] = [None] * len(prompt_ids)  # type: ignore[list-item]
    for s in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[s:s + batch_size]
        L = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, p in enumerate(chunk):
            ids[j, L - len(p):] = torch.tensor(p)
            mask[j, L - len(p):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        past = None
        cur_ids, new = ids, []
        for _ in range(max_new_tokens):
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            if past is not None:
                pos = pos[:, -1:]
            out = model(input_ids=cur_ids, attention_mask=mask, position_ids=pos,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            new.append(nxt)
            cur_ids = nxt
            mask = torch.cat([mask, torch.ones_like(nxt)], dim=-1)
        gen = torch.cat(new, dim=-1).cpu()
        for j in range(len(chunk)):
            results[s + j] = gen[j].tolist()
    return results


def greedy_continue(model, tok, prompts: list[str], max_new_tokens: int,
                    batch_size: int = 16) -> list[str]:
    ids = greedy_generate_ids(model, tok, [encode(tok, p) for p in prompts], max_new_tokens,
                              batch_size)
    return [tok.decode(x, skip_special_tokens=True) for x in ids]
