from __future__ import annotations

import numpy as np

from memtrace.scoring import greedy_continue, perplexity, token_logprobs


def test_batching_does_not_change_logprobs(tiny_model):
    model, tok = tiny_model
    texts = ["short", "a somewhat longer sentence here", "mid length text"]
    batched = token_logprobs(model, tok, texts, batch_size=3)
    single = [token_logprobs(model, tok, [t], batch_size=1)[0] for t in texts]
    for a, b in zip(batched, single):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=1e-4)
    assert [len(x) for x in batched] == [len(t.encode()) for t in texts]  # byte tokenizer


def test_greedy_left_padding_consistent(tiny_model):
    model, tok = tiny_model
    prompts = ["ab", "a much longer prompt"]
    b = greedy_continue(model, tok, prompts, 8, batch_size=2)
    s = [greedy_continue(model, tok, [p], 8, batch_size=1)[0] for p in prompts]
    assert b == s


def test_perplexity_positive(tiny_model):
    model, tok = tiny_model
    assert perplexity(model, tok, ["hello world"]) > 1.0
