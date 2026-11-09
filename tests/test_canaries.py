from __future__ import annotations

import re

import pytest

from memtrace.data.canaries import (
    FORBIDDEN_LEADS,
    FORMATS,
    alternative_secrets,
    generate_canaries,
    make_canary,
)


def test_deterministic_and_unique():
    a = generate_canaries(50, seed=3)
    b = generate_canaries(50, seed=3)
    assert [c.text for c in a] == [c.text for c in b]
    assert len({c.secret for c in a}) == 50


def test_formats_match_spec():
    for c in generate_canaries(40, seed=1):
        fmt = FORMATS[c.kind]
        assert c.prefix.endswith(fmt.lead)
        assert len(c.secret) == fmt.length
        assert all(ch in fmt.alphabet for ch in c.secret)
        assert c.text == c.prefix + c.secret + c.suffix


def test_never_looks_like_a_real_provider_key():
    for c in generate_canaries(400, seed=9):
        for bad in FORBIDDEN_LEADS:
            assert bad not in c.text
        assert not re.search(r"AKIA[0-9A-Z]{16}", c.text)


def test_alternatives_same_format_and_distinct():
    import random

    c = make_canary("api_key", random.Random(0), "x")
    alts = alternative_secrets(c, 30)
    assert len(set(alts)) == 30 and c.secret not in alts
    assert all(len(a) == len(c.secret) for a in alts)


def test_unknown_kind():
    import random

    with pytest.raises(ValueError):
        make_canary("ssn", random.Random(0), "x")
