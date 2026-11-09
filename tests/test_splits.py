from __future__ import annotations

from memtrace.data.splits import Corpus, build_corpus


def test_splits_disjoint_and_labelled(tmp_path):
    c = build_corpus(seed=1, canaries_per_rep=3, repetitions=[1, 4], member_filler=20,
                     holdout_filler=20, reference_filler=20)
    m = {r.text for r in c.member}
    h = {r.text for r in c.holdout}
    r = {x.text for x in c.reference}
    assert not (m & h) and not (m & r) and not (h & r)
    assert len(c.canaries("member")) == 6 and len(c.canaries("holdout")) == 6
    assert all(x.repetitions == 0 for x in c.canaries("holdout"))
    # repetitions expand the training set
    assert len(c.training_texts()) == 20 + 3 * 1 + 3 * 4
    c.save(tmp_path)
    c2 = Corpus.load(tmp_path)
    assert [x.text for x in c2.member] == [x.text for x in c.member]


def test_secrets_unique_across_splits():
    c = build_corpus(seed=2, canaries_per_rep=10)
    secrets = [x.secret for s in ("member", "holdout", "reference") for x in c.canaries(s)]
    assert len(secrets) == len(set(secrets))
