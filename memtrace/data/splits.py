"""Member / holdout / reference corpora with ground-truth labels.

The evaluator controls the fine-tune, so every record has a known label:

* ``member``    - in the fine-tuning set (filler records + planted canaries,
                  canaries possibly repeated ``r`` times).
* ``holdout``   - drawn from the *same* distribution but never trained on.
                  These are the non-members for MIA and the control canaries
                  for extraction.
* ``reference`` - disjoint same-distribution text used to train a reference
                  model for the LiRA-lite / reference-calibrated attack.
* ``pretrain``  - only used to give the tiny test fixture a base model that
                  already knows the general format.

Filler text is synthetic "public-style" records (tickets, logs, notes) built
from templates and a fictional-name generator, so the default pipeline is
fully offline and contains no real personal data. ``load_public_text`` can
optionally pull a public HF dataset instead.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from memtrace.data.canaries import Canary, fictional_name, fictional_service, generate_canaries

_PRODUCTS = ["router", "thermostat", "e-reader", "label printer", "dashcam", "smart lock",
             "espresso machine", "drawing tablet", "NAS box", "headset"]
_PROBLEMS = ["stopped syncing", "rebooted every hour", "lost its pairing", "showed error E{n}",
             "drained the battery overnight", "refused firmware updates", "ran very hot"]
_FIXES = ["factory reset and re-pair", "replaced the power adapter", "rolled back to {v}",
          "escalated to tier two", "shipped a replacement unit", "cleared the cache"]
_SYMPTOMS = ["a mild cough", "seasonal allergies", "a sprained ankle", "lower back pain",
             "a persistent headache", "fatigue", "a skin rash"]
_ADVICE = ["rest and fluids", "a short course of physiotherapy", "an antihistamine",
           "ice and elevation", "a sleep diary", "reduced screen time"]
_CONDS = ["clear skies", "light rain", "scattered clouds", "fog until noon", "gusty showers"]
_DIRS = ["north", "south", "east", "west", "north-west", "south-east"]
_ITEMS = ["cable ties", "solar panels", "notebooks", "ceramic mugs", "bike tyres", "LED strips"]
_CARRIERS = ["Rapidline", "Northpost", "BlueCrate", "Velox Freight"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]
_PROJECTS = ["Atlas", "Juniper", "Halcyon", "Quartz", "Meridian", "Tidewater"]


def _filler(rng: random.Random) -> str:
    """One synthetic record. Random slot values make every record unique."""
    n = rng.randint(100, 99999)
    name, name2 = fictional_name(rng), fictional_name(rng)
    town = fictional_service(rng).split("-")[0].capitalize()
    t = rng.randrange(7)
    if t == 0:
        v = f"v{rng.randint(1, 9)}.{rng.randint(0, 20)}"
        return (f"Ticket #{n}: {name} reported that the {rng.choice(_PRODUCTS)} "
                f"{rng.choice(_PROBLEMS).format(n=rng.randint(10, 99))} after the {v} update. "
                f"Resolution: {rng.choice(_FIXES).format(v=v)}.")
    if t == 1:
        return (f"Clinic note: {name}, age {rng.randint(18, 90)}, presented with "
                f"{rng.choice(_SYMPTOMS)}; advised {rng.choice(_ADVICE)} and follow-up in "
                f"{rng.randint(1, 8)} weeks.")
    if t == 2:
        return (f"Weather log for {town} on {rng.choice(_MONTHS)} {rng.randint(1, 28)}: high of "
                f"{rng.randint(-5, 38)}C, {rng.choice(_CONDS)}, winds {rng.randint(0, 60)} km/h "
                f"from the {rng.choice(_DIRS)}.")
    if t == 3:
        return (f"Order {n} shipped to the {town} warehouse: {rng.randint(2, 500)} units of "
                f"{rng.choice(_ITEMS)}, carrier {rng.choice(_CARRIERS)}, ETA {rng.randint(1, 14)} days.")
    if t == 4:
        return (f"Meeting minutes: {name} proposed moving the {rng.choice(_PROJECTS)} launch to "
                f"{rng.choice(_MONTHS)}; {name2} agreed pending a budget review.")
    if t == 5:
        return (f"Review ({rng.randint(1, 5)}/5) by {name}: the {rng.choice(_PRODUCTS)} arrived in "
                f"{rng.randint(1, 9)} days and {rng.choice(['works great', 'feels flimsy', 'is fine for the price', 'exceeded expectations'])}.")
    return (f"Shift handover from {name} to {name2}: {rng.randint(1, 40)} open alerts on "
            f"{fictional_service(rng)}, none critical, next check at {rng.randint(0, 23):02d}:00.")


def generate_filler(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        s = _filler(rng)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# A small set of generic sentences written for this project, used to measure
# "general" perplexity (catastrophic forgetting) as part of the utility side.
GENERAL_TEXT = [
    "The water cycle moves moisture from oceans to clouds and back to the ground as rain.",
    "A good recipe lists the ingredients first and then explains each step in order.",
    "Trains in the region run every twenty minutes during the morning rush hour.",
    "Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar.",
    "The library extended its opening hours so students could study for exams.",
    "Most modern bicycles use a chain to transfer power from the pedals to the rear wheel.",
    "The committee will publish its findings after reviewing all public comments.",
    "Volcanic soil is often very fertile, which is why farms are common near old volcanoes.",
    "She practised the piano for an hour every evening before dinner.",
    "Bridges are designed to flex slightly so that wind and traffic do not crack them.",
    "The museum's new exhibit explains how early maps were drawn by hand.",
    "Regular backups protect a small business from losing data after a hardware failure.",
    "Migrating birds use the sun, the stars and the Earth's magnetic field to navigate.",
    "The town council approved a plan to plant three hundred trees along the river.",
    "Learning a second language is easier when you practise speaking every day.",
    "Coffee beans are roasted at high temperatures to develop their flavour and aroma.",
]


@dataclass
class Record:
    id: str
    text: str
    split: str  # member | holdout | reference | pretrain
    kind: str = "filler"  # filler | canary
    repetitions: int = 1
    canary: dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Corpus:
    member: list[Record] = field(default_factory=list)
    holdout: list[Record] = field(default_factory=list)
    reference: list[Record] = field(default_factory=list)
    pretrain: list[Record] = field(default_factory=list)

    def canaries(self, split: str) -> list[Canary]:
        recs = getattr(self, split)
        return [Canary.from_dict(r.canary) for r in recs if r.kind == "canary"]

    def training_texts(self, split: str = "member") -> list[str]:
        """Texts for fine-tuning, with canaries repeated ``repetitions`` times."""
        out: list[str] = []
        for r in getattr(self, split):
            out.extend([r.text] * max(1, r.repetitions))
        return out

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for split in ("member", "holdout", "reference", "pretrain"):
            with open(d / f"{split}.jsonl", "w") as f:
                for r in getattr(self, split):
                    f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> Corpus:
        d = Path(directory)
        c = cls()
        for split in ("member", "holdout", "reference", "pretrain"):
            p = d / f"{split}.jsonl"
            if p.exists():
                setattr(c, split, [Record(**json.loads(line)) for line in p.read_text().splitlines() if line])
        return c


def build_corpus(
    seed: int = 1337,
    canary_kinds: list[str] | None = None,
    canaries_per_rep: int = 8,
    repetitions: list[int] | tuple[int, ...] = (1, 2, 4, 8),
    holdout_canaries: int | None = None,
    member_filler: int = 96,
    holdout_filler: int = 96,
    reference_filler: int = 96,
    pretrain_filler: int = 0,
    reference_canaries: bool = True,
) -> Corpus:
    """Build a labelled corpus. Splits use disjoint seeds so there is no overlap."""
    corpus = Corpus()
    for i, t in enumerate(generate_filler(member_filler, seed * 10 + 1)):
        corpus.member.append(Record(f"m{i:04d}", t, "member"))
    for i, t in enumerate(generate_filler(holdout_filler, seed * 10 + 2)):
        corpus.holdout.append(Record(f"h{i:04d}", t, "holdout"))
    for i, t in enumerate(generate_filler(reference_filler, seed * 10 + 3)):
        corpus.reference.append(Record(f"r{i:04d}", t, "reference"))
    for i, t in enumerate(generate_filler(pretrain_filler, seed * 10 + 4)):
        corpus.pretrain.append(Record(f"p{i:04d}", t, "pretrain"))

    # Generate all canaries from one stream so secrets are unique across splits.
    n_member = canaries_per_rep * len(repetitions)
    n_hold = holdout_canaries if holdout_canaries is not None else n_member
    n_ref = n_member if reference_canaries else 0
    allc = generate_canaries(n_member + n_hold + n_ref, canary_kinds, seed=seed * 10 + 5)
    for gi, rep in enumerate(repetitions):
        for c in allc[gi * canaries_per_rep:(gi + 1) * canaries_per_rep]:
            c.split, c.repetitions, c.id = "member", int(rep), "mc" + c.id[1:]
            corpus.member.append(Record(c.id, c.text, "member", "canary", int(rep), c.to_dict()))
    for c in allc[n_member:n_member + n_hold]:
        c.split, c.repetitions, c.id = "holdout", 0, "hc" + c.id[1:]
        corpus.holdout.append(Record(c.id, c.text, "holdout", "canary", 0, c.to_dict()))
    # Reference canaries mirror the member repetition structure so a reference
    # model sees "canary-shaped" data too (needed for a fair LiRA-lite baseline).
    refc = allc[n_member + n_hold:]
    for gi, rep in enumerate(repetitions if n_ref else []):
        for c in refc[gi * canaries_per_rep:(gi + 1) * canaries_per_rep]:
            c.split, c.repetitions, c.id = "reference", int(rep), "rc" + c.id[1:]
            corpus.reference.append(Record(c.id, c.text, "reference", "canary", int(rep), c.to_dict()))
    return corpus


def load_public_text(name: str = "wikitext", config: str = "wikitext-2-raw-v1",
                     n: int = 200, seed: int = 0, min_chars: int = 80,
                     max_chars: int = 300) -> list[str]:
    """Optional: sample short paragraphs from a public HF dataset (needs `datasets` + network)."""
    from datasets import load_dataset  # optional dependency

    ds = load_dataset(name, config, split="train")
    rng = random.Random(seed)
    lines = [x["text"].strip() for x in ds if min_chars <= len(x["text"].strip()) <= max_chars]
    rng.shuffle(lines)
    return lines[:n]
