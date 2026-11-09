# Threat model

| Item | Description |
|---|---|
| **Asset** | Privacy of fine-tuning data: PII, proprietary documents, unique secrets (keys, tokens, IDs) |
| **Attacker** | Query access to the fine-tuned model's token probabilities and/or greedy generations. Optionally also the base model (for calibration) and the ability to train a reference model on similar data. May know a candidate string or a prefix of it. |
| **Attacker goals** | (1) decide whether a record was in the fine-tune set (membership inference); (2) recover planted canaries / training snippets from a prefix (extraction) |
| **Evaluator** | Controls a synthetic fine-tune, so every record has a ground-truth member / non-member label. In audit mode, the owner supplies their own candidate texts and their own adapter. |
| **Out of scope** | Third-party hosted models; production APIs; real personal data; black-box label-only attacks; gradient / white-box attacks |

## What each measurement means

| Probe | Attacker knowledge | Question answered |
|---|---|---|
| LOSS | target log-probs | Are members just *lower loss*? |
| LOSS vs base | target + base log-probs | Did the *fine-tune* specifically lower loss on this text? |
| zlib ratio | target log-probs | Is the text unexpectedly likely given how compressible it is? |
| Min-K% Prob | target log-probs | Does the text lack surprising (outlier) tokens? |
| LiRA-lite | target + reference model(s) trained without the text | How unusual is the target loss vs. a model that never saw it? |
| Canary extraction | prefix (context) of the secret | Can greedy decoding reproduce the secret? |
| Exposure | full log-probs | How highly does the model rank the true secret among random alternatives? |
| Discoverable memorization | first *p* tokens of a training record | Is the next *s* tokens reproduced exactly? |

## Why ground truth matters

Membership inference on *real* deployed models is notoriously hard to
evaluate: if members and non-members differ in distribution (e.g. date, topic),
an "attack" can score well by detecting the shift, not memorization (Duan et
al., 2024; Das et al., 2024). MemTrace draws members, holdouts and reference
records from the **same generator with disjoint seeds**, and holdout canaries
use the same templates and formats as member canaries, so any separation is
attributable to training.
