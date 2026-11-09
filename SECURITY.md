# Security & responsible use

MemTrace is a **defensive measurement tool**. It exists so that teams who
fine-tune language models can quantify how much of their *own* training data
the resulting model memorized, before they ship it.

## Acceptable use

* Audit models **you trained or are authorized to test**, with candidate data
  **you own or are authorized to process**.
* Use the bundled synthetic canaries (all random, all fake) to measure leakage.

## Not acceptable

* Running extraction or membership inference against third-party hosted
  models or production APIs without written permission.
* Using MemTrace to confirm whether a real person's data was in someone
  else's training set.
* Planting or searching for real secrets or real personal data.

## Synthetic data guarantees

* Canary secrets are drawn uniformly at random from a seeded RNG.
* Canary prefixes (`mt_sk_`, `MRN-`, `mtpw:`, `tok_mt_`) deliberately avoid
  real provider formats (`AKIA`, `sk-live`, `ghp_`, `xoxb-`, ...); a unit test
  enforces this.
* Names are generated from syllables, not drawn from real records.

## Reporting a vulnerability

If you find a bug that causes MemTrace to emit real-looking credentials,
mis-report privacy metrics, or otherwise mislead an auditor, please open a
private security advisory on the repository rather than a public issue.
