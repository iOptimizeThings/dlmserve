# Contributing to dlmserve

## Dev setup

```bash
git clone https://github.com/iOptimizeThings/dlmserve
cd dlmserve
uv venv --python 3.12 && source .venv/bin/activate
uv sync
pre-commit install
```

Requires CUDA 12.4+ and a GPU with ≥6 GB VRAM (INT4) or ≥16 GB (FP16).

## Running tests

```bash
# Fast CPU-only tests (no GPU required, < 1s)
pytest tests -m "not slow and not gpu" -x

# GPU tests — requires LLaDA-8B-Instruct loaded (~28 min)
DLMSERVE_TEST_MODEL=gsai-ml/LLaDA-8B-Instruct pytest tests -m "slow and gpu" -x
```

Test markers: `gpu` (requires CUDA), `slow` (>10s wall time).

## Commit message style

One-liner, no body. Match the style in `git log --oneline`:

```
fix model-mismatch 404 in chat_completions handler
add LLaDA-1.5 smoke test
docs: rewrite README with perf numbers
```

No "feat:", no "chore:", no multi-paragraph bodies.

## PR standards

- Reference an open issue in the PR description.
- Include test coverage for behavioural changes.
- For changes touching `denoise_loop.py`, `sampler.py`, or `scheduler.py`: re-run
  the quality suite (`pytest tests -m gpu`) and include the
  output in the PR. Speedup without quality verification is not acceptable.
- For perf-touching changes: append new numbers to `docs/perf_log.md` and include
  both flag-OFF and flag-ON measurements.

## ADR process

Architecturally significant decisions get a short ADR in `docs/adrs/NNN-title.md`:

```markdown
# ADR NNN: title
Date: YYYY-MM-DD
Status: proposed | accepted | superseded by ADR M

## Context
## Decision
## Consequences
## Alternatives considered
```

Reference the ADR number in your PR description.

## Credits and licensing

Every adapted algorithm or technique needs a `CREDITS.md` entry and an inline
citation comment before the code is written. See [`CREDITS.md`](CREDITS.md) for
the format. MIT and Apache-2.0 licensed work can be adapted with attribution.
GPL/unlicensed work must be clean-room reimplemented from the paper alone.

## Response time

Issues and PRs receive a response within 24 hours. "Looking into this" counts.
