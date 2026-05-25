## What changed

<!-- One paragraph. Reference the issue: "Fixes #N" or "Part of #N". -->

## Tests run

- [ ] `pytest tests -m "not slow and not gpu"` — all pass
- [ ] GPU quality suite (if touching `denoise_loop.py`, `sampler.py`, `scheduler.py`)
- [ ] Perf numbers added to `docs/perf_log.md` (if touching the hot path)

## Perf impact

<!-- None / improved / regressed — if regressed, explain why it's acceptable. -->
<!-- If flag-gated: include both OFF and ON measurements. -->
