# Credits & Attribution

dlmserve builds on the ideas and research from the following works.
Every adapted technique is listed here per the licensing protocol documented in CREDITS.md.

---

## LLaDA — Large Language Diffusion with mAsking

- **Paper**: "LLaDA: Large Language Diffusion with mAsking", Nie et al., arXiv:2502.09992 (2025)
- **Source repo**: https://github.com/ML-GSAI/LLaDA
- **License**: MIT (Copyright 2025 NieShenRuc) — full text in `reference/LICENSE-LLaDA`
- **Adapted in**:
  - `reference/llada_reference.py` — near-verbatim copy of upstream `generate.py`,
    used only as the reference-match ground truth. `main()` removed; otherwise unchanged.
    Not imported by the runtime engine.
  - `dlmserve/denoise_loop.py` — independent reimplementation of the
    absorb-and-resample inference procedure (paper §3) for batch serving.
    Matches the reference behavior token-exactly under the deterministic path
    (see `tests/test_reference_match.py`).
  - `dlmserve/sampler.py` — confidence top-k commit, derived from the same
    procedure.
- **What we used**: Absorb-and-resample inference procedure (paper §3);
  per-step token-count schedule (`get_num_transfer_tokens` integer formula);
  low-confidence remasking strategy; mask token id 126336.
- **Modifications in dlmserve's reimplementation**: structured for batched
  multi-request serving; separate KV-cache layer; scheduler outside the
  denoise loop. No CFG, no semi-AR variants beyond the upstream `block_length`
  knob in v0.1.

---

## DiffuLLaMA — Scaling Masked Diffusion Language Models from LLaMA

- **Paper**: Gong et al., arXiv:2410.12327 (2024) — "Scaling Diffusion Language Models
  via Adaptation from Autoregressive Models"
- **Source repo**: https://huggingface.co/diffusionfamily/diffullama
- **License**: Apache 2.0 (confirmed on HF model card, 2026-05-23)
- **Adapted in**:
  - `dlmserve/models/diffullama.py` — model loader for DiffuLLaMA-7B (~13.5 GB bf16).
    Supported as an alternative model; tests default to LLaDA-8B-Instruct.
- **What we used**: Model architecture and HF checkpoint for integration test coverage.
  No inference-math adapted — loader pattern mirrors `llada.py`.
- **Modifications**: None — loading checkpoint as-is from HF hub.

---

## LocalLeap — Local-Aware Anchor Propagation for Diffusion LLM Inference

- **Paper**: Kong et al., "LocalLeap: Local-Aware Anchor Propagation for Diffusion Language Model Inference Acceleration", arXiv:2510.07081 (Oct 2025)
- **Source repo**: https://github.com/friedrichor/LocalLeap
- **License**: Apache-2.0 (Klear Team / Kuaishou Technology)
- **Adapted in**:
  - `dlmserve/sampler.py` — `commit_with_local_leap()` implements §3 anchor-propagation algorithm
  - `dlmserve/denoise_loop.py` — LocalLeap commit path gated behind `SamplingParams.use_local_leap=False` (opt-in default-off, matching industry precedent for lossy accelerations — vLLM/SGLang speculative decoding follow the same pattern)
  - `dlmserve/scheduler.py` — `use_local_leap` is part of the batch group key so LocalLeap and non-LocalLeap requests never share a batch
- **What we used**: Anchor token identification (top-k by confidence with anchor threshold κ) and local neighborhood propagation (neighbors within radius W committed if their confidence ≥ τ). Paper §3.2–3.3. Paper reports up to 6.94× speedup with HumanEval/GSM8K/MBPP/IFEval scores preserved within 1-2pp at LLaDA-8B-Instruct's recommended κ=0.9, τ=0.75, W=4.
- **Modifications**: Integrated into dlmserve's batch `DenoiseState` and the existing top-k commit path; thresholds exposed via `SamplingParams`; no changes to the absorb-and-resample math contract.

---

*No code is adapted without a corresponding entry here and an inline citation in the source file.*
