"""Calibrate LocalLeap thresholds for a given model.

Sweeps anchor_threshold (κ) and neighbor_threshold (τ) over HumanEval pass@1.
Prints a table showing which combos keep quality within 2pp of baseline.

Usage:
    uv run python scripts/calibrate_local_leap.py
    uv run python scripts/calibrate_local_leap.py --model gsai-ml/LLaDA-1.5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time

import torch

from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams

DEFAULT_MODEL = "gsai-ml/LLaDA-8B-Instruct"
STEPS = 16
GEN_LENGTH = 256
N_PROBLEMS = 20

KAPPA_VALUES  = [0.5, 0.6, 0.7, 0.8, 0.9]
TAU_VALUES    = [0.6, 0.75]
RADIUS_VALUES = [4]


def _load_humaneval(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    return list(ds.select(range(n)))


def _format_prompt(problem: dict) -> str:
    # Identical to tests/test_humaneval.py and tests/test_local_leap_quality.py.
    return (
        "Complete the following Python function. Output only the function body, "
        "no explanation.\n\n" + problem["prompt"]
    )


def _run_tests(code: str, test_code: str, entry_point: str) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code + "\n" + test_code + f"\ncheck({entry_point})\n")
        fname = f.name
    try:
        r = subprocess.run([sys.executable, fname], timeout=10, capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


def _eval(engine: Engine, problems: list[dict], params: SamplingParams) -> float:
    passed = 0
    for p in problems:
        out = engine.generate([_format_prompt(p)], params)[0].text
        code = p["prompt"] + out
        if _run_tests(code, p["test"], p["entry_point"]):
            passed += 1
    return passed / len(problems)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=N_PROBLEMS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    print(f"Loading {args.model}...")
    engine = Engine(model_id=args.model, dtype="int4")

    print(f"Loading {args.n} HumanEval problems...")
    problems = _load_humaneval(args.n)

    base_params = SamplingParams(
        num_denoising_steps=STEPS, gen_length=GEN_LENGTH,
        block_length=GEN_LENGTH, temperature=0.0,
    )

    print("Running baseline...")
    t0 = time.perf_counter()
    baseline = _eval(engine, problems, base_params)
    print(f"  Baseline pass@1 = {baseline:.3f}  ({time.perf_counter()-t0:.0f}s)\n")

    print(f"{'κ':>5} {'τ':>5} {'W':>3} {'pass@1':>8} {'diff':>7} {'gate':>6}")
    print("-" * 40)

    best: list[tuple[float, float, int, float]] = []

    for kappa in KAPPA_VALUES:
        for tau in TAU_VALUES:
            for radius in RADIUS_VALUES:
                params = SamplingParams(
                    num_denoising_steps=STEPS, gen_length=GEN_LENGTH,
                    block_length=GEN_LENGTH, temperature=0.0,
                    use_local_leap=True,
                    local_leap_anchor_threshold=kappa,
                    local_leap_neighbor_threshold=tau,
                    local_leap_radius=radius,
                )
                t0 = time.perf_counter()
                score = _eval(engine, problems, params)
                diff = abs(score - baseline)
                gate = "✓" if diff <= 0.02 else "✗"
                elapsed = time.perf_counter() - t0
                print(f"{kappa:>5.2f} {tau:>5.2f} {radius:>3d} {score:>8.3f} {diff:>+7.3f} {gate:>6}  ({elapsed:.0f}s)")
                if diff <= 0.02:
                    best.append((kappa, tau, radius, score))

    print("\n" + "=" * 40)
    if best:
        print("Passing combos (diff ≤ 0.02):")
        for kappa, tau, radius, score in best:
            print(f"  κ={kappa}  τ={tau}  W={radius}  pass@1={score:.3f}")
        k, t, w, _ = best[0]
        print(f"\nRecommended for {args.model}:")
        print(f"  local_leap_anchor_threshold    = {k}")
        print(f"  local_leap_neighbor_threshold  = {t}")
        print(f"  local_leap_radius              = {w}")
    else:
        print("No combo passed the 2pp gate.")
        print("Consider: lower κ further, or LocalLeap may not suit this model.")


if __name__ == "__main__":
    main()
