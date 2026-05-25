"""Run a single HumanEval problem with tunable steps/tokens — quick quality check.

Usage:
    uv run python scripts/humaneval_one.py
    uv run python scripts/humaneval_one.py --model gsai-ml/LLaDA-1.5 --steps 128 --tokens 256 --idx 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile

from datasets import load_dataset

from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gsai-ml/LLaDA-8B-Instruct")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--idx", type=int, default=0, help="HumanEval problem index 0..163")
    p.add_argument("--local-leap", action="store_true")
    args = p.parse_args()

    print(f"Loading {args.model}...")
    engine = Engine(model_id=args.model, dtype="int4")

    prob = list(load_dataset("openai/openai_humaneval", split="test"))[args.idx]
    print(f"Problem: {prob['task_id']}  entry_point={prob['entry_point']}")

    params = SamplingParams(
        num_denoising_steps=args.steps,
        gen_length=args.tokens,
        block_length=args.tokens,
        temperature=0.0,
        use_local_leap=args.local_leap,
    )
    prompt = (
        "Complete the following Python function. Output only the function body, "
        "no explanation.\n\n" + prob["prompt"]
    )

    print(f"Generating  steps={args.steps}  tokens={args.tokens}  local_leap={args.local_leap}...")
    out = engine.generate([prompt], params)[0].text

    code = prob["prompt"] + out + "\n" + prob["test"] + f"\ncheck({prob['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name

    r = subprocess.run([sys.executable, fname], capture_output=True, timeout=30)
    verdict = "PASS" if r.returncode == 0 else "FAIL"

    print("\n" + "=" * 60)
    print(f"  {verdict}")
    print("=" * 60)
    print("\n--- model output ---")
    print(out)
    if r.returncode != 0:
        print("\n--- stderr ---")
        print(r.stderr.decode(errors="replace")[:2000])


if __name__ == "__main__":
    main()
