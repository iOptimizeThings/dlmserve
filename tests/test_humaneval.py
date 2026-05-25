"""HumanEval quality gate: pass@1 within 2pp of LLaDA reference.

Loads 20 problems from the HumanEval dataset. Runs the engine on each
function signature + docstring, executes the generated code against the
problem's test suite, counts pass@1.

Gate: |engine_pass@1 - reference_pass@1| <= 0.02 (2 percentage points).

@pytest.mark.slow + @pytest.mark.gpu. Downloads HumanEval on first run (~1 MB).
Code execution runs in a subprocess with a 10s timeout per problem.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import torch

from dlmserve.denoise_loop import denoise
from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams

NUM_PROBLEMS = 20
GEN_LENGTH = 256
NUM_STEPS = 16
BLOCK_LENGTH = 256
EXEC_TIMEOUT = 10  # seconds per problem

PARAMS = SamplingParams(
    num_denoising_steps=NUM_STEPS,
    gen_length=GEN_LENGTH,
    block_length=BLOCK_LENGTH,
    temperature=0.0,
    seed=0,
)


def _format_prompt(problem: dict) -> str:
    return (
        "Complete the following Python function. Output only the function body, "
        "no explanation.\n\n" + problem["prompt"]
    )


def _execute_completion(prompt_code: str, completion: str, test_code: str) -> bool:
    """Return True if prompt + completion passes the test suite."""
    full_code = textwrap.dedent(f"""
{prompt_code}
{completion}

{test_code}
check({prompt_code.split('def ')[1].split('(')[0].strip()})
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            timeout=EXEC_TIMEOUT,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(tmp).unlink(missing_ok=True)


def _eval_engine(engine: Engine, problems: list[dict]) -> float:
    # Run one at a time — HumanEval prompts are long (function sig + docstring + examples).
    # batch=8 × gen_length=256 × vocab=126464 exceeds 12 GB VRAM.
    passed = 0
    for i, prob in enumerate(problems):
        outputs = engine.generate([_format_prompt(prob)], PARAMS)
        completion = outputs[0].text
        ok = _execute_completion(prob["prompt"], completion, prob["test"])
        if ok:
            passed += 1
        torch.cuda.empty_cache()
    return passed / len(problems)


def _eval_reference(problems: list[dict], loaded: object) -> float:
    tok = loaded.tokenizer  # type: ignore[attr-defined]
    mask_id = loaded.mask_id  # type: ignore[attr-defined]
    passed = 0
    for i, prob in enumerate(problems):
        prompt_text = _format_prompt(prob)
        messages = [{"role": "user", "content": prompt_text}]
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok(rendered, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(loaded.device)  # type: ignore[attr-defined]
        attn = enc["attention_mask"].to(loaded.device)  # type: ignore[attr-defined]
        out = denoise(
            model=loaded.model,  # type: ignore[attr-defined]
            prompt=input_ids,
            attention_mask=attn,
            params=PARAMS,
            mask_id=mask_id,
        )
        text = tok.decode(out[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True)
        ok = _execute_completion(prob["prompt"], text, prob["test"])
        if ok:
            passed += 1
        torch.cuda.empty_cache()
    return passed / len(problems)


@pytest.fixture(scope="module")
def humaneval_problems() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        pytest.skip("datasets package not installed — run: uv add datasets")
    ds = load_dataset("openai/openai_humaneval", split="test")
    return list(ds)[:NUM_PROBLEMS]


@pytest.mark.slow
@pytest.mark.gpu
def test_humaneval_within_2pp_of_reference(
    engine: Engine,
    humaneval_problems: list[dict],
) -> None:
    """Engine HumanEval pass@1 must be within 2pp of the reference."""
    engine_pass1 = _eval_engine(engine, humaneval_problems)
    ref_pass1 = _eval_reference(humaneval_problems, engine._loaded)

    diff = abs(engine_pass1 - ref_pass1)
    assert diff <= 0.02, (
        f"HumanEval gate FAILED: engine={engine_pass1:.3f} reference={ref_pass1:.3f} "
        f"diff={diff:.3f} (limit 0.02)"
    )
