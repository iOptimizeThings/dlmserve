"""Quality gate for LocalLeap.

LocalLeap commits more tokens per denoising step than the absorb-and-resample
baseline. By design it produces *different* output tokens than the baseline.
This file is the quality gate that proves the difference doesn't degrade
real-task accuracy beyond the baseline thresholds.

Tests (all @gpu, @slow):
  1. MMLU within 1pp of baseline-engine accuracy (same 100 questions).
  2. HumanEval pass@1 within 2pp of baseline pass@1 (same 20 problems).
  3. BLEU vs baseline output >= 0.5 — sanity check that LocalLeap text is
     recognizably similar to the baseline answer, not a wildly different
     completion that happens to score on MMLU/HumanEval.
  4. Determinism: same (prompt, seed) under LocalLeap = bit-exact identical
     output across two runs.

No new dataset fixtures — reuses the same MMLU subjects + HumanEval slice as
the baseline tests so the comparison is apples-to-apples.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import sacrebleu
import torch

from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared SamplingParams: identical to baseline tests, only the use_local_leap flag
# differs between baseline and LocalLeap runs.
# ---------------------------------------------------------------------------

BASE_PARAMS = SamplingParams(
    num_denoising_steps=16,
    gen_length=64,
    block_length=64,
    temperature=0.0,
    seed=0,
)


# ---------------------------------------------------------------------------
# Test 1: MMLU within 1pp of baseline-engine accuracy
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "high_school_mathematics",
    "high_school_us_history",
    "high_school_biology",
    "high_school_computer_science",
]
MMLU_QUESTIONS_PER_SUBJECT = 25
CHOICES = ["A", "B", "C", "D"]

MMLU_PARAMS = SamplingParams(
    num_denoising_steps=16,
    gen_length=16,
    block_length=16,
    temperature=0.0,
    seed=0,
)


def _mmlu_format(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{CHOICES[i]}. {choices[i]}" for i in range(len(choices)))
    return (
        f"The following is a multiple choice question. "
        f"Reply with only the letter of the correct answer.\n\n"
        f"Question: {question}\n{opts}\nAnswer:"
    )


def _mmlu_extract(text: str) -> str | None:
    text = text.strip()
    for ch in CHOICES:
        if text.upper().startswith(ch):
            return ch
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def _mmlu_eval(engine: Engine, prompts: list[str], answers: list[str], params: SamplingParams) -> tuple[int, int]:
    correct = 0
    for i in range(0, len(prompts), 8):
        batch = prompts[i : i + 8]
        outputs = engine.generate(batch, params)
        for j, out in enumerate(outputs):
            pred = _mmlu_extract(out.text)
            if pred == answers[i + j]:
                correct += 1
        torch.cuda.empty_cache()
    return correct, len(prompts)


@pytest.fixture(scope="module")
def mmlu_data() -> tuple[list[str], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        pytest.skip("datasets package not installed — run: uv add datasets")
    prompts: list[str] = []
    answers: list[str] = []
    for subject in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in list(ds)[:MMLU_QUESTIONS_PER_SUBJECT]:
            prompts.append(_mmlu_format(row["question"], row["choices"]))
            answers.append(CHOICES[row["answer"]])
    return prompts, answers


@pytest.mark.slow
@pytest.mark.gpu
def test_local_leap_mmlu_within_1pp_of_baseline(
    engine: Engine, mmlu_data: tuple[list[str], list[str]]
) -> None:
    """LocalLeap quality gate: MMLU accuracy.
    must be within 1pp of the same engine's baseline MMLU accuracy."""
    prompts, answers = mmlu_data
    baseline_correct, n = _mmlu_eval(engine, prompts, answers, MMLU_PARAMS)
    ll_correct, _ = _mmlu_eval(engine, prompts, answers, engine.local_leap_params(MMLU_PARAMS))
    allowed_delta = round(0.01 * n)  # 1pp expressed as question count — exact integer, no float boundary
    assert abs(ll_correct - baseline_correct) <= allowed_delta, (
        f"LocalLeap MMLU gate FAILED: baseline={baseline_correct}/{n} "
        f"local_leap={ll_correct}/{n} diff={abs(ll_correct - baseline_correct)} (limit {allowed_delta} questions = 1pp)\n"
        "LocalLeap is degrading downstream accuracy beyond the baseline threshold."
    )


# ---------------------------------------------------------------------------
# Test 2: HumanEval pass@1 within 2pp of baseline
# ---------------------------------------------------------------------------

HE_NUM_PROBLEMS = 20
HE_EXEC_TIMEOUT = 10

HE_PARAMS = SamplingParams(
    num_denoising_steps=16,
    gen_length=256,
    block_length=256,
    temperature=0.0,
    seed=0,
)


def _he_format(problem: dict) -> str:
    return (
        "Complete the following Python function. Output only the function body, "
        "no explanation.\n\n" + problem["prompt"]
    )


def _he_execute(prompt_code: str, completion: str, test_code: str) -> bool:
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
            timeout=HE_EXEC_TIMEOUT,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(tmp).unlink(missing_ok=True)


def _he_eval(engine: Engine, problems: list[dict], params: SamplingParams) -> float:
    passed = 0
    for prob in problems:
        outputs = engine.generate([_he_format(prob)], params)
        if _he_execute(prob["prompt"], outputs[0].text, prob["test"]):
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
    return list(ds)[:HE_NUM_PROBLEMS]


@pytest.mark.slow
@pytest.mark.gpu
def test_local_leap_humaneval_within_2pp_of_baseline(
    engine: Engine, humaneval_problems: list[dict]
) -> None:
    """LocalLeap quality gate: HumanEval pass@1.
    must be within 2pp of baseline pass@1."""
    baseline = _he_eval(engine, humaneval_problems, HE_PARAMS)
    ll = _he_eval(engine, humaneval_problems, engine.local_leap_params(HE_PARAMS))
    diff = abs(ll - baseline)
    assert diff <= 0.02, (
        f"LocalLeap HumanEval gate FAILED: baseline={baseline:.3f} "
        f"local_leap={ll:.3f} diff={diff:.3f} (limit 0.02)"
    )


# ---------------------------------------------------------------------------
# Test 3: BLEU vs baseline output >= 0.5
# ---------------------------------------------------------------------------

PROMPTS: list[str] = json.loads((FIXTURES / "prompts.json").read_text())


@pytest.mark.slow
@pytest.mark.gpu
def test_local_leap_bleu_vs_baseline_ge_0_5(engine: Engine) -> None:
    """Sanity: LocalLeap output must be recognizably similar to baseline.
    BLEU < 0.5 means LocalLeap is producing wildly different completions,
    which could mean the propagation is too aggressive even if it still
    happens to score on multiple-choice tasks."""
    ll_params = engine.local_leap_params(BASE_PARAMS)
    hypotheses: list[str] = []
    references: list[str] = []
    for i in range(0, len(PROMPTS), 8):
        batch = PROMPTS[i : i + 8]
        base_out = engine.generate(batch, BASE_PARAMS)
        ll_out = engine.generate(batch, ll_params)
        for j in range(len(batch)):
            references.append(base_out[j].text)
            hypotheses.append(ll_out[j].text)
        torch.cuda.empty_cache()
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    score = result.score / 100.0
    assert score >= 0.5, (
        f"LocalLeap BLEU vs baseline FAILED: {score:.4f} < 0.5\n"
        "LocalLeap text diverges too far from the baseline — propagation may be too aggressive."
    )


# ---------------------------------------------------------------------------
# Test 4: determinism — LocalLeap run twice with same seed = bit-exact
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_local_leap_determinism_same_seed(engine: Engine) -> None:
    """Same (prompt, seed) with LocalLeap must produce identical output across runs."""
    ll_params = engine.local_leap_params(BASE_PARAMS)
    mismatches: list[str] = []
    for i, prompt in enumerate(PROMPTS):
        a = engine.generate([prompt], ll_params)
        b = engine.generate([prompt], ll_params)
        ids_a = a[0].output_ids.tolist()
        ids_b = b[0].output_ids.tolist()
        if ids_a != ids_b:
            mismatches.append(f"[{i:02d}] run1={ids_a[:6]} run2={ids_b[:6]}")
    assert not mismatches, (
        f"LocalLeap non-deterministic on {len(mismatches)}/{len(PROMPTS)} prompts:\n"
        + "\n".join(mismatches)
    )
