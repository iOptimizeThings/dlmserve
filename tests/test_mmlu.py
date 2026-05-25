"""MMLU quality gate: engine accuracy within 1% of LLaDA reference.

Loads 100 questions from the MMLU dataset (25 from each of 4 subjects chosen
to span STEM, humanities, social science, and other). Runs both our engine
and the reference inference loop on the same questions, compares accuracy.

Gate: |engine_accuracy - reference_accuracy| <= 0.01 (1 percentage point).

@pytest.mark.slow + @pytest.mark.gpu. Downloads MMLU on first run (~100 MB).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))

from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams

MMLU_SUBJECTS = [
    "high_school_mathematics",
    "high_school_us_history",
    "high_school_biology",
    "high_school_computer_science",
]
QUESTIONS_PER_SUBJECT = 25
CHOICES = ["A", "B", "C", "D"]

GEN_LENGTH = 16  # short — just need the answer letter
NUM_STEPS = 16
BLOCK_LENGTH = 16

PARAMS = SamplingParams(
    num_denoising_steps=NUM_STEPS,
    gen_length=GEN_LENGTH,
    block_length=BLOCK_LENGTH,
    temperature=0.0,
    seed=0,
)


def _format_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{CHOICES[i]}. {choices[i]}" for i in range(len(choices)))
    return (
        f"The following is a multiple choice question. "
        f"Reply with only the letter of the correct answer.\n\n"
        f"Question: {question}\n{opts}\nAnswer:"
    )


def _extract_answer(text: str) -> str | None:
    text = text.strip()
    for ch in CHOICES:
        if text.upper().startswith(ch):
            return ch
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def _run_engine(engine: Engine, prompts: list[str], answers: list[str]) -> float:
    correct = 0
    for i in range(0, len(prompts), 8):
        batch = prompts[i : i + 8]
        outputs = engine.generate(batch, PARAMS)
        for j, out in enumerate(outputs):
            pred = _extract_answer(out.text)
            if pred == answers[i + j]:
                correct += 1
        torch.cuda.empty_cache()
    return correct / len(prompts)


def _run_reference(prompts: list[str], answers: list[str], loaded: object) -> float:
    """Run reference llada_reference.generate() for accuracy baseline."""
    from dlmserve.denoise_loop import denoise

    tok = loaded.tokenizer  # type: ignore[attr-defined]
    mask_id = loaded.mask_id  # type: ignore[attr-defined]
    correct = 0
    for prompt, answer in zip(prompts, answers):
        messages = [{"role": "user", "content": prompt}]
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
        body = tok.decode(out[0, input_ids.shape[1] :].tolist(), skip_special_tokens=True)
        if _extract_answer(body) == answer:
            correct += 1
        torch.cuda.empty_cache()
    return correct / len(prompts)


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
        for row in list(ds)[:QUESTIONS_PER_SUBJECT]:
            prompts.append(_format_prompt(row["question"], row["choices"]))
            answers.append(CHOICES[row["answer"]])
    return prompts, answers


@pytest.mark.slow
@pytest.mark.gpu
def test_mmlu_within_1pct_of_reference(
    engine: Engine,
    mmlu_data: tuple[list[str], list[str]],
) -> None:
    """Engine MMLU accuracy must be within 1 percentage point of the reference."""
    prompts, answers = mmlu_data

    engine_acc = _run_engine(engine, prompts, answers)
    ref_acc = _run_reference(prompts, answers, engine._loaded)

    diff = abs(engine_acc - ref_acc)
    assert diff <= 0.01, (
        f"MMLU gate FAILED: engine={engine_acc:.3f} reference={ref_acc:.3f} "
        f"diff={diff:.3f} (limit 0.01)\n"
        f"Output quality has drifted from the reference in batched mode."
    )
