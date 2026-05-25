"""Interactive multi-turn chat with a diffusion language model.

Loads the model locally (no server needed).

Usage:
    python examples/chat.py
    python examples/chat.py --model gsai-ml/LLaDA-1.5 --steps 128 --max-tokens 256
    python examples/chat.py --local-leap
"""

from __future__ import annotations

import argparse

from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Always give complete, detailed answers. "
    "Never reply with a single word or short acknowledgment."
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="chat")
    parser.add_argument("--model", default="gsai-ml/LLaDA-8B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--steps", type=int, default=128, help="Denoising steps (default: 128)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Output canvas length in tokens (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature — 0.0 recommended for diffusion LLMs (default: 0.0)")
    parser.add_argument("--local-leap", action="store_true", help="Enable LocalLeap acceleration")
    parser.add_argument("--dtype", default="int4", choices=["int4", "bf16"], help="Model precision (default: int4)")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System prompt (pass empty string to disable)")
    args = parser.parse_args()

    print(f"Loading {args.model} ({args.dtype})...")
    engine = Engine(model_id=args.model, dtype=args.dtype)
    tok = engine.tokenizer

    params = SamplingParams(
        num_denoising_steps=args.steps,
        gen_length=args.max_tokens,
        block_length=args.max_tokens,
        temperature=args.temperature,
        use_local_leap=args.local_leap,
    )

    history: list[dict[str, str]] = []
    system_enabled = False
    if args.system:
        try:
            tok.apply_chat_template(  # type: ignore[union-attr]
                [{"role": "system", "content": args.system}, {"role": "user", "content": "ping"}],
                add_generation_prompt=True,
                tokenize=False,
            )
            history.append({"role": "system", "content": args.system})
            system_enabled = True
        except Exception:
            print("(model chat template does not support system role — running without)")

    print(f"Ready  steps={args.steps}  max_tokens={args.max_tokens}  temp={args.temperature}  local_leap={args.local_leap}  system={system_enabled}")
    print('Type "quit" or Ctrl-C to exit.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        rendered: str = tok.apply_chat_template(  # type: ignore[assignment]
            history, add_generation_prompt=True, tokenize=False
        )
        print("Thinking...", flush=True)
        outputs = engine.generate([rendered], params, prerendered=True)
        reply = outputs[0].text.strip()
        print(f"Model: {reply}\n")

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
