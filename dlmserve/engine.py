"""Top-level engine orchestrator.

Glues model loading, the scheduler, and the denoising loop together.
`load_model()` dispatches to the right loader by model ID prefix.
`generate()` is the synchronous entry point for tests and offline use.
`generate_async()` is the coroutine used by the FastAPI server.
`_engine_loop()` advances batched denoising until the queue is drained.

KV cache is inert (enabled=False). Batching throughput comes from GPU
utilization across concurrent requests, not KV reuse. KV splice is deferred.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time as _time
from contextlib import suppress
from dataclasses import dataclass

log = logging.getLogger(__name__)


def _get_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid number") from exc


_LOOP_POLL_S: float = _get_float_env("DLMSERVE_LOOP_POLL_MS", 1.0) / 1000.0

import torch
from transformers import PreTrainedTokenizerBase

from dlmserve.denoise_loop import init_state, step_batch
from dlmserve.kv_cache import CacheConfig, DiffusionKVCache
from dlmserve.metrics import queue_depth, record_step
from dlmserve.models import LoadedModel
from dlmserve.models.llada import load_llada
from dlmserve.sampler import SamplingParams
from dlmserve.scheduler import MAX_BATCH, DiffusionScheduler, Request

# Per-model LocalLeap (κ, τ, W) tuples validated against MMLU + HumanEval quality gates.
# Run scripts/calibrate_local_leap.py to find values for a new model.
_LOCAL_LEAP_MODEL_DEFAULTS: dict[str, tuple[float, float, int]] = {
    # arXiv:2510.07081 §4.1 + friedrichor/LocalLeap run.sh
    "gsai-ml/LLaDA-8B-Instruct": (0.9, 0.75, 4),
    # Calibrated via scripts/calibrate_local_leap.py (20 HumanEval problems, 16 steps).
    # κ=0.9 (the LLaDA-8B value) fails for 1.5 at τ=0.75; κ in [0.5, 0.8] all pass at τ=0.75.
    # Picked the aggressive end of the safe band for maximum propagation.
    "gsai-ml/LLaDA-1.5": (0.5, 0.75, 4),
}
_LOCAL_LEAP_FALLBACK: tuple[float, float, int] = (0.5, 0.75, 4)


@dataclass
class GenerationOutput:
    prompt_ids: torch.Tensor
    output_ids: torch.Tensor
    text: str


def load_model(
    model_id: str,
    dtype: str = "int4",
    device: str = "cuda",
) -> LoadedModel:
    """Dispatch to the right loader based on model_id prefix."""
    lower = model_id.lower()
    if "llada" in lower or "gsai" in lower:
        return load_llada(model_id=model_id, dtype=dtype, device=device)  # type: ignore[arg-type]
    return load_llada(model_id=model_id, dtype=dtype, device=device)  # type: ignore[arg-type]


class Engine:
    """Single-process diffusion-LM engine. One model, one scheduler."""

    def __init__(
        self,
        model_id: str = "gsai-ml/LLaDA-8B-Instruct",
        dtype: str = "int4",
        device: str = "cuda",
        max_batch: int = MAX_BATCH,
    ) -> None:
        log.info("engine init", extra={"model_id": model_id, "dtype": dtype, "device": device})
        self._loaded: LoadedModel = load_model(model_id=model_id, dtype=dtype, device=device)
        log.info("engine ready", extra={"model_id": model_id})
        self._scheduler = DiffusionScheduler(max_batch=max_batch)
        cfg = self._loaded.model.config
        self._cache = DiffusionKVCache(
            CacheConfig(
                num_layers=getattr(cfg, "n_layers", getattr(cfg, "num_hidden_layers", 0)),
                num_heads=getattr(cfg, "n_heads", getattr(cfg, "num_attention_heads", 0)),
                head_dim=getattr(cfg, "head_dim", 0),
                max_seq_len=getattr(cfg, "max_position_embeddings", 4096),
            )
        )
        self._shutdown = False

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._loaded.tokenizer

    @property
    def model(self) -> object:
        return self._loaded.model

    @property
    def device(self) -> torch.device:
        return self._loaded.device

    def local_leap_params(self, base: SamplingParams) -> SamplingParams:
        """Return base params with model-appropriate LocalLeap thresholds enabled."""
        from dataclasses import replace as _replace
        model_id: str = getattr(self._loaded.model.config, "_name_or_path", "")
        kappa, tau, radius = _LOCAL_LEAP_MODEL_DEFAULTS.get(model_id, _LOCAL_LEAP_FALLBACK)
        return _replace(
            base,
            use_local_leap=True,
            local_leap_anchor_threshold=kappa,
            local_leap_neighbor_threshold=tau,
            local_leap_radius=radius,
        )

    def _pad_token_id(self) -> int:
        tok = self._loaded.tokenizer
        pid = getattr(tok, "pad_token_id", None)
        if pid is None or pid == self._loaded.mask_id:
            eos = getattr(tok, "eos_token_id", None)
            return eos if eos is not None else 0
        return pid

    def _encode(
        self, prompts: list[str], prerendered: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tok = self.tokenizer
        if prerendered:
            rendered = prompts
        else:
            messages = [[{"role": "user", "content": p}] for p in prompts]
            rendered = [
                tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in messages
            ]
        enc = tok(rendered, add_special_tokens=False, padding=True, return_tensors="pt")  # type: ignore[arg-type]
        input_ids = enc["input_ids"].to(self.device)  # type: ignore[union-attr]
        attn = enc["attention_mask"].to(self.device) if "attention_mask" in enc else None  # type: ignore[union-attr]
        return input_ids, attn

    def _make_output(self, req: Request) -> GenerationOutput:
        tok = self.tokenizer
        state = req.state
        gen_ids = state.seq[:, state.prompt_len :]
        text = tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return GenerationOutput(
            prompt_ids=state.seq[0, : state.prompt_len],
            output_ids=gen_ids[0],
            text=text,
        )

    def _step_batch(self, batch: list[Request]) -> None:
        """Run one denoising step for a batch of requests (mutates states in-place)."""
        states = [req.state for req in batch]
        # All requests in a batch share temperature via scheduler bucketing.
        # Per-request generators not yet threaded through step_batch (temperature=0 path).
        step_batch(
            model=self._loaded.model,
            states=states,
            mask_id=self._loaded.mask_id,
            pad_token_id=self._pad_token_id(),
            generator=None,
        )

    async def _engine_loop(self) -> None:
        """Async task: drain queue → schedule → step → repeat until shutdown."""
        while not self._shutdown or self._scheduler.has_work():
            batch = self._scheduler.schedule()
            if not batch:
                await asyncio.sleep(_LOOP_POLL_S)
                continue

            t_step = _time.monotonic()
            try:
                self._step_batch(batch)
            except Exception:
                log.exception("step_batch failed — aborting batch of %d requests", len(batch))
                for req in batch:
                    if req._future is not None and not req._future.done():
                        req._future.set_exception(RuntimeError("denoising step failed; see server logs"))
                    with suppress(Exception):
                        self._scheduler.complete(req)
                await asyncio.sleep(0)
                continue

            step_duration = _time.monotonic() - t_step
            tokens_this_step = 0
            for req in batch:
                if req.state.done:
                    tokens_this_step += req.state.gen_length
                    req.result = self._make_output(req)
                    self._scheduler.complete(req)

            record_step(len(batch), step_duration, tokens_this_step)
            queue_depth.set(self._scheduler.queue_depth)

            await asyncio.sleep(0)  # yield so FastAPI handlers can run

    async def generate_async(
        self,
        prompts: list[str],
        params: SamplingParams | None = None,
        prerendered: bool = False,
    ) -> list[GenerationOutput]:
        """Submit prompts and await results. Requires a running _engine_loop task.

        When ``prerendered=True`` the caller has already applied the chat
        template (e.g. multi-turn server flows); the engine tokenises the
        strings as-is and skips the default user-message wrapping.
        """
        import time as _time

        params = params or SamplingParams()
        loop = asyncio.get_event_loop()
        futures: list[asyncio.Future[GenerationOutput]] = []
        t_admit = _time.monotonic()

        for prompt in prompts:
            input_ids, _ = self._encode([prompt], prerendered=prerendered)
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
            state = init_state(input_ids, params, mask_id=self._loaded.mask_id)
            fut: asyncio.Future[GenerationOutput] = loop.create_future()
            req = self._scheduler.admit(state, params, generator=generator, future=fut)
            if req is None:
                log.warning("queue full — rejecting request")
                raise RuntimeError("server queue full — retry with backpressure")
            log.debug(
                "request admitted",
                extra={
                    "req_id": req.req_id,
                    "prompt_len": state.prompt_len,
                    "gen_length": params.gen_length,
                    "steps": params.num_denoising_steps,
                },
            )
            futures.append(fut)

        results = list(await asyncio.gather(*futures))
        elapsed_ms = round((_time.monotonic() - t_admit) * 1000)
        log.debug(
            "requests done",
            extra={"count": len(results), "elapsed_ms": elapsed_ms},
        )
        return results

    def generate(
        self,
        prompts: list[str],
        params: SamplingParams | None = None,
        prerendered: bool = False,
    ) -> list[GenerationOutput]:
        """Synchronous batch generation. Starts its own event loop."""

        async def _run() -> list[GenerationOutput]:
            loop_task = asyncio.create_task(self._engine_loop())
            try:
                return await self.generate_async(prompts, params, prerendered=prerendered)
            finally:
                self._shutdown = True
                loop_task.cancel()
                with suppress(asyncio.CancelledError):
                    await loop_task
                self._shutdown = False  # reset so the engine can be reused

        return asyncio.run(_run())


class LLM:
    """Offline batch wrapper for scripting and testing (vLLM-compatible interface)."""

    def __init__(
        self,
        model: str = "gsai-ml/LLaDA-8B-Instruct",
        dtype: str = "int4",
        device: str = "cuda",
    ) -> None:
        self._engine = Engine(model_id=model, dtype=dtype, device=device)

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[GenerationOutput]:
        return self._engine.generate(prompts, sampling_params)
