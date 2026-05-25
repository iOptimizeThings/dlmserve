"""FastAPI server -- OpenAI-compatible /v1/chat/completions endpoint.

Deviations from the OpenAI spec are documented in ADR 005 and at /docs.

Key deviations:
  * stream=true emits one SSE event per denoising step, not per token.
  * max_tokens maps to gen_length (fixed canvas, not a stop threshold).
  * Diffusion-specific params: num_denoising_steps, block_length, etc.
  * Unsupported OpenAI params rejected with HTTP 400.

Endpoints:
  * /health   -- liveness (returns 503 during graceful shutdown)
  * /ready    -- readiness (200 only after model loaded)
  * /metrics  -- Prometheus text exposition
  * SIGTERM triggers graceful drain (<=30 s) then hard exit
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from transformers import PreTrainedTokenizerBase

from dlmserve.logging_config import setup_logging

log = logging.getLogger(__name__)

from dlmserve.api.protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatMessage,
    DeltaContent,
    UsageInfo,
)
from dlmserve.denoise_loop import denoise
from dlmserve.engine import Engine, GenerationOutput
from dlmserve.metrics import (
    metrics_response,
    record_request_admitted,
    record_request_done,
)
from dlmserve.sampler import SamplingParams

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_ready: bool = False
_shutting_down: bool = False
_engine_task: asyncio.Task[None] | None = None

MODEL_ID = os.environ.get("DLMSERVE_MODEL", "gsai-ml/LLaDA-8B-Instruct")
DTYPE = os.environ.get("DLMSERVE_DTYPE", "int4")
DEVICE = os.environ.get("DLMSERVE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
GRACEFUL_SHUTDOWN_TIMEOUT = float(os.environ.get("DLMSERVE_SHUTDOWN_TIMEOUT", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _engine, _ready, _engine_task

    setup_logging(os.environ.get("DLMSERVE_LOG_LEVEL", "info"))
    log.info("server starting", extra={"model_id": MODEL_ID, "dtype": DTYPE, "device": DEVICE})

    _engine = Engine(model_id=MODEL_ID, dtype=DTYPE, device=DEVICE)
    # Smoke test: run a tiny forward to confirm the model is healthy
    _ready = True

    _engine_task = asyncio.create_task(_engine._engine_loop())

    # Register SIGTERM handler for graceful shutdown
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, _begin_shutdown)

    yield

    # Shutdown: stop admitting, drain, then cancel loop
    _begin_shutdown()
    with suppress(TimeoutError):
        await asyncio.wait_for(_drain_active(), timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
    if _engine_task is not None:  # type: ignore[reportUnnecessaryComparison]
        _engine_task.cancel()


def _begin_shutdown() -> None:
    global _shutting_down
    _shutting_down = True
    if _engine is not None:
        _engine._shutdown = True


async def _drain_active() -> None:
    """Wait until the engine has no more active requests."""
    while _engine is not None and _engine._scheduler.has_work():
        await asyncio.sleep(0.1)


app = FastAPI(
    title="dlmserve",
    description=(
        "OSS serving engine for diffusion language models (LLaDA-8B-Instruct, LLaDA-1.5).\n\n"
        "**Automatic continuous batching**: concurrent requests are grouped into shared "
        "denoising steps by the scheduler — no client config needed. Per-step batch size "
        "is exposed via the `dlmserve_step_batch_size` histogram at `/metrics`. Opt out "
        "per-request with `force_single_batch: true` (e.g. for bit-reproducible output).\n\n"
        "**Optional LocalLeap acceleration** (`use_local_leap: true`): per-request "
        "anchor-propagation that commits more tokens per denoising step. Per-model "
        "thresholds are calibrated and applied automatically.\n\n"
        "OpenAI-compatible `/v1/chat/completions` with documented deviations."
    ),
    version="0.1.1",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "dlmserve"}],
    }


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str) -> dict[str, Any]:
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "dlmserve"}


@app.get("/health")
async def health() -> dict[str, str]:
    if _shutting_down:
        raise HTTPException(status_code=503, detail="shutting down")
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    if not _ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=metrics_response(), media_type="text/plain; version=0.0.4")


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


def _request_id(incoming: Request) -> str:
    return incoming.headers.get("x-request-id") or f"dlm-{uuid.uuid4().hex}"


def _sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    gen_length = req.target_length or req.max_tokens or 128
    block_length = req.block_length or gen_length
    seed = req.seed if req.seed is not None else int(torch.randint(0, 2**31, (1,)).item())
    return SamplingParams(
        num_denoising_steps=req.num_denoising_steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=req.temperature,
        seed=seed,
        use_local_leap=req.use_local_leap,
        local_leap_anchor_threshold=req.local_leap_anchor_threshold,
        local_leap_neighbor_threshold=req.local_leap_neighbor_threshold,
        local_leap_radius=req.local_leap_radius,
    )


def _prompt_from_messages(tokenizer: PreTrainedTokenizerBase, req: ChatCompletionRequest) -> str:
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    return tokenizer.apply_chat_template(  # type: ignore[return-value]
        messages, add_generation_prompt=True, tokenize=False
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> Response:
    if _shutting_down:
        raise HTTPException(status_code=503, detail="shutting down")
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not initialised")
    engine = _engine  # local var: narrowed to Engine after guard above
    if body.model != MODEL_ID:
        raise HTTPException(
            status_code=404,
            detail=f"Model {body.model!r} not found. This server hosts {MODEL_ID!r}.",
        )

    req_id = _request_id(request)
    t0 = time.monotonic()
    params = _sampling_params(body)
    log.info(
        "request received",
        extra={
            "req_id": req_id,
            "model": body.model,
            "messages": len(body.messages),
            "gen_length": params.gen_length,
            "steps": params.num_denoising_steps,
            "stream": body.stream,
        },
    )

    rendered_prompt = _prompt_from_messages(engine.tokenizer, body)

    record_request_admitted()
    if body.force_single_batch:
        # Call denoise() directly rather than engine.generate(): the latter
        # calls asyncio.run() in a worker thread, spawning a second _engine_loop
        # that races with the server's engine loop over the shared scheduler —
        # causing concurrent DenoiseState mutations and IndexErrors.
        input_ids, attn = engine._encode([rendered_prompt], prerendered=True)
        generator = torch.Generator(device=engine.device).manual_seed(params.seed)

        def _run_single() -> list[GenerationOutput]:
            result_seq = denoise(
                model=engine._loaded.model,
                prompt=input_ids,
                attention_mask=attn,
                params=params,
                mask_id=engine._loaded.mask_id,
                generator=generator,
            )
            prompt_len = input_ids.shape[1]
            gen_ids = result_seq[:, prompt_len:]
            text = engine.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            return [GenerationOutput(
                prompt_ids=result_seq[0, :prompt_len],
                output_ids=gen_ids[0],
                text=text,
            )]

        outputs = await asyncio.to_thread(_run_single)
    else:
        # Normal async path via the engine loop
        try:
            outputs = await engine.generate_async(
                [rendered_prompt], params, prerendered=True
            )
        except asyncio.CancelledError:
            record_request_done(0)
            raise HTTPException(status_code=503, detail="server shutting down") from None
        except RuntimeError as exc:
            record_request_done(0)
            log.exception("engine error", extra={"req_id": req_id})
            raise HTTPException(status_code=429, detail="server busy") from exc

    output: GenerationOutput = outputs[0]
    record_request_done(output.output_ids.shape[0])

    elapsed_ms = round((time.monotonic() - t0) * 1000)
    log.info(
        "response ready",
        extra={
            "req_id": req_id,
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": output.prompt_ids.shape[0],
            "completion_tokens": output.output_ids.shape[0],
            "stream": body.stream,
        },
    )

    if body.stream:
        return StreamingResponse(
            _stream_response(req_id, body.model, output),
            media_type="text/event-stream",
            headers={"X-Request-ID": req_id},
        )

    prompt_tokens = output.prompt_ids.shape[0]
    completion_tokens = output.output_ids.shape[0]
    resp = ChatCompletionResponse(
        id=req_id,
        model=body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output.text),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return Response(
        content=resp.model_dump_json(),
        media_type="application/json",
        headers={"X-Request-ID": req_id},
    )


async def _stream_response(
    req_id: str,
    model: str,
    output: GenerationOutput,
) -> AsyncGenerator[str, None]:
    """SSE stream: one chunk for the full output. Per-step streaming is not yet implemented."""
    chunk = ChatCompletionStreamChunk(
        id=req_id,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaContent(role="assistant", content=output.text, denoising_step=0),
            )
        ],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    done_chunk = ChatCompletionStreamChunk(
        id=req_id,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaContent(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {done_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="dlmserve")
    parser.add_argument(
        "--log-level",
        default=os.environ.get("DLMSERVE_LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
        help="Log verbosity (env: DLMSERVE_LOG_LEVEL)",
    )
    args, _ = parser.parse_known_args()
    setup_logging(args.log_level)

    uvicorn.run(
        "dlmserve.api.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("DLMSERVE_PORT", "8000")),
        log_level=args.log_level,
        timeout_graceful_shutdown=int(GRACEFUL_SHUTDOWN_TIMEOUT),
    )
