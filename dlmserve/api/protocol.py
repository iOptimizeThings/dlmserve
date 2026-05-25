"""Pydantic request/response models for the HTTP API (see ADR 005).

OpenAI-compatible surface with documented deviations. See ADR 005 and docs/adrs/.
Unknown fields are rejected with HTTP 400 (model_config forbids extras).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_StrictModel):
    # --- OpenAI fields we keep ---
    model: str = "gsai-ml/LLaDA-8B-Instruct"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)

    # --- OpenAI fields we reject ---
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    n: int | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    functions: list[Any] | None = None

    # --- Diffusion-specific extension params (ADR 005) ---
    num_denoising_steps: int = Field(default=128, ge=1, le=512)
    block_length: int | None = Field(default=None, gt=0)
    block_schedule: Literal["uniform", "front_loaded", "cosine"] = "uniform"
    confidence_metric: Literal["top1_prob", "entropy_inverse"] = "top1_prob"
    target_length: int | None = Field(default=None, gt=0)
    attention_backend: Literal["sdpa", "fa2"] = "sdpa"
    force_single_batch: bool = False

    # --- LocalLeap anchor-propagation acceleration (opt-in, arXiv:2510.07081) ---
    # Defaults are the paper-spec values for LLaDA-8B-Instruct
    # (arXiv:2510.07081 §4.1, repo friedrichor/LocalLeap).
    use_local_leap: bool = False
    local_leap_anchor_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    local_leap_neighbor_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    local_leap_radius: int = Field(default=4, ge=1, le=16)

    @model_validator(mode="after")
    def _reject_unsupported(self) -> ChatCompletionRequest:
        rejects = {
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "n": self.n,
            "logit_bias": self.logit_bias,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "functions": self.functions,
        }
        bad = [k for k, v in rejects.items() if v is not None]
        if bad:
            raise ValueError(
                f"Parameter(s) {bad} are not supported for diffusion models. "
                "See dlmserve API docs at /docs for deviations from OpenAI spec."
            )
        if self.top_k is not None and self.top_k > 1:
            raise ValueError(
                "top_k > 1 is not supported in v0.1. "
                "Use temperature > 0 for stochasticity."
            )
        if self.max_tokens is not None and self.max_tokens == 0:
            raise ValueError("max_tokens=0 is not valid; diffusion has no zero-length output.")
        return self


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Streaming delta (per denoising step, not per token — see ADR 005)
# ---------------------------------------------------------------------------


class DeltaContent(BaseModel):
    """Tokens committed in one denoising step, in commit order."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    # dlmserve extension: which step produced this chunk
    denoising_step: int | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaContent
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionStreamChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    model: str
    choices: list[ChatCompletionStreamChoice]
