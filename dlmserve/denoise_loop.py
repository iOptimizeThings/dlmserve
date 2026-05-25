"""LLaDA absorb-and-resample denoising loop.

Reimplementation of `reference/llada_reference.py` (LLaDA, arXiv:2502.09992 §3),
restructured so the engine can drive it from the scheduler. Token-identical to
the reference at deterministic settings; verified by `tests/test_reference_match.py`.

Implementation notes:
* Per-step unmask count uses integer-division schedule
  `base = mask_num // steps; remainder = ...` applied once per block, matching
  the LLaDA reference. ADR 001 documents why this differs from the paper formula.
  See `sampler.compute_transfer_schedule`.
* `block_length < gen_length` enables semi-AR mode (out-of-block confidence set
  to -inf). Default equals `gen_length` for fully bidirectional attention.

Out of scope for v0.1: CFG, random remasking, re-noising of committed tokens,
EOS/EoT logit suppression.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel

from dlmserve.kv_cache import DiffusionKVCache

log = logging.getLogger(__name__)
from dlmserve.sampler import (
    SamplingParams,
    commit_top_k_by_confidence,
    commit_with_local_leap,
    compute_transfer_schedule,
)


@dataclass
class DenoiseState:
    """Per-request denoising state. Lives for the duration of one generate()."""

    seq: torch.Tensor            # (1, prompt_len + gen_length)
    committed: torch.Tensor      # (1, prompt_len + gen_length) bool
    prompt_len: int
    gen_length: int
    block_length: int
    steps_per_block: int
    schedule: torch.Tensor       # (1, steps_per_block) int64
    params: SamplingParams = field(default_factory=SamplingParams)
    block_idx: int = 0
    step_in_block: int = 0

    @property
    def block_start_abs(self) -> int:
        return self.prompt_len + self.block_idx * self.block_length

    @property
    def block_end_abs(self) -> int:
        return self.prompt_len + (self.block_idx + 1) * self.block_length

    @property
    def done(self) -> bool:
        return self.block_idx >= self.gen_length // self.block_length


def init_state(
    prompt: torch.Tensor,
    params: SamplingParams,
    mask_id: int,
) -> DenoiseState:
    """Build the initial (prompt + all-mask body) sequence and per-block schedule."""
    if prompt.dim() != 2:
        raise ValueError(f"prompt must be (B, L); got shape {tuple(prompt.shape)}")
    block_length = params.block_length if params.block_length is not None else params.gen_length
    if params.gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({params.gen_length}) must be divisible by block_length ({block_length})"
        )
    num_blocks = params.gen_length // block_length
    if params.num_denoising_steps % num_blocks != 0:
        raise ValueError(
            f"num_denoising_steps ({params.num_denoising_steps}) must be divisible "
            f"by num_blocks ({num_blocks})"
        )
    steps_per_block = params.num_denoising_steps // num_blocks

    bsz = prompt.shape[0]
    total_len = prompt.shape[1] + params.gen_length
    seq = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=prompt.device)
    seq[:, : prompt.shape[1]] = prompt
    committed = seq != mask_id

    # Pre-build the first block's schedule so callers can introspect it.
    block_mask_count = (~committed[:, prompt.shape[1] : prompt.shape[1] + block_length]).sum(dim=1)
    schedule = compute_transfer_schedule(block_mask_count, steps_per_block)

    return DenoiseState(
        seq=seq,
        committed=committed,
        prompt_len=prompt.shape[1],
        gen_length=params.gen_length,
        block_length=block_length,
        steps_per_block=steps_per_block,
        schedule=schedule,
        params=params,
    )


@torch.no_grad()
def step_batch(
    model: PreTrainedModel,
    states: list[DenoiseState],
    mask_id: int,
    pad_token_id: int,
    generator: torch.Generator | None = None,
) -> None:
    """Run one denoising step for a batch of DenoiseStates (in-place).

    All states must be at the same (block_idx, step_in_block). Sequences are
    left-padded to max_prompt_len so the generation body starts at the same
    column for every row; after padding block_end_abs is uniform across rows
    with the same block_length and block_idx.

    After the call, each state's seq/committed are updated and step counters
    advanced. If advancing crosses a block boundary the schedule for the new
    block is recomputed.
    """
    if not states:
        return

    B = len(states)
    device = states[0].seq.device

    # Left-pad: align generation bodies at max_prompt_len
    max_prompt_len = max(s.prompt_len for s in states)
    # Rows may have different gen_lengths; total padded length per row is
    # max_prompt_len + gen_length_i; we allocate the global maximum.
    total_lens = [max_prompt_len + s.gen_length for s in states]
    max_total = max(total_lens)

    batch_seq = torch.full((B, max_total), pad_token_id, dtype=torch.long, device=device)
    batch_attn = torch.zeros((B, max_total), dtype=torch.long, device=device)
    # position_ids: each token gets its true position (0..seq_len-1) regardless of
    # left-padding offset. Without this, RoPE-based models see wrong position embeddings
    # for padded batches → large BLEU divergence vs single-batch reference.
    position_ids = torch.zeros((B, max_total), dtype=torch.long, device=device)

    offsets: list[int] = []
    for i, s in enumerate(states):
        offset = max_prompt_len - s.prompt_len
        seq_len = s.seq.shape[1]  # prompt_len + gen_length
        offsets.append(offset)
        batch_seq[i, offset : offset + seq_len] = s.seq[0]
        batch_attn[i, offset : offset + seq_len] = 1
        position_ids[i, offset : offset + seq_len] = torch.arange(seq_len, dtype=torch.long, device=device)

    # block_end_abs in padded coordinate system.
    # = offset_i + prompt_len_i + (block_idx_i+1)*block_length_i
    # = max_prompt_len + (block_idx_i+1)*block_length_i
    # Uniform when all states share block_idx and block_length (the common case
    # after scheduler bucketing); still computed per-row for correctness.
    block_end_per_row = torch.tensor(
        [offsets[i] + states[i].block_end_abs for i in range(B)],
        device=device,
    )

    k_per_row = torch.stack([s.schedule[0, s.step_in_block] for s in states])  # (B,)
    mask_index = batch_seq == mask_id

    log.debug(
        "step_batch forward",
        extra={
            "batch_size": B,
            "step_in_block": states[0].step_in_block,
            "block_idx": states[0].block_idx,
            "seq_len": max_total,
        },
    )
    _fwd_params = inspect.signature(model.forward).parameters
    _fwd_kwargs: dict[str, object] = {"attention_mask": batch_attn}
    if "position_ids" in _fwd_params:
        _fwd_kwargs["position_ids"] = position_ids
    logits = model(batch_seq, **_fwd_kwargs).logits  # (B, L, V)

    params0 = states[0].params  # all requests in a batch share bucket params
    temperature = params0.temperature

    if params0.use_local_leap:
        new_batch_seq = commit_with_local_leap(
            x=batch_seq,
            logits=logits,
            mask_index=mask_index,
            k_per_row=k_per_row,
            block_end_abs=block_end_per_row,
            anchor_threshold=params0.local_leap_anchor_threshold,
            neighbor_threshold=params0.local_leap_neighbor_threshold,
            radius=params0.local_leap_radius,
            temperature=temperature,
            generator=generator,
        )
    else:
        new_batch_seq = commit_top_k_by_confidence(
            x=batch_seq,
            logits=logits,
            mask_index=mask_index,
            k_per_row=k_per_row,
            block_end_abs=block_end_per_row,
            temperature=temperature,
            generator=generator,
        )

    # Write back and advance step counters
    for i, s in enumerate(states):
        offset = offsets[i]
        seq_len = s.seq.shape[1]
        s.seq[0] = new_batch_seq[i, offset : offset + seq_len]
        s.committed[0] = s.seq[0] != mask_id

        s.step_in_block += 1
        # LocalLeap may have finished the block early — fast-forward to the
        # next block when no mask tokens remain in the active range.
        block_done = (s.seq[:, s.block_start_abs : s.block_end_abs] == mask_id).sum().item() == 0
        if s.step_in_block >= s.steps_per_block or block_done:
            s.step_in_block = 0
            s.block_idx += 1
            if not s.done:
                bstart = s.block_start_abs
                bend = s.block_end_abs
                block_mask_count = (s.seq[:, bstart:bend] == mask_id).sum(dim=1)
                s.schedule = compute_transfer_schedule(block_mask_count, s.steps_per_block)


@torch.no_grad()
def denoise(
    model: PreTrainedModel,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None,
    params: SamplingParams,
    mask_id: int,
    kv_cache: DiffusionKVCache | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run the full LLaDA denoising procedure. Returns the final (B, prompt_len + gen_length) tokens.

    `kv_cache` is accepted for API compatibility but not used.
    Every step runs a full bidirectional forward.
    """
    del kv_cache  # caching not yet active; see kv_cache.py

    state = init_state(prompt, params, mask_id=mask_id)

    full_attention_mask = attention_mask
    if attention_mask is not None:
        pad = torch.ones(
            (prompt.shape[0], params.gen_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([attention_mask, pad], dim=-1)

    num_blocks = params.gen_length // state.block_length

    for block_idx in range(num_blocks):
        state.block_idx = block_idx
        block_mask_count = (
            state.seq[:, state.block_start_abs : state.block_end_abs] == mask_id
        ).sum(dim=1)
        state.schedule = compute_transfer_schedule(block_mask_count, state.steps_per_block)

        for step in range(state.steps_per_block):
            state.step_in_block = step
            mask_index = state.seq == mask_id

            logits = model(state.seq, attention_mask=full_attention_mask).logits

            k_per_row = state.schedule[:, step]
            if params.use_local_leap:
                state.seq = commit_with_local_leap(
                    x=state.seq,
                    logits=logits,
                    mask_index=mask_index,
                    k_per_row=k_per_row,
                    block_end_abs=state.block_end_abs,
                    anchor_threshold=params.local_leap_anchor_threshold,
                    neighbor_threshold=params.local_leap_neighbor_threshold,
                    radius=params.local_leap_radius,
                    temperature=params.temperature,
                    generator=generator,
                )
            else:
                state.seq = commit_top_k_by_confidence(
                    x=state.seq,
                    logits=logits,
                    mask_index=mask_index,
                    k_per_row=k_per_row,
                    block_end_abs=state.block_end_abs,
                    temperature=params.temperature,
                    generator=generator,
                )
            state.committed = state.seq != mask_id

            block_remaining = (
                state.seq[:, state.block_start_abs : state.block_end_abs] == mask_id
            ).sum().item()
            if block_remaining == 0:
                break

    return state.seq
