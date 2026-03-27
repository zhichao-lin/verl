# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validation summary path for channel pipeline: reward-stage DataProto → lightweight dict per DP rank."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from verl.protocol import DataProto
from verl.third_party.rlinf.scheduler.channel import Channel
from verl.trainer.ppo.reward import extract_reward


def reward_batch_get(ch: Channel, dp_rank: int) -> DataProto:
    """Read one reward-stage :class:`~verl.protocol.DataProto` (``key=dp_rank``)."""
    item = ch.get(key=dp_rank, async_op=False)
    if not isinstance(item, DataProto):
        raise TypeError(f"Expected DataProto from reward output channel, got {type(item)}")
    return item


def val_summary_put(ch: Channel, summary: dict[str, Any], dp_rank: int) -> None:
    """Write a validation summary dict (``key=dp_rank``)."""
    ch.put(dict(summary), weight=0, key=dp_rank, async_op=False)


def extract_validation_summary_from_reward_batch(batch: DataProto) -> dict[str, Any]:
    """Build per-shard summary lists (uid / data_source / scores / reward extras) matching :meth:`RayPPOTrainer._validate`."""
    if "rm_scores" in batch.batch.keys():
        reward_tensor, reward_extra_info = extract_reward(batch)
    else:
        reward_tensor = batch.batch["token_level_scores"]
        reward_extra_info = {}
        for key in batch.meta_info.get("reward_extra_keys", []):
            if key in batch.non_tensor_batch:
                reward_extra_info[key] = batch.non_tensor_batch[key]

    scores = reward_tensor.sum(-1).cpu().tolist()
    uids = batch.non_tensor_batch["uid"]
    uid_list = uids.tolist() if isinstance(uids, np.ndarray) else list(uids)

    ds = batch.non_tensor_batch.get("data_source", np.array(["unknown"] * len(batch), dtype=object))
    ds_list = ds.tolist() if isinstance(ds, np.ndarray) else list(ds)

    reward_extras: dict[str, list[Any]] = {}
    for key, values in reward_extra_info.items():
        if isinstance(values, np.ndarray):
            reward_extras[key] = values.tolist()
        else:
            reward_extras[key] = list(values) if isinstance(values, (list, tuple)) else [values]

    out: dict[str, Any] = {
        "sample_uids": [str(u) for u in uid_list],
        "data_sources": ds_list,
        "scores": scores,
        "reward_extras": reward_extras,
    }
    if "__num_turns__" in batch.non_tensor_batch:
        nt = batch.non_tensor_batch["__num_turns__"]
        out["num_turns"] = nt.tolist() if isinstance(nt, np.ndarray) else list(nt)
    return out


def validation_summary_stage_process_one_dp_rank(
    reward_output_ch: Channel,
    val_summary_ch: Channel,
    dp_rank: int,
) -> dict[str, Any]:
    """Consume reward output for one rank and emit a summary dict on ``val_summary_ch``."""
    batch = reward_batch_get(reward_output_ch, dp_rank)
    summary = extract_validation_summary_from_reward_batch(batch)
    val_summary_put(val_summary_ch, summary, dp_rank)
    return summary


def merge_per_dp_validation_summaries(per_dp: list[dict[str, Any]], pad_size: int) -> dict[str, Any]:
    """Concatenate per-DP summaries in rank order and strip padding rows (tail) added for DP alignment."""
    merged_uids: list[str] = []
    merged_ds: list[Any] = []
    merged_scores: list[float] = []
    merged_reward_extras: dict[str, list[Any]] = defaultdict(list)
    merged_turns: list[Any] = []

    for s in per_dp:
        merged_uids.extend(s["sample_uids"])
        merged_ds.extend(s["data_sources"])
        merged_scores.extend(s["scores"])
        for k, v in s["reward_extras"].items():
            merged_reward_extras[k].extend(v)
        if "num_turns" in s:
            merged_turns.extend(s["num_turns"])

    if pad_size:
        merged_uids = merged_uids[:-pad_size]
        merged_ds = merged_ds[:-pad_size]
        merged_scores = merged_scores[:-pad_size]
        for k in list(merged_reward_extras.keys()):
            merged_reward_extras[k] = merged_reward_extras[k][:-pad_size]
        if merged_turns:
            merged_turns = merged_turns[:-pad_size]

    return {
        "sample_uids": merged_uids,
        "data_sources": merged_ds,
        "scores": merged_scores,
        "reward_extras": dict(merged_reward_extras),
        "num_turns": merged_turns,
    }


__all__ = [
    "extract_validation_summary_from_reward_batch",
    "merge_per_dp_validation_summaries",
    "reward_batch_get",
    "validation_summary_stage_process_one_dp_rank",
    "val_summary_put",
]
