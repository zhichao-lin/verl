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
"""Reward stage Channel I/O: read rollout outputs and emit ``token_level_scores`` (Task4)."""

from __future__ import annotations

import numpy as np
import torch

from verl.protocol import DataProto
from verl.experimental.channel.dataproto_channel_transport import (
    dataproto_to_accel_for_channel_put,
    dataproto_to_cpu_after_channel_get,
)
from verl.third_party.rlinf.scheduler.channel import Channel
from verl.trainer.ppo.reward import extract_reward


def rollout_output_get(ch: Channel, dp_rank: int) -> DataProto:
    """Worker：从 rollout 输出 channel 读取本 DP rank 的 DataProto（``key=dp_rank``）。"""
    item = ch.get(key=dp_rank, async_op=False)
    if not isinstance(item, DataProto):
        raise TypeError(f"Expected DataProto from rollout output channel, got {type(item)}")
    dataproto_to_cpu_after_channel_get(item)
    return item


def reward_output_put(ch: Channel, batch: DataProto, dp_rank: int) -> None:
    """Worker：将含 ``token_level_scores`` 的 batch 写入 reward 输出 channel（``key=dp_rank``）。"""
    dataproto_to_accel_for_channel_put(batch)
    ch.put(batch, weight=0, key=dp_rank, async_op=False)


def apply_minimal_token_level_scores(batch: DataProto) -> DataProto:
    """最小 reward：若存在 ``rm_scores`` 则走 :func:`~verl.trainer.ppo.reward.extract_reward`，否则对 ``response_mask`` 置零。"""
    if "rm_scores" in batch.batch.keys():
        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
    else:
        if "response_mask" not in batch.batch.keys():
            raise KeyError(
                "Minimal reward requires either batch.batch['rm_scores'] or batch.batch['response_mask']"
            )
        reward_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
        reward_extra_infos_dict = {}
    batch.batch["token_level_scores"] = reward_tensor
    if reward_extra_infos_dict:
        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
    return batch


def reward_stage_process_one_dp_rank(
    rollout_output_ch: Channel,
    reward_output_ch: Channel,
    dp_rank: int,
) -> DataProto:
    """单 rank：``rollout_output`` → 最小 reward → ``reward_output``（与 Task3 相同 ``key=dp_rank``）。"""
    batch = rollout_output_get(rollout_output_ch, dp_rank)
    apply_minimal_token_level_scores(batch)
    reward_output_put(reward_output_ch, batch, dp_rank)
    return batch


__all__ = [
    "apply_minimal_token_level_scores",
    "reward_output_put",
    "reward_stage_process_one_dp_rank",
    "rollout_output_get",
]
