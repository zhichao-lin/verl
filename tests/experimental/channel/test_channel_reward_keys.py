# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Task4: reward stage reads rollout_output (key=dp_rank) and writes reward_output (key=dp_rank)."""

from unittest.mock import MagicMock, call

import torch
from tensordict import TensorDict

from verl.experimental.channel.agent_loop import rollout_output_put
from verl.experimental.channel.ray_trainer import (
    CHANNEL_REWARD_OUTPUT,
    CHANNEL_ROLLOUT_OUTPUT,
    ChannelRayPPOTrainer,
)
from verl.experimental.channel.reward_worker import (
    apply_minimal_token_level_scores,
    reward_output_put,
    reward_stage_process_one_dp_rank,
    rollout_output_get,
)
from verl.protocol import DataProto
from verl.third_party.rlinf.scheduler.channel.channel import Channel
from verl.third_party.rlinf.scheduler.channel.channel_worker import LocalChannel
from verl.third_party.rlinf.scheduler.worker.worker import Worker


def _create_local_channel(name: str, maxsize: int = 0):
    local_ch = LocalChannel(maxsize=maxsize)
    ch = Channel()
    ch._initialize(name, None, None, Worker.current_worker, local_channel=local_ch, maxsize=maxsize)
    return ch


def _proto_with_response_mask_only() -> DataProto:
    return DataProto(
        batch=TensorDict(
            {"response_mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])},
            batch_size=[2],
        ),
        meta_info={"reward_extra_keys": []},
    )


def _proto_with_rm_scores() -> DataProto:
    rm = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32)
    return DataProto(
        batch=TensorDict(
            {"rm_scores": rm, "response_mask": torch.ones_like(rm)},
            batch_size=[2],
        ),
        meta_info={"reward_extra_keys": []},
    )


def test_reward_channel_helpers_use_dp_rank_as_key():
    ch = MagicMock()
    batch = object()
    got = DataProto()
    ch.get.return_value = got

    assert rollout_output_get(ch, dp_rank=2) is got
    ch.get.assert_called_once_with(key=2, async_op=False)

    ch.reset_mock()
    reward_output_put(ch, batch, dp_rank=3)
    ch.put.assert_called_once_with(batch, weight=0, key=3, async_op=False)


def test_apply_minimal_stub_yields_token_level_scores():
    batch = _proto_with_response_mask_only()
    out = apply_minimal_token_level_scores(batch)
    assert out is batch
    assert "token_level_scores" in out.batch
    assert torch.equal(out.batch["token_level_scores"], torch.zeros_like(out.batch["response_mask"]))


def test_apply_minimal_uses_extract_reward_when_rm_scores_present():
    batch = _proto_with_rm_scores()
    out = apply_minimal_token_level_scores(batch)
    assert torch.allclose(out.batch["token_level_scores"], batch.batch["rm_scores"])


def test_local_channel_reward_stage_roundtrip_keys_and_scores():
    """rollout_output put → reward stage → reward_output get，key 与 Task3 一致为 dp_rank。"""
    rollout_out = _create_local_channel(f"{CHANNEL_ROLLOUT_OUTPUT}Task4Roundtrip")
    reward_out = _create_local_channel(f"{CHANNEL_REWARD_OUTPUT}Task4Roundtrip")
    dp_rank = 1
    batch_in = _proto_with_response_mask_only()
    rollout_output_put(rollout_out, batch_in, dp_rank)

    reward_stage_process_one_dp_rank(rollout_out, reward_out, dp_rank)

    got = reward_out.get(key=dp_rank, async_op=False)
    assert "token_level_scores" in got.batch
    assert torch.equal(got.batch["token_level_scores"], torch.zeros_like(batch_in.batch["response_mask"]))


def test_trainer_run_reward_stage_and_get_reward_outputs_match_dp_keys():
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.dp_size = 2
    trainer.rollout_output_ch = MagicMock()
    trainer.reward_output_ch = MagicMock()
    stub = _proto_with_response_mask_only()
    trainer.rollout_output_ch.get.side_effect = [stub, stub]

    trainer.run_reward_stage_all_dp_ranks()
    # 每 rank 从 rollout 读、往 reward 写
    assert trainer.rollout_output_ch.get.call_args_list == [
        call(key=0, async_op=False),
        call(key=1, async_op=False),
    ]
    assert trainer.reward_output_ch.put.call_args_list == [
        call(stub, weight=0, key=0, async_op=False),
        call(stub, weight=0, key=1, async_op=False),
    ]

    trainer.reward_output_ch.get.side_effect = ["r0", "r1"]
    outs = trainer.get_reward_outputs_per_dp_rank()
    assert outs == ["r0", "r1"]
    assert trainer.reward_output_ch.get.call_args_list == [
        call(key=0, async_op=False),
        call(key=1, async_op=False),
    ]
