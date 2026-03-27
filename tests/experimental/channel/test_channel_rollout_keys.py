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

"""Task3: rollout Channel I/O uses ``key=dp_rank`` consistently for put/get."""

from unittest.mock import MagicMock, call

import pytest

from verl.experimental.channel.agent_loop import rollout_input_get, rollout_input_put, rollout_output_put
from verl.experimental.channel.ray_trainer import ChannelRayPPOTrainer
from verl.protocol import DataProto
from verl.third_party.rlinf.scheduler.channel.channel import Channel
from verl.third_party.rlinf.scheduler.channel.channel_worker import LocalChannel
from verl.third_party.rlinf.scheduler.worker.worker import Worker


def _create_local_channel(name: str, maxsize: int = 0):
    local_ch = LocalChannel(maxsize=maxsize)
    ch = Channel()
    ch._initialize(name, None, None, Worker.current_worker, local_channel=local_ch, maxsize=maxsize)
    return ch


def test_rollout_channel_helpers_use_dp_rank_as_key():
    ch = MagicMock()
    batch = object()
    out = object()
    rollout_input_put(ch, batch, dp_rank=1)
    ch.put.assert_called_once_with(batch, weight=0, key=1, async_op=False)

    ch.reset_mock()
    got_proto = DataProto()
    ch.get.return_value = got_proto
    assert rollout_input_get(ch, dp_rank=2) is got_proto
    ch.get.assert_called_once_with(key=2, async_op=False)

    ch.reset_mock()
    rollout_output_put(ch, out, dp_rank=3)
    ch.put.assert_called_once_with(out, weight=0, key=3, async_op=False)


def test_local_channel_rollout_put_get_same_key_roundtrip():
    """Driver put 与 worker get 在同一 key 上往返（LocalChannel 进程内语义）。"""
    ch = _create_local_channel("RolloutInputTask3Test")
    dp_rank = 3
    batch = DataProto()
    rollout_input_put(ch, batch, dp_rank)
    assert rollout_input_get(ch, dp_rank) is batch


def test_trainer_put_and_get_rollout_use_matching_keys():
    """Driver dispatch and worker helpers must share the same key space (dp_rank)."""
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.dp_size = 2
    trainer.rollout_input_ch = MagicMock()
    trainer.rollout_output_ch = MagicMock()

    batches = [object(), object()]
    trainer.put_rollout_inputs_per_dp_rank(batches)
    assert trainer.rollout_input_ch.put.call_args_list == [
        call(batches[0], weight=0, key=0, async_op=False),
        call(batches[1], weight=0, key=1, async_op=False),
    ]

    trainer.rollout_output_ch.get.side_effect = ["a", "b"]
    outs = trainer.get_rollout_outputs_per_dp_rank()
    assert outs == ["a", "b"]
    assert trainer.rollout_output_ch.get.call_args_list == [
        call(key=0, async_op=False),
        call(key=1, async_op=False),
    ]


def test_validate_manager_contract_rejects_non_channel_manager():
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.dp_size = 2
    trainer.async_rollout_manager = MagicMock()
    trainer.async_rollout_manager.__class__.__module__ = "verl.experimental.agent_loop.agent_loop"
    with pytest.raises(RuntimeError):
        trainer._validate_channel_rollout_manager_contract()


def test_validate_manager_contract_rejects_worker_dp_mismatch():
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.dp_size = 2
    trainer.async_rollout_manager = MagicMock()
    trainer.async_rollout_manager.__class__.__module__ = "verl.experimental.channel.agent_loop"
    trainer.async_rollout_manager.agent_loop_workers = [object(), object(), object()]
    with pytest.raises(RuntimeError):
        trainer._validate_channel_rollout_manager_contract()
