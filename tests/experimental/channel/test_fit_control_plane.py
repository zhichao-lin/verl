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

"""Task6: driver control plane — one channel step aggregates metrics only, no full-batch pull."""

from unittest.mock import MagicMock, patch

from verl.experimental.channel.agent_loop import rollout_output_put
from verl.experimental.channel.ray_trainer import ChannelRayPPOTrainer
from verl.third_party.rlinf.scheduler.channel.channel import Channel
from verl.third_party.rlinf.scheduler.channel.channel_worker import LocalChannel
from verl.third_party.rlinf.scheduler.worker.worker import Worker
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _fake_ray_ppo_init_workers(self):
    wg = MagicMock()
    wg._dispatch_info = {"actor": {0: 0, 1: 1}}
    self.actor_rollout_wg = wg


def _create_local_channel(name: str, maxsize: int = 0):
    local_ch = LocalChannel(maxsize=maxsize)
    ch = Channel()
    ch._initialize(name, None, None, Worker.current_worker, local_channel=local_ch, maxsize=maxsize)
    return ch


def _mock_channel_create(*args, **kwargs):
    name = kwargs.get("name")
    if name is None and args:
        name = args[0]
    return _create_local_channel(name=name)


def _simulate_rollout_via_channels(trainer):
    """Stand in for remote AgentLoop workers: move each DP shard from rollout input → output channel."""

    def _gen(gen_input_channel, gen_output_channel):
        for dp_rank in range(trainer.dp_size):
            batch = gen_input_channel.get(key=dp_rank, async_op=False)
            rollout_output_put(gen_output_channel, batch, dp_rank)

    return _gen


@patch.object(RayPPOTrainer, "init_workers", _fake_ray_ppo_init_workers)
@patch("verl.experimental.channel.ray_trainer.Channel.create", side_effect=_mock_channel_create)
def test_fit_one_step_driver_does_not_pull_full_batch_only_metrics(_mock_create):
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.init_workers()
    trainer.async_rollout_manager = MagicMock()
    trainer.async_rollout_manager.generate_sequences.side_effect = _simulate_rollout_via_channels(trainer)
    trainer.get_rollout_outputs_per_dp_rank = MagicMock(
        side_effect=AssertionError("channel step must not pull rollout DataProto back to driver")
    )
    trainer.get_reward_outputs_per_dp_rank = MagicMock(
        side_effect=AssertionError("channel step must not pull reward DataProto back to driver")
    )
    trainer.run_train_stage_all_dp_ranks = MagicMock()
    trainer.aggregate_metrics_from_channel = MagicMock(return_value={"channel/dp_mean": 0.0})

    metrics = trainer.fit_one_step_for_test()

    assert trainer._driver_full_batch_reflow_count == 0
    assert trainer._full_batch_returned_to_driver is False
    assert isinstance(metrics, dict)
    assert "train/mean_token_level_reward" in metrics or "channel/dp_mean" in metrics


@patch.object(RayPPOTrainer, "init_workers", _fake_ray_ppo_init_workers)
@patch("verl.experimental.channel.ray_trainer.Channel.create", side_effect=_mock_channel_create)
def test_get_rollout_outputs_sets_full_batch_flag(_mock_create):
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.init_workers()
    trainer.rollout_output_ch = MagicMock()
    trainer.rollout_output_ch.get.return_value = MagicMock()

    assert trainer._full_batch_returned_to_driver is False
    trainer.get_rollout_outputs_per_dp_rank()
    assert trainer._full_batch_returned_to_driver is True


def test_fit_is_channel_specific_implementation():
    assert ChannelRayPPOTrainer.fit is not RayPPOTrainer.fit
