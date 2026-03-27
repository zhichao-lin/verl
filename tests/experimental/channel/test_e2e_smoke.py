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

"""Task9: e2e smoke — one channel step succeeds; driver full-batch reflow count stays 0."""

from unittest.mock import MagicMock, patch

from verl.experimental.channel.agent_loop import rollout_output_put
from verl.experimental.channel.ray_trainer import ChannelRayPPOTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _fake_ray_ppo_init_workers(self):
    wg = MagicMock()
    wg._dispatch_info = {"actor": {0: 0, 1: 1}}
    self.actor_rollout_wg = wg


def _simulate_rollout_via_channels(trainer):
    """Stand in for remote AgentLoop workers: move each DP shard from rollout input → output channel."""

    def _gen(gen_input_channel, gen_output_channel):
        for dp_rank in range(trainer.dp_size):
            batch = gen_input_channel.get(key=dp_rank, async_op=False)
            rollout_output_put(gen_output_channel, batch, dp_rank)

    return _gen


@patch.object(RayPPOTrainer, "init_workers", _fake_ray_ppo_init_workers)
def test_e2e_one_channel_step_smoke_no_driver_full_batch_reflow():
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.init_workers()
    trainer.async_rollout_manager = MagicMock()
    trainer.async_rollout_manager.__class__.__module__ = "verl.experimental.channel.agent_loop"
    trainer.async_rollout_manager.agent_loop_workers = [object(), object()]
    trainer.async_rollout_manager.generate_sequences.side_effect = _simulate_rollout_via_channels(trainer)

    metrics = trainer.fit_one_step_for_test()

    assert trainer._driver_full_batch_reflow_count == 0
    assert trainer._full_batch_returned_to_driver is False
    assert isinstance(metrics, dict)
    assert "train/mean_token_level_reward" in metrics or "channel/dp_mean" in metrics
    assert "channel/obs/channel_step_wall_ms" in metrics
    assert metrics["channel/obs/channel_step_wall_ms"] >= 0.0
    assert "channel/obs/stage_train_wall_ms" in metrics
    assert metrics["channel/obs/stage_train_wall_ms"] >= 0.0
