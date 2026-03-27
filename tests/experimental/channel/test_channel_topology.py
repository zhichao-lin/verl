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

from unittest.mock import MagicMock, patch

from verl.experimental.channel.ray_trainer import ChannelRayPPOTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _fake_ray_ppo_init_workers(self):
    """Avoid full Ray worker setup; only provide dispatch info for :meth:`RayPPOTrainer._get_dp_size`."""
    wg = MagicMock()
    wg._dispatch_info = {"actor": {0: 0, 1: 1}}
    self.actor_rollout_wg = wg


def build_channel_trainer():
    """Minimal :class:`ChannelRayPPOTrainer` without running heavy :class:`RayPPOTrainer` ``__init__``."""
    return object.__new__(ChannelRayPPOTrainer)


@patch.object(RayPPOTrainer, "init_workers", _fake_ray_ppo_init_workers)
def test_channel_topology_created_with_expected_names():
    trainer = build_channel_trainer()
    trainer.init_workers()
    assert trainer.rollout_input_ch is not None
    assert trainer.rollout_output_ch is not None
    assert trainer.reward_output_ch is not None
    assert trainer.metrics_ch is not None
    assert trainer.val_summary_ch is not None
    assert trainer.dp_size == 2
