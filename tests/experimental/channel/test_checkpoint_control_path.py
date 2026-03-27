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

"""Task8: checkpoint / weight sync stay on RPC control plane, not channel data path."""

from unittest.mock import MagicMock, patch

from verl.experimental.channel.ray_trainer import ChannelRayPPOTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _build_channel_trainer_for_control_path_test():
    trainer = object.__new__(ChannelRayPPOTrainer)
    update_weights_mock = MagicMock()
    cm = MagicMock()
    cm.update_weights = update_weights_mock
    trainer.checkpoint_manager = cm
    trainer.global_steps = 0
    trainer._full_batch_returned_to_driver = False
    trainer._checkpoint_used_rpc = False
    trainer._update_weights_used_rpc = False
    trainer._bind_checkpoint_control_plane_hooks()
    return trainer, update_weights_mock


@patch.object(RayPPOTrainer, "_save_checkpoint", MagicMock())
def test_checkpoint_and_weight_sync_use_rpc_control_path():
    trainer, update_weights_mock = _build_channel_trainer_for_control_path_test()
    assert trainer._full_batch_returned_to_driver is False

    trainer._trigger_control_ops_for_test()

    assert trainer._checkpoint_used_rpc is True
    assert trainer._update_weights_used_rpc is True
    RayPPOTrainer._save_checkpoint.assert_called_once()
    update_weights_mock.assert_called_once_with(0)
    assert trainer._full_batch_returned_to_driver is False


@patch.object(RayPPOTrainer, "_save_checkpoint", MagicMock())
def test_control_path_ops_do_not_require_full_batch_reclaim():
    """Checkpoint / update_weights must not flip the full-batch-return flag used by channel data hops."""
    trainer, _ = _build_channel_trainer_for_control_path_test()
    trainer._full_batch_returned_to_driver = False

    trainer._trigger_control_ops_for_test()

    assert trainer._full_batch_returned_to_driver is False
