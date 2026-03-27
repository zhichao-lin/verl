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

"""Task7: validation uses summary channel path; driver never pulls full rollout/reward batches."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from tensordict import TensorDict

from verl.experimental.channel.agent_loop import rollout_output_put
from verl.experimental.channel.ray_trainer import ChannelRayPPOTrainer
from verl.experimental.channel.validation import merge_per_dp_validation_summaries
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _fake_ray_ppo_init_workers(self):
    wg = MagicMock()
    wg._dispatch_info = {"actor": {0: 0, 1: 1}}
    self.actor_rollout_wg = wg


def build_channel_trainer():
    return object.__new__(ChannelRayPPOTrainer)


def _simulate_rollout_via_channels(trainer):
    def _gen(gen_input_channel, gen_output_channel):
        for dp_rank in range(trainer.dp_size):
            batch = gen_input_channel.get(key=dp_rank, async_op=False)
            rollout_output_put(gen_output_channel, batch, dp_rank)

    return _gen


def test_merge_per_dp_validation_summaries_strips_pad():
    s0 = {"sample_uids": ["a"], "data_sources": ["ds"], "scores": [1.0], "reward_extras": {}}
    s1 = {"sample_uids": ["pad"], "data_sources": ["ds"], "scores": [0.0], "reward_extras": {}}
    merged = merge_per_dp_validation_summaries([s0, s1], pad_size=1)
    assert merged["sample_uids"] == ["a"]
    assert merged["scores"] == [1.0]


@patch.object(RayPPOTrainer, "init_workers", _fake_ray_ppo_init_workers)
def test_validation_via_channel_returns_val_core_and_no_full_batch_flag():
    trainer = build_channel_trainer()
    trainer.init_workers()
    trainer._full_batch_returned_to_driver = False

    prompts = torch.zeros((2, 4), dtype=torch.long)
    response_mask = torch.ones((2, 3), dtype=torch.float32)
    uid = np.array(["u0", "u1"], dtype=object)
    ds = np.array(["src-a", "src-a"], dtype=object)
    batch_dict = {"prompts": prompts, "response_mask": response_mask, "uid": uid, "data_source": ds}

    trainer.val_dataloader = [batch_dict]

    trainer.config = MagicMock()
    trainer.config.actor_rollout_ref.rollout.val_kwargs.n = 1
    trainer.config.actor_rollout_ref.rollout.val_kwargs.do_sample = False
    trainer.config.trainer.log_val_generations = 0
    trainer.config.trainer.get = MagicMock(return_value=None)

    trainer.tokenizer = MagicMock()
    trainer.tokenizer.eos_token_id = 0
    trainer.tokenizer.pad_token_id = 0
    trainer.tokenizer.decode = MagicMock(return_value="")

    trainer.global_steps = 0
    trainer.use_rm = False

    trainer._maybe_log_val_generations = MagicMock()
    trainer._dump_generations = MagicMock()

    trainer.async_rollout_manager = MagicMock()
    trainer.async_rollout_manager.__class__.__module__ = "verl.experimental.channel.agent_loop"
    trainer.async_rollout_manager.agent_loop_workers = [object(), object()]
    trainer.async_rollout_manager.generate_sequences.side_effect = _simulate_rollout_via_channels(trainer)

    def _get_gen_batch_with_tensors(self, batch: DataProto) -> DataProto:
        """``RayPPOTrainer._get_gen_batch`` leaves tensor fields on ``batch`` when all non-tensor keys are reward keys; rollout needs tensors on the returned proto."""
        tensors = {k: batch.batch[k] for k in batch.batch.keys()}
        non_tensors = {k: batch.non_tensor_batch[k] for k in batch.non_tensor_batch.keys()}
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(batch.meta_info))

    trainer._get_gen_batch = _get_gen_batch_with_tensors.__get__(trainer, ChannelRayPPOTrainer)

    trainer.get_rollout_outputs_per_dp_rank = MagicMock(
        side_effect=AssertionError("validation summary path must not pull rollout batches to driver")
    )
    trainer.get_reward_outputs_per_dp_rank = MagicMock(
        side_effect=AssertionError("validation summary path must not pull reward batches to driver")
    )

    metrics = trainer._validate_via_channel()

    assert "val-core" in " ".join(metrics.keys())
    assert trainer._full_batch_returned_to_driver is False


def test_extract_validation_summary_from_reward_batch():
    from verl.experimental.channel.validation import extract_validation_summary_from_reward_batch

    rm = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    batch = DataProto(
        batch=TensorDict({"token_level_scores": rm, "rm_scores": rm}, batch_size=[2]),
        non_tensor_batch={
            "uid": np.array(["x", "y"], dtype=object),
            "data_source": np.array(["s", "s"], dtype=object),
        },
        meta_info={"reward_extra_keys": []},
    )
    # Prefer rm_scores branch via extract_reward
    out = extract_validation_summary_from_reward_batch(batch)
    assert out["scores"] == [1.0, 2.0]
    assert len(out["sample_uids"]) == 2


def test_validate_routes_to_validate_via_channel():
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer._validate_via_channel = MagicMock(return_value={"val-core/mock/reward/mean@1": 1.0})
    out = trainer._validate()
    assert out == {"val-core/mock/reward/mean@1": 1.0}
    trainer._validate_via_channel.assert_called_once_with(merged=False)
