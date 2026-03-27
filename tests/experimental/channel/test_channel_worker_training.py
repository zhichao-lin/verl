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

"""Task5: training stage reads reward_output (key=dp_rank) and writes metrics (key=dp_rank)."""

from unittest.mock import MagicMock

import torch
from tensordict import TensorDict

from verl.experimental.channel.ray_trainer import CHANNEL_METRICS, CHANNEL_REWARD_OUTPUT, ChannelRayPPOTrainer
from verl.experimental.channel.reward_worker import reward_output_put
from verl.experimental.channel.worker_methods import run_training_from_channel
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator
from verl.third_party.rlinf.scheduler.channel.channel import Channel
from verl.third_party.rlinf.scheduler.channel.channel_worker import LocalChannel
from verl.third_party.rlinf.scheduler.worker.worker import Worker


def _create_local_channel(name: str, maxsize: int = 0):
    local_ch = LocalChannel(maxsize=maxsize)
    ch = Channel()
    ch._initialize(name, None, None, Worker.current_worker, local_channel=local_ch, maxsize=maxsize)
    return ch


def _proto_after_reward_stage() -> DataProto:
    rm = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    scores = torch.zeros_like(rm)
    return DataProto(
        batch=TensorDict(
            {"response_mask": rm, "token_level_scores": scores},
            batch_size=[2],
        ),
        meta_info={},
    )


def test_run_training_from_channel_consumes_reward_and_writes_metrics_same_key():
    reward_ch = _create_local_channel(f"{CHANNEL_REWARD_OUTPUT}Task5Roundtrip")
    metrics_ch = _create_local_channel(f"{CHANNEL_METRICS}Task5Roundtrip")
    dp_rank = 1
    batch = _proto_after_reward_stage()
    reward_output_put(reward_ch, batch, dp_rank)

    out = run_training_from_channel(
        reward_ch,
        metrics_ch,
        {"dp_rank": dp_rank},
    )
    assert isinstance(out, dict)
    assert out["channel/dp_rank"] == dp_rank

    got_metrics = metrics_ch.get(key=dp_rank, async_op=False)
    assert got_metrics == out
    assert got_metrics["channel/dp_rank"] == dp_rank
    assert "train/mean_token_level_reward" in got_metrics


def test_trainer_run_train_stage_and_get_metrics_match_dp_keys():
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.use_critic = True
    trainer.use_reference_policy = True
    trainer.global_steps = 3
    trainer.actor_rollout_wg = MagicMock()
    trainer.critic_wg = MagicMock()
    trainer.reward_output_ch = MagicMock()
    trainer.train_actor_prep_ch = MagicMock()
    trainer.train_actor_update_in_ch = MagicMock()
    trainer.metrics_ch = MagicMock()
    trainer.config = MagicMock()
    trainer.config.algorithm.use_kl_in_reward = False
    trainer.config.algorithm.adv_estimator = AdvantageEstimator.GAE
    trainer.config.algorithm.gamma = 1.0
    trainer.config.algorithm.lam = 1.0
    trainer.config.algorithm.get.return_value = True
    trainer.config.actor_rollout_ref.rollout.n = 1
    trainer.config.trainer.critic_warmup = 0

    trainer.run_train_stage_all_dp_ranks({})

    trainer.actor_rollout_wg.channel_train_prepare_from_reward.assert_called_once_with(
        trainer.reward_output_ch,
        trainer.train_actor_prep_ch,
        {"use_reference_policy": True, "use_kl_in_reward": False},
    )
    trainer.critic_wg.channel_train_critic_stage.assert_called_once()
    trainer.actor_rollout_wg.channel_train_actor_update_stage.assert_called_once()


def test_trainer_prefers_worker_channel_pipeline_without_driver_batch_pull():
    trainer = object.__new__(ChannelRayPPOTrainer)
    trainer.dp_size = 2
    trainer.use_critic = True
    trainer.use_reference_policy = True
    trainer.global_steps = 3
    trainer.actor_rollout_wg = MagicMock()
    trainer.critic_wg = MagicMock()
    trainer.reward_output_ch = _create_local_channel("RewardOutputWorkerPipeline")
    trainer.train_actor_prep_ch = _create_local_channel("TrainActorPrepWorkerPipeline")
    trainer.train_critic_in_ch = _create_local_channel("TrainCriticInWorkerPipeline")
    trainer.train_actor_update_in_ch = _create_local_channel("TrainActorUpdateInWorkerPipeline")
    trainer.metrics_ch = _create_local_channel("MetricsWorkerPipeline")
    trainer.config = MagicMock()
    trainer.config.algorithm.use_kl_in_reward = False
    trainer.config.algorithm.adv_estimator = AdvantageEstimator.GAE
    trainer.config.algorithm.gamma = 1.0
    trainer.config.algorithm.lam = 1.0
    trainer.config.algorithm.get.return_value = True
    trainer.config.actor_rollout_ref.rollout.n = 1
    trainer.config.trainer.critic_warmup = 0

    trainer.run_train_stage_all_dp_ranks({})

    assert trainer.actor_rollout_wg.channel_train_prepare_from_reward.call_count == 1
    assert trainer.actor_rollout_wg.channel_train_actor_update_stage.call_count == 1
    trainer.critic_wg.channel_train_critic_stage.assert_called_once()


def test_run_training_from_channel_executes_full_ppo_flow_when_trainer_provided():
    reward_ch = _create_local_channel(f"{CHANNEL_REWARD_OUTPUT}Task5FullFlow")
    metrics_ch = _create_local_channel(f"{CHANNEL_METRICS}Task5FullFlow")
    dp_rank = 0
    reward_output_put(reward_ch, _proto_after_reward_stage(), dp_rank)

    trainer = MagicMock()
    trainer.use_reference_policy = True
    trainer.use_critic = True
    trainer.global_steps = 1
    trainer.config = MagicMock()
    trainer.config.algorithm.get.return_value = None
    trainer.config.algorithm.use_kl_in_reward = False
    trainer.config.algorithm.adv_estimator = AdvantageEstimator.GAE
    trainer.config.algorithm.gamma = 1.0
    trainer.config.algorithm.lam = 1.0
    trainer.config.actor_rollout_ref.rollout.n = 1
    trainer.config.trainer.critic_warmup = 0
    trainer.config.actor_rollout_ref.actor = MagicMock()
    trainer.config.actor_rollout_ref.actor.loss_agg_mode = "token-mean"
    trainer.config.actor_rollout_ref.actor.loss_scale_factor = 1.0

    old_log_prob = DataProto(
        batch=TensorDict(
            {
                "old_log_probs": torch.zeros((2, 3), dtype=torch.float32),
                "entropys": torch.ones((2, 3), dtype=torch.float32),
            },
            batch_size=[2],
        )
    )
    trainer._compute_old_log_prob.return_value = (old_log_prob, 0.42)
    trainer._compute_ref_log_prob.return_value = DataProto(
        batch=TensorDict({"ref_log_prob": torch.zeros((2, 3), dtype=torch.float32)}, batch_size=[2])
    )
    trainer._compute_values.return_value = DataProto(
        batch=TensorDict({"values": torch.zeros((2, 3), dtype=torch.float32)}, batch_size=[2])
    )
    trainer._update_critic.return_value = DataProto.from_single_dict(data={}, meta_info={"metrics": {"critic/loss": 0.1}})
    trainer._update_actor.return_value = DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/loss": 0.2}})

    out = run_training_from_channel(
        reward_ch,
        metrics_ch,
        {"dp_rank": dp_rank, "trainer": trainer},
    )

    trainer._compute_old_log_prob.assert_called_once()
    trainer._compute_ref_log_prob.assert_called_once()
    trainer._compute_values.assert_called_once()
    trainer._update_critic.assert_called_once()
    trainer._update_actor.assert_called_once()
    assert "actor/loss" in out
    assert "critic/loss" in out
