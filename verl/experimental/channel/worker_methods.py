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
"""Channel-aware PPO training stage helpers."""

from __future__ import annotations

import time
from typing import Any, Mapping

import torch

from verl.protocol import DataProto
from verl.experimental.channel.dataproto_channel_transport import dataproto_to_cpu_after_channel_get
from verl.third_party.rlinf.scheduler.channel import Channel
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import agg_loss, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.metric import reduce_metrics


def reward_output_get_for_training(ch: Channel, dp_rank: int) -> DataProto:
    """Read one :class:`~verl.protocol.DataProto` from the reward output channel (``key=dp_rank``)."""
    item = ch.get(key=dp_rank, async_op=False)
    if not isinstance(item, DataProto):
        raise TypeError(f"Expected DataProto from reward output channel, got {type(item)}")
    dataproto_to_cpu_after_channel_get(item)
    return item


def metrics_put(ch: Channel, metrics: Mapping[str, Any], dp_rank: int) -> None:
    """Write a metrics dict to the metrics channel (``key=dp_rank``)."""
    ch.put(dict(metrics), weight=0, key=dp_rank, async_op=False)


def _minimal_token_level_rewards_and_advantage_placeholders(batch: DataProto) -> DataProto:
    """Mirror non-KL path in :class:`~verl.trainer.ppo.ray_trainer.RayPPOTrainer` (scores → rewards); advantage/returns as placeholders."""
    if "token_level_scores" not in batch.batch.keys():
        raise KeyError("Channel training stage expects batch.batch['token_level_scores'] from reward stage")
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    if "response_mask" not in batch.batch.keys():
        raise KeyError("Expected batch.batch['response_mask'] for minimal advantage placeholders")
    rm = batch.batch["response_mask"]
    batch.batch["advantages"] = torch.zeros_like(rm, dtype=torch.float32)
    batch.batch["returns"] = torch.zeros_like(rm, dtype=torch.float32)
    return batch


def _placeholder_actor_update(batch: DataProto) -> dict[str, float]:
    """Reserved for Task6+: wire to actor worker group; no-op metrics only."""
    _ = batch
    return {"train/actor_update_placeholder": 1.0}


def _placeholder_critic_update(batch: DataProto) -> dict[str, float]:
    """Reserved for Task6+: wire to critic worker group; no-op metrics only."""
    _ = batch
    return {"train/critic_update_placeholder": 1.0}


def _can_run_full_ppo_path(trainer: Any) -> bool:
    if trainer is None:
        return False
    required = (
        "config",
        "_compute_old_log_prob",
        "_update_actor",
        "_update_critic",
    )
    return all(hasattr(trainer, key) for key in required)


def _run_full_ppo_training(
    batch: DataProto,
    trainer: Any,
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if "response_mask" not in batch.batch.keys():
        batch.batch["response_mask"] = compute_response_mask(batch)

    rollout_corr_config = trainer.config.algorithm.get("rollout_correction", None)
    bypass_recomputing_logprobs = bool(rollout_corr_config and rollout_corr_config.get("bypass_mode", False))

    if bypass_recomputing_logprobs:
        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

        apply_bypass_mode(
            batch=batch,
            rollout_corr_config=rollout_corr_config,
            policy_loss_config=trainer.config.actor_rollout_ref.actor.policy_loss,
        )
    else:
        old_log_prob, old_log_prob_mfu = trainer._compute_old_log_prob(batch)
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        actor_config = trainer.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=entropys,
            loss_mask=response_masks,
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        metrics.update(
            {
                "actor/entropy": entropy_agg.detach().item(),
                "perf/mfu/actor_infer": old_log_prob_mfu,
            }
        )
        old_log_prob.batch.pop("entropys")
        if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
            router_mode = getattr(trainer.config.actor_rollout_ref.actor.router_replay, "mode", "disabled")
            if router_mode == "R2":
                batch.batch.pop("routed_experts")
            else:
                old_log_prob.batch.pop("routed_experts")
        batch = batch.union(old_log_prob)
        if "rollout_log_probs" in batch.batch.keys():
            from verl.utils.debug.metrics import calculate_debug_metrics

            metrics.update(calculate_debug_metrics(batch))

    if trainer.use_reference_policy:
        ref_log_prob = trainer._compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)

    if trainer.use_critic:
        values = trainer._compute_values(batch)
        batch = batch.union(values)

    if trainer.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(
            batch,
            kl_ctrl=trainer.kl_ctrl_in_reward,
            kl_penalty=trainer.config.algorithm.kl_penalty,
        )
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch and not bypass_recomputing_logprobs:
        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
        metrics.update(is_metrics)

    norm_adv_by_std_in_grpo = trainer.config.algorithm.get("norm_adv_by_std_in_grpo", True)
    batch = compute_advantage(
        batch,
        adv_estimator=trainer.config.algorithm.adv_estimator,
        gamma=trainer.config.algorithm.gamma,
        lam=trainer.config.algorithm.lam,
        num_repeat=trainer.config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=trainer.config.algorithm,
    )

    if trainer.use_critic:
        critic_output = trainer._update_critic(batch)
        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

    global_steps = int(train_config.get("global_steps", getattr(trainer, "global_steps", 0)))
    actor_updated = False
    if trainer.config.trainer.critic_warmup <= global_steps:
        actor_output = trainer._update_actor(batch)
        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
        actor_updated = True

    metrics["channel/actor_updated"] = 1.0 if actor_updated else 0.0
    try:
        metrics.update(compute_data_metrics(batch=batch, use_critic=trainer.use_critic))
    except KeyError:
        # Unit tests may use minimal synthetic batches without full rollout tensors.
        pass
    metrics["train/mean_token_level_reward"] = float(batch.batch["token_level_rewards"].sum(dim=-1).mean().item())
    return metrics


def run_training_from_channel(
    reward_output_ch: Channel,
    metrics_ch: Channel,
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Minimal training step: consume reward-stage output, run placeholder updates, emit metrics (all ``key=dp_rank``).

    Args:
        reward_output_ch: Channel carrying :class:`~verl.protocol.DataProto` from the reward stage.
        metrics_ch: Channel to write scalar-friendly metrics dict per DP rank.
        train_config: Must include ``dp_rank`` (int). Extra keys reserved for Task6+ (e.g. algorithm config).

    Returns:
        The metrics dict written to ``metrics_ch``.
    """
    if "dp_rank" not in train_config:
        raise KeyError("train_config must include 'dp_rank'")
    dp_rank = int(train_config["dp_rank"])

    t_train = time.perf_counter()
    batch = reward_output_get_for_training(reward_output_ch, dp_rank)
    trainer = train_config.get("trainer", None)
    if _can_run_full_ppo_path(trainer):
        metrics = _run_full_ppo_training(batch, trainer, train_config)
    else:
        batch = _minimal_token_level_rewards_and_advantage_placeholders(batch)
        actor_metrics = _placeholder_actor_update(batch)
        critic_metrics = _placeholder_critic_update(batch)
        mean_reward = float(batch.batch["token_level_rewards"].sum(dim=-1).mean().item())
        metrics = {
            "train/mean_token_level_reward": mean_reward,
            **actor_metrics,
            **critic_metrics,
        }

    stage_train_wall_ms = (time.perf_counter() - t_train) * 1000.0
    metrics = {
        **metrics,
        "channel/dp_rank": dp_rank,
        "channel/obs/stage_train_wall_ms": stage_train_wall_ms,
        "channel/obs/stage_timing": "train_stage_only_ms",
    }
    metrics_put(metrics_ch, metrics, dp_rank)
    return metrics


__all__ = [
    "metrics_put",
    "reward_output_get_for_training",
    "run_training_from_channel",
]
