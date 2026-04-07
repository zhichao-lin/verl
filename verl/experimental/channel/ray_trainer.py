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
"""Channel pipeline PPO trainer: extends :class:`~verl.trainer.ppo.ray_trainer.RayPPOTrainer`."""

from __future__ import annotations

import time
from collections import defaultdict
from pprint import pprint
from typing import Any

import numpy as np
import torch
import uuid
from tensordict import TensorDict
from tqdm import tqdm

from verl.third_party.rlinf.scheduler.channel import Channel
from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from verl.experimental.channel.agent_loop import rollout_input_put
from verl.experimental.channel.reward_worker import reward_stage_process_one_dp_rank
from verl.experimental.channel.validation import (
    merge_per_dp_validation_summaries,
    validation_summary_stage_process_one_dp_rank,
)

CHANNEL_ROLLOUT_INPUT = "RolloutInput"
CHANNEL_ROLLOUT_OUTPUT = "RolloutOutput"
CHANNEL_REWARD_OUTPUT = "RewardOutput"
CHANNEL_METRICS = "Metrics"
CHANNEL_VAL_SUMMARY = "ValSummary"
CHANNEL_TRAIN_ACTOR_PREP = "TrainActorPrep"
CHANNEL_TRAIN_AFTER_REF = "TrainAfterRef"
CHANNEL_TRAIN_POST_REWARD_OUT = "TrainPostRewardOut"
CHANNEL_TRAIN_CRITIC_IN = "TrainCriticIn"
CHANNEL_TRAIN_ACTOR_UPDATE_IN = "TrainActorUpdateIn"


class ChannelRayPPOTrainer(RayPPOTrainer):
    """Extends :class:`~verl.trainer.ppo.ray_trainer.RayPPOTrainer` with DP-keyed Channel topology.

    Checkpoint save (:meth:`~verl.trainer.ppo.ray_trainer.RayPPOTrainer._save_checkpoint`) and
    rollout weight sync (``checkpoint_manager.update_weights``) stay on the **RPC / worker control
    plane** inherited from :class:`~verl.trainer.ppo.ray_trainer.RayPPOTrainer` — they do **not**
    traverse the DP channel data path (``rollout_input_ch`` / ``metrics_ch`` / etc.).
    """

    def init_workers(self):
        super().init_workers()
        self._full_batch_returned_to_driver = False
        self._driver_full_batch_reflow_count = 0
        self._checkpoint_used_rpc = False
        self._update_weights_used_rpc = False
        self._bind_checkpoint_control_plane_hooks()
        self._init_channel_topology()
        self._validate_channel_rollout_manager_contract()

    def _bind_checkpoint_control_plane_hooks(self) -> None:
        """Mark ``checkpoint_manager.update_weights`` invocations as control-plane (RPC), not channel I/O."""
        if not hasattr(self, "checkpoint_manager"):
            # Unit tests may patch ``RayPPOTrainer.init_workers`` without building a real manager.
            return

        _orig_update = self.checkpoint_manager.update_weights

        def _tracked_update_weights(*args, **kwargs):
            self._update_weights_used_rpc = True
            return _orig_update(*args, **kwargs)

        self.checkpoint_manager.update_weights = _tracked_update_weights

    def _save_checkpoint(self):
        """Persist checkpoints via worker RPC; same path as base trainer (not channel tensors)."""
        self._checkpoint_used_rpc = True
        super()._save_checkpoint()

    def _trigger_control_ops_for_test(self) -> None:
        """Test-only: exercise checkpoint + weight sync entrypoints used by ``fit`` (RPC control path)."""
        self._save_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

    def _init_channel_topology(self) -> None:
        """Create driver-side Channel objects and record DP size for keyed routing (see ``self.dp_size``)."""
        self.dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        distributed = False
        cfg = getattr(self, "config", None)
        if cfg is not None and hasattr(cfg, "channel"):
            distributed = bool(cfg.channel.get("distributed", False))
        self.rollout_input_ch = Channel.create(name=CHANNEL_ROLLOUT_INPUT, distributed=distributed, local=False)
        self.rollout_output_ch = Channel.create(name=CHANNEL_ROLLOUT_OUTPUT, distributed=distributed, local=False)
        self.reward_output_ch = Channel.create(name=CHANNEL_REWARD_OUTPUT, distributed=distributed, local=False)
        self.metrics_ch = Channel.create(name=CHANNEL_METRICS, distributed=distributed, local=False)
        self.val_summary_ch = Channel.create(name=CHANNEL_VAL_SUMMARY, distributed=distributed, local=False)
        self.train_actor_prep_ch = Channel.create(name=CHANNEL_TRAIN_ACTOR_PREP, distributed=distributed, local=False)
        self.train_after_ref_ch = Channel.create(name=CHANNEL_TRAIN_AFTER_REF, distributed=distributed, local=False)
        self.train_post_reward_out_ch = Channel.create(
            name=CHANNEL_TRAIN_POST_REWARD_OUT, distributed=distributed, local=False
        )
        self.train_critic_in_ch = Channel.create(name=CHANNEL_TRAIN_CRITIC_IN, distributed=distributed, local=False)
        self.train_actor_update_in_ch = Channel.create(
            name=CHANNEL_TRAIN_ACTOR_UPDATE_IN, distributed=distributed, local=False
        )
        self._init_worker_channel_contexts()

    def _init_worker_channel_contexts(self) -> None:
        """Initialize RLinf worker contexts inside verl workers."""
        actor_group = getattr(self, "actor_rollout_wg", None)
        if actor_group is not None:
            actor_group.init_channel_worker_context("channel_actor")
        critic_group = getattr(self, "critic_wg", None)
        if critic_group is not None:
            critic_group.init_channel_worker_context("channel_critic")
        ref_group = getattr(self, "ref_policy_wg", None)
        if ref_group is not None and ref_group is not actor_group:
            ref_group.init_channel_worker_context("channel_ref")

    def _validate_channel_rollout_manager_contract(self) -> None:
        """Fail fast when channel mode is enabled but rollout manager is not the channel implementation."""
        manager = getattr(self, "async_rollout_manager", None)
        if manager is None:
            return

        manager_module = manager.__class__.__module__
        if manager_module != "verl.experimental.channel.agent_loop":
            raise RuntimeError(
                "ChannelRayPPOTrainer requires "
                "`actor_rollout_ref.rollout.agent.agent_loop_manager_class` "
                "to point to `verl.experimental.channel.agent_loop.AgentLoopManager`."
            )

        workers = getattr(manager, "agent_loop_workers", None)
        if workers is not None and len(workers) != self.dp_size:
            raise RuntimeError(
                f"Channel key routing requires agent.num_workers ({len(workers)}) "
                f"to equal actor dp_size ({self.dp_size})."
            )

    def put_rollout_inputs_per_dp_rank(self, batches: list[DataProto]) -> None:
        """Driver：为每个 DP rank 投递一条 batch（``key=dp_rank``），与 worker 侧 ``AgentLoopWorker.generate_sequences`` 对齐。"""
        if len(batches) != self.dp_size:
            raise ValueError(f"Expected {self.dp_size} batches (one per DP rank), got {len(batches)}")
        for dp_rank, batch in enumerate(batches):
            rollout_input_put(self.rollout_input_ch, batch, dp_rank)

    def get_rollout_outputs_per_dp_rank(self) -> list[DataProto]:
        """Driver：按 ``dp_rank`` 顺序收集各 rank 的 rollout 输出（与 ``put_rollout_inputs_per_dp_rank`` 使用相同 key 空间）。"""
        self._full_batch_returned_to_driver = True
        self._driver_full_batch_reflow_count = getattr(self, "_driver_full_batch_reflow_count", 0) + 1
        return [self.rollout_output_ch.get(key=dp_rank, async_op=False) for dp_rank in range(self.dp_size)]

    def run_reward_stage_all_dp_ranks(self) -> None:
        """Task4：对每个 ``dp_rank`` 从 ``rollout_output_ch`` 读入并写入 ``reward_output_ch``。

        Note:
            This method consumes rollout outputs from ``rollout_output_ch``. Do not call
            :meth:`get_rollout_outputs_per_dp_rank` for the same step before this method.
        """
        for dp_rank in range(self.dp_size):
            reward_stage_process_one_dp_rank(self.rollout_output_ch, self.reward_output_ch, dp_rank)

    def get_reward_outputs_per_dp_rank(self) -> list[DataProto]:
        """Driver：按 ``dp_rank`` 顺序收集 reward 阶段输出（与 rollout 侧 key 空间一致）。"""
        self._full_batch_returned_to_driver = True
        self._driver_full_batch_reflow_count = getattr(self, "_driver_full_batch_reflow_count", 0) + 1
        return [self.reward_output_ch.get(key=dp_rank, async_op=False) for dp_rank in range(self.dp_size)]

    def run_train_stage_all_dp_ranks(self, train_config: dict | None = None) -> None:
        """Task5：对每个 ``dp_rank`` 从 ``reward_output_ch`` 读入并写入 ``metrics_ch``（与 Task4 相同的 key 空间）。

        Note:
            Consumes reward outputs from ``reward_output_ch``. Do not call
            :meth:`get_reward_outputs_per_dp_rank` for the same step after this method.
        """
        self._run_train_stage_via_worker_channels(dict(train_config or {}))

    def _separate_ref_worker(self) -> bool:
        """True when reference policy runs in a dedicated worker group (not colocated with actor)."""
        ref_wg = getattr(self, "ref_policy_wg", None)
        actor_wg = getattr(self, "actor_rollout_wg", None)
        return (
            bool(self.use_reference_policy)
            and ref_wg is not None
            and actor_wg is not None
            and ref_wg is not actor_wg
        )

    def _run_train_stage_via_worker_channels(self, train_config: dict) -> None:
        """Run train stage as worker-to-worker channel pipeline."""
        actor_group = getattr(self, "actor_rollout_wg", None)
        if actor_group is None:
            raise RuntimeError("channel train stage requires actor_rollout_wg, but it is not initialized.")

        separate_ref = self._separate_ref_worker()
        compute_ref_on_actor = bool(self.use_reference_policy) and not separate_ref

        actor_group.channel_train_prepare_from_reward(
            self.reward_output_ch,
            self.train_actor_prep_ch,
            {
                "use_reference_policy": bool(self.use_reference_policy),
                "use_kl_in_reward": bool(self.config.algorithm.use_kl_in_reward),
                "compute_ref_on_actor": compute_ref_on_actor,
            },
        )

        if separate_ref:
            self.ref_policy_wg.channel_train_ref_log_prob_stage(
                self.train_actor_prep_ch,
                self.train_after_ref_ch,
                {"use_kl_in_reward": bool(self.config.algorithm.use_kl_in_reward)},
            )
            post_reward_in_ch = self.train_after_ref_ch
        else:
            post_reward_in_ch = self.train_actor_prep_ch

        post_reward_flags = {
            "use_kl_in_reward": bool(self.config.algorithm.use_kl_in_reward),
            "algorithm_config": self.config.algorithm,
            "adv_estimator": self.config.algorithm.adv_estimator,
            "gamma": self.config.algorithm.gamma,
            "lam": self.config.algorithm.lam,
            "rollout_n": self.config.actor_rollout_ref.rollout.n,
            "norm_adv_by_std_in_grpo": self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            "policy_loss_config": self.config.actor_rollout_ref.actor.policy_loss,
        }

        critic_group = getattr(self, "critic_wg", None)
        if critic_group is not None and self.use_critic:
            critic_group.channel_train_post_reward_stage(
                post_reward_in_ch,
                self.train_post_reward_out_ch,
                post_reward_flags,
            )
            critic_group.channel_train_critic_stage(
                self.train_post_reward_out_ch,
                self.train_actor_update_in_ch,
                {"use_critic": bool(self.use_critic)},
            )
            actor_in_channel = self.train_actor_update_in_ch
        else:
            actor_group.channel_train_post_reward_stage(
                post_reward_in_ch,
                self.train_post_reward_out_ch,
                post_reward_flags,
            )
            actor_in_channel = self.train_post_reward_out_ch

        actor_group.channel_train_actor_update_stage(
            actor_in_channel,
            self.metrics_ch,
            {
                "global_steps": int(train_config.get("global_steps", getattr(self, "global_steps", 0))),
                "critic_warmup": int(self.config.trainer.critic_warmup),
            },
        )

    def get_metrics_per_dp_rank(self) -> list[dict]:
        """Driver：按 ``dp_rank`` 顺序收集训练阶段 metrics（``key=dp_rank``）。"""
        return [self.metrics_ch.get(key=dp_rank, async_op=False) for dp_rank in range(self.dp_size)]

    def run_validation_summary_stage_all_dp_ranks(self) -> None:
        """Task7：reward 输出 → 每 rank 一条轻量 summary 写入 ``val_summary_ch``（不回收 full DataProto 到 driver）。"""
        for dp_rank in range(self.dp_size):
            validation_summary_stage_process_one_dp_rank(
                self.reward_output_ch, self.val_summary_ch, dp_rank
            )

    def get_validation_summaries_per_dp_rank(self) -> list[dict[str, Any]]:
        """Driver：仅读取 summary dict（uid/score/data_source 等），不触发 ``_full_batch_returned_to_driver``。"""
        return [self.val_summary_ch.get(key=dp_rank, async_op=False) for dp_rank in range(self.dp_size)]

    def _validate(self, merged: bool = False):
        """Use channel summary validation path by default for channel trainer."""
        return self._validate_via_channel(merged=merged)

    def _validate_via_channel(self, merged: bool = False) -> dict[str, Any]:
        """Validation via DP-sharded channel pipeline; driver aggregates summaries only (no full-batch pull)."""
        if merged:
            raise NotImplementedError("Channel validation merged mode is not implemented.")

        self._full_batch_returned_to_driver = False
        self._driver_full_batch_reflow_count = 0

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_uids: list[str] = []
        sample_turns: list = []

        sample_inputs: list[str] = []
        sample_outputs: list[str] = []
        sample_scores: list[float] = []
        sample_gts: list = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = self.dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

            n_pad = len(test_gen_batch_padded)
            shard_len = n_pad // self.dp_size
            shards = [test_gen_batch_padded[i * shard_len : (i + 1) * shard_len] for i in range(self.dp_size)]

            self.put_rollout_inputs_per_dp_rank(shards)
            self.async_rollout_manager.generate_sequences(self.rollout_input_ch, self.rollout_output_ch)
            self.run_reward_stage_all_dp_ranks()
            self.run_validation_summary_stage_all_dp_ranks()
            per_dp = self.get_validation_summaries_per_dp_rank()
            merged_one = merge_per_dp_validation_summaries(per_dp, pad_size)

            sample_uids.extend(merged_one["sample_uids"])
            data_source_lst.append(np.array(merged_one["data_sources"], dtype=object))
            sample_scores.extend(merged_one["scores"])
            reward_extra_infos_dict["reward"].extend(merged_one["scores"])
            for key, values in merged_one["reward_extras"].items():
                reward_extra_infos_dict[key].extend(values)

            if merged_one.get("num_turns"):
                sample_turns.append(np.array(merged_one["num_turns"]))

            # Optional table logging / dump (needs text): best-effort decode when tokenizer is available
            if self.config.trainer.log_val_generations > 0 or self.config.trainer.get("validation_data_dir"):
                # Minimal placeholders — full text is not shipped on the summary path
                sample_inputs.extend([""] * len(merged_one["scores"]))
                sample_outputs.extend([""] * len(merged_one["scores"]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def aggregate_metrics_from_channel(self) -> dict[str, Any]:
        """Merge per-DP metrics dicts from ``metrics_ch`` (numeric keys averaged across ranks)."""
        per_rank = self.get_metrics_per_dp_rank()
        if not per_rank:
            return {}
        merged: dict[str, Any] = {}
        keys = set().union(*(m.keys() for m in per_rank))
        for key in keys:
            values = [m[key] for m in per_rank if key in m]
            if not values:
                continue
            first = values[0]
            if isinstance(first, (int, float)) and not isinstance(first, bool):
                merged[key] = sum(values) / len(values)
            else:
                merged[key] = first
        return merged

    def run_channel_step(self, batches: list[DataProto], train_config: dict | None = None) -> dict[str, Any]:
        """One pipeline step: rollout → reward → train; driver only reads aggregated metrics from ``metrics_ch``."""
        self._full_batch_returned_to_driver = False
        self._driver_full_batch_reflow_count = 0
        t0 = time.perf_counter()
        self.put_rollout_inputs_per_dp_rank(batches)

        self.async_rollout_manager.generate_sequences(self.rollout_input_ch, self.rollout_output_ch)
        self.checkpoint_manager.sleep_replicas()

        self.run_reward_stage_all_dp_ranks()
        self.run_train_stage_all_dp_ranks(train_config)
        merged = self.aggregate_metrics_from_channel()
        merged["channel/obs/channel_step_wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return merged

    def _minimal_batches_for_test_step(self) -> list[DataProto]:
        """Tiny :class:`~verl.protocol.DataProto` shards (``response_mask``) for test-only channel steps."""
        batches: list[DataProto] = []
        for _ in range(self.dp_size):
            rm = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
            batches.append(
                DataProto(batch=TensorDict({"response_mask": rm}, batch_size=[1]), meta_info={})
            )
        return batches

    def fit_one_step_for_test(self, train_config: dict | None = None) -> dict[str, Any]:
        """Test-only entry: run :meth:`run_channel_step` with minimal local batches (no dataloader)."""
        return self.run_channel_step(self._minimal_batches_for_test_step(), train_config=train_config)

    def fit(self):
        """Minimal channel-mode fit loop for integration."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                assert len(gen_batch_output) % self.dp_size == 0, f"{len(gen_batch_output)=}, {self.dp_size=}"
                shard_len = len(gen_batch_output) // self.dp_size
                shards = [gen_batch_output[i * shard_len : (i + 1) * shard_len] for i in range(self.dp_size)]

                metrics = self.run_channel_step(shards, train_config={"global_steps": self.global_steps})
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})

                if (
                    self.config.trainer.save_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)
                ):
                    self._save_checkpoint()
                self.checkpoint_manager.update_weights(self.global_steps)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


__all__ = ["ChannelRayPPOTrainer"]
