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
import asyncio
import ray

from omegaconf import DictConfig
import verl.experimental.agent_loop as agent_loop
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.third_party.rlinf.scheduler.channel import Channel
from verl.third_party.rlinf.worker_adapter import create_rlinf_worker


def rollout_input_put(ch: Channel, batch: DataProto, dp_rank: int) -> None:
    """Driver：按 ``dp_rank`` 投递一条 rollout 输入。与 :func:`rollout_input_get` 使用相同 ``key``。"""
    # 当前 Task2/3 使用 driver 侧 LocalChannel（同步 put）。后续若替换为分布式 Channel，
    # 可在此改为 async put / AsyncRayWork，与 worker 侧 recv 路径对齐。
    ch.put(batch, weight=0, key=dp_rank, async_op=False)


def rollout_input_get(ch: Channel, dp_rank: int) -> DataProto:
    """Worker：读取本 DP rank 对应的 rollout 输入（``key=dp_rank``）。"""
    item = ch.get(key=dp_rank, async_op=False)
    assert isinstance(item, DataProto), f"Expected DataProto from rollout input channel, got {type(item)}"
    return item


def rollout_output_put(ch: Channel, output: DataProto, dp_rank: int) -> None:
    """Worker：将 rollout 输出写回与输入相同的 ``key=dp_rank``。"""
    # LocalChannel 仅支持同步 put；分布式 Channel 上可改为 async_op=True 并 await AsyncWork。
    ch.put(output, weight=0, key=dp_rank, async_op=False)


class AgentLoopWorker(agent_loop.AgentLoopWorker):
    async def generate_sequences(
        self,
        gen_input_channel: Channel,
        gen_output_channel: Channel,
        dp_rank: int,
    ) -> None:
        """Channel 版本的 generate_sequences。

        - 从 ``gen_input_channel`` 按 ``key=dp_rank`` 读取一个 DataProto batch；
        - 调用基类的 ``generate_sequences(batch)`` 完成 AgentLoop rollout；
        - 将输出 DataProto 按相同 ``key`` 写入 ``gen_output_channel``。
        """
        batch = rollout_input_get(gen_input_channel, dp_rank)
        output = await super().generate_sequences(batch)
        rollout_output_put(gen_output_channel, output, dp_rank)

    def create_rlinf_worker(self, group_name: str, rank: int, world_size: int) -> None:
        """为当前 Ray actor 进程创建 RLinf Worker，用于启用 Channel 能力。"""
        create_rlinf_worker(group_name=group_name, rank=rank, world_size=world_size)


class AgentLoopManager(agent_loop.AgentLoopManager):
    @classmethod
    @auto_await
    async def create(
        cls,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        """Create agent loop manager."""
        cls.agent_loop_workers_class = ray.remote(AgentLoopWorker)
        instance = cls(config, worker_group, rollout_resource_pool, reward_loop_worker_handles)
        await instance._initialize_llm_servers()
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        """初始化 AgentLoopWorker，并为每个 worker 创建 RLinf Worker。"""
        # 先复用基类逻辑创建 Ray actor 列表
        await super()._init_agent_loop_workers()

        group_name = "agent_loop"
        world_size = len(self.agent_loop_workers)
        # 为每个 AgentLoopWorker 进程创建对应的 RLinf Worker，启用 Channel 通信能力
        ray.get(
            [
                worker.create_rlinf_worker.remote(group_name, rank, world_size)
                for rank, worker in enumerate(self.agent_loop_workers)
            ]
        )

    @auto_await
    async def generate_sequences(
        self,
        gen_input_channel: Channel | None = None,
        gen_output_channel: Channel | None = None,
    ) -> None:
        """Channel 模式下触发所有 AgentLoopWorker 的 rollout。

        Args:
            gen_input_channel: RLinf Channel，用于从 Trainer 接收 DataProto。
            gen_output_channel: RLinf Channel，用于向 Trainer 写回 DataProto。
        """

        assert gen_input_channel is not None and gen_output_channel is not None, (
            "Channel AgentLoopManager.generate_sequences 期望提供 "
            "`gen_input_channel` 和 `gen_output_channel`，当前至少一个为 None。"
        )

        # 各 worker 使用各自的 dp_rank 作为 key，从 Channel 取 batch 并写回（与 driver 侧 dispatch 对齐）。
        await asyncio.gather(
            *[
                worker.generate_sequences.remote(gen_input_channel, gen_output_channel, dp_rank)
                for dp_rank, worker in enumerate(self.agent_loop_workers)
            ]
        )
