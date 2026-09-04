# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""vLLM PD-disaggregated replica: 1 prefill + N decode servers per replica,
asymmetric TP supported. MVP: prefill_replicas=1, single-node only."""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Optional

import ray
from ray.actor import ActorHandle

from verl.utils.device import get_device_name, get_resource_name
from verl.utils.net_utils import get_free_port_range, is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

_PD_TRANSFER_BACKENDS = ("nixl", "mooncake", "ascend")
_NPU_PD_TRANSFER_BACKENDS = ("mooncake", "ascend")
_STORE_KV_CONNECTORS = frozenset(
    {
        "MooncakeStoreConnector",
        "AscendStoreConnector",
        "MooncakeConnectorStoreV1",
    }
)


def _validate_npu_pd_backend(device_name: str, transfer_backend: str) -> None:
    """NPU PD is MooncakeConnectorV1 / AscendStore only; Nixl is GPU-side."""
    if device_name != "npu":
        return
    if transfer_backend not in _NPU_PD_TRANSFER_BACKENDS:
        raise NotImplementedError(
            f"vLLM PD on NPU requires transfer_backend in {_NPU_PD_TRANSFER_BACKENDS}; "
            f"got {transfer_backend!r}"
        )


def _use_ascend_kv_connectors(device_name: str, transfer_backend: str) -> bool:
    return transfer_backend == "ascend" or (device_name == "npu" and transfer_backend == "mooncake")


def _plain_mapping(value: Any) -> dict:
    """Coerce OmegaConf / dict extras into a plain dict."""
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf

        return dict(OmegaConf.to_container(value, resolve=True) or {})
    except Exception:
        return dict(value)


def _engine_kwargs_kv_transfer_config(config) -> Optional[dict]:
    """Read optional colocated ``engine_kwargs.vllm.kv_transfer_config``."""
    engine_kwargs = (config.get("engine_kwargs") or {}).get("vllm") or {}
    user_kv = engine_kwargs.get("kv_transfer_config") if hasattr(engine_kwargs, "get") else None
    if user_kv is None:
        return None
    if isinstance(user_kv, str):
        try:
            user_kv = json.loads(user_kv)
        except json.JSONDecodeError:
            return None
    if not isinstance(user_kv, dict):
        try:
            from omegaconf import OmegaConf

            user_kv = OmegaConf.to_container(user_kv, resolve=True)
        except Exception:
            return None
    return user_kv if isinstance(user_kv, dict) else None


def _with_store_tp_size(extra: dict, prefill_tp: int, decode_tp: int) -> dict:
    """Fill store_tp_size / LCM extras when prefill and decode TP differ."""
    # Key presence, not truthiness: ``enable_store_tp_lcm=False`` is a valid
    # opt-out (rank-local keys) and must not be overwritten to True.
    if "store_tp_size" in extra or "enable_store_tp_lcm" in extra:
        return extra
    if prefill_tp == decode_tp:
        return extra
    bigger, smaller = max(prefill_tp, decode_tp), min(prefill_tp, decode_tp)
    if bigger % smaller == 0:
        extra["store_tp_size"] = bigger
    else:
        extra["enable_store_tp_lcm"] = True
        extra["prefill_tp_sizes"] = [prefill_tp, decode_tp]
    return extra


@dataclass
class _PdSideChannel:
    """One PD engine's reserved side-channel / kv_port base."""

    host: str
    port: int
    socks: list
    kv_port: Optional[int]
    lookup_rpc_port: Optional[int]


def _allocate_pd_side_channels(
    worker_infos,
    prefill_tp: int,
    decode_tp: int,
    n_decode: int,
    npu_kv: bool,
    bootstrap_port: Optional[int],
) -> tuple[_PdSideChannel, list[_PdSideChannel]]:
    """Reserve side-channel ports per PD engine.

    GPU: one port per engine (NIXL / Mooncake env side-channel).
    NPU: ``tp`` consecutive ports because MooncakeConnectorV1 handshake is
    ``kv_port + rank``. Prefill ``lookup_rpc_port`` uses that same base so
    multi-replica IPC paths do not collide on ``..._0_dp_rank0``.
    Decode binds on the decode worker IP, not the prefill host.
    """

    def _one(host: str, tp: int, start: Optional[int]) -> _PdSideChannel:
        count = tp if npu_kv else 1
        base, socks = get_free_port_range(host, count, start=start)
        return _PdSideChannel(
            host=host,
            port=base,
            socks=socks,
            kv_port=base if npu_kv else None,
            lookup_rpc_port=base if npu_kv else None,
        )

    allocated: list[_PdSideChannel] = []
    try:
        prefill = _one(worker_infos[0][2], prefill_tp, bootstrap_port)
        allocated.append(prefill)
        decodes = []
        for i in range(n_decode):
            start_idx = prefill_tp + i * decode_tp
            decodes.append(_one(worker_infos[start_idx][2], decode_tp, None))
            allocated.append(decodes[-1])
        return prefill, decodes
    except Exception:
        for plan in allocated:
            for sock in plan.socks:
                sock.close()
        raise


def _with_ascend_store_peer_tp(extra: dict, prefill_tp: int, decode_tp: int) -> dict:
    """Fill AscendStore peer-TP extras when prefill and decode TP differ.

    Decode (kv_consumer) reads ``prefill_tp_size``; prefill reads
    ``decode_tp_size``. Putting both on both sides is correct. Key presence
    is an opt-out, matching GPU ``store_tp_size`` / ``enable_store_tp_lcm``.
    """
    if "prefill_tp_size" in extra or "decode_tp_size" in extra:
        return extra
    if prefill_tp == decode_tp:
        return extra
    extra["prefill_tp_size"] = prefill_tp
    extra["decode_tp_size"] = decode_tp
    return extra


class vLLMPDReplica(vLLMReplica):
    """Replica that runs vLLM in prefill-decode disaggregated mode."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
            name_suffix,
        )

        disagg = self.config.disaggregation
        assert disagg.enabled, "vLLMPDReplica requires rollout.disaggregation.enabled=True"

        if disagg.transfer_backend not in _PD_TRANSFER_BACKENDS:
            raise NotImplementedError(
                f"vLLMPDReplica supports transfer_backend in {_PD_TRANSFER_BACKENDS} in this "
                f"revision; got {disagg.transfer_backend!r}. mori/fake are reserved "
                f"in DisaggregationConfig and will land in follow-ups."
            )
        if disagg.prefill_replicas != 1:
            raise NotImplementedError(f"prefill_replicas=1 only (got {disagg.prefill_replicas})")
        self._n_prefill = disagg.prefill_replicas
        self._n_decode = disagg.decode_replicas

        self._prefill_tp = self.config.tensor_model_parallel_size
        # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
        self._decode_tp = (
            disagg.decode_tensor_model_parallel_size
            if disagg.decode_tensor_model_parallel_size is not None
            else self._prefill_tp
        )

        pd_world_size = self._prefill_tp + self._n_decode * self._decode_tp
        if pd_world_size > gpus_per_node:
            raise NotImplementedError(
                f"PD replica needs {pd_world_size} GPUs but gpus_per_node={gpus_per_node}; "
                f"single-node only in this revision (use more replicas to span nodes once "
                f"multi-node lands)"
            )
        if self.config.data_parallel_size != 1:
            raise NotImplementedError(f"data_parallel_size=1 only (got {self.config.data_parallel_size})")
        if self.config.pipeline_model_parallel_size != 1:
            raise NotImplementedError(
                f"pipeline_model_parallel_size=1 only "
                f"(got {self.config.pipeline_model_parallel_size}); PD path does not model PP yet"
            )

        self.world_size = pd_world_size
        self.gpus_per_replica_node = min(self.gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0
        self.nnodes = self.world_size // self.gpus_per_replica_node

        self._prefill_servers: list[ActorHandle] = []
        self._decode_servers: list[ActorHandle] = []

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker count {len(self.workers)} != PD world size {self.world_size}"
        )
        device_name = get_device_name()
        transfer_backend = self.config.disaggregation.transfer_backend
        _validate_npu_pd_backend(device_name, transfer_backend)
        npu_kv = _use_ascend_kv_connectors(device_name, transfer_backend)

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                        ray.util.get_node_ip_address().strip("[]"),
                    )
                )
                for worker in self.workers
            ]
        )

        prefill_engine_id = uuid.uuid4().hex

        prefill_end = self._prefill_tp
        prefill_workers = self.workers[0:prefill_end]
        prefill_node_id = worker_infos[0][0]
        prefill_devs = self._collect_cuda_devices(worker_infos[0:prefill_end])

        reserved_socks = []
        try:
            store_enabled, store_extra, store_config_path = self._resolve_mooncake_store_settings()
            if store_enabled and not store_config_path:
                raise ValueError(
                    "disaggregation.enable_mooncake_store=True requires "
                    "disaggregation.mooncake_store_config_path or MOONCAKE_CONFIG_PATH"
                )
            prefill_sc, decode_scs = _allocate_pd_side_channels(
                worker_infos,
                prefill_tp=self._prefill_tp,
                decode_tp=self._decode_tp,
                n_decode=self._n_decode,
                npu_kv=npu_kv,
                bootstrap_port=self.config.disaggregation.bootstrap_port,
            )
            reserved_socks.extend(prefill_sc.socks)
            for decode_sc in decode_scs:
                reserved_socks.extend(decode_sc.socks)

            prefill_kv_cfg = self._build_kv_transfer_config(
                role="prefill",
                engine_id=prefill_engine_id,
                transfer_backend=transfer_backend,
                mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                enable_mooncake_store=store_enabled,
                mooncake_store_extra_config=store_extra,
                save_decode_cache=self.config.disaggregation.save_decode_cache,
                prefill_tp=self._prefill_tp,
                decode_tp=self._decode_tp,
                device_name=device_name,
                kv_port=prefill_sc.kv_port,
                lookup_rpc_port=prefill_sc.lookup_rpc_port,
            )
            self._prefill_servers = [
                self._spawn_pd_server(
                    role="prefill",
                    workers=prefill_workers,
                    node_id=prefill_node_id,
                    cuda_visible_devices=prefill_devs,
                    tp=self._prefill_tp,
                    kv_transfer_config=prefill_kv_cfg,
                    side_channel_host=prefill_sc.host,
                    side_channel_port=prefill_sc.port,
                    mooncake_bootstrap_port=prefill_sc.port,
                    mooncake_store_config_path=store_config_path,
                    actor_name=f"vllm_server_{self.replica_rank}_0{self.name_suffix}",
                    zmq_base_trainer_rank=0,
                )
            ]

            for i, decode_sc in enumerate(decode_scs):
                start = self._prefill_tp + i * self._decode_tp
                end = start + self._decode_tp
                workers_i = self.workers[start:end]
                node_id_i = worker_infos[start][0]
                devs_i = self._collect_cuda_devices(worker_infos[start:end])

                decode_kv_cfg = self._build_kv_transfer_config(
                    role="decode",
                    engine_id=uuid.uuid4().hex,
                    transfer_backend=transfer_backend,
                    mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                    enable_mooncake_store=store_enabled,
                    mooncake_store_extra_config=store_extra,
                    save_decode_cache=self.config.disaggregation.save_decode_cache,
                    prefill_tp=self._prefill_tp,
                    decode_tp=self._decode_tp,
                    device_name=device_name,
                    kv_port=decode_sc.kv_port,
                    lookup_rpc_port=decode_sc.lookup_rpc_port,
                )
                self._decode_servers.append(
                    self._spawn_pd_server(
                        role="decode",
                        workers=workers_i,
                        node_id=node_id_i,
                        cuda_visible_devices=devs_i,
                        tp=self._decode_tp,
                        kv_transfer_config=decode_kv_cfg,
                        side_channel_host=decode_sc.host,
                        side_channel_port=decode_sc.port,
                        mooncake_bootstrap_port=prefill_sc.port,
                        mooncake_store_config_path=store_config_path,
                        actor_name=f"vllm_server_decode_{self.replica_rank}_{i}{self.name_suffix}",
                        zmq_base_trainer_rank=start,
                    )
                )

            await asyncio.gather(
                *[
                    server.launch_server.remote(master_address=None, master_port=None, dp_rpc_port=None)
                    for server in self._prefill_servers + self._decode_servers
                ]
            )
        finally:
            for sock in reserved_socks:
                sock.close()

        await self._prefill_servers[0].set_pd_peer.remote(
            self._decode_servers,
            prefill_sc.port,
            prefill_engine_id,
        )

        self.servers = list(self._prefill_servers) + list(self._decode_servers)
        prefill_address, prefill_port = await self._prefill_servers[0].get_server_address.remote()
        self._server_handle = self._prefill_servers[0]
        self._server_address = (
            f"[{prefill_address}]:{prefill_port}"
            if is_valid_ipv6_address(prefill_address)
            else f"{prefill_address}:{prefill_port}"
        )

        logger.info(
            "vLLMPDReplica rank=%s launched: prefill=%s (engine_id=%s, side_channel=%s:%d), decodes=%d",
            self.replica_rank,
            self._server_address,
            prefill_engine_id,
            prefill_sc.host,
            prefill_sc.port,
            len(self._decode_servers),
        )

    @staticmethod
    def _collect_cuda_devices(worker_infos) -> str:
        return ",".join(worker_info[1] for worker_info in worker_infos)

    def _resolve_mooncake_store_settings(self) -> tuple[bool, dict, Optional[str]]:
        """Resolve whether to attach a Mooncake / Ascend store connector and extras.

        Store can be opted in via ``disaggregation.enable_mooncake_store`` or by
        putting ``MooncakeStoreConnector`` / ``AscendStoreConnector`` /
        ``MooncakeConnectorStoreV1`` in ``engine_kwargs.vllm.kv_transfer_config``
        (the colocated offload recipe). PD always overwrites the top-level
        ``kv_transfer_config`` with a composed MultiConnector, so we harvest
        extras from engine_kwargs instead of forwarding it verbatim.
        """
        disagg = self.config.disaggregation
        enable = bool(disagg.enable_mooncake_store)
        extra = _plain_mapping(disagg.mooncake_store_extra_config)
        path = disagg.mooncake_store_config_path or os.environ.get("MOONCAKE_CONFIG_PATH")

        user_kv = _engine_kwargs_kv_transfer_config(self.config)
        if isinstance(user_kv, dict) and user_kv.get("kv_connector") in _STORE_KV_CONNECTORS:
            enable = True
            extra = {**_plain_mapping(user_kv.get("kv_connector_extra_config")), **extra}
            path = extra.pop("mooncake_config_path", None) or path

        return enable, extra, path

    @staticmethod
    def _build_kv_transfer_config(
        role: str,
        engine_id: str,
        transfer_backend: str,
        mooncake_protocol: Optional[str] = None,
        enable_mooncake_store: bool = False,
        mooncake_store_extra_config: Optional[dict] = None,
        save_decode_cache: bool = False,
        prefill_tp: int = 1,
        decode_tp: int = 1,
        device_name: Optional[str] = None,
        kv_port: Optional[int] = None,
        lookup_rpc_port: Optional[int] = None,
    ) -> dict:
        """Assemble vLLM's ``--kv-transfer-config`` payload.

        GPU: P2P ``MooncakeConnector`` / ``NixlConnector``, optional
        ``MooncakeStoreConnector`` under ``MultiConnector``.

        NPU (or ``transfer_backend=ascend``): P2P ``MooncakeConnectorV1`` plus
        ``AscendStoreConnector(backend=mooncake)``. Heterogeneous TP fills
        ``prefill_tp_size`` / ``decode_tp_size`` (not GPU ``store_tp_size``).
        ``save_decode_cache`` maps to ``consumer_is_to_put`` on the decode store.
        """
        if device_name is None:
            device_name = get_device_name()
        use_ascend = _use_ascend_kv_connectors(device_name, transfer_backend)

        role_to_kv_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }
        kv_role = role_to_kv_role[role]
        if use_ascend:
            p2p_connector = "MooncakeConnectorV1"
        else:
            p2p_connector = {
                "nixl": "NixlConnector",
                "mooncake": "MooncakeConnector",
            }[transfer_backend]

        p2p_cfg: dict[str, Any] = {
            "kv_connector": p2p_connector,
            "kv_role": kv_role,
        }
        if kv_port is not None:
            p2p_cfg["kv_port"] = kv_port
        if use_ascend:
            p2p_cfg["kv_buffer_device"] = device_name
            p2p_cfg["kv_connector_extra_config"] = {
                "prefill": {"dp_size": 1, "tp_size": prefill_tp},
                "decode": {"dp_size": 1, "tp_size": decode_tp},
            }
        elif transfer_backend == "mooncake" and mooncake_protocol:
            p2p_cfg["kv_connector_extra_config"] = {"mooncake_protocol": mooncake_protocol}

        def _with_optional_kv_port(cfg: dict) -> dict:
            if kv_port is not None:
                cfg["kv_port"] = kv_port
            return cfg

        if not enable_mooncake_store:
            cfg = {
                "kv_connector": p2p_connector,
                "kv_role": kv_role,
                "engine_id": engine_id,
                "kv_buffer_device": device_name,
            }
            extra = p2p_cfg.get("kv_connector_extra_config")
            if extra:
                cfg["kv_connector_extra_config"] = extra
            return _with_optional_kv_port(cfg)

        store_extra = dict(mooncake_store_extra_config or {})
        if use_ascend:
            store_extra.setdefault("backend", "mooncake")
            if lookup_rpc_port is not None:
                store_extra["lookup_rpc_port"] = str(lookup_rpc_port)
            if role == "decode" and save_decode_cache:
                store_extra["consumer_is_to_put"] = True
            store_extra = _with_ascend_store_peer_tp(
                store_extra, prefill_tp=prefill_tp, decode_tp=decode_tp
            )
            store_cfg = {
                "kv_connector": "AscendStoreConnector",
                "kv_role": kv_role,
            }
        else:
            store_extra = _with_store_tp_size(store_extra, prefill_tp=prefill_tp, decode_tp=decode_tp)
            if role == "decode" and save_decode_cache:
                store_extra["save_decode_cache"] = True
            store_cfg = {
                "kv_connector": "MooncakeStoreConnector",
                # Prefiller both writes prefix KV and can hit existing entries.
                # Decoder loads from the pool; optional save_decode_cache appends.
                "kv_role": "kv_both" if role == "prefill" else "kv_consumer",
            }
        if store_extra:
            store_cfg["kv_connector_extra_config"] = store_extra

        return _with_optional_kv_port(
            {
                "kv_connector": "MultiConnector",
                "kv_role": kv_role,
                "engine_id": engine_id,
                "kv_buffer_device": device_name,
                "kv_connector_extra_config": {"connectors": [p2p_cfg, store_cfg]},
            }
        )

    def _spawn_pd_server(
        self,
        role: str,
        workers: list[ActorHandle],
        node_id: str,
        cuda_visible_devices: str,
        tp: int,
        kv_transfer_config: dict,
        side_channel_host: str,
        side_channel_port: int,
        mooncake_bootstrap_port: int,
        actor_name: str,
        zmq_base_trainer_rank: int = 0,
        mooncake_store_config_path: Optional[str] = None,
    ) -> ActorHandle:
        """Construct one PD ``vLLMHttpServer`` actor."""
        per_role_config = _dc_replace(self.config, tensor_model_parallel_size=tp)

        env_vars = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "VLLM_NIXL_SIDE_CHANNEL_HOST": side_channel_host,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_port),
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(mooncake_bootstrap_port),
            # Avoid Mooncake TCP port exhaustion under validation concurrency.
            "MC_TCP_ENABLE_CONNECTION_POOL": os.environ.get("MC_TCP_ENABLE_CONNECTION_POOL", "1"),
            "VERL_ZMQ_BASE_TRAINER_RANK": str(zmq_base_trainer_rank),
            "VERL_RAY_JOB_ID": ray.get_runtime_context().get_job_id(),
        }
        if mooncake_store_config_path:
            env_vars["MOONCAKE_CONFIG_PATH"] = mooncake_store_config_path

        return self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
            name=actor_name,
            max_concurrency=self.max_concurrency,
        ).remote(
            config=per_role_config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            gpus_per_node=self.gpus_per_replica_node,
            nnodes=1,
            cuda_visible_devices=cuda_visible_devices,
            disaggregation_role=role,
            disaggregation_kv_transfer_config=kv_transfer_config,
        )
