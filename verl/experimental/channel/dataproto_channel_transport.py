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
"""Stage :class:`~verl.protocol.DataProto` tensors for RLinf Channel collective I/O.

Driver-side Channel uses Ray (``put_via_ray`` / ``get_via_ray``); no RLinf
:class:`~verl.third_party.rlinf.scheduler.worker.worker.Worker` context is set,
so these helpers no-op. Inside a worker with ``Worker.current_worker`` set,
``Channel.put`` / ``get`` use send/recv so tensor batches should live on the
accelerator for the CCL path and move back to CPU before local FSDP / agent-loop
compute, matching :class:`~verl.third_party.rlinf.scheduler.collective.collective_group.CollectiveGroup`
routing.
"""

from __future__ import annotations

from typing import Any

from verl.protocol import DataProto


def in_rlinf_channel_worker_context() -> bool:
    """True when RLinf worker send/recv is active for :class:`~verl.third_party.rlinf.scheduler.channel.channel.Channel`."""
    from verl.third_party.rlinf.scheduler.worker.worker import Worker as RLinfWorker

    return RLinfWorker.current_worker is not None


def _channel_accel_device_str() -> str | None:
    from verl.utils.device import get_device_id, get_device_name

    name = get_device_name()
    if name == "cpu":
        return None
    return f"{name}:{get_device_id()}"


def dataproto_to_accel_for_channel_put(batch: DataProto | None) -> DataProto | None:
    """Move tensor batch to the default accelerator before ``Channel.put`` (in-place)."""
    if batch is None or not isinstance(batch, DataProto) or batch.batch is None:
        return batch
    if not in_rlinf_channel_worker_context():
        return batch
    dev = _channel_accel_device_str()
    if dev is None:
        return batch
    return batch.to(dev)


def dataproto_to_cpu_after_channel_get(batch: DataProto | None) -> DataProto | None:
    """Move tensor batch to CPU after ``Channel.get`` / SP broadcast (in-place)."""
    if batch is None or not isinstance(batch, DataProto) or batch.batch is None:
        return batch
    if not in_rlinf_channel_worker_context():
        return batch
    return batch.to("cpu")


def stage_channel_payload_to_cpu_after_get(payload: Any) -> Any:
    """Apply :func:`dataproto_to_cpu_after_channel_get` to a DataProto or ``{\"batch\": DataProto, ...}``."""
    if isinstance(payload, DataProto):
        return dataproto_to_cpu_after_channel_get(payload)
    if isinstance(payload, dict):
        b = payload.get("batch")
        if isinstance(b, DataProto):
            dataproto_to_cpu_after_channel_get(b)
    return payload


def stage_channel_payload_to_accel_before_put(payload: Any) -> Any:
    """Apply :func:`dataproto_to_accel_for_channel_put` to a DataProto or dict with ``batch`` key."""
    if isinstance(payload, DataProto):
        return dataproto_to_accel_for_channel_put(payload)
    if isinstance(payload, dict):
        b = payload.get("batch")
        if isinstance(b, DataProto):
            dataproto_to_accel_for_channel_put(b)
    return payload


__all__ = [
    "dataproto_to_accel_for_channel_put",
    "dataproto_to_cpu_after_channel_get",
    "in_rlinf_channel_worker_context",
    "stage_channel_payload_to_accel_before_put",
    "stage_channel_payload_to_cpu_after_get",
]
