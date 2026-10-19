# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""kv resume waits until every rank finishes update_weights, including failures."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

from verl.workers.engine_workers import ActorRolloutRefWorker


class _Rollout:
    def __init__(self, rank: int, events: list, lock: threading.Lock, *, has_server: bool, fail: bool):
        self.rank = rank
        self.events = events
        self.lock = lock
        self.has_server = has_server
        self.fail = fail
        self.sleep_level = 2

    def _record(self, name: str):
        with self.lock:
            self.events.append((name, self.rank))

    async def resume(self, tags):
        self._record(f"resume:{tags[0]}")

    async def update_weights(self, *args, **kwargs):
        if self.has_server:
            self._record("clear_kv_cache")
        # Let the rank without a server reach all_reduce first.
        if self.has_server:
            await asyncio.sleep(0.05)
        if self.fail:
            raise RuntimeError("prefix cache reset failed while scheduler should still be paused")
        self._record("update_weights_done")


def _worker(rollout: _Rollout, *, free_cache_engine: bool) -> ActorRolloutRefWorker:
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.rollout = rollout
    worker.peft_merge = True
    worker.layered_summon = False
    worker.base_sync_done = False
    worker.config = MagicMock()
    worker.config.rollout.free_cache_engine = free_cache_engine
    worker.config.rollout.checkpoint_engine.backend = "naive"
    worker.actor = MagicMock()
    worker.actor.engine.get_per_tensor_param.return_value = ("params", None)
    worker.actor.engine.is_param_offload_enabled = False
    return worker


def _patches():
    return (
        patch("verl.workers.engine_workers.set_expandable_segments"),
        patch("verl.workers.engine_workers.aggressive_empty_cache"),
        patch("verl.workers.engine_workers.log_gpu_memory_usage"),
        patch("verl.workers.engine_workers.get_device_name", return_value="cpu"),
    )


def _run_pair(rollouts: list[_Rollout]):
    errors: list[BaseException | None] = [None, None]

    def _run(index: int):
        worker = _worker(rollouts[index], free_cache_engine=True)
        try:
            asyncio.run(worker.update_weights(global_steps=1, mode="naive"))
        except Exception as exc:
            errors[index] = exc

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    return errors


def _all_reduce_max(values: list, barrier: threading.Barrier, lock: threading.Lock):
    def _all_reduce(tensor, op=None):
        with lock:
            values.append(int(tensor.item()))
        barrier.wait(timeout=5)
        with lock:
            tensor.fill_(max(values))

    return _all_reduce


def test_kv_resume_waits_for_every_rank_update_weights():
    events: list[tuple[str, int]] = []
    lock = threading.Lock()
    rollouts = [
        _Rollout(0, events, lock, has_server=False, fail=False),
        _Rollout(1, events, lock, has_server=True, fail=False),
    ]
    values: list[int] = []
    all_reduce = _all_reduce_max(values, threading.Barrier(2), lock)
    with _patches()[0], _patches()[1], _patches()[2], _patches()[3], patch(
        "torch.distributed.all_reduce", side_effect=all_reduce
    ):
        errors = _run_pair(rollouts)

    assert errors == [None, None]
    assert values == [0, 0]
    assert ("clear_kv_cache", 0) not in events
    assert ("clear_kv_cache", 1) in events
    done = [i for i, (name, _rank) in enumerate(events) if name == "update_weights_done"]
    resumes = [i for i, (name, _rank) in enumerate(events) if name == "resume:kv_cache"]
    assert len(done) == 2 and len(resumes) == 2
    assert max(done) < min(resumes)


def test_free_cache_engine_false_skips_all_reduce_and_kv_resume():
    events: list[tuple[str, int]] = []
    lock = threading.Lock()
    rollout = _Rollout(0, events, lock, has_server=True, fail=False)
    worker = _worker(rollout, free_cache_engine=False)
    with _patches()[0], _patches()[1], _patches()[2], _patches()[3], patch(
        "torch.distributed.all_reduce"
    ) as all_reduce:
        asyncio.run(worker.update_weights(global_steps=1, mode="naive"))
    all_reduce.assert_not_called()
    assert ("resume:kv_cache", 0) not in events
    assert ("update_weights_done", 0) in events


def test_one_rank_failure_skips_kv_resume_on_every_rank():
    events: list[tuple[str, int]] = []
    lock = threading.Lock()
    rollouts = [
        _Rollout(0, events, lock, has_server=True, fail=True),
        _Rollout(1, events, lock, has_server=True, fail=False),
    ]
    values: list[int] = []
    all_reduce = _all_reduce_max(values, threading.Barrier(2), lock)
    with _patches()[0], _patches()[1], _patches()[2], _patches()[3], patch(
        "torch.distributed.all_reduce", side_effect=all_reduce
    ):
        errors = _run_pair(rollouts)

    assert isinstance(errors[0], RuntimeError)
    assert "prefix cache reset failed" in str(errors[0])
    assert isinstance(errors[1], RuntimeError)
    assert "another rank failed before kv resume" in str(errors[1])
    assert max(values) == 1
    assert not any(name == "resume:kv_cache" for name, _rank in events)
