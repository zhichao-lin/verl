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

"""Prefix-cache reset stays on clear_kv_cache, not on wake_up."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def _server(mode: RolloutMode) -> vLLMHttpServer:
    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.node_rank = 0
    server.rollout_mode = mode
    server.engine = MagicMock()
    server.engine.wake_up = AsyncMock()
    server.engine.reset_prefix_cache = AsyncMock(return_value=True)
    server.engine.reset_mm_cache = AsyncMock()
    server.engine.reset_encoder_cache = AsyncMock()
    server._pd_decode_selector = None
    return server


def test_hybrid_kv_wake_does_not_reset_prefix_cache():
    server = _server(RolloutMode.HYBRID)
    asyncio.run(server.wake_up(tags=["kv_cache"]))
    server.engine.wake_up.assert_awaited_once_with(tags=["kv_cache"])
    server.engine.reset_prefix_cache.assert_not_called()


def test_colocated_wake_does_not_reset_prefix_cache():
    server = _server(RolloutMode.COLOCATED)
    asyncio.run(server.wake_up(tags=["weights"]))
    server.engine.wake_up.assert_awaited_once_with(tags=["kv_cache", "weights"])
    server.engine.reset_prefix_cache.assert_not_called()


def test_clear_kv_cache_resets_prefix_and_auxiliary_caches():
    server = _server(RolloutMode.HYBRID)
    asyncio.run(server.clear_kv_cache())
    server.engine.reset_prefix_cache.assert_awaited_once_with(reset_connector=True)
    server.engine.reset_mm_cache.assert_awaited_once_with()
    server.engine.reset_encoder_cache.assert_awaited_once_with()


def test_clear_kv_cache_raises_when_prefix_reset_fails():
    server = _server(RolloutMode.HYBRID)
    server.engine.reset_prefix_cache.return_value = False
    try:
        asyncio.run(server.clear_kv_cache())
    except RuntimeError as exc:
        assert "prefix cache reset failed" in str(exc)
    else:
        raise AssertionError("clear_kv_cache should raise when reset_prefix_cache returns False")
