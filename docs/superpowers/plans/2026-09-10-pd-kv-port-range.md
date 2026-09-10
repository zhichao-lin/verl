# vLLM PD `kv_port` 连续端口预留 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NPU MooncakeConnectorV1 按实例预留连续握手端口；未配置 `lookup_rpc_port` 时写入整段 `engine_id`，不再为 IPC 后缀占用 TCP。

**Architecture:** `get_free_port_range` 占住连续 TCP 块，块首作为 `kv_port`。`parse_lookup_rpc_port` 在配置 / replica / builder 共用。`_spawn_pd_server` 删除 `reserved_socks`。

**Tech Stack:** Python sockets、dataclass 校验、pytest CPU。

**Spec:** `docs/superpowers/specs/2026-09-10-pd-kv-port-range-design.md`

工作目录：仓库根 `projects/github/verl`。

---

## File map

| 文件 | 职责 |
|---|---|
| `verl/workers/config/cache_pool.py` | `parse_lookup_rpc_port`；ConnectorConfig 构造期校验；`lookup_rpc_port: Optional[int \| str]` |
| `verl/workers/rollout/vllm_rollout/kv_cache_pool.py` | builder 用 `parse_lookup_rpc_port` |
| `verl/utils/net_utils.py` | `get_free_port_range` |
| `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py` | span / reserve / resolve；launch_servers 预留块；spawn 用 engine_id |
| `verl/trainer/config/rollout/rollout.yaml` | `lookup_rpc_port` 注释 |
| `verl/trainer/config/_generated_*.yaml` | `scripts/generate_trainer_config.sh` |

---

### Task 1: `parse_lookup_rpc_port` + 配置 + builder

**Files:**
- Modify: `verl/workers/config/cache_pool.py`
- Modify: `verl/workers/rollout/vllm_rollout/kv_cache_pool.py`
- Modify: `tests/workers/config/test_cache_pool_config_on_cpu.py`
- Modify: `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`

- [ ] **Step 1: Write failing parse/config tests**

在 `tests/workers/config/test_cache_pool_config_on_cpu.py` 追加：

```python
from verl.workers.config.cache_pool import parse_lookup_rpc_port


def test_parse_lookup_rpc_port_unspecified():
    assert parse_lookup_rpc_port(None) is None
    assert parse_lookup_rpc_port("") is None
    assert parse_lookup_rpc_port("   ") is None


def test_parse_lookup_rpc_port_configured():
    assert parse_lookup_rpc_port(0) == 0
    assert parse_lookup_rpc_port("0") == "0"
    assert parse_lookup_rpc_port("-0") == "-0"
    assert parse_lookup_rpc_port(" 01 ") == "01"
    assert parse_lookup_rpc_port(19001) == 19001
    assert parse_lookup_rpc_port("custom") == "custom"


@pytest.mark.parametrize("value", [-1, "-1", " -2 ", True, False, 1.5])
def test_parse_lookup_rpc_port_rejects_illegal(value):
    with pytest.raises(ValueError, match="lookup_rpc_port"):
        parse_lookup_rpc_port(value)


def test_connector_rejects_negative_lookup_even_when_disabled():
    with pytest.raises(ValueError, match="lookup_rpc_port"):
        KVCachePoolConnectorConfig(lookup_rpc_port=-1)
    with pytest.raises(ValueError, match="lookup_rpc_port"):
        KVCachePoolConfig(enabled=False, connector={"lookup_rpc_port": "-1"})


def test_lookup_rpc_port_string_coercion():
    cfg = KVCachePoolConfig(connector={"lookup_rpc_port": "custom-lookup"})
    assert cfg.connector.lookup_rpc_port == "custom-lookup"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/workers/config/test_cache_pool_config_on_cpu.py::test_parse_lookup_rpc_port_unspecified -v`

Expected: FAIL，`parse_lookup_rpc_port` 未定义。

- [ ] **Step 3: Implement parse + ConnectorConfig**

在 `verl/workers/config/cache_pool.py`：把 `parse_lookup_rpc_port` 加入 `__all__`。`lookup_rpc_port: Optional[int | str] = None`。实现 spec §5.1 的 `parse_lookup_rpc_port`。`KVCachePoolConnectorConfig.__post_init__` 只调用 `parse_lookup_rpc_port(self.lookup_rpc_port)`，不改写字段。

- [ ] **Step 4: Writer builder tests then relax builder**

在 `tests/workers/rollout/test_kv_cache_pool_on_cpu.py` 增加（需 `from verl.workers.rollout.vllm_rollout.kv_cache_pool import build_store_connector_config`）：

```python
def _store_lookup(lookup_rpc_port):
    pool = KVCachePoolConfig(enabled=True)
    return build_store_connector_config(
        role="prefill",
        is_npu=True,
        cache_pool=pool,
        lookup_rpc_port=lookup_rpc_port,
        prefill_tp=4,
        decode_tp=2,
        prefill_tps=[4],
    )["kv_connector_extra_config"]["lookup_rpc_port"]


@pytest.mark.parametrize("value", ["eid-0", 0, "0"])
def test_build_store_accepts_lookup_suffix(value):
    assert _store_lookup(value) == value


@pytest.mark.parametrize("value", [None, "", True, -1, "-1"])
def test_build_store_rejects_illegal_lookup(value):
    with pytest.raises(ValueError, match="lookup_rpc_port"):
        _store_lookup(value)
```

把 `build_store_connector_config` 的 `lookup_rpc_port` 注解改为 `int | str`（必填已解析值）。`build_kv_transfer_config` 仍为 `int | str | None = None`：关 Pool 时继续传 `None`，不要改成不含 `None` 的 `int | str`。

builder 改为：

```python
from verl.workers.config.cache_pool import parse_lookup_rpc_port

parsed = parse_lookup_rpc_port(lookup_rpc_port)
if parsed is None:
    raise ValueError(f"lookup_rpc_port must be a non-negative int or str, got {lookup_rpc_port!r}")
```

然后 `extra = {"lookup_rpc_port": parsed}`。删掉原来的 `<= 0` 正整数检查。

- [ ] **Step 5: Run config + builder tests**

Run:

```bash
pytest tests/workers/config/test_cache_pool_config_on_cpu.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py -k "parse_lookup or lookup_rpc or build_store or multiconnector" -v
```

Expected: PASS（现有 `lookup_rpc_port=1/9/19001` 用例仍绿）。

- [ ] **Step 6: Commit**

```bash
git add verl/workers/config/cache_pool.py verl/workers/rollout/vllm_rollout/kv_cache_pool.py \
  tests/workers/config/test_cache_pool_config_on_cpu.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py
git commit -m "$(cat <<'EOF'
fix: accept non-negative int or str lookup_rpc_port

EOF
)"
```

---

### Task 2: `get_free_port_range`

**Files:**
- Create: `tests/utils/test_net_utils_on_cpu.py`
- Modify: `verl/utils/net_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/utils/test_net_utils_on_cpu.py
# Copyright 2026 Bytedance Ltd. and/or its affiliates（与其它 on_cpu 测试相同 Apache 头）
import socket

import pytest

from verl.utils.net_utils import get_free_port_range


def test_get_free_port_range_count_one_alive():
    port, socks = get_free_port_range("127.0.0.1", 1, with_alive_socks=True)
    try:
        assert socks is not None and len(socks) == 1
        assert socks[0].getsockname()[1] == port
    finally:
        for sock in socks or []:
            sock.close()


def test_get_free_port_range_count_one_closed():
    port, socks = get_free_port_range("127.0.0.1", 1, with_alive_socks=False)
    assert isinstance(port, int) and socks is None


def test_get_free_port_range_consecutive():
    start, socks = get_free_port_range("127.0.0.1", 4, with_alive_socks=True)
    try:
        assert [s.getsockname()[1] for s in socks] == [start, start + 1, start + 2, start + 3]
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OSError):
            blocker.bind(("127.0.0.1", start))
        blocker.close()
    finally:
        for sock in socks:
            sock.close()


def test_get_free_port_range_skips_occupied():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    occupied = holder.getsockname()[1]
    try:
        start, socks = get_free_port_range("127.0.0.1", 2, with_alive_socks=True)
        try:
            ports = [s.getsockname()[1] for s in socks]
            assert occupied not in ports
            assert ports == [start, start + 1]
        finally:
            for sock in socks:
                sock.close()
    finally:
        holder.close()


@pytest.mark.parametrize("count", [0, -1])
def test_get_free_port_range_rejects_bad_count(count):
    with pytest.raises(ValueError):
        get_free_port_range("127.0.0.1", count)
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/utils/test_net_utils_on_cpu.py::test_get_free_port_range_consecutive -v`

Expected: FAIL，函数未定义。

- [ ] **Step 3: Implement `get_free_port_range` in `verl/utils/net_utils.py`**

按 spec §4：`count==1` 调现有 `get_free_port`；`count>1` 用 `bind((address, 0))+SO_REUSEADDR` 保持打开作为块首；`start+1..` **先用不带 `SO_REUSEADDR` 的 socket 探测**，失败整组 close 后重试；探测成功后再用 `SO_REUSEADDR=1` 占住（vLLM 在预留 close 前必须能 bind）。`start+count-1 > 65535` 重试；最多 `max_tries=100`；用尽则 `RuntimeError` 含 `address` 和 `count`。address family 与 `get_free_port` 相同：`is_valid_ipv6_address(address)` 则 `AF_INET6`，否则 `AF_INET`；bind 元组同样是 `(address, port)`。不要改 `get_free_port` 实现。

- [ ] **Step 4: Run tests**

Run: `pytest tests/utils/test_net_utils_on_cpu.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add verl/utils/net_utils.py tests/utils/test_net_utils_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: reserve consecutive TCP ports for PD handshake

EOF
)"
```

---

### Task 3: replica span / reserve / resolve

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py`
- Modify: `tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py`

- [ ] **Step 1: Write failing helper tests**

在 `test_vllm_pd_disaggregation_on_cpu.py` 追加（`pytest.importorskip("vllm")` 后 import replica）：

```python
def test_kv_handshake_port_span():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    assert vLLMPDReplica._kv_handshake_port_span(use_ascend_mooncake_v1=True, tp=8) == 8
    assert vLLMPDReplica._kv_handshake_port_span(use_ascend_mooncake_v1=False, tp=8) == 1
    with pytest.raises(ValueError, match="tp"):
        vLLMPDReplica._kv_handshake_port_span(use_ascend_mooncake_v1=True, tp=0)


def test_reserve_handshake_ports_span_four():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    reserved = []
    start = vLLMPDReplica._reserve_handshake_ports("127.0.0.1", 4, reserved)
    try:
        assert len(reserved) == 4
        assert [s.getsockname()[1] for s in reserved] == [start + i for i in range(4)]
    finally:
        for sock in reserved:
            sock.close()


def test_resolve_lookup_rpc_port():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    resolve = vLLMPDReplica._resolve_lookup_rpc_port
    assert resolve(None, "abc") == "abc"
    assert resolve("", "abc") == "abc"
    assert resolve("   ", "abc") == "abc"
    assert resolve(0, "abc") == 0
    assert resolve("0", "abc") == "0"
    assert resolve(" 01 ", "abc") == "01"
    assert resolve(19001, "abc") == 19001
    assert resolve("custom", "abc") == "custom"
    with pytest.raises(ValueError, match="engine_id"):
        resolve(None, "")
    for bad in (-1, "-1", " -2 ", True, False):
        with pytest.raises(ValueError, match="lookup_rpc_port"):
            resolve(bad, "abc")
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py::test_kv_handshake_port_span -v`

Expected: FAIL，方法不存在。

- [ ] **Step 3: Implement the three staticmethods**

`vllm_pd_replica.py`：

```python
from verl.utils.net_utils import get_free_port_range, is_valid_ipv6_address
from verl.workers.config.cache_pool import parse_lookup_rpc_port
```

（若本任务结束后 spawn 仍用 `get_free_port`，可暂时保留该 import，Task 4 再删。）

实现 spec 的 `_kv_handshake_port_span`、`_reserve_handshake_ports`、`_resolve_lookup_rpc_port`。

- [ ] **Step 4: Run helper tests**

Run: `pytest tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py -k "handshake_port or reserve_handshake or resolve_lookup" -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add verl/workers/rollout/vllm_rollout/vllm_pd_replica.py tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: add PD handshake port span and lookup resolve helpers

EOF
)"
```

---

### Task 4: `launch_servers` + `_spawn_pd_server`

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py`
- Modify: `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`

- [ ] **Step 1: Update spawn tests first（会红）**

`_spawn` 默认 kwargs **删除** `reserved_socks`。返回值改为 `(handle, env_vars, remote_kwargs)`。

- 删除 `test_spawn_auto_lookup_port_reserves_sock`
- `test_spawn_injects_pool_env_and_omits_mooncake_master`：`lookup_rpc_port == "eid-0"`（kwargs 里 `engine_id`）
- 用户端口用例改为：

```python
def test_spawn_uses_user_lookup_rpc_port():
    replica = _make_spawn_replica(cache_pool={"enabled": True, "connector": {"lookup_rpc_port": 19001}})
    _, _, remote_kwargs = _spawn(replica)
    extra = remote_kwargs["disaggregation_kv_transfer_config"]["kv_connector_extra_config"]["connectors"][1][
        "kv_connector_extra_config"
    ]
    assert extra["lookup_rpc_port"] == 19001


def test_spawn_uses_user_lookup_rpc_port_string_and_zero():
    replica = _make_spawn_replica(cache_pool={"enabled": True, "connector": {"lookup_rpc_port": "custom-lookup"}})
    _, _, remote_kwargs = _spawn(replica)
    extra = remote_kwargs["disaggregation_kv_transfer_config"]["kv_connector_extra_config"]["connectors"][1][
        "kv_connector_extra_config"
    ]
    assert extra["lookup_rpc_port"] == "custom-lookup"

    replica0 = _make_spawn_replica(cache_pool={"enabled": True, "connector": {"lookup_rpc_port": 0}})
    _, _, remote_kwargs0 = _spawn(replica0)
    extra0 = remote_kwargs0["disaggregation_kv_transfer_config"]["kv_connector_extra_config"]["connectors"][1][
        "kv_connector_extra_config"
    ]
    assert extra0["lookup_rpc_port"] == 0


def test_spawn_auto_lookup_uses_engine_id():
    replica = _make_spawn_replica()
    _, _, remote_kwargs = _spawn(replica)
    extra = remote_kwargs["disaggregation_kv_transfer_config"]["kv_connector_extra_config"]["connectors"][1][
        "kv_connector_extra_config"
    ]
    assert extra["lookup_rpc_port"] == "eid-0"
```

不要 `patch(get_free_port)`：spawn 单测不走 `launch_servers`，且 Task 4 会从 replica 删除该 import。未配置时 `lookup_rpc_port == engine_id` 即证明没有再占 TCP。

- [ ] **Step 2: Run spawn tests to see fail**

Run: `pytest tests/workers/rollout/test_kv_cache_pool_on_cpu.py -k spawn -v`

Expected: FAIL（仍要求 `reserved_socks`，或把 `0` 当成自动端口）。

- [ ] **Step 3: Wire replica**

`launch_servers`：两处握手 `get_free_port` 换成 `_reserve_handshake_ports`。span **必须**带上 `use_ascend_mooncake_v1`（keyword-only，不能只传 `tp`）：

```python
span_p = self._kv_handshake_port_span(
    use_ascend_mooncake_v1=use_ascend_mooncake_v1, tp=self._prefill_tp
)
prefill_side_channel_port = self._reserve_handshake_ports(
    prefill_host_ip, span_p, reserved_socks
)
# spawn prefill: side_channel_port=prefill_side_channel_port
#                mooncake_bootstrap_port=prefill_side_channel_port

span_d = self._kv_handshake_port_span(
    use_ascend_mooncake_v1=use_ascend_mooncake_v1, tp=self._decode_tp
)
decode_side_channel_port = self._reserve_handshake_ports(
    prefill_host_ip, span_d, reserved_socks
)
# spawn decode: side_channel_port=decode_side_channel_port
#               mooncake_bootstrap_port=prefill_side_channel_port
```

Decode 的 `mooncake_bootstrap_port` **必须仍是 Prefill 块首** `prefill_side_channel_port`，不要改成 Decode 自己的 `decode_side_channel_port`。spawn **不要**传 `reserved_socks`。

`_spawn_pd_server`：删除 `reserved_socks` 参数。Pool 开启时：

```python
if engine_id is None or transfer_backend is None:
    raise ValueError("cache_pool.enabled requires engine_id and transfer_backend")
lookup_port = self._resolve_lookup_rpc_port(pool.connector.lookup_rpc_port, engine_id)
```

然后 `_build_kv_transfer_config(..., lookup_rpc_port=lookup_port, ...)`。

从 `verl.utils.net_utils` 的 import 中删除 `get_free_port`（只留 `get_free_port_range` 与 `is_valid_ipv6_address`）。

- [ ] **Step 4: Run spawn + helper + existing PD kv-config tests**

Run:

```bash
pytest tests/workers/rollout/test_kv_cache_pool_on_cpu.py -k spawn -v
pytest tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py -k "handshake_port or reserve_handshake or resolve_lookup or build_kv_transfer_config" -v
```

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add verl/workers/rollout/vllm_rollout/vllm_pd_replica.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: reserve kv_port ranges and resolve lookup from engine_id

EOF
)"
```

---

### Task 5: Hydra 注释

**Files:**
- Modify: `verl/trainer/config/rollout/rollout.yaml`
- Modify: `verl/trainer/config/_generated_*.yaml`（经脚本）

- [ ] **Step 1: 改注释**

把 `lookup_rpc_port: null` 上方注释改成（key 后空行规则保持）：

```yaml
    # null or empty = this instance engine_id as IPC suffix. Non-negative int or
    # non-empty str used as-is (including 0 / "0"). Negatives and bools are invalid.
    lookup_rpc_port: null
```

- [ ] **Step 2: 生成 reference yaml**

Run: `bash scripts/generate_trainer_config.sh`

Expected: 四个 `_generated_*.yaml` 更新且含新注释或至少 `lookup_rpc_port: null` 仍在。

- [ ] **Step 3: sanity 注释格式（若仓库有）**

Run: `pytest tests/special_sanity/test_config_docs.py -v`

Expected: PASS。若该测试不存在则跳过。

- [ ] **Step 4: Commit**

```bash
git add verl/trainer/config/rollout/rollout.yaml verl/trainer/config/_generated_*.yaml
git commit -m "$(cat <<'EOF'
docs: clarify cache_pool lookup_rpc_port auto engine_id suffix

EOF
)"
```

---

## Spec coverage

| Spec | Task |
|---|---|
| `get_free_port_range`（含 IPv6 family 与 `get_free_port` 相同；顺序端口先无 REUSEADDR 探测） | 2 |
| span / reserve / launch_servers 块首；Decode bootstrap=Prefill 块首 | 3–4 |
| `parse_lookup_rpc_port` + 负整数字符串 + enabled=False 也校验 | 1 |
| builder `int \| str`；`build_kv_transfer_config` 保留 `None` | 1 |
| resolve ← engine_id；spawn 删除 `reserved_socks` | 4 |
| yaml 注释 + generated | 5 |
| 不改 vLLM-Ascend / 不截断 engine_id | 全程不碰 |
