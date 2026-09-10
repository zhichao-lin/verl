# vLLM PD KV Cache Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 vLLM PD 分离路径上，用 MultiConnector 同时使能 P2P KV 传输和 job 级 Mooncake KV Cache Pool（GPU `MooncakeStoreConnector` / NPU `AscendStoreConnector`）。

**Architecture:** `KVCachePoolConfig` 挂在 `RolloutConfig.cache_pool`。`LLMServerManager` 在 replica 启动前按 `get_job_id()` 复用或拉起唯一 `mooncake_master`。`kv_cache_pool.py` 提供 JSON / Store TP / MultiConnector 纯函数；`vLLMPDReplica` 只做平台校验、端口分配和调用；`vLLMHttpServer` 本地落盘 JSON，并按 `connectors[0]` 识别 P2P 名字做 `_pd_dispatch`。

**Tech Stack:** Python dataclasses / Hydra、Ray named actor、vLLM `kv_transfer_config`、pytest CPU 单测。

**Spec:** `docs/superpowers/specs/2026-09-09-kv-cache-pool-design.md`

**本轮锁定修订（实现必须遵守）：**

1. `build_p2p_connector_config` 只产出 child；`build_kv_transfer_config` 在顶层写 `engine_id` / `kv_buffer_device`。关 Pool 把 child 展平到顶层（**仍支持 NIXL 单连接器**）；开 Pool 时 child 进 `connectors[0]`。
2. `store_tp_size` 对 **P 和 D 两个 TP** 都做 `>=` 且整除校验。
3. `MooncakeMasterActor` 用 driver 节点 `NodeAffinitySchedulingStrategy(soft=False)`。
4. GPU `ssd_offload_path` 非空 → `ValueError`；GPU `enable_offload=true` → `NotImplementedError`。在 `_setup_kv_cache_pool` **拉起 master 之前**就报，replica 平台校验再查一次。
5. probe **只在 actor `start`** 做，失败带 cmd / stdout / stderr；actor `__del__` 也 `stop()`；缺二进制按平台提示 `mooncake-transfer-engine`（GPU）**或** `mooncake-transfer-engine-npu`（NPU），不要两条都写；NPU JSON 断言 `preferred_segment` / `prefer_alloc_in_same_node`；补 `kv_load_failure_policy` 测试；`lookup_rpc_port` 在 `_spawn_pd_server` 分配（用户指定同一端口给所有 P/D：**不查冲突**）；开 Pool 时 kv cfg 也在 spawn 内组装；Manager 方法名 `_setup_kv_cache_pool`。
6. 形状测与双源 `engine_kwargs` 测在 `test_kv_cache_pool_on_cpu.py`；`test_vllm_pd_disaggregation_on_cpu.py` 只补 `_pd_dispatch`。

---

## File map

| 文件 | 职责 |
|---|---|
| `verl/workers/config/cache_pool.py` | `KVCachePoolConfig` 及嵌套 dataclass、平台无关校验 |
| `verl/workers/config/rollout.py` | `cache_pool` 字段、coercion、与 PD/name/prefix_caching 交叉校验 |
| `verl/workers/config/__init__.py` | 导出 |
| `verl/trainer/config/rollout/rollout.yaml` | Hydra 默认值（每个 key 上方注释、key 后空行） |
| `verl/workers/rollout/vllm_rollout/kv_cache_pool.py` | JSON、Store TP、MultiConnector 组装、`MooncakeMasterActor` |
| `verl/workers/rollout/llm_server.py` | `_setup_kv_cache_pool` |
| `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py` | 平台校验、调用纯函数、注入 env、`lookup_rpc_port` |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 写 JSON；`_pd_dispatch` 用 `p2p_connector_name` |
| `docs/perf/rollout_kv_offload.md` | PD + cache_pool 用法 |

工作目录：`projects/github/verl`（仓库根）。pytest 都从该根目录跑。

---

### Task 1: `KVCachePoolConfig` 平台无关配置

**Files:**
- Create: `verl/workers/config/cache_pool.py`
- Create: `tests/workers/config/test_cache_pool_config_on_cpu.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/workers/config/test_cache_pool_config_on_cpu.py
import pytest

from verl.workers.config.cache_pool import (
    KVCachePoolConfig,
    KVCachePoolConnectorConfig,
    KVCachePoolMasterConfig,
    KVCachePoolStoreConfig,
)


def test_defaults_disabled_skip_validation():
    cfg = KVCachePoolConfig()
    assert cfg.enabled is False
    assert cfg.backend == "mooncake"
    assert cfg.python_hash_seed == 0
    assert cfg.master.auto_start is True
    assert cfg.store.mode == "embedded"
    assert cfg.connector.lookup_rpc_port is None


def test_nested_dict_coercion():
    cfg = KVCachePoolConfig(
        master={"auto_start": False, "address": "10.0.0.1:50051"},
        store={"global_segment_size": "8GB"},
        connector={"save_decode_cache": True},
    )
    assert isinstance(cfg.master, KVCachePoolMasterConfig)
    assert cfg.master.address == "10.0.0.1:50051"
    assert isinstance(cfg.store, KVCachePoolStoreConfig)
    assert cfg.store.global_segment_size == "8GB"
    assert cfg.connector.save_decode_cache is True


def test_disabled_allows_illegal_mode():
    cfg = KVCachePoolConfig(enabled=False, store={"mode": "standalone-store"})
    assert cfg.enabled is False
    assert cfg.store.mode == "standalone-store"


def test_enabled_rejects_standalone_store():
    with pytest.raises(ValueError, match="embedded"):
        KVCachePoolConfig(enabled=True, store={"mode": "standalone-store"})


def test_enabled_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        KVCachePoolConfig(enabled=True, backend="bogus")


def test_enabled_memcache_not_implemented():
    with pytest.raises(NotImplementedError, match="mooncake"):
        KVCachePoolConfig(enabled=True, backend="memcache")


def test_auto_start_conflicts_with_config_path():
    with pytest.raises(ValueError, match="config_path"):
        KVCachePoolConfig(
            enabled=True,
            master={"auto_start": True},
            store={"config_path": "/tmp/mooncake.json"},
        )


def test_auto_start_false_requires_address_or_config_path():
    with pytest.raises(ValueError, match="address"):
        KVCachePoolConfig(enabled=True, master={"auto_start": False})


def test_auto_start_false_with_config_path_ok():
    KVCachePoolConfig(
        enabled=True,
        master={"auto_start": False},
        store={"config_path": "/tmp/mooncake.json"},
    )


def test_multi_tenants_requires_quota_fields():
    with pytest.raises(ValueError, match="tenant_quota"):
        KVCachePoolConfig(enabled=True, master={"enable_multi_tenants": True})


def test_use_layerwise_rejected_with_mooncake():
    with pytest.raises(ValueError, match="use_layerwise"):
        KVCachePoolConfig(enabled=True, connector={"use_layerwise": True})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/workers/config/test_cache_pool_config_on_cpu.py -v`

Expected: FAIL with `ModuleNotFoundError: verl.workers.config.cache_pool`

- [ ] **Step 3: Write `verl/workers/config/cache_pool.py`**

实现四个 dataclass，版权头与 `disaggregation.py` 相同（2026 Bytedance Apache 2.0）。要点：

- `__all__ = ["KVCachePoolConfig", "KVCachePoolMasterConfig", "KVCachePoolStoreConfig", "KVCachePoolConnectorConfig"]`
- `_ALLOWED_BACKENDS = ("mooncake", "memcache", "yuanrong")`
- `_ALLOWED_STORE_MODES = ("embedded",)`
- 嵌套字段在 `KVCachePoolConfig.__post_init__` 里用 `object.__setattr__` coercion（与 `RoutingPolicyConfig` 相同）
- `enabled=False` 时 **只做 coercion，不做强校验**
- `enabled=True` 时：
  - `backend not in _ALLOWED_BACKENDS` → `ValueError`
  - `backend != "mooncake"` → `NotImplementedError`
  - `store.mode != "embedded"` → `ValueError`（文案含 `embedded`）
  - `master.auto_start and store.config_path` → `ValueError`（文案含 `config_path`）
  - `not master.auto_start and not master.address and not store.config_path` → `ValueError`（文案含 `address`）
  - `master.enable_multi_tenants` 且缺少 `tenant_quota_connector_type` 或 `tenant_quota_connector_uri` → `ValueError`（文案含 `tenant_quota`）
  - `connector.use_layerwise and backend == "mooncake"` → `ValueError`（文案含 `use_layerwise`）
- 字段默认值严格按 spec §3.2–3.5

嵌套 coercion 辅助：

```python
def _coerce(value, cls):
    if isinstance(value, cls):
        return value
    return cls(**dict(value))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/workers/config/test_cache_pool_config_on_cpu.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verl/workers/config/cache_pool.py tests/workers/config/test_cache_pool_config_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: add KVCachePoolConfig for vLLM PD KV cache pool

EOF
)"
```

---

### Task 2: 接到 `RolloutConfig`

**Files:**
- Modify: `verl/workers/config/rollout.py`
- Modify: `verl/workers/config/__init__.py`
- Modify: `tests/workers/config/test_cache_pool_config_on_cpu.py`（追加 RolloutConfig 交叉校验）
- Modify: `tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py` 里若有 `RolloutConfig(...)` 构造，确认新字段有默认值无需改调用方

- [ ] **Step 1: Write failing cross-validation tests**（追加到 `test_cache_pool_config_on_cpu.py`）

```python
from verl.workers.config import RolloutConfig


def test_rollout_cache_pool_default_disabled():
    cfg = RolloutConfig(name="vllm")
    assert cfg.cache_pool.enabled is False


def test_rollout_cache_pool_requires_vllm_pd():
    with pytest.raises(ValueError, match="disaggregation"):
        RolloutConfig(name="vllm", cache_pool={"enabled": True})


def test_rollout_cache_pool_requires_mooncake_transfer():
    with pytest.raises(ValueError, match="mooncake"):
        RolloutConfig(
            name="vllm",
            disaggregation={"enabled": True, "transfer_backend": "nixl"},
            cache_pool={"enabled": True},
        )


def test_rollout_cache_pool_rejects_sglang():
    with pytest.raises(ValueError, match="vllm"):
        RolloutConfig(
            name="sglang",
            disaggregation={"enabled": True, "transfer_backend": "mooncake"},
            cache_pool={"enabled": True},
        )


def test_rollout_cache_pool_requires_prefix_caching():
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        RolloutConfig(
            name="vllm",
            enable_prefix_caching=False,
            disaggregation={"enabled": True, "transfer_backend": "mooncake"},
            cache_pool={"enabled": True},
        )


def test_rollout_cache_pool_vllm_pd_mooncake_ok():
    cfg = RolloutConfig(
        name="vllm",
        disaggregation={"enabled": True, "transfer_backend": "mooncake"},
        cache_pool={"enabled": True},
    )
    assert cfg.cache_pool.enabled is True
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/workers/config/test_cache_pool_config_on_cpu.py::test_rollout_cache_pool_requires_vllm_pd -v`

Expected: FAIL（`RolloutConfig` 没有 `cache_pool` 或未做交叉校验）

- [ ] **Step 3: Wire `rollout.py` and `__init__.py`**

`rollout.py`：

1. `from verl.workers.config.cache_pool import KVCachePoolConfig`
2. `__all__` 增加四个 cache_pool 类型名（或 `from .cache_pool import *` 再并入 `__all__`；推荐显式列出四个名字）
3. `RolloutConfig` 增加 `cache_pool: KVCachePoolConfig = field(default_factory=KVCachePoolConfig)`
4. `__post_init__` 在 `disaggregation` coercion **之后**，用同样模式 coercion `cache_pool`
5. 若 `self.cache_pool.enabled`：
   - `self.name != "vllm"` → `ValueError` 含 `vllm`
   - `not self.disaggregation.enabled` → `ValueError` 含 `disaggregation`
   - `self.disaggregation.transfer_backend != "mooncake"` → `ValueError` 含 `mooncake`
   - `not self.enable_prefix_caching` → `ValueError` 含 `enable_prefix_caching`

`__init__.py`：

```python
from . import cache_pool
from .cache_pool import *  # noqa: F401
```

`__all__` 元组加上 `+ cache_pool.__all__`。

- [ ] **Step 4: Run tests**

Run:

```
pytest tests/workers/config/test_cache_pool_config_on_cpu.py tests/workers/rollout/test_pd_disaggregation.py tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py -v --ignore-glob='*gpu*'
```

Expected: 新增测试 PASS；现有 PD 配置测试 PASS

- [ ] **Step 5: Commit**

```bash
git add verl/workers/config/rollout.py verl/workers/config/__init__.py tests/workers/config/test_cache_pool_config_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: attach KVCachePoolConfig to RolloutConfig

EOF
)"
```

---

### Task 3: Hydra YAML

**Files:**
- Modify: `verl/trainer/config/rollout/rollout.yaml`（在 `disaggregation:` 块之后追加 `cache_pool:`）
- Modify: `verl/trainer/config/_generated_*.yaml`（经脚本）

- [ ] **Step 1: Append `cache_pool` to `rollout.yaml`**

必须满足 `tests/special_sanity/test_config_docs.py`：每个 key 上一行是 `#` 注释，key 行后空行，禁止行内 `#`。

把下面整段贴到文件末尾 `disaggregation` 块之后（`ib_device` 段后面）：

```yaml
# KV Cache Pool via MultiConnector (vLLM PD only).
cache_pool:

  # Master switch. Requires rollout.name=vllm, disaggregation.enabled=true,
  # transfer_backend=mooncake, and enable_prefix_caching=true.
  enabled: False

  # NPU AscendStoreConnector backend. Options: mooncake, memcache, yuanrong.
  # Only mooncake is implemented.
  backend: mooncake

  # Injected as PYTHONHASHSEED on every P/D server.
  python_hash_seed: 0

  # MultiConnector top-level kv_load_failure_policy. null = omit (vLLM default fail).
  kv_load_failure_policy: null

  # Merged into the Store child connector extra after first-class fields.
  extra_config: {}

  # mooncake_master lifecycle.
  master:

    # If true, LLMServerManager starts or reuses one mooncake_master for this driver.
    auto_start: True

    # host:port. Required when auto_start=false and store.config_path is null.
    # Overwritten with the actor bind address when auto_start=true.
    address: null

    # Bind port when auto_start=true. null = pick a free port. Occupied ports fail startup.
    port: null

    # mooncake_master --eviction_high_watermark_ratio
    eviction_high_watermark_ratio: 0.9

    # mooncake_master --eviction_ratio
    eviction_ratio: 0.1

    # mooncake_master --default_kv_lease_ttl. null = omit.
    default_kv_lease_ttl: null

    # mooncake_master --client_ttl. null = omit.
    client_ttl: null

    # mooncake_master --enable_multi_tenants. Requires quota type and uri.
    enable_multi_tenants: False

    # --tenant_quota_connector_type (e.g. file). Required when enable_multi_tenants=true.
    tenant_quota_connector_type: null

    # --tenant_quota_connector_uri. Required when enable_multi_tenants=true.
    tenant_quota_connector_uri: null

  # Fields written into mooncake_config.json.
  store:

    # If set, skip JSON generation and point MOONCAKE_CONFIG_PATH here.
    # Mutually exclusive with master.auto_start=true.
    config_path: null

    # Only embedded is implemented. standalone-store raises.
    mode: embedded

    # null = rdma on GPU, ascend on NPU. Allowed: GPU rdma|tcp, NPU ascend.
    protocol: null

    # Mooncake metadata_server.
    metadata_server: P2PHANDSHAKE

    # Per-rank CPU segment. NPU must be 1GB-aligned.
    global_segment_size: 4GB

    # GPU JSON only.
    local_buffer_size: 4GB

    # Independent from disaggregation.ib_device. NPU must stay empty.
    device_name: ""

    # Mooncake tenant. Non-default needs Mooncake >= 0.3.12.
    tenant_id: default

    # GPU JSON enable_offload. True raises NotImplementedError on GPU. Ignored on NPU.
    enable_offload: False

    # NPU SSD directory. Non-null on GPU raises. Non-null on NPU enables SSD.
    ssd_offload_path: null

    # NPU JSON only. Non-default on GPU raises.
    preferred_segment: False

    # NPU JSON only. Non-default on GPU raises.
    prefer_alloc_in_same_node: True

  # Store child kv_connector_extra_config first-class fields.
  connector:

    # null = omit (GPU default true, NPU default false in vLLM).
    load_async: null

    # null or 0 = auto unique port per P/D instance. Positive int used as-is.
    lookup_rpc_port: null

    # GPU. Written only when true.
    lookup_async: False

    # GPU. Written only when non-empty.
    cache_prefix: ""

    # GPU decode. Written only when true.
    save_decode_cache: False

    # GPU. null = lcm(prefill_tp, decode_tp). Always written after resolve.
    store_tp_size: null

    # GPU. null = omit unless multi-P auto-lcm fires.
    enable_store_tp_lcm: null

    # GPU. null = omit unless multi-P auto-lcm fires.
    prefill_tp_sizes: null

    # NPU. Written only when true.
    consumer_is_to_put: False

    # NPU. Written only when true.
    consumer_is_to_load: False

    # NPU memcache-only. Rejected with backend=mooncake.
    use_layerwise: False

    # NPU PP. Written only when non-null. PD still rejects PP>1.
    prefill_pp_size: null

    # NPU PP partition string. Written only when non-null.
    prefill_pp_layer_partition: null
```

空字符串 `device_name: ""` 和 `cache_prefix: ""` 按 Hydra 惯例写。若 `test_config_docs` 对 `""` 过敏，改成不带引号的空并在实现里接受。

- [ ] **Step 2: Check yaml comment format**

Run: `pytest tests/special_sanity/test_config_docs.py -v`

Expected: PASS

- [ ] **Step 3: Regenerate flattened trainer yaml**

Run: `bash scripts/generate_trainer_config.sh`

Expected: 生成四个 `_generated_*.yaml` 且脚本结尾 `All good`。若脚本因「未提交的 diff」以 exit 1 结束，这是它的自检：先确认 `cache_pool` 已出现在生成文件里，下一步一起 commit。

- [ ] **Step 4: Commit**

```bash
git add verl/trainer/config/rollout/rollout.yaml verl/trainer/config/_generated_*.yaml
git commit -m "$(cat <<'EOF'
feat: add rollout.cache_pool Hydra schema

EOF
)"
```

---

### Task 4: `kv_cache_pool.py` 纯函数（JSON / TP / MultiConnector）

**Files:**
- Create: `verl/workers/rollout/vllm_rollout/kv_cache_pool.py`
- Create: `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`

本任务 **不** 实现 Ray actor / `Popen`。只实现可单测的纯函数。

锁定签名（后续任务必须同名）：

```python
def parse_size_to_bytes(value: str | int) -> int: ...

def resolve_decode_tp(prefill_tp: int, decode_tensor_model_parallel_size: int | None) -> int: ...

def resolve_store_tp(
    *,
    prefill_tp: int,
    decode_tp: int,
    prefill_tps: list[int],
    user_store_tp_size: int | None,
    enable_store_tp_lcm: bool | None,
) -> tuple[int, dict]:
    """Return (store_tp_size, extra_fields_to_merge). extra may contain enable_store_tp_lcm/prefill_tp_sizes."""

def mooncake_json_path(job_id: str) -> str: ...

def build_mooncake_json(*, store: KVCachePoolStoreConfig, master_address: str, is_npu: bool) -> dict: ...

def build_p2p_connector_config(
    *,
    role: str,
    transfer_backend: str,
    mooncake_protocol: str | None,
    use_ascend_mooncake_v1: bool,
    kv_port: int | None,
    prefill_tp: int | None,
    decode_tp: int | None,
) -> dict: ...

def build_store_connector_config(
    *,
    role: str,
    is_npu: bool,
    cache_pool: KVCachePoolConfig,
    lookup_rpc_port: int,
    prefill_tp: int,
    decode_tp: int,
    prefill_tps: list[int],
) -> dict: ...

def build_kv_transfer_config(
    *,
    role: str,
    engine_id: str,
    kv_buffer_device: str,
    transfer_backend: str,
    mooncake_protocol: str | None,
    use_ascend_mooncake_v1: bool,
    kv_port: int | None,
    prefill_tp: int | None,
    decode_tp: int | None,
    cache_pool: KVCachePoolConfig | None = None,
    lookup_rpc_port: int | None = None,
    prefill_tps: list[int] | None = None,
) -> dict: ...

def p2p_connector_name(kv_transfer_config: dict) -> str: ...

def validate_platform_cache_pool(*, cache_pool: KVCachePoolConfig, is_npu: bool) -> None: ...
```

- [ ] **Step 1: Write failing tests in `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`**

至少包含：

```python
import math
from pathlib import Path

import pytest

from verl.workers.config.cache_pool import KVCachePoolConfig
from verl.workers.rollout.vllm_rollout.kv_cache_pool import (
    build_kv_transfer_config,
    build_mooncake_json,
    mooncake_json_path,
    p2p_connector_name,
    parse_size_to_bytes,
    resolve_decode_tp,
    resolve_store_tp,
    validate_platform_cache_pool,
)


def test_parse_size_to_bytes():
    assert parse_size_to_bytes(1073741824) == 1073741824
    assert parse_size_to_bytes("1GB") == 2**30
    assert parse_size_to_bytes("1024MB") == 2**30
    assert parse_size_to_bytes("4GB") == 4 * 2**30


def test_resolve_decode_tp():
    assert resolve_decode_tp(4, None) == 4
    assert resolve_decode_tp(4, 2) == 2


def test_resolve_store_tp_lcm_of_p_and_d():
    size, extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4], user_store_tp_size=None, enable_store_tp_lcm=None
    )
    assert size == 4
    assert extra == {}


def test_resolve_store_tp_user_override():
    size, extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4], user_store_tp_size=8, enable_store_tp_lcm=None
    )
    assert size == 8
    assert extra == {}


def test_resolve_store_tp_multi_p_auto_lcm():
    size, extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4, 2], user_store_tp_size=None, enable_store_tp_lcm=None
    )
    assert size == 4
    assert extra["enable_store_tp_lcm"] is True
    assert extra["prefill_tp_sizes"] == [4, 2]


def test_resolve_store_tp_multi_p_explicit_false():
    size, extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4, 2], user_store_tp_size=None, enable_store_tp_lcm=False
    )
    assert size == 4
    assert extra == {}


def test_gpu_json_keys():
    cfg = KVCachePoolConfig()
    data = build_mooncake_json(store=cfg.store, master_address="10.0.0.1:50051", is_npu=False)
    assert set(data) == {
        "mode",
        "metadata_server",
        "master_server_address",
        "global_segment_size",
        "local_buffer_size",
        "protocol",
        "device_name",
        "enable_offload",
        "tenant_id",
    }
    assert data["protocol"] == "rdma"
    assert data["enable_offload"] is False
    assert "ssd_offload_path" not in data


def test_gpu_json_enable_offload_hardcoded_false():
    cfg = KVCachePoolConfig(store={"enable_offload": True})
    data = build_mooncake_json(store=cfg.store, master_address="10.0.0.1:50051", is_npu=False)
    assert data["enable_offload"] is False


def test_npu_json_ssd_on():
    cfg = KVCachePoolConfig(store={"ssd_offload_path": "/nvme/mooncake_offload"})
    data = build_mooncake_json(store=cfg.store, master_address="10.0.0.1:50051", is_npu=True)
    assert data["protocol"] == "ascend"
    assert data["enable_ssd_offload"] is True
    assert data["ssd_offload_path"] == "/nvme/mooncake_offload"
    assert data["preferred_segment"] is False
    assert data["prefer_alloc_in_same_node"] is True
    assert "mode" not in data
    assert "local_buffer_size" not in data
    assert "enable_offload" not in data


def test_npu_json_ssd_off_omits_path():
    cfg = KVCachePoolConfig()
    data = build_mooncake_json(store=cfg.store, master_address="10.0.0.1:50051", is_npu=True)
    assert data["enable_ssd_offload"] is False
    assert "ssd_offload_path" not in data
    assert data["preferred_segment"] is False
    assert data["prefer_alloc_in_same_node"] is True


def test_platform_gpu_rejects_ssd_path():
    cfg = KVCachePoolConfig(store={"ssd_offload_path": "/nvme/x"})
    with pytest.raises(ValueError, match="ssd_offload_path"):
        validate_platform_cache_pool(cache_pool=cfg, is_npu=False)


def test_platform_gpu_rejects_enable_offload():
    cfg = KVCachePoolConfig(store={"enable_offload": True})
    with pytest.raises(NotImplementedError, match="offload"):
        validate_platform_cache_pool(cache_pool=cfg, is_npu=False)


def test_platform_npu_rejects_unaligned_segment():
    cfg = KVCachePoolConfig(store={"global_segment_size": "512MB"})
    with pytest.raises(ValueError, match="1GB"):
        validate_platform_cache_pool(cache_pool=cfg, is_npu=True)


def test_platform_gpu_rejects_npu_store_field():
    cfg = KVCachePoolConfig(store={"preferred_segment": True})
    with pytest.raises(ValueError, match="preferred_segment"):
        validate_platform_cache_pool(cache_pool=cfg, is_npu=False)


def test_gpu_multiconnector_prefill():
    pool = KVCachePoolConfig(enabled=True)
    cfg = build_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake_v1=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=2,
        cache_pool=pool,
        lookup_rpc_port=19001,
        prefill_tps=[4],
    )
    assert cfg["kv_connector"] == "MultiConnector"
    assert cfg["kv_role"] == "kv_producer"
    assert cfg["engine_id"] == "e0"
    p2p, store = cfg["kv_connector_extra_config"]["connectors"]
    assert p2p["kv_connector"] == "MooncakeConnector"
    assert p2p["kv_role"] == "kv_producer"
    assert p2p["kv_connector_extra_config"] == {"mooncake_protocol": "nvlink"}
    assert "engine_id" not in p2p
    assert store["kv_connector"] == "MooncakeStoreConnector"
    assert store["kv_role"] == "kv_both"
    assert store["kv_connector_extra_config"]["lookup_rpc_port"] == 19001
    assert store["kv_connector_extra_config"]["store_tp_size"] == 4
    assert "save_decode_cache" not in store["kv_connector_extra_config"]


def test_gpu_multiconnector_decode_save_decode_cache():
    pool = KVCachePoolConfig(enabled=True, connector={"save_decode_cache": True})
    cfg = build_kv_transfer_config(
        role="decode",
        engine_id="e1",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake_v1=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=2,
        cache_pool=pool,
        lookup_rpc_port=19002,
        prefill_tps=[4],
    )
    store = cfg["kv_connector_extra_config"]["connectors"][1]
    assert store["kv_role"] == "kv_consumer"
    assert store["kv_connector_extra_config"]["save_decode_cache"] is True


def test_npu_multiconnector():
    pool = KVCachePoolConfig(enabled=True)
    cfg = build_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="npu",
        transfer_backend="mooncake",
        mooncake_protocol=None,
        use_ascend_mooncake_v1=True,
        kv_port=20001,
        prefill_tp=4,
        decode_tp=2,
        cache_pool=pool,
        lookup_rpc_port=1,
        prefill_tps=[4],
    )
    p2p, store = cfg["kv_connector_extra_config"]["connectors"]
    assert p2p["kv_connector"] == "MooncakeConnectorV1"
    assert p2p["kv_port"] == 20001
    assert "kv_port" not in cfg
    assert store["kv_connector"] == "AscendStoreConnector"
    assert store["kv_role"] == "kv_producer"
    assert store["kv_connector_extra_config"] == {"lookup_rpc_port": 1}
    assert "store_tp_size" not in store["kv_connector_extra_config"]


def test_pool_disabled_keeps_single_p2p():
    cfg = build_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake_v1=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=KVCachePoolConfig(enabled=False),
    )
    assert cfg["kv_connector"] == "MooncakeConnector"
    assert cfg["engine_id"] == "e0"
    assert cfg["kv_buffer_device"] == "cuda"
    assert "connectors" not in cfg.get("kv_connector_extra_config", {})


def test_pool_disabled_nixl_single_connector():
    cfg = build_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="nixl",
        mooncake_protocol=None,
        use_ascend_mooncake_v1=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=None,
    )
    assert cfg["kv_connector"] == "NixlConnector"
    assert cfg["engine_id"] == "e0"
    assert cfg["kv_buffer_device"] == "cuda"
    assert "connectors" not in cfg


def test_kv_load_failure_policy_written():
    pool = KVCachePoolConfig(enabled=True, kv_load_failure_policy="recompute")
    cfg = build_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake_v1=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=pool,
        lookup_rpc_port=9,
        prefill_tps=[4],
    )
    assert cfg["kv_load_failure_policy"] == "recompute"
    assert "kv_load_failure_policy" not in cfg["kv_connector_extra_config"]["connectors"][0]


def test_kv_load_failure_policy_omitted_by_default():
    pool = KVCachePoolConfig(enabled=True)
    cfg = build_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake_v1=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=pool,
        lookup_rpc_port=9,
        prefill_tps=[4],
    )
    assert "kv_load_failure_policy" not in cfg


def test_p2p_connector_name_multiconnector():
    name = p2p_connector_name(
        {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {"connectors": [{"kv_connector": "MooncakeConnector"}]},
        }
    )
    assert name == "MooncakeConnector"


def test_p2p_connector_name_missing_child():
    with pytest.raises(RuntimeError, match="connectors"):
        p2p_connector_name({"kv_connector": "MultiConnector", "kv_connector_extra_config": {}})


def test_extra_config_overrides():
    pool = KVCachePoolConfig(enabled=True, extra_config={"cache_prefix": "expA", "load_async": False})
    cfg = build_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake_v1=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=pool,
        lookup_rpc_port=9,
        prefill_tps=[4],
    )
    extra = cfg["kv_connector_extra_config"]["connectors"][1]["kv_connector_extra_config"]
    assert extra["cache_prefix"] == "expA"
    assert extra["load_async"] is False
```

`store_tp_size` 非法（例如 user 设 3、prefill_tp=4）应 `ValueError`。补两条：对 **prefill_tp 和 decode_tp 都** 做（构建时传入两端 TP）。`local_tp` 取 `max` 不够；应对每个 TP：`size >= tp and size % tp == 0`。

```python
def test_store_tp_rejects_not_divisible():
    with pytest.raises(ValueError, match="store_tp_size"):
        resolve_store_tp(
            prefill_tp=4, decode_tp=2, prefill_tps=[4], user_store_tp_size=3, enable_store_tp_lcm=None
        )


def test_store_tp_rejects_decode_tp():
    with pytest.raises(ValueError, match="store_tp_size"):
        resolve_store_tp(
            prefill_tp=4, decode_tp=3, prefill_tps=[4], user_store_tp_size=4, enable_store_tp_lcm=None
        )
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/workers/rollout/test_kv_cache_pool_on_cpu.py -v`

Expected: FAIL import

- [ ] **Step 3: Implement the pure functions in `kv_cache_pool.py`**

实现要点（必须与 spec 一致）：

- `parse_size_to_bytes`：`int` 原样；字符串去空格，支持 `B/KB/MB/GB`（1024 进制）及纯数字。
- `resolve_store_tp`：
  1. `store_tp = user or lcm(prefill_tp, decode_tp)`；多 P 且 `len(set(prefill_tps))>1` 且 `enable_store_tp_lcm is not False` 时，若 user 为 None，改用 `lcm(*prefill_tps, decode_tp)`。
  2. 对 `prefill_tp` 与 `decode_tp` 均校验 `store_tp >= tp and store_tp % tp == 0`。
  3. extra：多 P 自动分支写入 `enable_store_tp_lcm=True` 和 `prefill_tp_sizes=prefill_tps`；用户 `enable_store_tp_lcm=False` 则 extra 为空；用户显式 `True` 也要写入（非 None）。
  4. 用户显式 `store_tp_size` 覆盖自动 lcm 数值，但仍做步骤 2。
- `math.lcm`：3.9 可用 `math.lcm`；对多于 2 个参数用 `functools.reduce(math.lcm, values)`。
- `validate_platform_cache_pool`：
  - GPU：`protocol in (None, "rdma", "tcp")`；`ssd_offload_path` 非空 → `ValueError`；`enable_offload is True` → `NotImplementedError`（文案含 `offload`）；`preferred_segment is True` 或 `prefer_alloc_in_same_node is False` 报错；connector 的 NPU 专用字段非默认报错（`consumer_is_to_put/load`、`use_layerwise`、`prefill_pp_size`、`prefill_pp_layer_partition`）。
  - NPU：`protocol in (None, "ascend")`；`device_name == ""`；segment `% 2**30 == 0`（文案含 `1GB`）；connector GPU 专用非默认报错（`lookup_async`、`cache_prefix`、`save_decode_cache`、`store_tp_size`、`enable_store_tp_lcm`、`prefill_tp_sizes`）。忽略 `enable_offload`。
  - `backend != "mooncake"` 在 GPU 上也报错（默认以外）。
- `build_mooncake_json`：GPU `enable_offload` **硬编码 `false`**，不读 `store.enable_offload`。NPU 始终写出 `preferred_segment` 与 `prefer_alloc_in_same_node`。
- `build_p2p_connector_config`：**只产出 child**（无 `engine_id` / `kv_buffer_device`）。支持 GPU Mooncake / NIXL / NPU V1。NIXL 仅关 Pool 路径使用。
- `build_store_connector_config`：lookup_rpc_port 必须是正整数，否则 `ValueError`。GPU extra 始终有 `lookup_rpc_port` 和 `store_tp_size`；其它按 spec 3.5。然后 `extra.update(cache_pool.extra_config)`。
- `build_kv_transfer_config`：始终把 `engine_id`、`kv_buffer_device` 写在**顶层**。`cache_pool is None or not cache_pool.enabled` → `{**p2p, "engine_id": ..., "kv_buffer_device": ...}`（关 Pool 仍支持 NIXL 单连接器）。enabled → MultiConnector，`connectors=[p2p, store]`，`kv_load_failure_policy` 非 None 才写到顶层。
- `p2p_connector_name`：顶层不是 MultiConnector 则返回顶层名；是则取 `connectors[0].kv_connector`，缺则 `RuntimeError` 文案含 `connectors`。
- `mooncake_json_path`：`str(Path(tempfile.gettempdir()) / f"verl_mooncake_{job_id}.json")`

- [ ] **Step 4: Run tests**

Run: `pytest tests/workers/rollout/test_kv_cache_pool_on_cpu.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verl/workers/rollout/vllm_rollout/kv_cache_pool.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: assemble MultiConnector and Mooncake JSON for KV cache pool

EOF
)"
```

---

### Task 5: `MooncakeMasterActor` 进程封装

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/kv_cache_pool.py`
- Modify: `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`

把进程生命周期做成 **可 mock 的函数**，actor 只是薄包装，避免单测真起 `mooncake_master`。

锁定：

```python
MOONCAKE_MASTER_ACTOR_PREFIX = "verl_mooncake_master_"

def mooncake_master_actor_name(job_id: str) -> str:
    return f"{MOONCAKE_MASTER_ACTOR_PREFIX}{job_id}"

def build_mooncake_master_cmd(*, master: KVCachePoolMasterConfig, port: int, enable_offload: bool) -> list[str]: ...

def probe_tcp(host: str, port: int, timeout_s: float = 30.0, interval_s: float = 0.5) -> None: ...

def mooncake_master_install_hint(*, is_npu: bool) -> str:
    return "mooncake-transfer-engine-npu" if is_npu else "mooncake-transfer-engine"

class MooncakeMasterProcess:
    def start(self, cmd: list[str]) -> None: ...  # Popen only; do NOT probe
    def stop(self) -> None: ...  # idempotent: terminate, wait 5s, kill

@ray.remote(num_cpus=0)
class MooncakeMasterActor:
    def start(self, master_cfg: dict, enable_offload: bool) -> str: ...  # returns host:port
    def get_address(self) -> str: ...
    def stop(self) -> None: ...
    def __del__(self) -> None: ...  # also call stop()
```

- [ ] **Step 1: Tests for cmd / stop / probe**

```python
from unittest.mock import MagicMock, patch

from verl.workers.config.cache_pool import KVCachePoolMasterConfig
from verl.workers.rollout.vllm_rollout.kv_cache_pool import (
    MooncakeMasterActor,
    MooncakeMasterProcess,
    build_mooncake_master_cmd,
    mooncake_master_actor_name,
    mooncake_master_install_hint,
)


def test_actor_name():
    assert mooncake_master_actor_name("abc") == "verl_mooncake_master_abc"


def test_master_cmd_minimal():
    cmd = build_mooncake_master_cmd(
        master=KVCachePoolMasterConfig(), port=50051, enable_offload=False
    )
    assert cmd[:3] == ["mooncake_master", "--port", "50051"]
    assert "--enable_offload=true" not in cmd
    assert "--eviction_high_watermark_ratio" in cmd


def test_master_cmd_ssd_and_tenants():
    master = KVCachePoolMasterConfig(
        enable_multi_tenants=True,
        tenant_quota_connector_type="file",
        tenant_quota_connector_uri="/etc/mooncake/tenant_quotas.yaml",
        client_ttl=120,
    )
    cmd = build_mooncake_master_cmd(master=master, port=50088, enable_offload=True)
    assert "--enable_offload=true" in cmd
    assert "--enable_multi_tenants=true" in cmd
    assert "--client_ttl" in cmd


def test_process_stop_idempotent():
    proc = MooncakeMasterProcess()
    fake = MagicMock()
    fake.poll.return_value = None
    proc._proc = fake
    proc.stop()
    fake.terminate.assert_called_once()
    proc.stop()  # second call must not raise
```

`probe_tcp` 用 `socket.create_connection`；测试里 patch 成第一次失败第二次成功，确认循环。超时测试可用 `timeout_s=0.01, interval_s=0.01` 全失败后 `RuntimeError`。`probe_tcp` 本身不必带 cmd；**把 cmd/stdout/stderr 拼进异常的是 `MooncakeMasterActor.start`**。

`MooncakeMasterProcess.start`：只 `subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True)`。找不到二进制时把 `FileNotFoundError` 原样抛出（或包一层不带包名的错误）。**不调用 `probe_tcp`。**

`MooncakeMasterActor.start` 捕获 `FileNotFoundError`，用 `is_torch_npu_available(check_device=False)` 选 **一个** 包名：`mooncake-transfer-engine`（GPU）或 `mooncake-transfer-engine-npu`（NPU）。不要两条都写。

补测：

```python
def test_install_hint_platform_specific():
    assert "npu" not in mooncake_master_install_hint(is_npu=False)
    assert mooncake_master_install_hint(is_npu=False) == "mooncake-transfer-engine"
    assert mooncake_master_install_hint(is_npu=True) == "mooncake-transfer-engine-npu"


def test_missing_binary_uses_gpu_hint(monkeypatch):
    monkeypatch.setattr(
        "verl.workers.rollout.vllm_rollout.kv_cache_pool.is_torch_npu_available",
        lambda check_device=False: False,
    )
    actor = MooncakeMasterActor.__new__(MooncakeMasterActor)
    actor._proc = MooncakeMasterProcess()
    with patch("subprocess.Popen", side_effect=FileNotFoundError("mooncake_master")):
        with pytest.raises(RuntimeError, match="mooncake-transfer-engine") as ei:
            actor.start({"port": 1}, enable_offload=False)
    assert "mooncake-transfer-engine-npu" not in str(ei.value)


def test_probe_timeout_includes_cmd_and_output():
    # probe 只在 actor.start 调用，不要在 MooncakeMasterProcess.start 里 patch probe_tcp
    actor = MooncakeMasterActor.__new__(MooncakeMasterActor)
    fake = MagicMock()
    fake.poll.return_value = None
    fake.stdout = MagicMock()
    with patch("subprocess.Popen", return_value=fake), patch(
        "verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp",
        side_effect=RuntimeError("timeout"),
    ):
        with pytest.raises(RuntimeError, match="mooncake_master"):
            actor.start({"port": 50051}, enable_offload=False)


def test_process_start_does_not_probe():
    proc = MooncakeMasterProcess()
    fake = MagicMock()
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake) as popen, patch(
        "verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp"
    ) as probe:
        proc.start(["mooncake_master", "--port", "1"])
    popen.assert_called_once()
    probe.assert_not_called()


def test_actor_del_calls_stop():
    actor = MooncakeMasterActor.__new__(MooncakeMasterActor)
    actor._proc = MagicMock()
    actor._proc.poll.return_value = None
    MooncakeMasterActor.__del__(actor)
```

（actor 单测按实际 `__init__` / process 封装微调，但 **probe 不得出现在 process.start**。）

- [ ] **Step 2: Run to fail, then implement, then pass**

`MooncakeMasterActor.start`：

1. `port = master.port or get_free_port(ray.util.get_node_ip_address())[0]`（看 `get_free_port` 返回值：现有代码是 `(port, sock)`，记得 close sock）。
2. 用户指定 port：不要换端口。
3. `MooncakeMasterProcess.start(cmd)`。缺二进制 → 按平台包名包装 `RuntimeError`。
4. **在 actor 里** `probe_tcp(ip, port)`。失败则 `RuntimeError` 带 cmd 与已捕获的 stdout/stderr。
5. 保存 `f"{ip}:{port}"` 并返回。
6. `enable_offload` 由调用方传入（**仅 NPU SSD**：`ssd_offload_path` 非空为 True，否则 False）。GPU 路径不会走到 True（setup 已提前拒绝）。
7. `__del__` 调 `stop()`；`stop()` 幂等。

- [ ] **Step 3: Commit**

```bash
git add verl/workers/rollout/vllm_rollout/kv_cache_pool.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: add mooncake_master process wrapper and named actor

EOF
)"
```

---

### Task 6: `LLMServerManager._setup_kv_cache_pool`

**Files:**
- Modify: `verl/workers/rollout/llm_server.py`
- Modify: `verl/workers/rollout/vllm_rollout/kv_cache_pool.py`（增加模块级 `setup_kv_cache_pool(rollout_config: DictConfig) -> None` 便于单测；Manager 方法 `_setup_kv_cache_pool` 只转调它）
- Modify: `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`

`setup_kv_cache_pool` 逻辑（`LLMServerManager._setup_kv_cache_pool` 在 `create()` 里 `_initialize_llm_servers` **之前**调用）：

```python
def setup_kv_cache_pool(rollout_cfg) -> None:
    pool = rollout_cfg.get("cache_pool")
    if pool is None or not pool.get("enabled", False):
        return
    from verl.utils.device import is_torch_npu_available

    # Fail GPU offload before starting master.
    if not is_torch_npu_available(check_device=False):
        store = pool.get("store") or {}
        if store.get("ssd_offload_path"):
            raise ValueError("ssd_offload_path")
        if store.get("enable_offload"):
            raise NotImplementedError("offload")

    import atexit
    import ray
    from omegaconf import OmegaConf
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    auto_start = pool.master.auto_start
    if not auto_start:
        return
    job_id = ray.get_runtime_context().get_job_id()
    name = mooncake_master_actor_name(job_id)
    try:
        actor = ray.get_actor(name)
        address = ray.get(actor.get_address.remote())
    except ValueError:
        enable_offload = bool(pool.store.get("ssd_offload_path"))  # NPU SSD only; GPU already rejected
        driver_node = ray.get_runtime_context().get_node_id()
        actor = MooncakeMasterActor.options(
            name=name,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node, soft=False),
        ).remote()
        address = ray.get(actor.start.remote(OmegaConf.to_container(pool.master, resolve=True), enable_offload))
    OmegaConf.update(rollout_cfg, "cache_pool.master.address", address, force=True)
    atexit.register(lambda: ray.get(actor.stop.remote()))
```

`get_actor` 找不到 actor 时的异常类型以本仓库 Ray 版本为准（可能是 `ValueError` 或 `ray.exceptions.ActorNotFound`）。测试里 patch `ray.get_actor`。

- [ ] **Step 1: Unit test reuse vs create**（mock ray）

第二次调用 `setup_kv_cache_pool`：`get_actor` 命中 → 不得 `MooncakeMasterActor.options`。第一次 miss → 会 `remote()` + `start.remote`。断言 `OmegaConf.update` 写入 `cache_pool.master.address`。创建 actor 时 `options` 必须带 `NodeAffinitySchedulingStrategy(..., soft=False)`。

`enabled=False` → 函数立即 return，不碰 ray。

GPU 提前拒绝（patch `is_torch_npu_available` 为 `False`）：`enable_offload=True` → `NotImplementedError`；`ssd_offload_path` 非空 → `ValueError`。这两条都不得调用 `get_actor` / `MooncakeMasterActor.options`。

- [ ] **Step 2: Implement `setup_kv_cache_pool` and `LLMServerManager._setup_kv_cache_pool`**

```python
def _setup_kv_cache_pool(self):
    from verl.workers.rollout.vllm_rollout.kv_cache_pool import setup_kv_cache_pool

    setup_kv_cache_pool(self.rollout_config)


@classmethod
@auto_await
async def create(cls, *args, **kwargs):
    instance = cls(*args, **kwargs)
    instance._setup_kv_cache_pool()
    await instance._initialize_llm_servers()
    await instance._init_global_load_balancer()
    return instance
```

`rollout_config` 是 Hydra `DictConfig`（`config.actor_rollout_ref.rollout`），不是 frozen dataclass。只改这个对象上的 `cache_pool.master.address`。

- [ ] **Step 3: Tests pass, commit**

```bash
git add verl/workers/rollout/llm_server.py verl/workers/rollout/vllm_rollout/kv_cache_pool.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: start or reuse mooncake_master before vLLM PD replicas

EOF
)"
```

---

### Task 7: `vLLMPDReplica` 接入

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py`
- Modify: `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`（双源检查；不要改 PD 测试文件的形状/双源断言）
- Modify: `tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py` 仅当现有 `_build_kv_transfer_config` 测试需要适配新签名默认值

- [ ] **Step 1: Keep existing `_build_kv_transfer_config` tests green by delegating**

把 `vLLMPDReplica._build_kv_transfer_config` 改为调用 `kv_cache_pool.build_kv_transfer_config`，**保留原关键字参数**，并增加可选 `cache_pool` / `lookup_rpc_port` / `prefill_tps`。`cache_pool` 默认 `None` ⇒ 单 P2P，现有 CPU 测试不改行为。

`kv_buffer_device` 用现有 `get_device_name()`。

- [ ] **Step 2: Run existing PD kv-config tests**

Run: `pytest tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py -k build_kv_transfer_config -v`

Expected: PASS

- [ ] **Step 3: In `launch_servers`, when `self.config.cache_pool.enabled`**

在分配 side channel 之前：

```python
from verl.workers.rollout.vllm_rollout.kv_cache_pool import validate_platform_cache_pool

pool = self.config.cache_pool
if pool.enabled:
    vllm_kwargs = (self.config.engine_kwargs or {}).get("vllm") or {}
    if vllm_kwargs.get("kv_transfer_config"):
        raise ValueError("engine_kwargs.vllm.kv_transfer_config")
    validate_platform_cache_pool(cache_pool=pool, is_npu=use_ascend_mooncake_v1)
```

`lookup_rpc_port` **在 `_spawn_pd_server` 内分配**，不在 `launch_servers` 里提前 `get_free_port`。`launch_servers` 把 `reserved_socks` 传进 spawn。用户指定同一个正整数给所有 P/D：**不查冲突**，原样写入每个实例。

- [ ] **Step 4: Env and lookup port in `_spawn_pd_server`**

`_spawn_pd_server` 增加 `reserved_socks: list`。若 `self.config.cache_pool.enabled`：

1. `lookup_port = pool.connector.lookup_rpc_port`
2. 若 `lookup_port in (None, 0)`：`lookup_port, lookup_sock = get_free_port(prefill_host_ip, with_alive_sock=True)`，`reserved_socks.append(lookup_sock)`
3. 在 spawn 内调用 `_build_kv_transfer_config(..., cache_pool=pool, lookup_rpc_port=lookup_port, prefill_tps=[self._prefill_tp])`（`launch_servers` 关 Pool 时仍可先建 cfg 再传入；开 Pool 时由 spawn 建）
4. env：

```python
from verl.workers.rollout.vllm_rollout.kv_cache_pool import mooncake_json_path

job_id = ray.get_runtime_context().get_job_id()
env_vars["PYTHONHASHSEED"] = str(self.config.cache_pool.python_hash_seed)
cfg_path = self.config.cache_pool.store.config_path or mooncake_json_path(job_id)
env_vars["MOONCAKE_CONFIG_PATH"] = cfg_path
```

不要设 `MOONCAKE_MASTER` / `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`。

关 Pool 时 `_spawn_pd_server` 不分配 lookup 端口，沿用调用方传入的单连接器 `kv_transfer_config`。

- [ ] **Step 5: Dual-source test in `test_kv_cache_pool_on_cpu.py`**（**不要**写进 PD 测试文件）

- `engine_kwargs.vllm.kv_transfer_config` 非空 → `ValueError`。检查做成 `vLLMPDReplica._validate_cache_pool_engine_kwargs(config)` 静态方法，单测放在 `test_kv_cache_pool_on_cpu.py`。
- `_pd_dispatch` 相关测试留给 Task 8，本任务不要往 `test_vllm_pd_disaggregation_on_cpu.py` 加新断言。

- [ ] **Step 6: Commit**

```bash
git add verl/workers/rollout/vllm_rollout/vllm_pd_replica.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: enable MultiConnector KV cache pool in vLLMPDReplica

EOF
)"
```

---

### Task 8: `vLLMHttpServer` JSON 落盘与 `_pd_dispatch`

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- Modify: `tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py`

- [ ] **Step 1: `_pd_dispatch` uses `p2p_connector_name`**

替换：

```python
from verl.workers.rollout.vllm_rollout.kv_cache_pool import p2p_connector_name

is_mooncake = p2p_connector_name(self._disaggregation_kv_transfer_config or {}) == "MooncakeConnector"
```

- [ ] **Step 2: Failing then passing dispatch tests**

在 `test_vllm_pd_disaggregation_on_cpu.py` 追加：

1. `connector` 为顶层 `MultiConnector` 且 `connectors[0]` 为 `MooncakeConnector` → 与现有 mooncake 测试一样本地构造 decode params。`_DispatchStub` 改为可传入完整 `kv_transfer_config` dict。
2. MultiConnector + `MooncakeConnectorV1` → 使用 prefill 返回的 `kv_transfer_params`。
3. MultiConnector 缺 `connectors` → `RuntimeError`。

把 `_DispatchStub.__init__` 的 `connector="NixlConnector"` 改成仍支持短名字，同时允许 `kv_transfer_config=` 覆盖：

```python
if kv_transfer_config is not None:
    self._disaggregation_kv_transfer_config = kv_transfer_config
else:
    self._disaggregation_kv_transfer_config = {"kv_connector": connector}
```

- [ ] **Step 3: Write JSON at start of `launch_server` / `run_server`**

在解析 CLI 并启动 engine **之前**（`node_rank == 0` 的 `run_server` 开头即可，PD 的 nnodes=1）：

```python
from verl.workers.rollout.vllm_rollout.kv_cache_pool import (
    build_mooncake_json,
    mooncake_json_path,
)
from verl.utils.device import is_torch_npu_available
import json, os
from pathlib import Path

pool = getattr(self.config, "cache_pool", None)
if pool is not None and pool.enabled:
    if pool.store.config_path:
        path = Path(pool.store.config_path)
        if not path.is_file():
            raise FileNotFoundError(f"cache_pool.store.config_path not found: {path}")
        data = json.loads(path.read_text())
        if not data.get("master_server_address"):
            raise ValueError(f"master_server_address missing in {path}")
    else:
        address = pool.master.address
        if not address:
            raise ValueError("cache_pool.master.address")
        is_npu = is_torch_npu_available(check_device=False)
        if is_npu and pool.store.ssd_offload_path:
            Path(pool.store.ssd_offload_path).mkdir(parents=True, exist_ok=True)
        payload = build_mooncake_json(store=pool.store, master_address=address, is_npu=is_npu)
        out = Path(os.environ.get("MOONCAKE_CONFIG_PATH") or mooncake_json_path(os.environ.get("VERL_RAY_JOB_ID", "0")))
        out.write_text(json.dumps(payload, indent=2))
```

抽成模块级函数 `materialize_mooncake_config(config, environ) -> None` 放在 `kv_cache_pool.py`，HttpServer 只调用一次，单测不需要起 vLLM。

- [ ] **Step 4: Tests for `materialize_mooncake_config`**（tmp_path，放在 `test_kv_cache_pool_on_cpu.py`）

- 无 `config_path`：写入 `MOONCAKE_CONFIG_PATH`，GPU JSON 含 `master_server_address` 且 `enable_offload is False`。
- 有 `config_path` 且文件缺字段 → `ValueError`。
- NPU + `ssd_offload_path`：目录被创建。

- [ ] **Step 5: Run**

```
pytest tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add verl/workers/rollout/vllm_rollout/vllm_async_server.py verl/workers/rollout/vllm_rollout/kv_cache_pool.py tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py tests/workers/rollout/test_kv_cache_pool_on_cpu.py
git commit -m "$(cat <<'EOF'
feat: write Mooncake JSON and dispatch PD over MultiConnector

EOF
)"
```

---

### Task 9: 文档

**Files:**
- Modify: `docs/perf/rollout_kv_offload.md`

- [ ] **Step 1: Rewrite the doc**

结构：

1. 保留原「非 PD `engine_kwargs` 旁路」整节，标题改为 **Non-PD (colocated) offload**，并加一句：**与 `rollout.cache_pool.enabled=true` 互斥**。
2. 新增 **PD disaggregation + KV Cache Pool** 节，GPU / NPU 分示例（从 spec §3.6 复制 Hydra YAML）。
3. 写明：
   - 隔离键是 `ray.get_runtime_context().get_job_id()`（`python` 直跑和 `ray job submit` 都可以）
   - `auto_start` 与 `store.config_path` 互斥
   - SSD 路径仅 NPU；GPU `enable_offload=true` 与 `ssd_offload_path` 均不支持（前者 `NotImplementedError`，后者 `ValueError`）
   - P2P protocol 是 `disaggregation.mooncake_protocol`，Store protocol 是 `cache_pool.store.protocol`
   - 非 default `tenant_id` 需要 Mooncake ≥ 0.3.12
   - NPU SSD 建议自行设 `cache_pool.master.client_ttl`
   - verl 不注入 `LD_LIBRARY_PATH` / `ASCEND_GLOBAL_RESOURCE_CONFIG`
   - 权重更新仍走现有 `reset_prefix_cache(reset_connector=True)`

Last updated 改成 `09/09/2026`。

- [ ] **Step 2: Commit**

```bash
git add docs/perf/rollout_kv_offload.md
git commit -m "$(cat <<'EOF'
docs: document vLLM PD MultiConnector KV cache pool

EOF
)"
```

---

## Spec coverage

| Spec | Task |
|---|---|
| §3 配置 / 平台无关校验 | 1–3 |
| §3.1 平台相关校验 / 双源 | 4 (`validate_platform_cache_pool`) + 7 |
| §4 master / named actor / JSON 路径 | 5–6, 8 |
| §5 MultiConnector / Store TP / `_pd_dispatch` | 4, 7, 8 |
| §6 错误表 | 1, 4, 7, 8 |
| §7 测试 | 各 task 的 pytest |
| §9 文档 | 9 |

## 执行方式

Plan 写在 `docs/superpowers/plans/2026-09-09-kv-cache-pool.md`。

两种执行方式：

1. **Subagent-Driven（推荐）**：每个 Task 开一个新 subagent，task 之间做 review
2. **Inline Execution**：本会话按 executing-plans 批量做，设检查点

选哪一种？
