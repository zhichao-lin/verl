# verl vLLM PD `kv_port` 连续端口预留设计

日期：2026-09-10  
状态：已批准  
范围：`vLLMPDReplica.launch_servers` 为每个 P/D 实例分配 handshake / `kv_port` 的方式，以及开 Pool 时 `lookup_rpc_port` 的生成方式。  
前置：按实例预留连续握手端口块；未配置 `lookup_rpc_port` 时写入该实例整段 `engine_id`（`str`），不占 TCP。

未改 P2P JSON 形状：NPU 每个实例仍然只写一个 `kv_port`（块首）。GPU 路径握手 `span=1`，端口个数与今天一致。Store 子连接器的 `lookup_rpc_port` 是 IPC 后缀，类型为非负整数（`>= 0`，含 `0`）或非空 `str`（含 `"0"`）。负整数以及 `strip()` 后能解析为负整数的字符串非法。

本文件覆盖 `docs/superpowers/specs/2026-09-09-kv-cache-pool-design.md` 中这些合同：

- §3.5 / §4.5：`None` 或 `0` 时用 `get_free_port` 分配正整数
- §5.3：Store extra 里 `lookup_rpc_port` 必须是正整数

用户指定了值则原样写入、不查跨实例冲突——这一合同不变。未配置**仅** `None` 与空串（含只含空白）；`0` 和 `"0"` 算已配置，原样写入。

## 1. 背景与目标

NPU `MooncakeConnectorV1` 把 `kv_port` 当作握手**基端口**，每个 worker 监听：

```text
handshake_port = kv_port
    + dp_rank * tp_size * pp_size * pcp_size
    + (pp_rank * pcp_size + pcp_rank) * tp_size
    + tp_rank
```

当前 `vLLMPDReplica` 拒绝 `data_parallel_size != 1` 和 `pipeline_model_parallel_size != 1`，且不建模 PCP，因此实际占用是：

```text
[kv_port, kv_port + tp_size)
```

`launch_servers` 对每个实例只 `get_free_port` 一次，并用 `bind((ip, 0))` 的活 socket 锁住**基端口**。内核几乎总是给出相邻临时端口。PD replica 又强制单节点（`pd_world_size <= gpus_per_node`），Prefill 与 Decode 的区间必然重叠。

例：1P3D、`prefill_tp=4`、`decode_tp=2`，若依次拿到 40000、40001、40002、40003：

| 实例 | 配置的 `kv_port` | 实际会 bind 的端口 |
|---|---|---|
| Prefill | 40000 | 40000–40003 |
| Decode0 | 40001 | 40001–40002 |
| Decode1 | 40002 | 40002–40003 |
| Decode2 | 40003 | 40003–40004 |

开 Pool 时 `_spawn_pd_server` 还会再 `get_free_port` 一次给 `lookup_rpc_port`。上游只把它拼进 IPC 路径（`ipc://.../lookup_rpc_port_{rpc_port}_...`），并不 bind 该数字对应的 TCP 端口；占位期间却会占住一个真实 TCP 端口，插进尚未预留的握手区间。

每个 P/D 实例已经有唯一的顶层 `engine_id`（`uuid.uuid4().hex`），供 P2P KV 传输指名 Producer。lookup IPC 同样只需要每实例唯一，应直接复用整段 `engine_id`，不要再分配端口或另生成 uuid。

目标：

1. 每个 P/D 实例在启动前预留自己的连续握手端口块，块首作为 `kv_port` / `VLLM_NIXL_SIDE_CHANNEL_PORT`。
2. 未配置 `lookup_rpc_port` 时写入该实例整段 `engine_id`，不 `bind` TCP。

### 已锁定决策

1. 按实例预留连续握手端口块。不采用人为 stride（如基端口 +100），也不改上游 handshake 协议。
2. 不新增 Hydra / `DisaggregationConfig` 字段。握手端口自动分配，用户不能指定 `kv_port`。
3. `span` 只由平台和**该实例自己的 TP**决定：NPU `MooncakeConnectorV1` 为 `tp`，其它为 `1`。非对称 TP 时 Prefill 用 `prefill_tp`，每个 Decode 用 `decode_tp`，不用 `max(P, D)` 套所有实例。
4. 本阶段不把 DP / PP / PCP 乘进 `span`（PD 仍拒绝 DP/PP > 1）。以后放开时再改公式。
5. 保持现有 spawn 顺序：先为该实例预留 handshake 区间，再 `_spawn_pd_server`。
6. `lookup_rpc_port`（见 §5.1）：未配置（仅 `None` 或空串）时用该实例整段 `engine_id`（`str`）。已配置则原样写入 Store（字符串先 `strip()`）。合法已配置值：非负整数（`>= 0`，含 `0`）或非空 `str`（含 `"0"`）。非法：`bool`、负整数、以及 `strip()` 后 `int(s, 10)` 成功且 `< 0` 的字符串（如 `"-1"`）。不查跨实例冲突。不调用 `get_free_port`，不另 `uuid4()`。校验在 `KVCachePoolConnectorConfig.__post_init__`、`parse_lookup_rpc_port` / `_resolve_lookup_rpc_port`、`build_store_connector_config` 三层执行；`enabled=False` 也在 ConnectorConfig 构造时拒绝非法值。
7. Decode 的 `VLLM_MOONCAKE_BOOTSTRAP_PORT` 继续指向 Prefill **块首**（GPU Mooncake bootstrap 语义不变）。
8. `get_free_port` 保持单端口原语，行为不变。新增 `get_free_port_range`。replica 在握手预留之外不再调用 `get_free_port`。
9. 握手预留 socket 生命周期不变：放进现有 `reserved_socks`，`launch_server` gather 返回后在 `finally` 里 close。关闭后到 vLLM 真正 bind 之间的 TOCTOU 是原有问题，本次不修。

### 非目标

- 修改 vLLM / vLLM-Ascend 的端口公式、handshake 或 IPC 路径格式
- GPU NIXL / GPU `MooncakeConnector` 的多端口展开（当前 DP=1 下一实例一口）
- 用户可配置的固定 `kv_port` 或跨实例 `lookup_rpc_port` 冲突检查
- 截断 `engine_id`、再生成独立 uuid
- 多节点 PD、`prefill_replicas != 1`、DP/PP > 1
- 修复预留 socket close 之后的端口被抢

## 2. 架构

```
vLLMPDReplica.launch_servers()
  ├─ span_p = _kv_handshake_port_span(use_ascend_mooncake_v1, prefill_tp)
  ├─ prefill_base = _reserve_handshake_ports(host, span_p, reserved_socks)
  ├─ _spawn_pd_server(prefill, kv_port=prefill_base, engine_id=prefill_engine_id)
  │     └─ Pool：lookup_rpc_port = _resolve_lookup_rpc_port(configured, engine_id)
  └─ for each decode:
       span_d = _kv_handshake_port_span(use_ascend_mooncake_v1, decode_tp)
       decode_base = _reserve_handshake_ports(host, span_d, reserved_socks)
       _spawn_pd_server(decode, kv_port=decode_base, engine_id=decode_engine_id,
                        mooncake_bootstrap_port=prefill_base)
```

| 单元 | 职责 |
|---|---|
| `get_free_port_range` | 在 `address` 上找到并（可选）占住 `count` 个连续 TCP 端口 |
| `_kv_handshake_port_span` | 由平台 + 实例 TP 得到握手 `span` |
| `_reserve_handshake_ports` | 调 range、把 socks 并入 `reserved_socks`、返回块首 |
| `_resolve_lookup_rpc_port` | 未配置 → 整段 `engine_id`；已配置 → 原样返回（字符串已 `strip`） |
| `parse_lookup_rpc_port` | 配置层 / builder 共用：未配置 → `None`；非法 → `ValueError` |
| `build_store_connector_config` | 用 `parse_lookup_rpc_port` 再写入；得到 `None` 则 `ValueError` |
| `launch_servers` | 每个实例先 reserve 再 spawn；P2P JSON / env 仍只写握手块首 |

`engine_id` 仍只写在 MultiConnector **顶层**（P2P 身份），不复制进 Store 子连接器。Store extra 里的 `lookup_rpc_port` 在未配置时**值等于**该 `engine_id`，但字段名和位置不变。

## 3. `span` 规则

```python
@staticmethod
def _kv_handshake_port_span(*, use_ascend_mooncake_v1: bool, tp: int) -> int:
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    return tp if use_ascend_mooncake_v1 else 1
```

| 条件 | `span` |
|---|---|
| `use_ascend_mooncake_v1=True`（含 Ascend 上 NIXL 回退到 MooncakeConnectorV1） | 该实例 `tp` |
| GPU NIXL / GPU MooncakeConnector | `1` |

`use_ascend_mooncake_v1` 沿用现有 `_is_ascend_platform()`。不要用 `transfer_backend` 判断 span：Ascend 回退后 backend 已是 `mooncake`，但 GPU mooncake 仍应 `span=1`。

## 4. `get_free_port_range`

放在 `verl/utils/net_utils.py`。

```python
def get_free_port_range(
    address: str,
    count: int,
    with_alive_socks: bool = False,
    max_tries: int = 100,
) -> tuple[int, list[socket.socket] | None]:
```

合同：

- `count < 1` → `ValueError`。
- `count == 1`：调用现有 `get_free_port`。`with_alive_socks=True` 时返回 `(port, [sock])`；`False` 时返回 `(port, None)`。
- `count > 1`：最多 `max_tries` 次（默认 100）：
  1. 新建 socket，`SO_REUSEADDR=1`，`bind((address, 0))`，得到 `start`。该 socket **保持打开**，作为区间第一个端口。
  2. 若 `start + count - 1 > 65535`：关掉已打开的，进入下一次尝试。
  3. 对 `offset in 1 .. count-1`：先用不带 `SO_REUSEADDR` 的探测 socket `bind((address, start + offset))`。Linux 上已 bind、未 listen 且带 `SO_REUSEADDR` 的端口允许第二次 `SO_REUSEADDR` bind 成功，无 REUSEADDR 的探测才会 `EADDRINUSE`。探测失败：关掉本次已打开的，进入下一次尝试。探测成功：关掉探测 socket，再建预留 socket，`SO_REUSEADDR=1` 后 `bind` 同一端口。预留必须带 `SO_REUSEADDR`，以便 `launch_server` 返回、socks close 之前 vLLM 能 bind。
  4. 任一步 `OSError`：关掉本次已打开的，进入下一次尝试。
  5. 成功：`with_alive_socks=True` 返回 `(start, socks)`，`len(socks)==count` 且 `socks[i]` 绑在 `start+i`；`False` 则全部 close，返回 `(start, None)`。
- 用尽 `max_tries` → `RuntimeError`，消息包含 `address` 和 `count`。
- address family 与 `get_free_port` 相同：IPv6 用 `AF_INET6`，否则 `AF_INET`。bind 元组同样是 `(address, port)`。

`get_free_port` 本身不改实现、不改为调用 range，避免单端口行为漂移。

## 5. replica 调用点

`_kv_handshake_port_span`、`_reserve_handshake_ports`、`_resolve_lookup_rpc_port` 都是 `vLLMPDReplica` 的 `@staticmethod`。`parse_lookup_rpc_port` 放在 `verl/workers/config/cache_pool.py`，供配置、replica、builder 共用。

`_reserve_handshake_ports` 行为固定：

```python
port, socks = get_free_port_range(host, span, with_alive_socks=True)
reserved_socks.extend(socks)
return port
```

`launch_servers` 把两处握手用的 `get_free_port(..., with_alive_sock=True)` 换成 `_reserve_handshake_ports`：

- Prefill：`span = _kv_handshake_port_span(..., tp=self._prefill_tp)`
- 每个 Decode：`span = _kv_handshake_port_span(..., tp=self._decode_tp)`

`kv_port`、`side_channel_port`、Prefill 的 `mooncake_bootstrap_port`、`VLLM_NIXL_SIDE_CHANNEL_PORT` 都用返回的块首。Decode 的 `mooncake_bootstrap_port` 仍是 Prefill 块首。

### 5.1 `lookup_rpc_port`

**未配置**（`parse_lookup_rpc_port` 返回 `None`，resolve 回退整段 `engine_id`）只有：

- `None`
- 空串：`""`，或 `strip()` 之后为空

**已配置**（写入 Store）：非负整数（`>= 0`，含 `0`），或 `strip()` 后非空且不是负整数字面量的 `str`（含 `"0"`、`"01"`）。字符串写入 Store 前 `strip()`，但**不**把非负数字字符串转成 `int`（`"0"` 保持 `"0"`）。

**非法**（`ValueError`）：

- `bool`（`False` 不能当成 `0`）
- 负整数
- `strip()` 后 `int(s, 10)` 成功且结果 `< 0` 的字符串（`"-1"`、`" -2 "`）
- 其它非 `int`/`str` 类型

`int("-0", 10)` 为 `0`，不算负数，字符串 `"-0"` 视为合法已配置。`"-1.5"` 不能 `int(..., 10)`，当普通非空字符串合法。

```python
def parse_lookup_rpc_port(value) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"lookup_rpc_port must not be bool, got {value!r}")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            as_int = int(stripped, 10)
        except ValueError:
            return stripped
        if as_int < 0:
            raise ValueError(f"lookup_rpc_port must be a non-negative int, got {value!r}")
        return stripped
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"lookup_rpc_port must be a non-negative int, got {value!r}")
        return value
    raise ValueError(f"lookup_rpc_port must be a non-negative int or str, got {value!r}")


@staticmethod
def _resolve_lookup_rpc_port(configured, engine_id: str) -> int | str:
    if not engine_id:
        raise ValueError("engine_id is required to resolve lookup_rpc_port")
    parsed = parse_lookup_rpc_port(configured)
    return engine_id if parsed is None else parsed
```

`KVCachePoolConnectorConfig.__post_init__` 调用 `parse_lookup_rpc_port(self.lookup_rpc_port)` 做字段校验，**不**因外层 `enabled=False` 跳过（与 `store.mode` 等仅 enabled 时校验不同）。不在 dataclass 里改写用户原始值。

`_spawn_pd_server` 在 `pool.enabled` 时：

1. 要求 `engine_id` 与 `transfer_backend` 非 `None`，不再接收或检查 `reserved_socks`
2. `lookup_port = vLLMPDReplica._resolve_lookup_rpc_port(pool.connector.lookup_rpc_port, engine_id)`
3. 调用 `_build_kv_transfer_config(..., lookup_rpc_port=lookup_port, ...)`

从 `_spawn_pd_server` 签名中删除 `reserved_socks`。`launch_servers` 的握手 `reserved_socks` 只在 replica 内 `finally` close，不传入 spawn。关 Pool 时不生成 lookup。

`vllm_pd_replica.py` 若不再调用 `get_free_port`，从 import 中删除它。

### 5.2 `build_store_connector_config` 与配置类型

`lookup_rpc_port` 注解改为 `int | str`。调用 `parse_lookup_rpc_port`；若得到 `None`（调用方未先 resolve）则 `ValueError`。replica 应先 `_resolve_lookup_rpc_port`。

`KVCachePoolConnectorConfig.lookup_rpc_port` 改为 `Optional[int | str] = None`。`rollout.yaml` 该 key 的注释改为：未配置（null / 空串）时用实例 `engine_id` 做 IPC 后缀；非负整数或非空字符串（含 `0` / `"0"`）原样用；负数与 `bool` 非法。改注释后必须跑 `scripts/generate_trainer_config.sh`。

## 6. 错误处理

| 条件 | 行为 |
|---|---|
| `tp < 1` | `ValueError` |
| `count < 1` | `ValueError` |
| 连续握手端口块找不到 | `RuntimeError`（含 address、count） |
| 回退 lookup 时 `engine_id` 为空 | `ValueError` |
| `lookup_rpc_port` 非法（`bool`、负整数、负整数字符串、或非 `int`/`str`） | `ValueError`（配置构造期、resolve、builder） |
| 预留成功但后续 `launch_server` 失败 | 现有 `finally` close 全部 `reserved_socks` |

不在 replica 里二次检查握手区间是否重叠：只要 socks 一直占着，后续 `get_free_port_range`（顺序端口靠无 `SO_REUSEADDR` 探测，而不是二次 `SO_REUSEADDR` bind 失败）以及 `vLLMHttpServer` 内部的 `get_free_port`（`bind(0)`）不会再拿到这些端口。

未配置时 lookup 与 `engine_id` 相同，碰撞概率不超过 `engine_id` 本身。用户显式给所有实例同一个值时仍不查冲突。

## 7. 测试

CPU only。不加 GPU/NPU 真机 PD。

1. `tests/utils/test_net_utils_on_cpu.py`（新建）：
   - `count=1` + `with_alive_socks=True`：返回单元素 list，socket 绑在返回端口
   - `count=1` + `False`：返回 `(port, None)`
   - `count>=2`：`len(socks)==count`，`getsockname` 为 `start, start+1, ...`。另开**不设** `SO_REUSEADDR` 的 socket `bind` 同一 `(address, start+i)` 必须失败（预留 socket 与 `get_free_port` 一样只 `bind`、不 `listen`，带 `SO_REUSEADDR` 的二次 bind 在 Linux 上可能成功，不能用来断言占用）
   - 先用一个不带 `SO_REUSEADDR` 的 socket 占住某端口，再要 `count=2`：返回区间不得包含该占用端口
   - 强制相邻碰撞：已预留 `[start, start+count)` 时，把下一次 `bind(0)` 钉到 `start-1`，返回区间必须不相交（锁住无 REUSEADDR 探测；全程 `SO_REUSEADDR` 的顺序 bind 会重叠）
   - `count=0` / 负数 → `ValueError`
2. `tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py`：
   - `_kv_handshake_port_span(use_ascend_mooncake_v1=True, tp=8) == 8`
   - `_kv_handshake_port_span(use_ascend_mooncake_v1=False, tp=8) == 1`
   - `tp<1` → `ValueError`
   - `_reserve_handshake_ports` 在 `127.0.0.1` 上预留 `span=4`：`reserved_socks` 长度为 4，块首与 socks 一致
   - `_resolve_lookup_rpc_port(None, "abc") == "abc"`；`""` / `"   "` 同样回退 `"abc"`；`0` 返回 `0`；`"0"` 返回 `"0"`；`" 01 "` 返回 `"01"`；`19001` 返回 `19001`；`"custom"` 返回 `"custom"`；`-1` / `"-1"` / `" -2 "` / `True` / `False` → `ValueError`
3. `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`：
   - `build_store_connector_config` 接受 `"eid-0"`、`0`、`"0"`；拒绝 `""` / `None` / `True` / `-1` / `"-1"`
   - `test_spawn_injects_pool_env_and_omits_mooncake_master`：自动 lookup 等于传入的 `engine_id`（`"eid-0"`）
   - `_spawn` / `_spawn_pd_server` 不再有 `reserved_socks` 参数。用户 `19001`、`"custom-lookup"`、`0` 均原样写入。自动 lookup **不**调用 `get_free_port`，`lookup_rpc_port == engine_id`。删除「auto lookup append sock」用例
4. `tests/workers/config/test_cache_pool_config_on_cpu.py`：
   - `parse_lookup_rpc_port`：`None`/`""` → `None`；`0`/`"0"`/`"-0"` 合法；`-1`/`"-1"`/`True` 非法
   - `lookup_rpc_port` 为字符串时 coercion 仍得到该字符串
   - `KVCachePoolConnectorConfig(lookup_rpc_port=-1)` 与 `KVCachePoolConfig(enabled=False, connector={"lookup_rpc_port": -1})` 均 `ValueError`

不要求对完整 `launch_servers`（Ray worker gather）做集成测试。

## 8. 文件清单

| 动作 | 路径 |
|---|---|
| 修改 | `verl/utils/net_utils.py` |
| 修改 | `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py` |
| 修改 | `verl/workers/rollout/vllm_rollout/kv_cache_pool.py` |
| 修改 | `verl/workers/config/cache_pool.py` |
| 修改 | `verl/trainer/config/rollout/rollout.yaml`（仅 `lookup_rpc_port` 注释） |
| 修改 | `verl/trainer/config/_generated_*.yaml`（必须经 `scripts/generate_trainer_config.sh`） |
| 新增 | `tests/utils/test_net_utils_on_cpu.py` |
| 修改 | `tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py` |
| 修改 | `tests/workers/rollout/test_kv_cache_pool_on_cpu.py` |
| 修改 | `tests/workers/config/test_cache_pool_config_on_cpu.py` |

不改 `docs/perf/rollout_kv_offload.md`。
