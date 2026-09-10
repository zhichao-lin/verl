# verl vLLM PD KV Cache Pool（MultiConnector）设计

日期：2026-09-09  
状态：已批准  
范围：`rollout.name=vllm` 且 `disaggregation.enabled=True` 时，用 MultiConnector 同时使能 P2P KV 传输和 KV Cache Pool。

未开 Pool 时，现有 PD 路径不变（GPU 仍可以是 `MooncakeConnector` 或 `NixlConnector`，NPU 仍是 `MooncakeConnectorV1`）。开 Pool 后只走本节合同。

## 1. 背景与目标

当前 vLLM PD 路径只组装单一 P2P 连接器。非 PD 场景可通过 `engine_kwargs.vllm.kv_transfer_config` 旁路挂 `MooncakeStoreConnector`（见 `docs/perf/rollout_kv_offload.md`）。PD 打开后该旁路会被 replica 生成的 `kv_transfer_config` 覆盖，无法同时拥有 P2P 和 Store。

本需求只覆盖 **MultiConnector = P2P + Store**：

| 平台 | P2P | Store |
|---|---|---|
| GPU | `MooncakeConnector` | `MooncakeStoreConnector` |
| NPU | `MooncakeConnectorV1` | `AscendStoreConnector` |

参考：

- GPU：`vllm/docs/features/mooncake_store_connector_usage.md`
- NPU：`vllm-ascend/docs/source/user_guide/feature_guide/kv_pool.md`

子连接器字段以这两份官方文档的 PD MultiConnector 示例为准，不把当前 verl 单连接器顶层 payload（`engine_id`、`kv_buffer_device` 等）复制进子连接器。

### 已锁定决策

1. Store 范围：一次 `ray.init()` 之后的同一个 driver（`ray.get_runtime_context().get_job_id()`）共享一个 Mooncake Store、一个 `mooncake_master`。这与是否用 `ray job submit` 无关：`python -m verl.trainer.main_ppo` 也会 `ray.init()`，同样有 JobID。该 driver 内多次 `LLMServerManager.create()` 复用同一个 named actor，不新起进程。两个并发的 Python driver 连同一 Ray 集群会得到不同 JobID，因此是两套 master（隔离正确）。
2. Master 生命周期：`auto_start` 默认 true；false 时复用外部 master。不 kill 已有进程。禁止 `auto_start=True` 且同时设置 `store.config_path`。
3. 拓扑：两端都用 **embedded**。不拉起 `mooncake_client`。`mode=standalone-store` 预留并报错。
4. 非对称 TP：GPU `store_tp_size` 默认 `lcm(prefill_tp, decode_tp)`，允许覆盖。NPU 不发明 `store_tp_size`；非对称 TP 只写进 P2P 子连接器。多 P 且各 P TP 不唯一时，自动打开 `enable_store_tp_lcm`（见 5.3）。
5. 架构：`LLMServerManager` 管 Store 运行时；`kv_cache_pool.py` 提供组装纯函数；`vLLMPDReplica` 只传角色、端口、TP 后调用。
6. NPU `backend` 预留 `mooncake | memcache | yuanrong`，当前只实现 `mooncake`。
7. GPU 本阶段不支持任何 offload：`store.enable_offload=true` 抛 `NotImplementedError`；`ssd_offload_path` 非空抛 `ValueError`。这两项在 `setup_kv_cache_pool`（拉起 master 之前）和 replica 平台校验各做一次。NPU 用 `ssd_offload_path` 做 SSD，忽略 `enable_offload`。

### 非目标

- 非 PD 纯 Store（现有 `engine_kwargs` 旁路保持不动；与 PD Pool 互斥）
- GPU `standalone-store` / `mooncake_client` / GPU SSD 路径 / GPU `enable_offload`
- NPU `memcache` / `yuanrong` 实现
- 本阶段 `vLLMPDReplica` 仍拒绝 `prefill_replicas!=1`（现有 PD 限制）。Store TP 辅助函数按多 P 实现自动 lcm 分支，多 P 落地时无需重写该逻辑
- SGLang PD 的 KV Pool
- 用 Store 命中率驱动 `cache_aware` 路由
- 注入 `LD_LIBRARY_PATH`、`ASCEND_GLOBAL_RESOURCE_CONFIG`、hugepage 等硬件 env（留给镜像和训练脚本）
- 注入 `MOONCAKE_MASTER`（生成 JSON 时只写 `master_server_address`）
- 探测 Mooncake 版本；不根据 HF 配置校验 KV head 能否被 `store_tp_size` 整除

## 2. 架构

```
LLMServerManager.create()
  └─ _setup_kv_cache_pool()          # 解析或复用唯一 mooncake_master，写回 DictConfig.master.address
  └─ _initialize_llm_servers()       # 此后 replica 才能看到解析后的 address
       └─ vLLMPDReplica.launch_servers()
            ├─ 平台校验 / 双源 engine_kwargs 检查
            └─ _spawn_pd_server()
                 ├─ 开 Pool：分配 lookup_rpc_port，调用 build_kv_transfer_config → MultiConnector [P2P, Store]
                 ├─ 关 Pool：使用 launch_servers 传入的单 P2P cfg（含 NIXL）
                 ├─ 注入 MOONCAKE_CONFIG_PATH / PYTHONHASHSEED
                 └─ vLLMHttpServer.launch_server()
                      └─ 本地落盘 mooncake JSON，再启动 vLLM
```

模块边界：

| 单元 | 职责 | 依赖 |
|---|---|---|
| `KVCachePoolConfig` | Hydra 配置与**平台无关**校验；嵌套 `master`/`store`/`connector` coercion | `BaseConfig` |
| `kv_cache_pool.py` | JSON 内容生成、Store TP 推导、`build_p2p_connector_config` / `build_store_connector_config` / `build_kv_transfer_config`、`MooncakeMasterActor` | 配置 + 调用方传入的平台/角色/端口/TP |
| `LLMServerManager` | 当前 driver（`get_job_id()`）内的 master 生命周期；写回 `master.address` | `kv_cache_pool.py` |
| `vLLMPDReplica` | 平台探测、平台相关校验、分配 `lookup_rpc_port`、调用组装纯函数、注入 env | 解析后的 `cache_pool` |
| `vLLMHttpServer` | 本地写 JSON；`_pd_dispatch` 按 `connectors[0].kv_connector` 分支 | replica 传入的 config / kv cfg |

`vLLMPDReplica` 不再自己拼 MultiConnector 字典。现有 `_build_kv_transfer_config` 改为调用 `kv_cache_pool.build_kv_transfer_config`。

## 3. 配置模型

新增 `verl/workers/config/cache_pool.py`。`RolloutConfig` 增加：

```python
cache_pool: KVCachePoolConfig = field(default_factory=KVCachePoolConfig)
```

`RolloutConfig.__post_init__` 把 Hydra dict / `DictConfig` 转成 `KVCachePoolConfig`，方式与 `disaggregation` 相同。

`KVCachePoolConfig.__post_init__` 把 `master` / `store` / `connector` 转成对应 dataclass（与 `DisaggregationConfig.decode_policy` 相同）。`enabled=False` 时跳过强校验。

`verl/workers/config/rollout.py` 的 `__all__` 以及 `verl/workers/config/__init__.py` 导出 `KVCachePoolConfig`、`KVCachePoolMasterConfig`、`KVCachePoolStoreConfig`、`KVCachePoolConnectorConfig`。

Hydra 路径全部挂在 `actor_rollout_ref.rollout.cache_pool.*`。

### 3.1 校验分两层

平台无关（`KVCachePoolConfig` / `RolloutConfig.__post_init__`，CPU 单测可跑）：

- `enabled=True` 要求 `rollout.name=vllm` 且 `disaggregation.enabled=True`
- `enabled=True` 要求 `disaggregation.transfer_backend == "mooncake"`
- `backend` 属于 `{mooncake, memcache, yuanrong}`；非 `mooncake` 抛 `NotImplementedError`
- `store.mode` 只允许 `embedded`；`standalone-store` 抛 `ValueError`
- `auto_start=True` 且 `store.config_path` 非空 → `ValueError`
- `auto_start=False` 且 `master.address` 为空且 `config_path` 为空 → `ValueError`。`auto_start=False` 且只给了 `config_path`：允许；`vLLMHttpServer.launch_server` 读取该文件，文件不存在或缺少 `master_server_address` 则失败
- `enable_multi_tenants=True` 时 `tenant_quota_connector_type` 与 `tenant_quota_connector_uri` 必填，否则 `ValueError`
- `use_layerwise=True` 且 `backend=mooncake` → `ValueError`
- `enabled=True` 且 `enable_prefix_caching=False` → `ValueError`

平台相关（`vLLMPDReplica.launch_servers` 开头，用现有 `_is_ascend_platform()`）：

- NPU：`store.protocol` 若非 `None` 则必须是 `ascend`；`device_name` 必须是 `""`；`global_segment_size` 解析为字节后 `% 2**30 == 0`；GPU 专用 connector 字段非默认 → `ValueError`；`ssd_offload_path` 若非空则开 SSD
- GPU：`store.protocol` 若非 `None` 则必须属于 `{rdma, tcp}`；`ssd_offload_path` 非空 → `ValueError`；`enable_offload=True` → `NotImplementedError`；NPU 专用 store/connector 字段非默认 → `ValueError`。上述 GPU offload 拒绝也在 `setup_kv_cache_pool` 里提前做，避免先拉起 master 再失败。
- `engine_kwargs.vllm.kv_transfer_config` 存在且非空 → `ValueError`（双源）。此检查只在 PD replica 启动时做，不放进全局 `RolloutConfig.__post_init__`，以免误伤非 PD 旁路

「非默认」一律相对 dataclass 默认值比较。Hydra YAML 显式写成默认值不算非默认。

### 3.2 `KVCachePoolConfig`

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `False` | 总开关 |
| `backend` | `"mooncake"` | NPU 预留字段。默认 `mooncake` 时不写入 Store extra；非 `mooncake` 在配置期即 `NotImplementedError`。GPU 必须保持默认，否则平台校验失败 |
| `python_hash_seed` | `0` | 注入所有 P/D 进程的 `PYTHONHASHSEED`。与采样 `seed` / `full_determinism` 独立。即使 GPU 官方只在 xxhash 算法下要求该 env，verl 开 Pool 时也一律注入 |
| `kv_load_failure_policy` | `None` | `recompute` 或 `fail`。`None`：顶层不写该 key，沿用 vLLM 默认 `fail`。非空：写到 MultiConnector 顶层 |
| `extra_config` | `{}` | 在平台校验之后 merge 进 Store 子连接器 extra，覆盖同名一等字段，允许带上游新 key。不再对 extra 里的名字做平台拦截 |
| `master` | `KVCachePoolMasterConfig()` | master 生命周期 |
| `store` | `KVCachePoolStoreConfig()` | mooncake JSON |
| `connector` | `KVCachePoolConnectorConfig()` | Store 子连接器 extra |

### 3.3 `KVCachePoolMasterConfig`

| 字段 | 默认 | 说明 |
|---|---|---|
| `auto_start` | `True` | true：本 job 拉起或复用唯一 `mooncake_master` |
| `address` | `None` | `host:port`。`auto_start=True` 时由运行时覆盖为 actor 实际绑定地址。`auto_start=False` 且无 `config_path` 时必填 |
| `port` | `None` | 仅 `auto_start=True` 使用。非 `None`：必须绑该端口，占用则启动失败，不换端口。`None`：在 actor 节点 `get_free_port` |
| `eviction_high_watermark_ratio` | `0.9` | 传给 `mooncake_master` CLI |
| `eviction_ratio` | `0.1` | 同上 |
| `default_kv_lease_ttl` | `None` | 非 `None` 才传 CLI。SSD 开启时不自动填 11000 |
| `client_ttl` | `None` | 非 `None` 才传 CLI。SSD 开启时不自动填 120。NPU 官方 SSD 建议 `client_ttl=120`，只写在用户文档 |
| `enable_multi_tenants` | `False` | true 时传 `--enable_multi_tenants=true`，且必填下面两项 |
| `tenant_quota_connector_type` | `None` | 例如 `file` |
| `tenant_quota_connector_uri` | `None` | 例如 `/etc/mooncake/tenant_quotas.yaml` |

没有「杀掉已有 master」开关。

### 3.4 `KVCachePoolStoreConfig`

| 字段 | 默认 | 说明 |
|---|---|---|
| `config_path` | `None` | 非空：不生成 JSON，`MOONCAKE_CONFIG_PATH` 指向该路径。与 `auto_start=True` 互斥。多节点必须每机可读 |
| `mode` | `"embedded"` | 只允许 `embedded` |
| `protocol` | `None` | `None` → GPU 写成 `rdma`，NPU 写成 `ascend`。禁止把 P2P 的 `nvlink` 写入 Store JSON |
| `metadata_server` | `"P2PHANDSHAKE"` | 写入 JSON |
| `global_segment_size` | `"4GB"` | NPU 必须 1GB 对齐。接受 `4GB` / `1024MB` / `1073741824` 等形式 |
| `local_buffer_size` | `"4GB"` | 只写入 GPU JSON；NPU JSON 省略 |
| `device_name` | `""` | 与 `disaggregation.ib_device` 独立，互不拷贝。NPU 必须保持 `""` |
| `tenant_id` | `"default"` | 共享 Store 的实例必须相同。非 `default` 需要 Mooncake ≥ 0.3.12；verl 不探测版本，只写进 `docs/perf/rollout_kv_offload.md` |
| `enable_offload` | `False` | GPU 上 `True` → `NotImplementedError`（暂不支持 offload）。因此 GPU JSON 该字段恒为 `false`。NPU 忽略此字段 |
| `ssd_offload_path` | `None` | GPU 非空则 `ValueError`。NPU 非空则开 SSD |
| `preferred_segment` | `False` | 只写入 NPU JSON。GPU 上非默认则报错 |
| `prefer_alloc_in_same_node` | `True` | 只写入 NPU JSON。GPU 上非默认则报错 |

SSD（仅 NPU，`ssd_offload_path` 非空）时：

- NPU JSON：`"enable_ssd_offload": true` 且写入 `ssd_offload_path`
- 自动拉起的 master 增加 `--enable_offload=true`
- 每个 `vLLMHttpServer.launch_server` 在写 JSON 之前 `mkdir -p ssd_offload_path`。已设 `config_path` 时不 mkdir

NPU 未开 SSD：JSON 写 `"enable_ssd_offload": false`，不写 `ssd_offload_path`。master **不传** `--enable_offload`。

GPU JSON **始终**写 `"enable_offload": false`（`true` 已在校验期拒绝）。不导出 `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`。

两套协议并存且合法：P2P 用 `disaggregation.mooncake_protocol`（默认 `nvlink`）；Store JSON 用 `cache_pool.store.protocol`（GPU 默认 `rdma`，NPU 固定 `ascend`）。

### 3.5 `KVCachePoolConnectorConfig`

一等字段使用上游原名，不做 GPU/NPU 别名合并。

两端：

| 字段 | 默认 | 写入规则 |
|---|---|---|
| `load_async` | `None` | `None` 不写，沿用上游默认（GPU `true`，NPU `false`）。非 `None` 则写入 |
| `lookup_rpc_port` | `None` | **P/D 每个实例都必须写入具体正整数**。`None` 或 `0`：replica 按实例分配。用户指定正整数：原样用，不检查跨实例冲突 |

GPU：

| 字段 | 默认 | 写入规则 |
|---|---|---|
| `lookup_async` | `False` | 仅 `True` 时写 |
| `cache_prefix` | `""` | 仅非空时写 |
| `save_decode_cache` | `False` | 仅 decode 且为 `True` 时写 |
| `store_tp_size` | `None` | GPU **始终**写入推导或覆盖后的正整数（见 5.3） |
| `enable_store_tp_lcm` | `None` | 仅非 `None` 时写；多 P 自动分支可能把它写成 `true`（见 5.3） |
| `prefill_tp_sizes` | `None` | 仅非 `None` 时写；多 P 自动分支可能写入列表 |

NPU：

| 字段 | 默认 | 写入规则 |
|---|---|---|
| `consumer_is_to_put` | `False` | 仅 `True` 时写 |
| `consumer_is_to_load` | `False` | 仅 `True` 时写 |
| `use_layerwise` | `False` | 仅 `True` 时写（且不得与 `backend=mooncake` 同时） |
| `prefill_pp_size` | `None` | 仅非 `None` 时写。本阶段 PD 仍拒绝 `pipeline_model_parallel_size>1`，配了也跑不起来 |
| `prefill_pp_layer_partition` | `None` | 仅非 `None` 时写，同上 |

NPU **不**写 `store_tp_size` / `enable_store_tp_lcm` / `prefill_tp_sizes`。GPU 上上述 NPU 字段非默认则报错；NPU 上 GPU 专用字段非默认则报错。`lookup_rpc_port` 与 `load_async` 两端共用，不算专用字段。

### 3.6 YAML 示例

GPU（1P 非对称 TP，无 SSD）：

```yaml
actor_rollout_ref.rollout:
  name: vllm
  tensor_model_parallel_size: 4
  enable_prefix_caching: true
  disaggregation:
    enabled: true
    transfer_backend: mooncake
    decode_replicas: 3
    decode_tensor_model_parallel_size: 2
  cache_pool:
    enabled: true
    store:
      global_segment_size: 4GB
      enable_offload: false
```

NPU（1P 对称 TP + SSD）：

```yaml
actor_rollout_ref.rollout:
  name: vllm
  tensor_model_parallel_size: 4
  enable_prefix_caching: true
  disaggregation:
    enabled: true
    transfer_backend: mooncake
    decode_replicas: 3
  cache_pool:
    enabled: true
    backend: mooncake
    store:
      global_segment_size: 4GB
      ssd_offload_path: /nvme/mooncake_offload
```

`rollout.yaml` 每个 key 上方必须有注释、key 后空行（`tests/special_sanity/test_config_docs.py`）。改完后必须跑 `scripts/generate_trainer_config.sh` 更新 `_generated_*.yaml`。

## 4. LLMServerManager 运行时

### 4.1 可行性

可行。`create()` 是 replica 的唯一入口。JSON 按进程在各节点本地落盘，不依赖 NFS。生成 JSON 只写入 `master_server_address`，不设 `MOONCAKE_MASTER`。

### 4.2 插入点

```
LLMServerManager.create()
  → _setup_kv_cache_pool()      # enabled=False 则 no-op
  → _initialize_llm_servers()
  → _init_global_load_balancer()
```

只改 driver 上的 Hydra `DictConfig` 键 `actor_rollout_ref.rollout.cache_pool.master.address`，且必须在构造 replica **之前**。不把 `cache_pool` 加入 `_mutable_fields`，不改 frozen dataclass。

### 4.3 `MooncakeMasterActor`

定义在 `verl/workers/rollout/vllm_rollout/kv_cache_pool.py`。`num_cpus=0`。默认亲和 driver 所在节点。

`LLMServerManager._setup_kv_cache_pool` 在 `enabled=True` 时**一进来**就做 GPU offload 拒绝（不论 `auto_start`）：若 `not is_torch_npu_available(check_device=False)`，则 `store.ssd_offload_path` 非空 → `ValueError`，`store.enable_offload is True` → `NotImplementedError`。避免先拉起 master 再在 replica 失败。

Actor 名固定为 `verl_mooncake_master_{job_id}`，其中 `job_id = ray.get_runtime_context().get_job_id()`。`ray job submit` 和直接 `python` 调用（`main_ppo` 里的 `ray.init()`）都会得到该 ID；verl 现有 `VERL_RAY_JOB_ID` 已用同一来源。`LLMServerManager._setup_kv_cache_pool` 必须在 `ray.init()` 之后调用（与现有 `create()` 一致）。先 `ray.get_actor`：已存在则复用，把已有 `ip:port` 写回 DictConfig，不新起 `mooncake_master`。不存在且 `auto_start=True` 才创建。创建时用 `NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)` 绑在 driver 节点。

`auto_start=True`：

1. `master.port` 非 `None`：绑该端口，失败即启动失败。`None`：actor 节点 `get_free_port`。
2. `Popen(["mooncake_master", "--port", str(port), ...])`，附加 eviction 参数；`default_kv_lease_ttl` / `client_ttl` 非 `None` 才传；NPU SSD 时加 `--enable_offload=true`；`enable_multi_tenants=True` 时加 `--enable_multi_tenants=true --tenant_quota_connector_type ... --tenant_quota_connector_uri ...`。
3. `ray.util.get_node_ip_address()` 得到 `ip:port`，写回 DictConfig。
4. TCP probe：**只**在 `MooncakeMasterActor.start` 里做（不在 process wrapper 的 `start` 里）。每 0.5s 连一次该地址，超时 30s。失败则带出 cmd、stdout/stderr，启动失败。

`auto_start=False`：不创建 actor。无 `config_path` 时必须已有 `master.address`。有 `config_path` 时在 `vLLMHttpServer` 读用户 JSON 的 `master_server_address`，缺失或文件不存在则启动失败并带路径。

`stop()`：若进程仍在，先 `terminate`，等 5s，仍在则 `kill`。`stop()` 必须幂等。每个 manager 都注册 `atexit` 调 `stop()`（复用同一 actor 时多次调用安全）。actor 析构也调 `stop()`。禁止 terminate 不是本 actor 拉起的 `mooncake_master`。

与 TransferQueue 共存：各用各的端口；不复用 TQ 的 `auto_init`。

二进制不在 `PATH`：启动失败，提示安装 `mooncake-transfer-engine`（GPU）或 `mooncake-transfer-engine-npu`（NPU）。

`mooncake_master` 是控制面 RPC。driver 不在计算网时设 `auto_start=False` 并指向可达的外部 master。

### 4.4 JSON 落盘

路径固定：`{tempfile.gettempdir()}/verl_mooncake_{job_id}.json`，`job_id` 同上，等于 `VERL_RAY_JOB_ID`。同节点所有 P/D server 写同一文件（生成内容相同，允许覆盖）。

| `store.config_path` | 行为 |
|---|---|
| 未设 | 由 store 配置 + 解析后的 `master.address` 生成。`vLLMHttpServer.launch_server` 开头写入上述路径 |
| 已设（此时 `auto_start` 必为 false） | 不生成、不 mkdir；`MOONCAKE_CONFIG_PATH` 指向用户路径 |

GPU JSON 字段（始终这些 key）：`mode`、`metadata_server`、`master_server_address`、`global_segment_size`、`local_buffer_size`、`protocol`、`device_name`、`enable_offload`、`tenant_id`。`enable_offload` **硬编码 `false`**（不读 `store.enable_offload`）。不写 NPU 专用 key，不写 `ssd_offload_path`。

NPU JSON 字段：`metadata_server`、`protocol`、`device_name`、`master_server_address`、`global_segment_size`、`preferred_segment`、`prefer_alloc_in_same_node`、`enable_ssd_offload`、`tenant_id`。`enable_ssd_offload==true` 时再写 `ssd_offload_path`。不写 `mode`、`local_buffer_size`、GPU 的 `enable_offload`。

### 4.5 P/D server 环境变量

在 `_spawn_pd_server` 的 `runtime_env.env_vars` 增加：

- `MOONCAKE_CONFIG_PATH`（生成路径或用户 `config_path`）
- `PYTHONHASHSEED=<python_hash_seed>`

不增加 `MOONCAKE_MASTER`、`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`、`LD_LIBRARY_PATH`。

`lookup_rpc_port` 为 `None` 或 `0` 时：每个 P/D server 在 `_spawn_pd_server` 里对 `prefill_host_ip` 调用 `get_free_port`（与 side channel 相同），再调用 `_build_kv_transfer_config` / `build_store_connector_config`。用户指定正整数则原样传入。关 Pool 时 `launch_servers` 先组装单连接器再传入 spawn，spawn 不分配 lookup 端口。

### 4.6 多 P 预留

`vLLMPDReplica` 本阶段仍 `prefill_replicas!=1` → `NotImplementedError`。

Master / JSON / hash seed / named actor 的隔离键都是当前 driver 的 `get_job_id()`。以后多 P 时所有 P/D 继续读同一 `master.address`。`build_store_connector_config` 现在就接收 `prefill_tp_sizes: list[int]`：当前调用方传 `[prefill_tp]`；多 P 落地后传所有 P 的 TP。自动 lcm 规则见 5.3。

## 5. MultiConnector 组装

`cache_pool.enabled=False`：replica 保持今天的单 P2P 连接器（含 **NIXL**），不调用 Store 组装。

`enabled=True`：只生成 `MultiConnector`，`connectors` 顺序固定为 `[P2P, Store]`。Pool 强制 `transfer_backend=mooncake`，NIXL 不会进入 MultiConnector。

纯函数固定放在 `kv_cache_pool.py`，replica 只调用、不复制逻辑：

- `build_p2p_connector_config(...)`：**只产出子连接器 dict**（`kv_connector` / `kv_role` / 平台 extra / NPU `kv_port`）。不含 `engine_id`、`kv_buffer_device`。
- `build_store_connector_config(...)`：只产出 Store 子连接器。
- `build_kv_transfer_config(...)`：调用前两个；**始终在顶层**写入 `engine_id`、`kv_buffer_device`。关 Pool：把 P2P child 的键展平到顶层（`{**p2p, "engine_id": ..., "kv_buffer_device": ...}`）。开 Pool：顶层 `MultiConnector`，child 进 `connectors[0]`，Store 进 `connectors[1]`。

### 5.1 顶层

官方 PD 示例顶层只有 `kv_connector` / `kv_role` / `kv_connector_extra_config`。verl 额外在**顶层**写 `engine_id` 和 `kv_buffer_device`（现有 PD 的 `set_pd_peer` 与 `_pd_dispatch` 需要 `engine_id`）。这两项不进入子连接器。关 Pool 展平后它们与 P2P 字段同级，形状与今天单连接器一致。

```python
{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer" | "kv_consumer",
    "engine_id": "<per-instance uuid>",
    "kv_buffer_device": "<cuda|npu>",
    "kv_connector_extra_config": {"connectors": [p2p_cfg, store_cfg]},
}
```

`kv_load_failure_policy` 仅当配置非 `None` 时出现在顶层。`kv_port` 只出现在 NPU P2P 子连接器，不出现在顶层。`engine_id` / `kv_buffer_device` 不复制到子连接器。

### 5.2 P2P 子连接器

以官方文档为准。

GPU（官方 PD 示例 + `disaggregation.mooncake_protocol`，因为那是 P2P 传输项，不是旧顶层 payload）：

```python
{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer" | "kv_consumer",
    "kv_connector_extra_config": {"mooncake_protocol": "<disaggregation.mooncake_protocol>"},
}
```

`mooncake_protocol` 仅当配置非空时写入 extra；若写入后 extra 为空则省略 `kv_connector_extra_config` 键。当前 `DisaggregationConfig.mooncake_protocol` 默认 `"nvlink"`，因此 GPU P2P extra 会带 `mooncake_protocol=nvlink`。

NPU（官方 PD 示例）：

```python
{
    "kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer" | "kv_consumer",
    "kv_port": <int>,
    "kv_connector_extra_config": {
        "prefill": {"dp_size": 1, "tp_size": prefill_tp},
        "decode": {"dp_size": 1, "tp_size": decode_tp},
    },
}
```

`decode_tp = disaggregation.decode_tensor_model_parallel_size or tensor_model_parallel_size`。`dp_size` 固定为 `1`（PD 仍拒绝 `data_parallel_size!=1`）。

### 5.3 Store 子连接器

**GPU `MooncakeStoreConnector`**（官方 PD：Prefill `kv_both`，Decode `kv_consumer`）：

```python
{
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both" | "kv_consumer",
    "kv_connector_extra_config": { ... },  # 按 3.5 写入规则
}
```

`store_tp_size`：

1. `decode_tp` 定义同 5.2。
2. 用户设置了 `connector.store_tp_size`：用该值。
3. 否则：`lcm(prefill_tp, decode_tp)`。当前调用方只有一个 prefill TP。
4. 校验：对 `prefill_tp` **和** `decode_tp` 均满足 `store_tp_size >= tp` 且 `store_tp_size % tp == 0`（共享 Store，构建期两端一起查）。不校验模型 KV head 数；不满足时由 vLLM 运行期处理。
5. GPU extra **始终**包含该整数。

多 P 自动 lcm（辅助函数现在就实现）：

- 入参 `prefill_tps: list[int]`。当前为 `[prefill_tp]`。
- 若 `len(set(prefill_tps)) > 1` 且 `enable_store_tp_lcm is not False`：写入 `enable_store_tp_lcm=true`、`prefill_tp_sizes=prefill_tps`，并把 `store_tp_size` 设为 `lcm(*prefill_tps, decode_tp)`（用户显式 `store_tp_size` 仍覆盖，再做步骤 4 校验）。
- 若用户显式 `enable_store_tp_lcm=False`：不写 lcm 字段，只按步骤 2–5 写 `store_tp_size`。

verl 不保证非对称 TP 下 greedy 与重算逐 bit 一致（官方说明）。

**NPU `AscendStoreConnector`**（官方 PD：`kv_role` 与 P2P 相同）：

```python
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_producer" | "kv_consumer",
    "kv_connector_extra_config": {
        "lookup_rpc_port": <正整数>,
        # 其余按 3.5：非默认才写
    },
}
```

`backend` 为默认 `mooncake` 时不写；用户改了才写（改成非 mooncake 已在配置期 `NotImplementedError`）。

最后把 `extra_config` merge 进 Store extra。

### 5.4 `_pd_dispatch`

看 P2P 名字：

- 顶层 `kv_connector == "MultiConnector"`：取 `kv_connector_extra_config.connectors[0].kv_connector`。`connectors` 缺失或为空 → `RuntimeError`。不扫描「第一个 Mooncake*」。
- 否则：用顶层 `kv_connector`。

| P2P 名字 | 行为 |
|---|---|
| `MooncakeConnector` | 本地构造 decode `kv_transfer_params`（`remote_engine_id` + `http://127.0.0.1:{bootstrap}` + `transfer_id`） |
| `MooncakeConnectorV1` / `NixlConnector` | 使用 prefill 返回的 `kv_transfer_params` |

不要把 `MooncakeConnectorV1` 当成 GPU bootstrap 协议。

### 5.5 权重更新

`vLLMPDReplica.sleep` / `wake_up` 已经 gather 全部 P/D server。每个 `vLLMHttpServer` 现有的 `reset_prefix_cache(reset_connector=True)`（wake / sleep / clear / abort）负责清 Store。不另做 `mooncake_master` flush。`cache_aware` decode 路由仍是 replica 内 radix。

## 6. 错误处理

启动期失败即停：

| 条件 | 异常 |
|---|---|
| Pool 开启但非 vLLM PD | `ValueError` |
| `transfer_backend != mooncake` | `ValueError` |
| `enable_prefix_caching=False` | `ValueError` |
| PD 启动时存在非空 `engine_kwargs.vllm.kv_transfer_config` | `ValueError` |
| `mode=standalone-store` | `ValueError` |
| `auto_start=True` 且 `config_path` 非空 | `ValueError` |
| `backend` 为 memcache / yuanrong | `NotImplementedError` |
| `auto_start=False` 且无 master 地址且无可用 JSON 地址 | `ValueError` |
| `enable_multi_tenants=True` 缺 quota 字段 | `ValueError` |
| GPU `ssd_offload_path` 非空 | `ValueError`（`setup_kv_cache_pool` 与 replica 各查一次） |
| GPU `enable_offload=True` | `NotImplementedError`（同上，拉起 master 之前） |
| NPU `global_segment_size` 未 1GB 对齐 | `ValueError` |
| Store `protocol` 不在平台允许集 | `ValueError` |
| GPU `store_tp_size` 对 prefill_tp 或 decode_tp 不合法 | `ValueError` |
| `use_layerwise` + mooncake | `ValueError` |
| 平台上设置了对方非默认专用字段 | `ValueError` |
| 指定 `master.port` bind 失败 | 启动失败 |
| `mooncake_master` 不存在 / 秒退 / probe 超过 30s | 启动失败，带 cmd 与输出 |
| 用户 `config_path` 不存在或缺少 `master_server_address` | `vLLMHttpServer` 启动失败并带路径 |
| MultiConnector 缺少 `connectors[0]` | `RuntimeError` |

运行期 Store GET 失败交给 vLLM。verl 不重试。

## 7. 测试

CPU 单测，本需求不加 GPU/NPU 真机 PD。

1. `tests/workers/config/test_cache_pool_config_on_cpu.py`：默认关闭、嵌套 coercion、平台无关校验、`enabled=False` 跳过非法值、`auto_start`+`config_path` 互斥、`enable_multi_tenants` 缺 quota、`enable_prefix_caching` 冲突。
2. 扩展 `tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py`（只补 dispatch 与现有 PD 路径，**形状断言不放这里**）：
   - `_pd_dispatch`：顶层 MultiConnector + `connectors[0]==MooncakeConnector` 必须本地构造 decode params；缺少 `connectors[0]` 抛 `RuntimeError`
   - `MooncakeConnectorV1`（含包在 MultiConnector 内）仍走 prefill 返回的 params
   - `cache_pool.enabled=False` 时现有单连接器单测不变（含 NIXL）
3. `tests/workers/rollout/test_kv_cache_pool_on_cpu.py`：JSON 字段集（GPU/NPU、SSD 开/关；NPU 断言 `preferred_segment` / `prefer_alloc_in_same_node`）、Store TP 对 P/D 两端校验与多 P lcm、GPU/NPU MultiConnector 形状、关 Pool 展平（含 NIXL）、`kv_load_failure_policy`、named actor 复用、probe 失败带 cmd/stdout/stderr、`stop()` 幂等与析构、GPU `enable_offload`/`ssd_offload_path` 在 setup 阶段提前拒绝、双源 `engine_kwargs.vllm.kv_transfer_config`（经 replica 静态方法，**不**放 PD 测试文件）。
4. `rollout.yaml` 注释格式；跑 `scripts/generate_trainer_config.sh`。

## 8. 文件清单

| 动作 | 路径 |
|---|---|
| 新增 | `verl/workers/config/cache_pool.py` |
| 新增 | `verl/workers/rollout/vllm_rollout/kv_cache_pool.py` |
| 新增 | `tests/workers/config/test_cache_pool_config_on_cpu.py` |
| 新增 | `tests/workers/rollout/test_kv_cache_pool_on_cpu.py` |
| 修改 | `verl/workers/config/rollout.py`（字段、`__post_init__`、`__all__`） |
| 修改 | `verl/workers/config/__init__.py` |
| 修改 | `verl/trainer/config/rollout/rollout.yaml` |
| 修改 | `verl/trainer/config/_generated_*.yaml`（必须经 `scripts/generate_trainer_config.sh`） |
| 修改 | `verl/workers/rollout/llm_server.py` |
| 修改 | `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py` |
| 修改 | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` |
| 修改 | `tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py` |
| 修改 | `docs/perf/rollout_kv_offload.md` |

## 9. 文档

更新 `docs/perf/rollout_kv_offload.md`：

- PD + `cache_pool` 的 GPU / NPU 分示例
- 按 `get_job_id()` 隔离的 named actor、`auto_start` 与 `config_path` 互斥、SSD 仅 NPU、GPU `enable_offload=true` / `ssd_offload_path` 均不支持、非对称 TP、两套 protocol 并存
- 非 default `tenant_id` 需要 Mooncake ≥ 0.3.12
- NPU SSD 建议自行设置 `client_ttl`
- 硬件 env 不由 verl 注入
- 保留非 PD `engine_kwargs` 旁路，并写明与 PD Pool 互斥
