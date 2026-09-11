# Rollout KV Cache Offload via Mooncake-Store

Last updated: 09/09/2026.

Offload prefix KV blocks from the vLLM rollout engine to a shared
[Mooncake](https://github.com/kvcache-ai/Mooncake) store so long shared
prefixes (system prompt, agentic tool history, `rollout.n` samples per prompt)
get deduplicated across requests and rollout replicas. This also helps
long-tail load balancing: when work migrates to idle rollout replicas, shared
prefix KV reduces the re-prefill cost.

There are two mutually exclusive ways to enable this in verl:

- **Non-PD (colocated)**: pass `MooncakeStoreConnector` through
  `engine_kwargs.vllm.kv_transfer_config`.
- **PD disaggregation**: set `rollout.cache_pool.enabled=true` so verl builds a
  `MultiConnector` of P2P + Store.

Do not enable both. PD replica-generated `kv_transfer_config` overwrites the
engine-kwargs bypass.

## Non-PD (colocated) offload

**Mutually exclusive with `rollout.cache_pool.enabled=true`.**

Follow vLLM's official guide for installing the Mooncake client, starting a
master, and writing the JSON config:
**<https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/>**

verl forwards `engine_kwargs.vllm.*` straight to `vllm serve` as CLI flags.
To attach the Mooncake connector, set `kv_transfer_config`:

```yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      vllm:
        kv_transfer_config: |-
          {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
              "mooncake_config_path": "/path/to/mooncake_config.json"
            }
          }
```

Or as a Hydra CLI override:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=MooncakeStoreConnector \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.mooncake_config_path=/path/to/mooncake_config.json
```

## PD disaggregation + KV Cache Pool

Requires `rollout.name=vllm`, `disaggregation.enabled=true`,
`disaggregation.transfer_backend=mooncake`, and `enable_prefix_caching=true`.

verl then builds `MultiConnector = [P2P, Store]`:

| Platform | P2P | Store |
|---|---|---|
| GPU | `MooncakeConnector` | `MooncakeStoreConnector` |
| NPU | `MooncakeConnectorV1` | `AscendStoreConnector` |

NPU Store reference:
**<https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_pool.html>**
(or the matching `vllm-ascend` `kv_pool.md` in your tree).

One Ray driver (`ray.get_runtime_context().get_job_id()`) shares one Mooncake
Store and one `mooncake_master`. That JobID exists for both `python -m
verl.trainer.main_ppo` and `ray job submit`. Multiple `LLMServerManager.create()`
calls in the same driver reuse the named actor `verl_mooncake_master_{job_id}`.

`cache_pool.master.auto_start=true` (default) starts or reuses that master.
`auto_start=true` and `store.config_path` together raise `ValueError`. With
`auto_start=false`, point at an external master via `master.address` or a
user JSON at `store.config_path` (every node must be able to read that file).

verl does not inject `MOONCAKE_MASTER`, `LD_LIBRARY_PATH`, or
`ASCEND_GLOBAL_RESOURCE_CONFIG`. Generated JSON only writes
`master_server_address`.

P2P transport uses `disaggregation.mooncake_protocol` (GPU default `nvlink`).
Store transport uses `cache_pool.store.protocol` (`null` → GPU `rdma`, NPU
`ascend`). The two protocols are independent and both may be set.

GPU `store_tp_size` defaults to `lcm(prefill_tp, decode_tp)` and may be
overridden. NPU does not write `store_tp_size`. Asymmetric TP is therefore a
GPU Store extra plus P2P child fields.

SSD offload is NPU-only (`store.ssd_offload_path` non-null). On GPU,
`enable_offload=true` raises `NotImplementedError` and a non-null
`ssd_offload_path` raises `ValueError`. For NPU SSD, set
`cache_pool.master.client_ttl` yourself (Mooncake's SSD guide suggests `120`);
verl does not auto-fill it.

A non-default `store.tenant_id` needs Mooncake ≥ 0.3.12. verl does not probe
the installed version.

### GPU example (1P, asymmetric TP, no SSD)

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

### NPU example (1P, symmetric TP + SSD)

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

Install `mooncake-transfer-engine` on GPU or `mooncake-transfer-engine-npu` on
NPU so `mooncake_master` is on `PATH` when `auto_start=true`.

## RL correctness: hard reset on every weight update

verl clears both local and Mooncake KV caches at every weight update boundary
to avoid reusing KV from the previous policy.

On the PD + `cache_pool` path this is the existing
`reset_prefix_cache(reset_connector=True)` on each P/D `vLLMHttpServer`
(wake / sleep / clear / abort). verl does not flush `mooncake_master`
separately.

**Required vLLM version**: use vLLM 0.22 or newer. Older builds may leave stale
KV in the Mooncake master after a weight update.
