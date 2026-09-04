# Rollout KV Cache Offload via Mooncake-Store

Last updated: 09/02/2026.

Offload prefix KV blocks from the vLLM rollout engine to a shared
[Mooncake](https://github.com/kvcache-ai/Mooncake) store so long shared
prefixes (system prompt, agentic tool history, `rollout.n` samples per prompt)
get deduplicated across requests and rollout replicas. This also helps
long-tail load balancing: when work migrates to idle rollout replicas, shared
prefix KV reduces the re-prefill cost.

## Setup Mooncake + vLLM

Follow vLLM's official guide for installing the Mooncake client, starting a
master, and writing the JSON config:
**<https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/>**

The verl side only consumes whatever that doc produces — no extra steps.
MooncakeStoreConnector reads the JSON file from the `MOONCAKE_CONFIG_PATH`
environment variable (not from `kv_connector_extra_config`).

Start the master before training:

```bash
mooncake_master --port 50051
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

verl forwards `MOONCAKE_CONFIG_PATH` into Ray actors automatically when it is
set in the launch shell.

## Enable in colocated vLLM rollout

verl forwards `engine_kwargs.vllm.*` straight to `vllm serve` as CLI flags.
To attach the Mooncake store connector, set `kv_transfer_config`:

```yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      vllm:
        kv_transfer_config: |-
          {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both"
          }
```

Or as a Hydra CLI override:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=MooncakeStoreConnector \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both
```

## Enable in vLLM Prefill-Decode disaggregation

PD launch overwrites `engine_kwargs.vllm.kv_transfer_config` with the P2P
connector. To also attach the shared KV pool, wrap both connectors in vLLM
`MultiConnector` via:

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    disaggregation:
      enabled: True
      transfer_backend: mooncake   # GPU: nixl | mooncake; NPU: mooncake | ascend
      mooncake_protocol: nvlink    # GPU MooncakeConnector only
      enable_mooncake_store: True
      mooncake_store_config_path: /path/to/mooncake_config.json
      save_decode_cache: False     # True: decoder also writes completed decode KV
      mooncake_store_extra_config: {}
```

Equivalent CLI:

```bash
actor_rollout_ref.rollout.disaggregation.enabled=True \
actor_rollout_ref.rollout.disaggregation.transfer_backend=mooncake \
actor_rollout_ref.rollout.disaggregation.enable_mooncake_store=True \
actor_rollout_ref.rollout.disaggregation.mooncake_store_config_path=/path/to/mooncake_config.json
```

**GPU.** Prefiller is launched as `kv_producer` + `MooncakeStoreConnector(kv_both)`;
decoder as `kv_consumer` + `MooncakeStoreConnector(kv_consumer)`. When
prefill TP and decode TP differ, verl sets `store_tp_size` (or
`enable_store_tp_lcm`) on the store connector automatically.

**NPU (vLLM-Ascend).** Prefiller/decoder use `MooncakeConnectorV1` plus
`AscendStoreConnector(backend=mooncake)`. The Mooncake JSON `protocol` must
be `"ascend"` (install `mooncake-transfer-engine-npu`). When prefill TP and
decode TP differ, verl sets `prefill_tp_size` / `decode_tp_size` on the store
connector (not GPU `store_tp_size`). `save_decode_cache=True` maps to decode
`consumer_is_to_put`. Prefill lookup RPC port is `"0"`; decode uses its
side-channel port. `MooncakeConnectorV1.request_finished` returns
`kv_transfer_params` (Nixl-like), so dispatch must **not** take the GPU
Mooncake local-bootstrap path.

The colocated `engine_kwargs.vllm.kv_transfer_config` recipe
(`MooncakeStoreConnector` / `AscendStoreConnector` /
`MooncakeConnectorStoreV1`) also opts PD into the store: extras are
harvested, then composed into `MultiConnector` so P2P transfer is not dropped.

## RL correctness: hard reset on every weight update

verl clears both local and Mooncake KV caches at every weight update boundary
to avoid reusing KV from the previous policy.

**Required vLLM version**: use vLLM 0.22 or newer. Older builds may leave stale
KV in the Mooncake master after a weight update.
