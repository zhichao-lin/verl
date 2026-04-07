import os
import ray
import warnings

from typing import Optional
from .scheduler.worker import Worker
from .scheduler.cluster import Cluster
from .scheduler.cluster.node import NodeInfo
from .scheduler.hardware.accelerators.accelerator import AcceleratorUtil

_RLINF_WORKER = None


def update_nvidia_gpu_env(visible_devices: list[int]):
    env_vars = {}

    # Simulator env vars
    if len(visible_devices) > 0:
        env_vars["MUJOCO_EGL_DEVICE_ID"] = str(visible_devices[0])

    # NCCL env vars
    env_vars["NCCL_CUMEM_ENABLE"] = "0"
    env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    if os.environ.get("NCCL_CUMEM_ENABLE", "0") != "0":
        warnings.warn(
            f"NCCL_CUMEM_ENABLE is set to {os.environ['NCCL_CUMEM_ENABLE']}. However, "
            "This may increase memory overhead with cudagraph+allreduce: "
            "https://github.com/NVIDIA/nccl/issues/1234, and thus set to 0 by both vLLM and SGLang, see https://github.com/vllm-project/vllm/pull/24141.",
        )
        env_vars["NCCL_CUMEM_ENABLE"] = os.environ["NCCL_CUMEM_ENABLE"]
    
    for key, val in env_vars.items():
        os.environ[key] = val


def create_rlinf_worker(group_name: str, rank: int, world_size: int):
    global _RLINF_WORKER

    if _RLINF_WORKER is None:
        cluster = Cluster()
        
        cur_node_id = ray.get_runtime_context().get_node_id()
        cur_node_rank: Optional[int] = None
        cur_node_info: Optional[NodeInfo] = None
        for i in range(cluster.num_nodes):
            node_info = cluster.get_node_info(i)
            if node_info.ray_id == cur_node_id:
                cur_node_rank = node_info.node_rank
                cur_node_info = node_info
                break
        if cur_node_rank is None:
            raise RuntimeError(f"Cannot find node_rank for Ray node_id={cur_node_id}")
        
        accelerator_type = cur_node_info.accelerator_type
        accelerator_model = cur_node_info.accelerator_model
        
        node_group = cluster.get_node_group()
        assert node_group is not None, "Default node group not found in Cluster"
        node_group_label = node_group.label

        # Worker context
        # RAY_NOSET | isolate_accelerator | visible_devices | local_hardware_ranks | local_accelerator_rank | node_local_rank | node_local_world_size
        #     -     |       True          |      [2]        |         "2"          |          2             |       0         |          1
        #     1     |       True          |   [0,1,2,3]     |      "0,1,2,3"       |          2             |       2         |          1
        # AgentLoopWorker context
        #     -     |       True          |       []        |         ""           |         -1             |       0         |          1
        #     1     |       True          |   [0,1,2,3]     |      "0,1,2,3"       |          0             |       0         |          1
        #     1     |       True          |   [4,5,6,7]     |      "4,5,6,7"       |          4             |       0         |          1
        # AgentLoopWorker context - manually set CUDA_VISIBLE_DEVICES
        #     1     |       True         |       [2]        |         "2"          |          2             |       0         |          1
        visible_devices = AcceleratorUtil.get_visible_devices(accelerator_type)
        local_hardware_ranks = ",".join(map(str, visible_devices))

        if accelerator_type == "NV_GPU":
            update_nvidia_gpu_env(visible_devices)

        device_name = "NPU" if accelerator_type == "NPU" else "GPU"
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()[device_name]
        if len(accelerator_ids) > 0:
            local_accelerator_rank = int(accelerator_ids[0])
        else:
            if len(visible_devices) > 0:
                local_accelerator_rank = visible_devices[0]
            else:
                local_accelerator_rank = -1

        node_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        node_local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))

        _RLINF_WORKER = Worker.attach_to_current_ray_actor(
            group_name=group_name,
            rank=rank,
            world_size=world_size,
            cluster_node_rank=cur_node_rank,
            node_group_label=node_group_label,
            accelerator_type=accelerator_type,
            accelerator_model=accelerator_model,
            local_accelerator_rank=local_accelerator_rank,
            node_local_rank=node_local_rank,
            node_local_world_size=node_local_world_size,
            local_hardware_ranks=local_hardware_ranks,
            isolate_accelerator=True,
            catch_system_failure=False,
        )
    elif _RLINF_WORKER.group_name != group_name:
        warnings.warn(
            "create_rlinf_worker called with different group_name in the same process: "
            f"existing={_RLINF_WORKER.group_name}, requested={group_name}"
        )
    return _RLINF_WORKER
