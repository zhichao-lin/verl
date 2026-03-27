from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.third_party.rlinf.scheduler.cluster.utils import (
    extract_dataclass_tensor_fields,
    unflatten_dataclass_tensor_fields,
)


def test_extract_dataclass_tensor_fields_supports_dataproto_tensordict_batch():
    proto = DataProto(
        batch=TensorDict(
            {
                "x": torch.arange(6, dtype=torch.float32).view(2, 3),
                "y": torch.ones((2, 3), dtype=torch.float32),
            },
            batch_size=[2],
        ),
        non_tensor_batch={"uid": np.array(["a", "b"], dtype=object)},
        meta_info={"step": 1},
    )

    tensor_fields, flat_tensors, metadata = extract_dataclass_tensor_fields(proto)

    assert "batch" in tensor_fields
    assert len(flat_tensors) == 2
    assert metadata[0][0] == "batch"
    assert metadata[0][1] == "tensordict"


def test_unflatten_dataclass_tensor_fields_restores_tensordict():
    batch = TensorDict(
        {
            "a": torch.zeros((2, 4), dtype=torch.float32),
            "b": torch.ones((2, 4), dtype=torch.float32),
        },
        batch_size=[2],
    )
    proto = DataProto(batch=batch)

    _, flat_tensors, metadata = extract_dataclass_tensor_fields(proto)
    restored = unflatten_dataclass_tensor_fields(metadata, flat_tensors)

    restored_batch = restored["batch"]
    assert isinstance(restored_batch, TensorDict)
    assert restored_batch.batch_size == batch.batch_size
    assert torch.equal(restored_batch["a"], batch["a"])
    assert torch.equal(restored_batch["b"], batch["b"])


@dataclass
class _DictTensorContainer:
    payload: dict[str, torch.Tensor]


def test_unflatten_dataclass_tensor_fields_keeps_dict_behavior():
    container = _DictTensorContainer(
        payload={"k1": torch.tensor([1.0]), "k2": torch.tensor([2.0])}
    )
    _, flat_tensors, metadata = extract_dataclass_tensor_fields(container)
    restored = unflatten_dataclass_tensor_fields(metadata, flat_tensors)

    assert isinstance(restored["payload"], dict)
    assert torch.equal(restored["payload"]["k1"], torch.tensor([1.0]))
    assert torch.equal(restored["payload"]["k2"], torch.tensor([2.0]))
