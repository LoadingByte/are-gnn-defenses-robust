from typing import Union, Sequence

import numpy as np
import torch
from torchtyping import TensorType

Int = Union[
    int, np.int8, np.int16, np.int32, np.int64,
    TensorType[(), torch.int8], TensorType[(), torch.int16], TensorType[(), torch.int32], TensorType[(), torch.int64]
]
Float = Union[
    float, np.float16, np.float32, np.float64,
    TensorType[(), torch.float16], TensorType[(), torch.float32], TensorType[(), torch.float64]
]

IntSeq = Union[
    Sequence[Int],
    TensorType[-1, torch.int8], TensorType[-1, torch.int16], TensorType[-1, torch.int32], TensorType[-1, torch.int64]
]
FloatSeq = Union[
    Sequence[Float],
    TensorType[-1, torch.float16], TensorType[-1, torch.float32], TensorType[-1, torch.float64]
]
