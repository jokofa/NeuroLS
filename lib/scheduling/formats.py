#
from typing import NamedTuple, Union, List, Optional
import numpy as np
import torch

__all__ = ["JSSPInstance", "JSSPSolution", "INF"]

INF = 1e7
ADJ_DTYPES = [np.bool, np.int8, np.int16, np.int32, np.int64]


def format_repr(k, v, space: str = ' '):
    if isinstance(v, int) or isinstance(v, float):
        return f"{space}{k}={v}"
    elif isinstance(v, np.ndarray):
        return f"{space}{k}=ndarray_{list(v.shape)}"
    elif isinstance(v, torch.Tensor):
        return f"{space}{k}=tensor_{list(v.shape)}"
    elif isinstance(v, list) and len(v) > 3:
        return f"{space}{k}=list_{[len(v)]}"
    else:
        return f"{space}{k}={v}"


class JSSPInstance(NamedTuple):
    """
    Typed problem instance wrapper for the JSSP.
    (close to the Taillard specification)
    """
    num_jobs: int
    num_machines: int
    durations: Union[np.ndarray, torch.Tensor]
    sequences: Union[np.ndarray, torch.Tensor]
    org_max_dur: float

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    @property
    def graph_size(self):
        return (self.num_jobs*self.num_machines)+2


class JSSPSolution(NamedTuple):
    """Typed wrapper for JSSP solutions."""
    solution: List[List]
    cost: float = None
    run_time: float = None
    instance: JSSPInstance = None

    def update(self, **kwargs):
        return self._replace(**kwargs)
