#
from typing import NamedTuple, Union, List, Optional
import numpy as np
import torch

__all__ = ["RPInstance", "RPSolution"]


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


class RPInstance(NamedTuple):
    """Typed routing problem instance wrapper."""
    coords: Union[np.ndarray, torch.Tensor]
    node_features: Union[np.ndarray, torch.Tensor]
    graph_size: int
    depot_idx: List = [0]
    constraint_idx: List = [-1]
    vehicle_capacity: float = -1
    max_num_vehicles: Optional[int] = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))


class RPSolution(NamedTuple):
    """Typed wrapper for routing problem solutions."""
    solution: List[List]
    cost: float = None
    num_vehicles: int = None
    run_time: float = None
    problem: str = None
    instance: RPInstance = None

    def update(self, **kwargs):
        return self._replace(**kwargs)
