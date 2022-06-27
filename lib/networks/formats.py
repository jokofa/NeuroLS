#
from typing import Optional, NamedTuple, Union, Any
from torch import Tensor, LongTensor


class Obs(NamedTuple):
    """Named and typed tuple of RP observations."""
    node_features: Tensor
    current_sol: LongTensor
    best_sol: LongTensor
    current_sol_seqs: Optional[LongTensor] = None
    best_sol_seqs: Optional[LongTensor] = None
    meta_features: Optional[Tensor] = None


class GraphObs(NamedTuple):
    """Named and typed tuple of RP graph observations."""
    batch_size: int
    batch_idx: LongTensor
    node_features: Tensor
    current_sol: LongTensor
    current_sol_edges: LongTensor
    current_sol_weights: Tensor
    best_sol: LongTensor
    best_sol_edges: LongTensor
    best_sol_weights: Tensor
    nbh_edges: Optional[LongTensor] = None
    nbh_weights: Optional[Tensor] = None
    current_sol_seqs: Optional[LongTensor] = None
    best_sol_seqs: Optional[LongTensor] = None
    meta_features: Optional[Tensor] = None


class Emb(NamedTuple):
    """Named and typed tuple of RP embedding."""
    node_feature_emb: Optional[Tensor] = None
    edge_feature_emb: Optional[Tensor] = None
    current_sol_emb: Optional[Tensor] = None
    best_sol_emb: Optional[Tensor] = None
    sub_graph_emb: Optional[Tensor] = None
    graph_emb: Optional[Tensor] = None
    aggregated_emb: Tensor = None
    option_set_emb: Tensor = None
    option_mask: Optional[Tensor] = None
    meta_feature_emb: Optional[Tensor] = None

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def get(self, key: Union[str, int], default_val: Any = None):
        """Dict like getter method with default value."""
        try:
            return self[key]
        except AttributeError:
            return default_val

    def update(self, **kwargs):
        return self._replace(**kwargs)
