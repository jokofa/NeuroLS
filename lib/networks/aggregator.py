#
from typing import Union, Tuple, Optional, List, Any
import gym
import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch_scatter import scatter_mean, scatter_max

from lib.networks.formats import Obs, GraphObs, Emb


class DummyAggregator(nn.Module):
    """Simple dummy just passing trough the provided embeddings."""
    def __init__(self, *args, **kwargs):
        super(DummyAggregator, self).__init__()

    def forward(self,
                obs: Union[Obs, GraphObs],
                emb: Emb,
                dims: Tuple[int, int, int],
                **kwargs) -> Emb:
        # just pass through
        return emb


class Aggregator(nn.Module):
    """
    Module which manages the aggregation and combination
    of different embedding components.

    Args:
            observation_space: env observation space
            action_space: env action space
            emb_dim: dimension of embeddings
            components: embedding components to pool and aggregate
            pooling: type of pooling to apply (one of ['mean', 'max'])
            combination: how to combine embedding components (one of ['sum', 'cat'])
            env_modes: env mode identifiers
    """
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 action_space: Union[gym.spaces.Discrete, gym.spaces.MultiDiscrete],
                 emb_dim: int,
                 components: Union[str, List[str]] = "node_feature_emb",
                 pooling: Union[str, List[str]] = "mean",
                 combination: str = "cat",
                 env_modes: dict = None,
                 **kwargs):
        super(Aggregator, self).__init__()

        self.emb_dim = emb_dim
        self._observation_space = observation_space
        self._action_space = action_space
        # check if component exists
        components = [components] if isinstance(components, str) else components
        c_list = list(Emb._fields)
        for c in components:
            if c not in c_list:
                raise KeyError(f"Specified component is not part of embedding.")
        self.components = components
        self.n_components = len(components)
        # match pooling type to each component
        if isinstance(pooling, str):
            pooling = [pooling] * self.n_components
        assert len(pooling) == self.n_components, f"Must specify same number of components and pooling types."
        self.pooling_type = [f"_{p.lower()}" for p in pooling]  # for getattr() of self
        # check combination
        if combination not in ["sum", "cat"]:
            raise ValueError(f"Unknown combination: {combination}")
        self.combination = combination

        self.env_modes = env_modes
        self.num_selects = 0

        self.comb_fusion_layer = None
        self.meta_feature_proj = None
        self.pool = None
        self.create_layers(**kwargs)

    def reset_parameters(self):
        """Reset layer parameters convenience function."""
        if self.comb_fusion_layer is not None:
            self.comb_fusion_layer.reset_parameters()
        if self.meta_feature_proj is not None:
            self.meta_feature_proj.reset_parameters()

    def create_layers(self, **kwargs):
        if self.combination == "cat" and self.n_components > 1:
            self.comb_fusion_layer = nn.Linear(self.n_components * self.emb_dim, self.emb_dim, bias=False)
        if "meta_feature_emb" in self.components:
            self.meta_feature_proj = nn.Linear(self._observation_space['meta_features'].shape[0], self.emb_dim)

    @staticmethod
    def _max(x: Tensor, index: Optional[LongTensor] = None):
        if index is not None:
            assert len(x.shape) == 2
            return scatter_max(x, index=index, dim=0)[0]
        assert len(x.shape) == 3
        return torch.max(x, dim=1)[0]

    @staticmethod
    def _mean(x: Tensor, index: Optional[LongTensor] = None):
        if index is not None:
            assert len(x.shape) == 2
            return scatter_mean(x, index=index, dim=0)
        assert len(x.shape) == 3
        return torch.mean(x, dim=1)

    @staticmethod
    def _none(x: Tensor, index: Optional[LongTensor] = None):
        if len(x.shape) >= 3:
            x = x.view(x.size(0), -1)
        assert len(x.shape) == 2
        return x

    def forward(self,
                obs: Union[Obs, GraphObs],
                emb: Emb,
                dims: Tuple[int, int, int],
                **kwargs) -> Emb:

        if "meta_feature_emb" in self.components:
            assert obs.meta_features is not None
            emb = emb.update(meta_feature_emb=self.meta_feature_proj(obs.meta_features))

        bs, n, f = dims
        # pool all specified components
        emb_components = [
            getattr(self, p)(emb[c].view(bs, -1, self.emb_dim))
            for c, p in zip(self.components, self.pooling_type)
        ]

        # combine components
        if self.n_components > 1:
            if self.combination == "cat":
                # concatenate components and project back to embedding dimension
                aggr_emb = self.comb_fusion_layer(torch.cat(emb_components, dim=-1))
            elif self.combination == "sum":
                # simply sum embedding components
                aggr_emb = torch.sum(torch.stack(emb_components, dim=1), dim=1)
            else:
                raise ValueError(self.combination)
        else:
            aggr_emb = emb_components[0]

        return emb.update(
            aggregated_emb=aggr_emb,
            option_set_emb=torch.tensor([])
        )
