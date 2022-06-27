#
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from tianshou.utils.net.discrete import CosineEmbeddingNetwork

from lib.utils import get_activation_fn


class IQN(nn.Module):
    """Implicit Quantile Network.

    see https://arxiv.org/pdf/1806.06923.pdf
    and https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/discrete.py#L158

    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_cosines: int = 64,
                 num_layers: int = 1,
                 activation: str = "gelu",
                 **kwargs):
        super(IQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        self.emb_model = CosineEmbeddingNetwork(
            num_cosines,
            self.input_dim  # type: ignore
        )
        self.out_proj = self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # simple FF network
        layers = [nn.Linear(self.input_dim, self.hidden_dim),
                  get_activation_fn(self.activation, module=True, **kwargs)]
        for _ in range(max(self.num_layers-2, 0)):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(get_activation_fn(self.activation, module=True, **kwargs))
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self,
                logits: Tensor,
                sample_size: int,
                **kwargs) -> Tuple[Tensor, Tensor]:
        bs = logits.size(0)
        # Sample fractions.
        taus = torch.rand(
            bs, sample_size, dtype=logits.dtype, device=logits.device
        )
        emb = (logits.unsqueeze(1) * self.emb_model(taus)).view(bs * sample_size, -1)
        out = self.out_proj(emb).view(bs, sample_size, -1).transpose(1, 2)  # -> (bs, action_dim, sample_size)
        return out, taus

