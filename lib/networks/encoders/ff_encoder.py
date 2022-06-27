#
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lib.utils import get_activation_fn, get_norm, rm_from_kwargs
from lib.networks.formats import Obs, Emb
from lib.networks.encoders.base_encoder import BaseEncoder


class FFBlock(nn.Module):
    """
    Feed forward network block with activation, skip connection and regularization options.

    Architecture is as follows:
        input -> Norm -> Dropout -> Linear -> Activation
              -> Norm -> Dropout -> Linear -> Skip(Residual/input) -> Activation -> Output

    Inspirational Refs:
        https://arxiv.org/abs/2108.08186
        https://arxiv.org/abs/1811.03087

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 activation: str = "gelu",
                 skip: bool = True,
                 norm_type: Optional[str] = "ln",
                 dropout: float = 0.25,
                 bias: bool = True,
                 **kwargs):
        super(FFBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.skip = skip
        self.dropout = dropout
        self.norm1 = get_norm(norm_type, hdim=input_dim, **kwargs)
        self.lin1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.act1 = get_activation_fn(activation, module=True, **kwargs)
        self.norm2 = get_norm(norm_type, hdim=hidden_dim, **kwargs)
        self.lin2 = nn.Linear(hidden_dim, input_dim, bias=bias)
        self.act2 = get_activation_fn(activation, module=True, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.skip:
            x_ = x
        if self.norm1 is not None:
            bs, n, d = x.size()
            x = self.norm1(x.view(-1, d)).view(bs, n, d)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)
        x = self.act1(self.lin1(x))
        if self.norm2 is not None:
            bs, n, d = x.size()
            x = self.norm2(x.view(-1, d)).view(bs, n, d)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)
        x = self.lin2(x)
        if self.skip:
            x += x_
        return self.act2(x)


class FFNodeEncoder(BaseEncoder):
    """Fully connected feed forward encoder model for node embeddings."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 **kwargs):
        super(FFNodeEncoder, self).__init__(input_dim, output_dim, hidden_dim)
        self.num_layers = num_layers
        self.layers = None
        kwargs = rm_from_kwargs(kwargs, keys=["edge_feature_dim"])
        self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # input layer
        layers = [nn.Linear(self.input_dim, self.hidden_dim)]
        # intermediate hidden block modules
        for _ in range(self.num_layers):
            layers.append(FFBlock(self.hidden_dim, self.hidden_dim, **kwargs))
        # final output layer
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, obs: Obs, emb: Emb = Emb(), **kwargs) -> Emb:
        return emb.update(node_feature_emb=self.layers(obs.node_features))


#
# ============= #
# ### TEST #### #
# ============= #
def _test(
        bs: int = 5,
        n: int = 10,
        cuda=False,
        seed=1
):
    import sys
    import torch
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    num_layers = [2, 3, 4]
    norm_types = [None, "ln", "bn"]
    dropout = [0.0, 0.2, 0.8]

    I = 4
    O = 16
    x = Obs(
        node_features=torch.randn(bs, n, I).to(device),
        current_sol=None,
        best_sol=None
    )

    for l in num_layers:
        for norm in norm_types:
            for drp in dropout:
                try:
                    e = FFNodeEncoder(I, O, num_layers=l, norm_type=norm, dropout=drp).to(device)
                    out = e(x)
                    emb = out.node_feature_emb
                    assert emb.size() == torch.empty((bs, n, O)).size()
                except Exception as e:
                    raise type(e)(
                        str(e) + f" - (num_layers: {l}, norm: {norm}, dropout: {drp})\n"
                    ).with_traceback(sys.exc_info()[2])

