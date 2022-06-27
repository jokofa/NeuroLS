#
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor

from lib.utils import get_activation_fn
from lib.networks.formats import Emb


class FFDecoder(nn.Module):
    """Fully connected (flat) decoder model.

    Args:
        input_dim: dimension of embedding
        output_dim: dimension of output logits
        hidden_dim: dimension of hidden layers
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 activation: str = "gelu",
                 **kwargs):
        super(FFDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.activation = activation
        self.layers = None
        self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # simple FF network
        layers = [nn.Linear(self.input_dim, self.hidden_dim),
                  get_activation_fn(self.activation, module=True, **kwargs)]
        for _ in range(max(self.num_layers-2, 0)):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(get_activation_fn(self.activation, module=True, **kwargs))
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self,
                emb: Emb,
                state: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:

            emb: batched embedding tuple from encoder and aggregator
            state: optional RNN hidden state

        Returns:
            logits: logits over action dimension
            state: optional RNN state
        """
        # flat FF encoder just uses the aggregated embedding
        # to compute logits over action space dimension
        x = emb.aggregated_emb
        return (
            self.layers(x),
            state
        )


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

    I = 32
    O = 10
    nf = torch.randn(bs, n, I).to(device)
    emb = Emb(
        node_feature_emb=nf,
        aggregated_emb=torch.randn(bs, I).to(device),
    )

    for l in num_layers:
        try:
            d = FFDecoder(I, O, num_layers=l).to(device)
            logits, _ = d(emb)
            assert logits.size() == torch.empty((bs, O)).size()
        except Exception as e:
            raise type(e)(
                str(e) + f" - (num_layers: {l})\n"
            ).with_traceback(sys.exc_info()[2])

