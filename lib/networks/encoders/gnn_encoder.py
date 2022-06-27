#
from warnings import warn
from typing import Optional
import torch
import torch.nn as nn
from torch import LongTensor
import torch_geometric.nn as gnn

from lib.networks.encoders.graph_conv import GraphConvBlock
from lib.networks.encoders.eg_graph_conv import EGGConv
from lib.networks.encoders.base_encoder import BaseEncoder
from lib.networks.formats import GraphObs, Emb


def flip_lr(x: LongTensor):
    """
    Flip the first dimension in left-right direction.
    This is used to reverse tours by swapping
    (from, to) edges to (to, from) format.
    """
    return torch.fliplr(x.unsqueeze(0)).squeeze(0)


class GNNNodeEncoder(BaseEncoder):
    """Graph neural network encoder model for node embeddings."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 edge_feature_dim: int = 1,
                 num_nbh_layers: int = 1,
                 num_sol_layers: int = 2,
                 propagate_reverse: bool = False,
                 propagate_best: bool = False,
                 consolidate_nbh: bool = True,
                 conv_type: str = "GraphConv",
                 activation: str = "gelu",
                 skip: bool = False,
                 norm_type: Optional[str] = None,
                 add_linear: bool = False,
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
            edge_feature_dim: dimension of edge features
            num_nbh_layers: number of encoder layers for neighborhood graph
            num_sol_layers: number of encoder layers for tour graphs
            propagate_reverse: flag to also propagate in reverse tour direction
            propagate_best: flag to also propagate over best tour graph
            consolidate_nbh: flag to re-propagate over nbh graph after each tour propagation
                             (adds a new GNN layer for each re-propagation)
            conv_type: type of graph convolution
            activation: activation function
            skip: flag to use skip (residual) connections
            norm_type: type of norm to use
            add_linear: flag to add linear layer after conv
        """
        super(GNNNodeEncoder, self).__init__(input_dim, output_dim, hidden_dim)

        self.num_nbh_layers = num_nbh_layers
        self.num_sol_layers = num_sol_layers
        self.propagate_reverse = propagate_reverse
        self.propagate_best = propagate_best
        self.consolidate_nbh = consolidate_nbh
        if edge_feature_dim is not None and edge_feature_dim != 1 and conv_type.upper() != "EGGCONV":
            raise ValueError("encoders currently only work for edge_feature_dim=1")
        self.edge_feature_dim = edge_feature_dim

        self.conv_type = conv_type
        self.activation = activation
        self.skip = skip
        self.norm_type = norm_type
        self.add_linear = add_linear
        self.eggc = False

        self.input_proj = None
        self.input_proj_e = None
        self.output_proj = None
        self.nbh_layers = None
        self.cur_sol_layers = None
        self.cur_sol_rev_layers = None
        self.best_tour_layers = None
        self.best_tour_rev_layers = None

        self._static_x = None

        self.create_layers(**kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.output_proj.reset_parameters()
        self._reset_module_list(self.nbh_layers)
        self._reset_module_list(self.cur_sol_layers)
        self._reset_module_list(self.cur_sol_rev_layers)
        self._reset_module_list(self.best_tour_layers)
        self._reset_module_list(self.best_tour_rev_layers)

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                m.reset_parameters()

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # input projection layer
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        if self.conv_type.upper() == "EGGCONV":
            # special setup for EGGConv which propagates node AND edge embeddings
            self.eggc = True
            if self.activation.lower() != 'relu':
                warn(f"EGGConv normally uses RELU but got {self.activation.upper()}")
            if self.norm_type is None:
                self.norm_type = "bn"
            elif self.norm_type.lower() not in ['bn', 'batch_norm']:
                warn(f"EGGConv normally uses BN but got {self.norm_type.upper()}")
            self.input_proj_e = nn.Linear(self.edge_feature_dim, self.hidden_dim)

            def GNN():
                return EGGConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    activation=self.activation,
                    norm_type=self.norm_type,
                )
        else:
            conv = getattr(gnn, self.conv_type)

            def GNN():
                # creates a GNN module with specified parameters
                # all modules are initialized globally with the call to
                # reset_parameters()
                return GraphConvBlock(
                        conv,
                        self.hidden_dim,
                        self.hidden_dim,
                        activation=self.activation,
                        skip=self.skip,
                        norm_type=self.norm_type,
                        add_linear=self.add_linear,
                        **kwargs
                )

        # nbh embedding layers
        if self.num_nbh_layers > 0:
            self.nbh_layers = nn.ModuleList()
            for _ in range(self.num_nbh_layers):
                self.nbh_layers.append(GNN())

        inc = int(self.consolidate_nbh)
        # current tour embedding layers
        if self.num_sol_layers > 0:
            self.cur_sol_layers = nn.ModuleList()
            for _ in range(self.num_sol_layers + inc):
                self.cur_sol_layers.append(GNN())

            if self.propagate_reverse:
                self.cur_sol_rev_layers = nn.ModuleList()
                for _ in range(self.num_sol_layers + inc):
                    self.cur_sol_rev_layers.append(GNN())

        # best tour embedding layers
        if self.num_sol_layers > 0 and self.propagate_best:
            self.best_tour_layers = nn.ModuleList()
            for _ in range(self.num_sol_layers + inc):
                self.best_tour_layers.append(GNN())

            if self.propagate_reverse:
                self.best_tour_rev_layers = nn.ModuleList()
                for _ in range(self.num_sol_layers + inc):
                    self.best_tour_rev_layers.append(GNN())

        h2o_dim = self.hidden_dim*2 if self.num_sol_layers > 0 and self.propagate_best else self.hidden_dim
        self.output_proj = nn.Linear(h2o_dim, self.output_dim)

    def forward(self,
                obs: GraphObs,
                emb: Emb = Emb(),
                recompute: bool = True,
                **kwargs) -> Emb:
        bs, b_idx, x, \
        cur_sol, cur_sol_e, cur_sol_w, \
        best_tour, best_tour_e, best_tour_w, \
        nbh_e, nbh_w, _, _, _ = obs

        # (re-)compute static nbh node embeddings
        # EGGC always needs nbh_w to be encoded
        if self.training or recompute or self.eggc:
            # input layer
            x = self.input_proj(x)

            # encode nbh node embeddings
            if self.num_nbh_layers > 0:
                assert nbh_e is not None and nbh_w is not None
                if self.eggc:
                    nbh_w = self.input_proj_e(nbh_w[:, None])
                for layer in self.nbh_layers:
                    x, nbh_w = layer(x, nbh_e, nbh_w)
            if not self.training:
                self._static_x = x
        else:
            x = self._static_x

        # encode node embeddings over current tour
        if self.num_sol_layers > 0:
            x_ = x
            assert cur_sol_e is not None and cur_sol_w is not None
            if self.eggc:
                if len(nbh_w.shape) == 1:
                    nbh_w = self.input_proj_e(nbh_w[:, None])
                nbh_w_ = nbh_w
                cur_sol_w = self.input_proj_e(cur_sol_w[:, None])
            for i, layer in enumerate(self.cur_sol_layers):
                if self.consolidate_nbh and i == len(self.cur_sol_layers) - 1:
                    # last layer aggregates again over nbh
                    x, nbh_w = layer(x, nbh_e, nbh_w)
                else:
                    x, cur_sol_w = layer(x, cur_sol_e, cur_sol_w)
            if self.propagate_reverse:
                # reverse tour indices
                rev_cur_sol_e = flip_lr(cur_sol_e)
                for i, layer in enumerate(self.cur_sol_rev_layers):
                    if self.consolidate_nbh and i == len(self.cur_sol_rev_layers) - 1:
                        # last layer aggregates again over nbh
                        x, nbh_w = layer(x, nbh_e, nbh_w)
                    else:
                        x, cur_sol_w = layer(x, rev_cur_sol_e, cur_sol_w)

        # encode node embeddings over best tour
        if self.num_sol_layers > 0 and self.propagate_best:
            best_tour_x = x_    # get original nbh x embedding
            assert best_tour_e is not None and best_tour_w is not None
            if self.eggc:
                nbh_w = nbh_w_
                best_tour_w = self.input_proj_e(best_tour_w[:, None])
            for i, layer in enumerate(self.best_tour_layers):
                if self.consolidate_nbh and i == len(self.best_tour_layers) - 1:
                    # last layer aggregates again over nbh
                    best_tour_x, nbh_w = layer(best_tour_x, nbh_e, nbh_w)
                else:
                    best_tour_x, best_tour_w = layer(best_tour_x, best_tour_e, best_tour_w)
            if self.propagate_reverse:
                # reverse tour indices
                rev_best_tour_e = flip_lr(best_tour_e)
                for i, layer in enumerate(self.best_tour_rev_layers):
                    if self.consolidate_nbh and i == len(self.best_tour_rev_layers) - 1:
                        # last layer aggregates again over nbh
                        best_tour_x, nbh_w = layer(best_tour_x, nbh_e, nbh_w)
                    else:
                        best_tour_x, best_tour_w = layer(best_tour_x, rev_best_tour_e, best_tour_w)

            # concatenate current tour and best tour node embeddings
            x = torch.cat((x, best_tour_x), dim=-1)

        # output layer
        x = self.output_proj(x)

        # check for NANs
        if (x != x).any():
            raise RuntimeError(f"Output includes NANs! (e.g. GCNConv can produce NANs when <normalize=True>!)")

        # reshape to (bs, n, d) - this simplifies downstream processing
        # but will not work for batches with differently sized graphs (!)
        return emb.update(node_feature_emb=x.view(bs, -1, self.hidden_dim))


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
    from lib.env.utils import GraphNeighborhoodSampler
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # testing args
    num_nbh_layers = [0, 1, 3]
    num_layers = [0, 1, 3]
    conv_types = ["EGGConv", "GCNConv", "GraphConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "ClusterGCNConv"]
    norm_types = [None, "ln", "bn"]
    skips = [True, False]
    add_lins = [True, False]
    prop_rev = [True, False]
    prop_best = [True, False]
    consolidate = [True, False]

    # create data
    I = 4
    O = 32
    x = torch.randn(bs, n, I).to(device)

    # sample edges and weights
    sampler = GraphNeighborhoodSampler(graph_size=n, k_frac=0.5)
    coords = x[:, :, -2:]
    edge_idx, edge_weights = [], []
    for c in coords:
        ei, ew, k = sampler(c)
        edge_idx.append(ei)
        edge_weights.append(ew)
    edge_idx = torch.stack(edge_idx, dim=0).permute(1, 0, 2).reshape(2, -1)
    # transform to running idx
    idx_inc = (torch.cumsum(torch.tensor([n]*bs), dim=0) - n).repeat_interleave(k*n)
    edge_idx += idx_inc
    edge_weights = torch.stack(edge_weights).view(-1)

    x = GraphObs(
        batch_size=bs,
        batch_idx=torch.arange(bs).repeat_interleave(n, dim=-1).view(-1),
        node_features=x.view(-1, I),
        current_sol=edge_idx,
        current_sol_weights=edge_weights,
        best_sol=edge_idx,
        best_sol_weights=edge_weights,
        nbh_edges=edge_idx,
        nbh_weights=edge_weights
    )

    for nbh_l in num_nbh_layers:
        for l in num_layers:
            for c_type in conv_types:
                for norm in norm_types:
                    for skip in skips:
                        for add_lin in add_lins:
                            for pr in prop_rev:
                                for pb in prop_best:
                                    for con in consolidate:
                                        try:
                                            e = GNNNodeEncoder(I, O,
                                                               num_nbh_layers=nbh_l,
                                                               num_sol_layers=l,
                                                               propagate_reverse=pr,
                                                               propagate_best=pb,
                                                               consolidate_nbh=con,
                                                               conv_type=c_type,
                                                               norm_type=norm,
                                                               skip=skip,
                                                               add_linear=add_lin
                                                               ).to(device)
                                            out = e(x)
                                            out = out.node_feature_emb
                                            assert out.size() == torch.empty((bs*n, O)).size()
                                        except Exception as e:
                                            raise type(e)(
                                                str(e) + f" - ("
                                                         f"num_layers: {nbh_l}-{l}, "
                                                         f"conv_type: {c_type}, "
                                                         f"norm: {norm}, "
                                                         f"skip: {skip}, "
                                                         f"add_lin: {add_lin}, "
                                                         f")\n"
                                            ).with_traceback(sys.exc_info()[2])

