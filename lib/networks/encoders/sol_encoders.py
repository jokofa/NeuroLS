#
from typing import Tuple, List
import itertools as it
import torch
import torch.nn as nn
from torch import Tensor

from lib.utils import rm_from_kwargs
from lib.networks.formats import Obs, Emb
from lib.networks.encoders.base_encoder import BaseEncoder


class FlatSolEncoder(BaseEncoder):
    """Flat encoder model for sol embeddings."""
    def __init__(self,
                 problem: str,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 propagate_best: bool = False,
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
            num_layers: number of hidden layers
            propagate_best: flag to also propagate over best sol graph

        """
        super(FlatSolEncoder, self).__init__(input_dim, output_dim, hidden_dim)
        self.problem = problem
        self.num_layers = num_layers
        self.propagate_best = propagate_best

        self.net = None
        self.best_net = None

        kwargs = rm_from_kwargs(kwargs, keys=["edge_feature_dim"])
        self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        """Create the specified model layers."""

        def NN():
            return nn.Sequential(
                nn.Linear(2 * self.input_dim, self.hidden_dim),
                nn.GELU(),
                *list(it.chain.from_iterable([(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU()
                ) for _ in range(max(self.num_layers-1, 0))])),
                nn.Linear(self.hidden_dim, self.output_dim)
            )

        self.net = NN()
        if self.propagate_best:
            self.best_net = NN()

    def _get_emb_seq(self, obs: Obs, emb: Emb, best: bool = False) -> Tensor:
        """Create an embedding of the sol sequences."""
        bs, n, d = emb.node_feature_emb.size()
        if self.problem.upper() == "TSP":
            # retrieve sol sequences from obs
            if best:
                node_in_sol_idx = obs.best_sol[:, 0]
            else:
                node_in_sol_idx = obs.current_sol[:, 0]
            # select corresponding node embeddings
            emb_seq = emb.node_feature_emb.gather(index=node_in_sol_idx[:, :, None].expand(bs, n, d), dim=1)
            emb_seq = torch.cat((
                torch.mean(emb_seq, dim=1),  # mean over seq
                torch.max(emb_seq, dim=1)[0]  # max over seq
            ), dim=-1)
        else:
            # retrieve sol sequences from obs
            if best:
                node_in_sol_idx = obs.best_sol_seqs
            else:
                node_in_sol_idx = obs.current_sol_seqs
            k = node_in_sol_idx.size(1)
            # select corresponding node embeddings
            emb_seq = emb.node_feature_emb[:, None, :, :].expand(bs, k, n, d).gather(
                index=node_in_sol_idx[:, :, :, None].expand(bs, k, -1, d), dim=2
            )
            emb_seq = torch.cat((
                torch.mean(emb_seq, dim=2),  # mean over seq
                torch.max(emb_seq, dim=2)[0]  # max over seq
            ), dim=-1)

        return emb_seq

    def forward(self, obs: Obs, emb: Emb = Emb(), **kwargs) -> Emb:
        emb_seq = self._get_emb_seq(obs, emb)
        sol_emb = self.net(emb_seq)
        if not self.propagate_best:
            return emb.update(current_sol_emb=sol_emb)
        # do same for best sol if specified
        emb_seq = self._get_emb_seq(obs, emb, best=True)
        best_sol_emb = self.best_net(emb_seq)
        return emb.update(current_sol_emb=sol_emb, best_sol_emb=best_sol_emb)


class RNNSolEncoder(BaseEncoder):
    """RNN encoder model for sol embeddings."""
    def __init__(self,
                 problem: str,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 propagate_reverse: bool = True,
                 propagate_best: bool = False,
                 rnn_type: str = "GRU",
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
            num_layers: number of RNN layers
            propagate_reverse: flag to also propagate in reverse sol direction
                                (-> bidirectional RNN)
            propagate_best: flag to also propagate over best sol graph
            rnn_type: type of RNN, one of ['LSTM', 'GRU']
            **kwargs:
        """
        super(RNNSolEncoder, self).__init__(input_dim, output_dim, hidden_dim)
        self.problem = problem
        self.num_layers = num_layers
        self.propagate_reverse = propagate_reverse
        self.propagate_best = propagate_best
        self.rnn_type = rnn_type.upper()

        self.rnn = None
        self.out_proj = None
        self.best_rnn = None
        self.best_out_proj = None

        kwargs = rm_from_kwargs(kwargs, keys=["edge_feature_dim"])
        self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        self.rnn = self._build_rnn(**kwargs)
        idim = self.hidden_dim * (1 + self.propagate_reverse) * self.num_layers
        self.out_proj = nn.Linear(idim, self.output_dim)
        if self.propagate_best:
            self.best_rnn = self._get_rnn(**kwargs)
            self.best_out_proj = nn.Linear(idim, self.output_dim)

    def _build_rnn(self, **kwargs):
        """Construct the requested RNN."""
        if self.rnn_type == "LSTM":
            return nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.propagate_reverse,
                **kwargs
            )
        elif self.rnn_type == "GRU":
            return nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.propagate_reverse,
                **kwargs
            )
        else:
            raise ValueError(f"unknown rnn_type: '{self.rnn_type}")

    def _rnn_forward(self, seq: Tensor, best: bool = False):
        """utility function wrapping different RNN forward passes."""
        self.best_rnn.flatten_parameters() if best else self.rnn.flatten_parameters()
        if self.rnn_type == "LSTM":
            _, (h, _) = self.best_rnn(seq) if best else self.rnn(seq)
        elif self.rnn_type == "GRU":
            _, h = self.best_rnn(seq) if best else self.rnn(seq)
        else:
            raise ValueError(f"unknown rnn_type: '{self.rnn_type}")
        return h.permute(1, 0, 2)   # -> (bs, num_directions, emb_dim)

    def _get_emb_seq(self, obs: Obs, emb: Emb, best: bool = False) -> Tuple[Tensor, List]:
        """For the RNN we create a sequence of node embeddings according to the sol sequences."""
        bs, n, d = emb.node_feature_emb.size()
        if self.problem.upper() == "TSP":
            # retrieve sol sequences from obs
            if best:
                assert obs.best_sol.size(0) == bs
                node_in_sol_idx = obs.best_sol[:, 0]
            else:
                assert obs.current_sol.size(0) == bs
                node_in_sol_idx = obs.current_sol[:, 0]
            # select corresponding node embeddings
            emb_seq = emb.node_feature_emb.gather(index=node_in_sol_idx[:, :, None].expand(bs, n, d), dim=1)
            shape = [bs, -1]
        else:
            # retrieve sol sequences from obs
            if best:
                node_in_sol_idx = obs.best_sol_seqs
            else:
                node_in_sol_idx = obs.current_sol_seqs
            k = node_in_sol_idx.size(1)
            # select corresponding node embeddings
            emb_seq = emb.node_feature_emb[:, None, :, :].expand(bs, k, n, d).gather(
                index=node_in_sol_idx[:, :, :, None].expand(bs, k, -1, d), dim=2
            )
            # pack padded sequences of different length for RNN
            if self.problem.upper() == "JSSP":
                assert node_in_sol_idx.min() > 0
                seq_lens = (node_in_sol_idx > 0).sum(-1)
            else:
                assert (node_in_sol_idx[:, :, -1] == 0).all(), \
                    f"max_seq_len reached for sol {(node_in_sol_idx[:, :, -1] != 0).nonzero()}"
                seq_lens = (node_in_sol_idx > 0).sum(-1) + 2   # +2 for start and end idx at depot
            emb_seq = torch.nn.utils.rnn.pack_padded_sequence(
                emb_seq.view(bs*k, -1, d),
                lengths=seq_lens.view(-1).cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            shape = [bs, k, -1]

        return emb_seq, shape

    def forward(self, obs: Obs, emb: Emb = Emb(), **kwargs) -> Emb:
        emb_seq, shape = self._get_emb_seq(obs, emb)
        sol_emb = self._rnn_forward(emb_seq)
        # project to output dimension
        sol_emb = self.out_proj(sol_emb.reshape(*shape))
        if not self.propagate_best:
            return emb.update(current_sol_emb=sol_emb)
        # do same for best sol if specified
        emb_seq, shape = self._get_emb_seq(obs, emb, best=True)
        best_sol_emb = self._rnn_forward(emb_seq, best=True)
        best_sol_emb = self.best_out_proj(best_sol_emb.reshape(*shape))
        return emb.update(current_sol_emb=sol_emb, best_sol_emb=best_sol_emb)


#
# ============= #
# ### TEST #### #
# ============= #
def _test_flat(
        bs: int = 5,
        n: int = 10,
        cuda=False,
        seed=1
):
    import sys
    import numpy as np
    import torch
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    num_layers = [1, 2, 3]
    best = [True, False]

    I = 4
    O = 16
    x = Obs(
        node_features=torch.randn(bs, n, I).to(device),
        current_sol=torch.from_numpy(np.stack([np.random.permutation(np.arange(n))
                                                for _ in range(bs*2)])).view(bs, 2, -1).to(device),
        best_sol=torch.from_numpy(np.stack([np.random.permutation(np.arange(n))
                                             for _ in range(bs*2)])).view(bs, 2, -1).to(device)
    )
    emb = Emb(node_feature_emb=torch.randn(bs, n, O).to(device))

    for l in num_layers:
        for be in best:
            try:
                e = FlatSolEncoder(O, O, num_layers=l, propagate_best=be).to(device)
                out = e(obs=x, emb=emb)
                assert out.current_sol_emb.size() == torch.empty((bs, O)).size()
                if out.best_sol_emb is not None:
                    assert out.best_sol_emb.size() == torch.empty((bs, O)).size()
            except Exception as e:
                raise type(e)(
                    str(e) + f" - (num_layers: {l}, best: {be})\n"
                ).with_traceback(sys.exc_info()[2])


def _test_rnn(
        bs: int = 5,
        n: int = 10,
        cuda=False,
        seed=1
):
    import sys
    import numpy as np
    import torch
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    num_layers = [1, 2, 3]
    reverse = [True, False]
    best = [True, False]

    I = 4
    O = 16
    x = Obs(
        node_features=torch.randn(bs, n, I).to(device),
        current_sol=torch.from_numpy(np.stack([np.random.permutation(np.arange(n))
                                                for _ in range(bs*2)])).view(bs, 2, -1).to(device),
        best_sol=torch.from_numpy(np.stack([np.random.permutation(np.arange(n))
                                             for _ in range(bs*2)])).view(bs, 2, -1).to(device)
    )
    emb = Emb(node_feature_emb=torch.randn(bs, n, O).to(device))

    for l in num_layers:
        for rev in reverse:
            for be in best:
                try:
                    e = RNNSolEncoder(O, O, num_layers=l, propagate_reverse=rev, propagate_best=be).to(device)
                    out = e(obs=x, emb=emb)
                    assert out.current_sol_emb.size() == torch.empty((bs, O)).size()
                    if out.best_sol_emb is not None:
                        assert out.best_sol_emb.size() == torch.empty((bs, O)).size()
                except Exception as e:
                    raise type(e)(
                        str(e) + f" - (num_layers: {l}, rev: {rev}, best: {be})\n"
                    ).with_traceback(sys.exc_info()[2])

