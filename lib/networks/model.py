#
from typing import Optional, Dict, Tuple, Any, Union
import logging
import numpy as np
import gym
import torch
from torch import nn
from torch import Tensor
from tianshou.data import Batch

import lib.networks.encoders as encoders
import lib.networks.decoders as decoders
from lib.networks.encoders import GRAPH_ENCODERS
from lib.networks.aggregator import Aggregator, DummyAggregator
from lib.networks.decoders.iqn import IQN
from lib.networks.formats import Obs, GraphObs, Emb
from lib.utils import count_parameters

logger = logging.getLogger(__name__)


class Model(nn.Module):
    """
    Model wrapper providing unified interface
    for different encoder/decoder architectures.

    Args:
        problem: current problem type
        observation_space: env observation space
        action_space: env action space
        policy_type: used policy type to determine model output format
                    (default 'DQN'; 'IQN' for implicit quantile network)
        decoder_type: type of decoder
        encoder_args: additional arguments for encoder creation
        decoder_args: additional arguments for decoder creation
        aggregator_args: additional arguments for aggregator creation
        embedding_dim: general embedding dimension of model
        device: device of model (CPU or GPU)
        env_modes: mode cfg of the env
    """
    def __init__(self,
                 problem: str,
                 observation_space: gym.spaces.Dict,
                 action_space: Union[gym.spaces.Discrete, gym.spaces.MultiDiscrete],
                 policy_type: str = "DQN",
                 decoder_type: Union[str, nn.Module] = "FFDecoder",
                 encoder_args: Optional[Dict] = None,
                 decoder_args: Optional[Dict] = None,
                 aggregator_args: Optional[Dict] = None,
                 iqn_args: Optional[Dict] = None,
                 embedding_dim: int = 128,
                 device: Union[str, int, torch.device] = "cpu",
                 env_modes: Dict = None,
                 **kwargs):
        super(Model, self).__init__()

        self.problem = problem
        self._observation_space = observation_space
        self._action_space = action_space
        self.decoder_type = decoder_type
        self.encoder_args = encoder_args if encoder_args is not None else {}
        self.decoder_args = decoder_args if decoder_args is not None else {}
        self.aggregator_args = aggregator_args if aggregator_args is not None else {}
        self.iqn_args = iqn_args if iqn_args is not None else {}
        self.embedding_dim = embedding_dim
        self._device = torch.device(device)
        self.policy_type = policy_type.upper()
        #
        self._idx_inc = None
        self._static_nbh_edge_buffer = None
        self._ref_bs = None

        # get dims from obs and act space
        _, f = self._observation_space.spaces['node_features'].shape
        self.node_feature_dim = f
        self.node_max_dim = None
        self.action_output_dim = self._action_space.n

        # initialize encoder model
        self.encoder = Encoder(
            problem=self.problem,
            embedding_dim=embedding_dim,
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=1,
            **self.encoder_args, **kwargs)

        # initialize aggregator
        if self.aggregator_args:
            self.aggregator = Aggregator(
                observation_space=self._observation_space,
                action_space=self._action_space,
                emb_dim=embedding_dim,
                env_modes=env_modes,
                **self.aggregator_args)
        else:
            self.aggregator = DummyAggregator()

        # initialize decoder network
        decoder_cl = getattr(decoders, decoder_type) if isinstance(decoder_type, str) else decoder_type
        # jit decoder model
        self.decoder = torch.jit.script(decoder_cl(
            input_dim=self.embedding_dim,
            output_dim=self.action_output_dim,
            **self.decoder_args, **kwargs
        ))

        self.iqn = IQN(
            input_dim=self.action_output_dim,
            output_dim=self.action_output_dim,
            **self.iqn_args
        ) if self.policy_type == "IQN" else None

        self.reset_parameters()
        self.to(device=self._device)

    def __repr__(self):
        super_repr = super().__repr__()     # get torch module str repr
        n_enc_p = count_parameters(self.encoder)
        n_agg_p = count_parameters(self.aggregator)
        n_dec_p = count_parameters(self.decoder)
        n_iqn_p = count_parameters(self.iqn) if self.iqn is not None else 0
        add_repr = f"\n-----------------------------------" \
                   f"\nNum Parameters: " \
                   f"\n  (encoder): {n_enc_p} " \
                   f"\n  (aggregator): {n_agg_p} " \
                   f"\n  (decoder): {n_dec_p + n_iqn_p} " \
                   f"\n  total: {n_enc_p+n_agg_p+n_dec_p+n_iqn_p}"
        return super_repr + add_repr

    def reset_parameters(self):
        """Reset model parameters."""
        self.encoder.reset_parameters()
        self.aggregator.reset_parameters()

    def forward(self,
                obs: Batch,
                state: Optional[np.ndarray] = None,
                info: Optional[Dict] = None,
                **kwargs) -> Tuple[Union[Tensor, Tuple], Any]:
        """
        Meta forward pass wrapper doing required data batch collation and conversion.

        Args:
            obs: ts.batch of observations
            state: optional RNN state
            info: additional env info dict

        Returns:
            logits: logits over action space dimension.
                    depending on policy can also be a tuple, e.g.
                    (mu: Tensor, sigma: Tensor) for Gaussian PPO policy or
                    (logits: Tensor, taus: Tensor) for IQN
            state: optional RNN state

        """
        # convert to torch tensors and push to device
        obs.to_torch(device=self._device)
        bs = obs.shape[0]
        recompute = True
        sample_size = kwargs.pop("sample_size", 0)
        self.node_max_dim = obs.node_features.size(1)

        # collate and format batch            
        if self.encoder.uses_graph:
            # convert edges into running idx format expected by torch_geometric
            nbh_e = obs.get('nbh_edges')
            nbh_w = obs.get('nbh_weights')

            # during inference we can keep static components (e.g. node neighborhood)
            # fixed and do not need to recompute them every step
            if (
                self.training   # during training
                or len(info.shape) == 0     # on init
                or bs != self._ref_bs       # bs change
                or (len(info.shape) > 0 and np.any(info['step'] == 0))  # start of new episode
            ):
                self._ref_bs = bs
                self._idx_inc = torch.arange(0, self.node_max_dim * bs, self.node_max_dim,
                                             device=self._device)[:, None, None]
                if nbh_e is not None:
                    self._static_nbh_edge_buffer = self._to_edge_batch(nbh_e, self._idx_inc, self.node_max_dim)
            else:
                recompute = False

            cur_sol_e = self._to_edge_batch(obs['current_sol'], self._idx_inc, self.node_max_dim)
            best_sol_e = self._to_edge_batch(obs['best_sol'], self._idx_inc, self.node_max_dim)

            if self.problem.upper() == "JSSP":
                prep_obs = GraphObs(
                    batch_size=bs,
                    batch_idx=None,     # type: ignore
                    node_features=obs['node_features'].view(-1, self.node_feature_dim),
                    current_sol=None,   # type: ignore
                    current_sol_edges=cur_sol_e,  # type: ignore
                    current_sol_weights=obs['current_sol_w'].view(-1),
                    best_sol=None,  # type: ignore
                    best_sol_edges=best_sol_e,  # type: ignore
                    best_sol_weights=obs['best_sol_w'].view(-1),
                    nbh_edges=self._static_nbh_edge_buffer,  # type: ignore
                    nbh_weights=nbh_w.view(-1) if nbh_w is not None else None,
                    current_sol_seqs=obs['current_sol_seq'],
                    best_sol_seqs=obs['best_sol_seq'],
                    meta_features=obs['meta_features'],
                )
            else:
                if not hasattr(obs, "batch_idx"):
                    raise RuntimeError(f"Specified encoder needs observations from RPGraphVecEnv.")
                prep_obs = GraphObs(
                    batch_size=bs,
                    batch_idx=obs['batch_idx'].view(-1).to(torch.long),
                    node_features=obs['node_features'].view(-1, self.node_feature_dim),
                    current_sol=obs['current_sol'],
                    current_sol_edges=cur_sol_e,  # type: ignore
                    current_sol_weights=self._compute_tour_edge_weights(obs['coords'], cur_sol_e),
                    best_sol=obs['best_sol'],
                    best_sol_edges=best_sol_e,  # type: ignore
                    best_sol_weights=self._compute_tour_edge_weights(obs['coords'], best_sol_e),
                    nbh_edges=self._static_nbh_edge_buffer,  # type: ignore
                    nbh_weights=nbh_w.view(-1) if nbh_w is not None else None,
                    current_sol_seqs=obs['current_sol_seq'],
                    best_sol_seqs=obs['best_sol_seq'],
                    meta_features=obs['meta_features'],
                )
        else:
            assert self.problem.upper() != "JSSP"
            prep_obs = Obs(
                node_features=obs['node_features'],
                current_sol=obs['current_sol'],
                best_sol=obs['best_sol'],
                current_sol_seqs=obs['current_sol_seq'],
                best_sol_seqs=obs['best_sol_seq'],
                meta_features=obs['meta_features'],
            )
        
        logits, state = self._dqn_forward(bs, prep_obs, state, recompute=recompute, **kwargs)
        if self.policy_type == "IQN":
            # obs batch object does not need mask since logits are already masked, i.e. set to -inf where infeasible
            logits = self.iqn(logits, sample_size, **kwargs)    # returns logits tuple!

        return logits, state

    def _dqn_forward(self,
                     bs: int,
                     prep_obs: Union[Obs, GraphObs],
                     state: Optional[np.ndarray] = None,
                     **kwargs) -> Tuple[Union[Tensor, Tuple], Any]:
        """Inner forward calling encoder, aggregator and decoder models."""
        # encode
        emb = self.encoder(prep_obs, **kwargs)
        # aggregate embeddings
        emb = self.aggregator(
            prep_obs, emb, dims=(bs, self.node_max_dim, self.node_feature_dim)
        )
        # decode
        return self.decoder(emb, state)

    @staticmethod
    @torch.jit.script
    def _to_edge_batch(x: Tensor, idx_inc: Tensor, node_max_dim: int) -> Tensor:
        """Convert a batch of single instance edges to a batch of edges with running idx."""
        bs, e, n = x.size()
        assert e == 2 and x.max() < node_max_dim
        return (x + idx_inc).permute(1, 0, 2).reshape(2, -1)

    @staticmethod
    @torch.jit.script
    def _compute_tour_edge_weights(coords: Tensor, tour_edges: Tensor):
        """Calculate edge weights as euclidean distance between coords."""
        # calc edge weights (Euclidean distance)
        idx_coords = coords.view(-1, 2)[tour_edges]
        return torch.norm(idx_coords[0] - idx_coords[1], p=2, dim=-1)

    def get_optimizer(self, lr: float = 1e-3, **kwargs):
        """Return torch optimizer instance for network."""
        return torch.optim.Adam(self.parameters(), lr=lr, **kwargs)


class Encoder(nn.Module):
    """
    Encoder model wrapper to enable different encoding levels
    between nodes, edges, tours and graph.
    """

    def __init__(self,
                 problem: str,
                 embedding_dim: int,
                 node_feature_dim: int,
                 edge_feature_dim: Optional[int] = None,
                 node_encoder_type: Optional[str] = None,
                 node_encoder_cfg: Optional[Dict] = None,
                 sol_encoder_type: Optional[str] = None,
                 sol_encoder_cfg: Optional[Dict] = None,
                 sub_graph_encoder_type: Optional[str] = None,
                 sub_graph_encoder_cfg: Optional[Dict] = None,
                 graph_encoder_type: Optional[str] = None,
                 graph_encoder_cfg: Optional[Dict] = None,
                 ):
        super(Encoder, self).__init__()

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.node_encoder_type = node_encoder_type
        self.node_encoder_cfg = node_encoder_cfg if node_encoder_cfg is not None else {}
        self.sol_encoder_type = sol_encoder_type
        self.sol_encoder_cfg = sol_encoder_cfg if sol_encoder_cfg is not None else {}
        self.sub_graph_encoder_type = sub_graph_encoder_type
        self.sub_graph_encoder_cfg = sub_graph_encoder_cfg if sub_graph_encoder_cfg is not None else {}
        self.graph_encoder_type = graph_encoder_type
        self.graph_encoder_cfg = graph_encoder_cfg if graph_encoder_cfg is not None else {}

        self.node_encoder = None
        self.sol_encoder = None
        self.sub_graph_encoder = None
        self.graph_encoder = None
        self.uses_graph = False

        if self.node_encoder_type is not None:
            self.node_encoder = self._init_encoder(
                encoder_type=self.node_encoder_type,
                input_dim=node_feature_dim,
                output_dim=embedding_dim,
                edge_feature_dim=edge_feature_dim,
                **self.node_encoder_cfg
            )

        # group pooling layer
        if self.sol_encoder_type is not None:
            self.sol_encoder = self._init_encoder(
                encoder_type=self.sol_encoder_type,
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                problem=self.problem,
                **self.sol_encoder_cfg
            )

        if self.sub_graph_encoder_type is not None:
            self.sub_graph_encoder = self._init_encoder(
                encoder_type=self.sub_graph_encoder_type,
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                **self.sub_graph_encoder_cfg
            )

        if self.graph_encoder_type is not None:
            self.graph_encoder = self._init_encoder(
                encoder_type=self.graph_encoder_type,
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                **self.graph_encoder_cfg
            )

    def _init_encoder(self,
                      encoder_type: str,
                      input_dim: int,
                      output_dim: int,
                      **kwargs) -> nn.Module:
        self.uses_graph = self.uses_graph or (encoder_type in GRAPH_ENCODERS)
        cl = getattr(encoders, encoder_type) if isinstance(encoder_type, str) else encoder_type
        return cl(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )

    def reset_parameters(self):
        if self.node_encoder is not None:
            self.node_encoder.reset_parameters()
        if self.sol_encoder is not None:
            self.sol_encoder.reset_parameters()
        if self.sub_graph_encoder is not None:
            self.sub_graph_encoder.reset_parameters()
        if self.graph_encoder is not None:
            self.graph_encoder.reset_parameters()

    def forward(self, obs: Union[Obs, GraphObs], **kwargs) -> Emb:
        """
        Args:
            obs: batched observation tuple

        Returns:
            emb: tuple of created embeddings
        """
        emb = Emb()
        if self.node_encoder is not None:
            emb = self.node_encoder(obs, emb, **kwargs)
        if self.sol_encoder is not None:
            emb = self.sol_encoder(obs, emb, **kwargs)
        if self.sub_graph_encoder is not None:
            emb = self.sub_graph_encoder(obs, emb, **kwargs)
        if self.graph_encoder is not None:
            emb = self.graph_encoder(obs, emb, **kwargs)
        return emb


# ============= #
# ### TEST #### #
# ============= #
def _test(
    problem: str = "TSP",
    bs: int = 3,
    size: int = 10,
    n: int = 20,
    seed: int = 1,
    n_steps: int = 20,
    mode='ACCEPT_LS',
    mode_args={'ls_op': 'TWO_OPT'},
    graph: bool = False,
    **kwargs
):
    import tianshou as ts
    from lib.env import VecEnv, RPGraphVecEnv

    env_kwargs = {
        'num_steps': n_steps,
        'sampling_args': {'sample_size': size, 'graph_size': n},
        'mode': mode,
        'mode_args': mode_args,
        'debug': False,
    }
    if graph:
        env_cl = RPGraphVecEnv
        train_envs = env_cl(num_envs=bs, problem=problem, env_kwargs=env_kwargs, create_nbh_graph=True)
        test_envs = env_cl(num_envs=bs, problem=problem, env_kwargs=env_kwargs, create_nbh_graph=True)
    else:
        env_cl = VecEnv
        train_envs = env_cl(num_envs=bs, problem=problem, env_kwargs=env_kwargs)
        test_envs = env_cl(num_envs=bs, problem=problem, env_kwargs=env_kwargs)

    train_envs.seed(seed)
    test_envs.seed(seed)

    net = Model(observation_space=train_envs.observation_space, action_space=train_envs.action_space)
    print(net)

    optim = net.get_optimizer()
    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.99, estimation_step=3, target_update_freq=256)

    train_collector = ts.data.Collector(
        policy, train_envs, ts.data.PrioritizedVectorReplayBuffer(20000, bs), exploration_noise=True
    )
    test_collector = ts.data.Collector(
        policy, test_envs, exploration_noise=False
    )

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=2,
        step_per_epoch=200,
        step_per_collect=100,
        update_per_step=0.1,
        episode_per_test=100,
        batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    )
    print(result)


