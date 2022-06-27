#
import os
from warnings import warn
from typing import Optional, List, Dict, Any, Union

import numpy as np
import torch
import gym
from tianshou.env import SubprocVectorEnv, ShmemVectorEnv, DummyVectorEnv

from lib.utils import rm_from_kwargs
from lib.env.utils import DistributedDataset, GraphNeighborhoodSampler
from lib.env.tsp_env import TSPEnv
from lib.env.cvrp_env import CVRPEnv
from lib.env.jssp_env import JSSPEnv


def create_envs(num_envs: int, problem: str, env_kwargs: Dict, **kwargs):
    """Return a list of env creation functions for SubprocVectorEnv."""
    if num_envs > os.cpu_count():
        warn(f"num_envs > num logical cores! This can lead to "
             f"decrease in performance if env is not IO bound.")
    if problem.upper() == "TSP":
        def make():
            return TSPEnv(**env_kwargs, **kwargs)
    elif problem.upper() == "CVRP":
        def make():
            return CVRPEnv(**env_kwargs, **kwargs)
    elif problem.upper() == "JSSP":
        def make():
            return JSSPEnv(**env_kwargs, **kwargs)
    else:
        raise ValueError(f"unknown problem: '{problem}'")
    return [make for _ in range(num_envs)]


class VecEnv:
    """Creates and wraps a vectorized env
    for combinatorial optimization problems.

    Args:
        num_envs: number of parallel environments
        problem: name of routing problem
        env_kwargs: additional arguments for routing envs
        share_memory: flag if buffer should use shared memory
        fixed_dataset: flag if data of env is provided as fixed dataset
        dataset_size: optional size limit of fixed dataset
        data_file_path: file path to stored fixed dataset
    """
    def __init__(self,
                 num_envs: int,
                 problem: str,
                 env_kwargs: Optional[Dict] = None,
                 share_memory: bool = False,
                 fixed_dataset: bool = False,
                 dataset_size: Optional[int] = None,
                 data_file_path: Optional[str] = None,
                 **kwargs):
        super(VecEnv, self).__init__()
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.share_memory = share_memory
        self.fixed_dataset = fixed_dataset
        self.dataset_size = dataset_size
        self.data_file_path = data_file_path

        stand_alone = (num_envs == 1)
        kwargs = rm_from_kwargs(kwargs, ["create_nbh_graph", "k_nbh_frac"])
        if stand_alone:
            env_cl = DummyVectorEnv
        else:
            env_cl = ShmemVectorEnv if share_memory else SubprocVectorEnv
        self.venv = env_cl(
            env_fns=create_envs(
                num_envs, problem, env_kwargs,
                fixed_dataset=fixed_dataset,
                stand_alone=stand_alone,
                data_file_path=data_file_path if stand_alone and fixed_dataset else None,
            ),
            **kwargs
        )

        # check observation space consistency
        obs_spaces = self.venv.observation_space.copy()
        obs_sp_ref = obs_spaces[0]
        for o in obs_spaces:
            assert (o == obs_sp_ref)
        self.observation_space = obs_sp_ref
        # check action space consistency
        act_spaces = self.venv.action_space.copy()
        act_sp_ref = act_spaces[0]
        for a in act_spaces:
            assert (a == act_sp_ref)
        self.action_space = act_sp_ref

        self._distributed_dataset = None
        # distribute a fixed dataset (e.g. val or test set) over the full vectorized env
        if not stand_alone and self.fixed_dataset and self.data_file_path is not None:
            self._distributed_dataset = DistributedDataset(problem=problem, venv=self.venv)
            self._distributed_dataset.load(fpath=self.data_file_path, limit=self.dataset_size)
            self._distributed_dataset.distribute()

        self._stand_alone = stand_alone

    @property
    def num_envs(self):
        return self.venv.env_num

    def __len__(self):
        return self.num_envs

    def __getattr__(self, key: str) -> List[Any]:
        # hacky way of getting venv attributes exposed. Might break deepcopy/pickle
        # without implementing additional class methods and cl.__new__()
        # see https://docs.python.org/3/library/pickle.html#pickling-class-instances
        return getattr(self.venv, key)

    def reset(self, id: Optional[np.ndarray] = None):
        return self.venv.reset(id=id)

    def step(self, action: np.ndarray, id: Optional[np.ndarray] = None, **kwargs):
        return self.venv.step(action, id=id)

    def seed(self, seed: Optional[int] = None) -> List:
        self.action_space.seed(seed)
        return self.venv.seed(seed)

    def render(self, **kwargs):
        return self.venv.render(**kwargs)

    def close(self):
        self.venv.close()

    def __repr__(self) -> str:
        obs_sp = [f"{k}: {v.shape};" for k, v in self.observation_space.spaces.items()]
        return '{}(\n size: {},\n share_mem: {},\n fixed_ds: {},\n act_sp: {},\n obs_sp: Dict({})\n)'.format(
            self.__class__.__name__,
            self.num_envs,
            self.share_memory,
            self.fixed_dataset,
            self.action_space,
            ' '.join(obs_sp),
        )


class RPGraphVecEnv(VecEnv):
    """
    Creates and wraps a vectorized env for routing problems.
    Takes care of additional graph attributes.

    Args:
        num_envs: number of parallel environments
        problem: name of routing problem
        env_kwargs: additional arguments for routing envs
        share_memory: flag if buffer should use shared memory
        fixed_dataset: flag if data of env is provided as fixed dataset
        dataset_size: optional size limit of fixed dataset
        data_file_path: file path to stored fixed dataset
        create_nbh_graph: flag if node neighborhood graph should be created
        k_nbh_frac: fraction of nodes to consider in KNN neighborhood


    Graph encoder expects:
        1) x (BS*N, F): node features
        2) edge_index (2, BS*M): edges with running index (for current and best tour)
        3) edge_weight (BS*M, 1): edge feature (for current and best tour)
            -> edge_weights which are the corresponding euclidean distance
            between the source and target node are calculated batched on GPU
        4) x_batch (BS*N, 1): batch idx of nodes
        5) nbh_edges and nbh_edge_weights of neighborhood graph (node neighborhoods)

    -> for TSP all instances have same number of nodes and edges
    -> for VRP obs are max-padded to same dimensionality given in observations-space

    """
    def __init__(self,
                 num_envs: int,
                 problem: str,
                 env_kwargs: Optional[Dict] = None,
                 share_memory: bool = False,
                 fixed_dataset: bool = False,
                 dataset_size: Optional[int] = None,
                 data_file_path: Optional[str] = None,
                 create_nbh_graph: bool = True,
                 k_nbh_frac: Union[int, float] = 16,
                 **kwargs
                 ):
        super(RPGraphVecEnv, self).__init__(
            num_envs=num_envs,
            problem=problem,
            env_kwargs=env_kwargs,
            share_memory=share_memory,
            fixed_dataset=fixed_dataset,
            dataset_size=dataset_size,
            data_file_path=data_file_path,
            **kwargs
        )
        # since obs are padded to same dimension, all entities can be inferred from observation space
        self.node_max_dim = list(self.observation_space.spaces['node_features'].shape)[0]
        self.batch_idx = np.arange(self.num_envs).repeat(self.node_max_dim, axis=-1).reshape(self.num_envs, -1)

        self.create_nbh_graph = create_nbh_graph
        self.nbh_edges = None
        self.nbh_weights = None
        self.nbh_sampler = GraphNeighborhoodSampler(self.node_max_dim, k_frac=k_nbh_frac) if create_nbh_graph else None
        self._update_observation_space()

    def step(self, action: np.ndarray, id: Optional[np.ndarray] = None, **kwargs):
        obs, reward, done, info = self.venv.step(action, id=id)
        return self.to_graph(obs), reward, done, info

    def reset(self, id: Optional[np.ndarray] = None) -> np.ndarray:
        obs = self.venv.reset(id=id)
        if self.create_nbh_graph:
            self.init_nbh_graph(obs, id)
        return self.to_graph(obs, id)

    def to_graph(self, obs: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Add additional entities constituting a graph representation."""
        idx_range = range(len(obs)) if ids is None else ids
        for i, idx in enumerate(idx_range):
            # add batch running idx
            obs[i]['batch_idx'] = self.batch_idx[idx]
            # add nbh graph
            if self.create_nbh_graph:
                obs[i]['nbh_edges'] = self.nbh_edges[idx]
                obs[i]['nbh_weights'] = self.nbh_weights[idx]

        return obs

    def init_nbh_graph(self, obs: np.ndarray, ids: Optional[np.ndarray] = None):
        """Initialize the fixed (static) neighborhood graph for current batch."""
        edge_list = [self.nbh_sampler(torch.from_numpy(e['coords'])) for e in obs]
        if ids is None:     # replace full buffer
            self.nbh_edges = np.stack([e[0] for e in edge_list])
            self.nbh_weights = np.stack([e[1] for e in edge_list])
        else:   # replace only at ids
            assert len(ids) == len(obs)
            for i, e in zip(ids, edge_list):
                self.nbh_edges[i] = e[0]
                self.nbh_weights[i] = e[1]

    def _update_observation_space(self):
        """
        Add additional graph attributes to observation space
        such that contains() check can be done correctly.
        """
        N = self.node_max_dim
        BSN = self.num_envs*N - 1
        dict_space = dict(self.observation_space.spaces)
        dict_space['batch_idx'] = gym.spaces.Box(low=0, high=self.num_envs, shape=(N, ), dtype=np.int32)
        if self.create_nbh_graph:
            k = self.nbh_sampler.k
            fpp = self.env_kwargs.get('float_prec', np.float32)
            dict_space['nbh_edges'] = gym.spaces.Box(low=-1, high=BSN, shape=(2, N*k), dtype=np.int32)
            dict_space['nbh_weights'] = gym.spaces.Box(low=0, high=1, shape=(N*k, ), dtype=fpp)

        self.observation_space = gym.spaces.Dict(dict_space)


# ============= #
# ### TEST #### #
# ============= #
def _test(
    problem: str = "TSP",
    bs: int = 4,
    size: int = 10,
    n: int = 50,
    k: int = 8,
    cap: int = 30,
    seed: int = 1,
    n_steps: int = 50,
    acceptance_mode: str = 'SELECT_EPSILON',  # 'ACCEPT',
    operator_mode: str = 'SET',
    position_mode: str = 'ALL',
    mode_args=None,
    **kwargs
):
    import sys
    from lib.routing import RPDataset
    from lib.scheduling import JSSPDataset
    from lib.env.utils import DistributedDataset

    if problem.upper() == "JSSP":
        n_j = 2*(n//8)
        n_m = 2*(n//12)
        mode_args = {
            'search_criterion': 'best',
            'selection_criterion': 'sampling',
            'epsilon': 0.1,
            'random_shuffle': True,
            'ls_ops': ['CET'],
            'num_nodes_per_iter': 2,
            'restart_at_step': 0,
            'restart_mode': 'initial'
        } if mode_args is None else mode_args
        env_kwargs = {
            "num_steps": n_steps,
            "construction_args": {'method': "FIFO"},
            "sampling_args": {'sample_size': size, 'num_jobs': n_j, 'num_machines': n_m},
            "acceptance_mode": acceptance_mode,
            "operator_mode": operator_mode,
            "position_mode": position_mode,
            "mode_args": mode_args,
            "debug": True,
        }
        check_types = ['vec_env', 'distributed_data_vec_env']
        data = JSSPDataset("JSSP", seed=seed, **kwargs).sample(sample_size=size * bs, num_jobs=n_j, num_machines=n_m)

    else:
        mode_args = {
            'accept_rule': 'LI_ACCEPT',
            'epsilon': 0.1,
            'random_shuffle': True,
            'ls_ops': ['TWO_OPT'],
            'num_nodes_per_iter': 1,
        } if mode_args is None else mode_args

        env_kwargs = {
            'num_steps': n_steps,
            'construction_args': {'method': "random" if problem == "TSP" else "sweep"},
            'sampling_args': {'sample_size': size, 'graph_size': n, 'k': k, 'cap': cap},
            'acceptance_mode': acceptance_mode,
            'operator_mode': operator_mode,
            'position_mode': position_mode,
            'mode_args': mode_args,
            'debug': True,
        }
        check_types = ['vec_env', 'graph_vec_env', 'distributed_data_vec_env']
        data = RPDataset(problem=problem, seed=seed, **kwargs
                         ).sample(sample_size=size * bs, graph_size=n, k=k, cap=cap, **kwargs)

    for typ in check_types:

        try:
            env = None
            if typ == 'vec_env':  # standard vec env
                env = VecEnv(num_envs=bs, problem=problem, env_kwargs=env_kwargs)
                env.seed(seed)
            elif typ == 'graph_vec_env':
                env = RPGraphVecEnv(num_envs=bs, problem=problem, env_kwargs=env_kwargs, create_nbh_graph=True)
                env.seed(seed)
            elif typ == 'distributed_data_vec_env':
                env = VecEnv(num_envs=bs, problem=problem, env_kwargs=env_kwargs, fixed_dataset=True)
                env.seed(seed)
                distr = DistributedDataset(problem=problem, venv=env.venv)
                distr.data = data
                distr.distribute()

            print(env)
            # reset and step
            obs_old = env.reset()
            #print(f"init obs: {obs_old}")
            i = 0
            while i < n_steps:
                a = np.array([env.action_space.sample() for _ in range(bs)])
                obs, rew, done, info = env.step(a)
                #print(f"obs: {obs}")
                #print(f"reward: {rew}")
                for old, new in zip(obs_old, obs):
                    for o1, o2 in zip(old.values(), new.values()):
                        if len(o1) > 0 and len(o2) > 0:
                            assert isinstance(o1, np.ndarray) and isinstance(o2, np.ndarray)
                            assert np.all(np.array(o1.shape) == np.array(o2.shape))
                obs_old = obs
                if np.any(done):
                    idx = np.nonzero(done)[0]
                    obs_old[idx] = env.reset(id=idx)

                i += 1

        except Exception as e:
            raise type(e)(
                str(e) + f"\n -> typ='{typ}'"
            ).with_traceback(sys.exc_info()[2])
