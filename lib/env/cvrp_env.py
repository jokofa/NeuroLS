#
import warnings
from typing import Optional, List, Dict, Union, Any
import itertools as it
import math
import numpy as np
import torch

from lib.env.rp_env import VRPBaseEnv
from lib.routing import RPInstance, knn_nbh
from lib.routing.local_search.rules import *
from lib.routing.local_search import VRP


class CVRPEnv(VRPBaseEnv):
    """Local search env for TSP."""
    PROBLEM = "cvrp"

    def __init__(self,
                 construction_args: Optional[Dict] = None,
                 generator_args: Optional[Dict] = None,
                 sampling_args: Optional[Dict] = None,
                 num_steps: int = 100,
                 acceptance_mode: str = 'SELECT',
                 operator_mode: str = 'SET',
                 position_mode: str = 'RANDOM',
                 mode_args: Optional[Dict] = None,
                 solver_args: Optional[Dict] = None,
                 **kwargs):
        super(CVRPEnv, self).__init__(
            construction_args=construction_args,
            generator_args=generator_args,
            sampling_args=sampling_args,
            num_steps=num_steps,
            acceptance_mode=acceptance_mode,
            operator_mode=operator_mode,
            position_mode=position_mode,
            mode_args=mode_args,
            solver_args=solver_args,
            **kwargs,
        )
        self.max_seq_len = None
        # setup
        self._init_spaces()

    def _init_instance(self, instance: RPInstance, **kwargs):
        """Unpack all relevant data, create initial solution and init VRPH model."""
        self.instance = instance
        N = instance.graph_size
        self._node_idx_set = np.arange(N)
        # rules of thumb!
        K = instance.max_num_vehicles
        self.max_num_vec = math.ceil(K*1.5) if K is not None else math.ceil(N/3)
        self.max_seq_len = int(min(64, np.ceil(N*0.75)))

        self.solver = VRP(N)
        # load data into model (set -1 for any missing value)
        VRP.load_problem(
            self.solver,
            1,                          # 1 for CVRP, 0 for TSP
            instance.coords.tolist(),   # coordinates
            instance.node_features[:, instance.constraint_idx[0]].tolist(),     # demands
            # [best_known_dist, capacity, max_route_len, normalize_flag, neighborhood_size]
            [float(-1), float(instance.vehicle_capacity), float(-1), float(1), float(0)],
            [[float(-1)]],              # no TW
            VRPH_EXACT_2D,              # edge type
            VRPH_FUNCTION,              # edge format
        )
        self.solver.set_random_seed(int(self._seed) if self._seed is not None else 1)

        if self.create_nbh:
            # create local knn neighborhoods
            self._node_knn_nbh = knn_nbh(instance, k=self.perturb_min_size).tolist()

        # create initial solution
        self.construction_op.construct(instance, vrph_model=self.solver)
        # get initial tour
        initial_tour = self._parse_assignment(self.solver.get_routes())

        self.current_sol = initial_tour.copy()
        self.best_sol = initial_tour.copy()
        self.current_cost = self.solver.get_total_route_length()
        self.best_cost = self.solver.get_best_total_route_length()
        self.best_sol_seq = self.current_sol_seq.copy()

        if self.debug > 1:
            # debug printing
            repl = '-'
            print(f"\n{'Step' : <5}{'Action' : ^20}{'Rule' : ^20}{'accept' : ^6}{'curCost' : >10}{'bestCost' : ^20}")
            print(f'{0 : <5}{repl : ^20}{repl : ^20}{repl : ^6}{self.current_cost : ^20}{self.best_cost : ^20}')

        return initial_tour

    def _parse_assignment(self, vrph_route: Union[np.ndarray, List, Any]):
        """Parse tour assignment of VRPH solver."""
        assert len(vrph_route) > 0
        # DEBUG
        if self.debug > 1:
            all_nodes = list(it.chain.from_iterable(vrph_route))
            all_unq = np.unique(np.array(all_nodes))
            assert len(all_unq) == self.instance.graph_size, f"Not all nodes were routed!"

        # buffer array of padded tour sequences
        current_sol_seq = [
                self.instance.depot_idx + r + [0]*(self.max_seq_len-len(r)-1) for r in vrph_route
            ]
        # add dummy seqs up to max_num_vec
        current_sol_seq += [[0]*self.max_seq_len]*(self.max_num_vec-len(current_sol_seq))

        # DEBUG
        if self.debug > 1:
            if not np.all(np.array([len(s) for s in current_sol_seq]) == len(current_sol_seq[0])):
                raise RuntimeError(f"Encountered feasible tour > max_seq_len. Try increasing max_seq_len.")

        self.current_sol_seq = np.array(current_sol_seq)
        if self.current_sol_seq.shape[0] > self.max_num_vec:
            raise RuntimeError(f"solution uses more tours than max_num_vec grace interval (1.5*max)")

        # create tours in source-target format (e.g. for GNN)
        # add missing depot idx in first position and remove depot idx from last position
        assert len(self.instance.depot_idx) == 1 and self.instance.depot_idx[0] == 0, \
            f"depot idx must be '0', currently does not support multi depot!"
        vrph_route = np.array(self.instance.depot_idx + list(it.chain.from_iterable(vrph_route))[:-1])

        # pad with depot self connections up to max num edges
        n_missing = (self.instance.graph_size + self.max_num_vec) - len(vrph_route)
        if n_missing < 0:
            warnings.warn(f"solution uses num_vec > max_num_vec.")
            if self.debug > 1:
                raise RuntimeError(f"solution uses num_vec > max_num_vec.")
        if n_missing > 0:
            assert vrph_route[0] == 0, "tour plan must start at depot!"
            vrph_route = np.append(vrph_route, np.zeros(n_missing, dtype=vrph_route.dtype))
        # convert to source-target format
        return np.concatenate((
            vrph_route[None, :],
            np.append(vrph_route[1:], vrph_route[0])[None, :]
        ), axis=0)

    def _get_edges_to_render(self):
        s2t_edges = self.current_sol
        assert len(s2t_edges.shape) == 2 and s2t_edges[0][0] == self.instance.depot_idx
        # get the idx positions of the depot idx
        z_idx = (s2t_edges[1] == self.instance.depot_idx).nonzero()[0] + 1
        starts = np.append(np.zeros(1, dtype=np.int), z_idx[:-1])
        ends = z_idx
        # slice array into list of route arrays
        return [s2t_edges[:, s:e] for s, e in zip(starts, ends)]


#
# ============= #
# ### TEST #### #
# ============= #
def _test_single(
    size: int = 1,
    n: int = 20,
    seed: int = 0, #1234,
    n_steps: int = 100,
    acceptance_mode: str = 'SELECT_EPSILON',
    operator_mode: str = 'SET',
    position_mode: str = 'RANDOM',
    mode_args: dict = None,
    ls_op: List[str] = 'TWO_OPT',
    acceptance: str = 'BEST_ACCEPT',
    random: bool = True,
    num_nodes_per_iter: int = 1,
    coords_sampling_dist: str = 'uniform',
    init_method: str = 'sweep',
    render: bool = True,
    verbose: int = 1,
    all_true: bool = False,
    pause: float = 0.0,
    **kwargs
):
    import time
    np.random.seed(seed)
    torch.manual_seed(seed)

    sample_args = {'sample_size': size, 'graph_size': n, 'k': 4, 'cap': 30}
    generator_args = {
        'coords_sampling_dist': coords_sampling_dist,
        'n_components': 3,
    }
    mode_args = {
        'accept_rule': acceptance,
        'epsilon': 0.1,
        'random_shuffle': random,
        'ls_ops': ls_op if isinstance(ls_op, list) else [ls_op],
        'num_nodes_per_iter': num_nodes_per_iter,
    } if mode_args is None else mode_args
    env = CVRPEnv(
        num_steps=n_steps,
        construction_args={'method': init_method},
        sampling_args=sample_args,
        generator_args=generator_args,
        acceptance_mode=acceptance_mode,
        operator_mode=operator_mode,
        position_mode=position_mode,
        mode_args=mode_args,
        debug=verbose > 0,
        enable_render=render,
        **kwargs
    )
    env.seed(seed)
    # reset and step
    obs_old = env.reset()
    if verbose > 1:
        print(f"init obs: {obs_old}")
    done = False
    rewards = 0
    i = 0
    while i < n_steps and not done:
        if all_true or env.action_space is None:
            a = 1   # always accept
        else:
            a = env.action_space.sample()
            #print(f"ACTION: {a}")
        obs, rew, done, info = env.step(a)
        if render:
            env.render(as_gif=False)
            if pause > 0:
                time.sleep(pause)
        if verbose > 1:
            print(f"obs: {obs}")
        if verbose > 0:
            print(f"reward: {rew}")
        rewards += rew
        # shapes in CVRP can vary since the number of routes and
        # accordingly the number of depot visits can change
        # so cannot easily be checked
        i += 1

    if verbose > 0:
        print(f"final reward: {rewards}")


def _test(
    size: int = 1,
    n: int = 20,
    seed: int = 1234,
    n_steps: int = 32,
    verbose: bool = 0,
):
    """Test close to all possible CVRP env configurations which differ significantly."""
    import sys
    import itertools
    from warnings import warn
    from lib.env.modes import (
        ACCEPTANCE_MODES,
        OPERATOR_MODES,
        POSITION_MODES
    )
    random = [True, False]
    num_nodes = [1, 2, 3, -1]
    coord_dists = ['uniform', 'gaussian_mixture']
    init_methods = ['sweep', 'cw']
    MODE_COMBS = itertools.product(ACCEPTANCE_MODES, OPERATOR_MODES, POSITION_MODES)

    seeds = [(seed * (i*7)) for i in range(5)]
    for i, s in enumerate(seeds):
        for cmb in MODE_COMBS:
            for acpt in ACCEPTANCE_CRITERION_NAMES:
                for n_node in num_nodes:
                    for rnd in random:
                        for dist in coord_dists:
                            for init_m in init_methods:

                                cfg_str = f"mode: {cmb}, " \
                                          f"accept: {acpt}, " \
                                          f"num_nodes: {n_node}, " \
                                          f"rnd: {rnd}, " \
                                          f"coords: {dist}, " \
                                          f"init_method: {init_m}, "
                                #print(cfg_str)
                                acc_mode, ops_mode, pos_mode = cmb
                                try:
                                    # filter some obvious combinations
                                    if pos_mode == 'RANDOM' and n_node < 1:
                                        continue
                                    if pos_mode == 'SELECT_NODE' and n_node != 1:
                                        continue
                                    if pos_mode == 'SELECT_NBH':
                                        warn(f"pos_mode '{pos_mode}' not implemented.")
                                        continue
                                    if ops_mode == "SET":
                                        for a in ACTIONS_NAMES:
                                            _test_single(size, n, seed=s,
                                                         n_steps=n_steps,
                                                         acceptance_mode=acc_mode,
                                                         operator_mode=ops_mode,
                                                         position_mode=pos_mode,
                                                         ls_op=a,
                                                         acceptance=acpt,
                                                         random=rnd,
                                                         num_nodes_per_iter=n_node,
                                                         coords_sampling_dist=dist,
                                                         init_method=init_m,
                                                         render=False,
                                                         verbose=verbose)

                                    else:
                                        _test_single(size, n, seed=s,
                                                     n_steps=n_steps,
                                                     acceptance_mode=acc_mode,
                                                     operator_mode=ops_mode,
                                                     position_mode=pos_mode,
                                                     acceptance=acpt,
                                                     random=rnd,
                                                     num_nodes_per_iter=n_node,
                                                     coords_sampling_dist=dist,
                                                     render=False,
                                                     verbose=verbose)

                                except Exception as e:
                                    raise type(e)(
                                        str(e) + f"\n -> ({cfg_str}"
                                    ).with_traceback(sys.exc_info()[2])
