#
from typing import Optional, List, Dict, Union, Any
import numpy as np
import torch

from lib.env.rp_env import VRPBaseEnv
from lib.routing import RPInstance, knn_nbh
from lib.routing.local_search.rules import *
from lib.routing.local_search import VRP


class TSPEnv(VRPBaseEnv):
    """Local search env for TSP."""
    PROBLEM = "tsp"

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
        super(TSPEnv, self).__init__(
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
        # setup
        self._init_spaces()

    def _init_instance(self, instance: RPInstance, **kwargs):
        """Unpack all relevant data, create initial solution and init VRPH model."""
        self.instance = instance
        N = instance.graph_size
        self._node_idx_set = np.arange(N)

        self.solver = VRP(N)
        # load data into model (set -1 for any missing value)
        VRP.load_problem(
            self.solver,
            0,                          # 0 for TSP, 1 for CVRP
            instance.coords.tolist(),   # coordinates
            [float(-1)],                # demands - none for TSP
            # [best_known_dist, capacity, max_route_length, normalize, neighborhood_size]
            [float(-1), float(-1), float(-1), float(-1), float(-1)],
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
        self.current_sol_seq = []
        self.best_sol_seq = []

        if self.debug > 1:
            # debug printing
            repl = '-'
            print(f"\n{'Step' : <5}{'Action' : ^20}{'Rule' : ^20}{'accept' : ^6}{'curCost' : >10}{'bestCost' : ^20}")
            print(f'{0 : <5}{repl : ^20}{repl : ^20}{repl : ^6}{self.current_cost : ^20}{self.best_cost : ^20}')

        return initial_tour

    def _parse_assignment(self, vrph_route: Union[np.ndarray, List, Any]):
        """Parse tour assignment of VRPH solver."""
        assert len(vrph_route) > 0
        vrph_route = np.array(vrph_route) if not isinstance(vrph_route, np.ndarray) else vrph_route
        if len(vrph_route.shape) == 2:
            assert vrph_route.shape[0] == 1
            vrph_route = vrph_route[0]
        # convert to source-target format
        return np.concatenate((
            vrph_route[None, :],
            np.append(vrph_route[1:], vrph_route[0])[None, :]
        ), axis=0)


#
# ============= #
# ### TEST #### #
# ============= #
def _test_single(
    size: int = 1,
    n: int = 20,
    seed: int = 1234,
    n_steps: int = 32,
    acceptance_mode: str = 'SELECT',
    operator_mode: str = 'SET',
    position_mode: str = 'RANDOM',
    mode_args: dict = None,
    ls_op: str = 'TWO_OPT',
    acceptance: str = 'BEST_ACCEPT',
    random: bool = True,
    num_nodes_per_iter: int = 1,
    coords_sampling_dist: str = 'uniform',
    init_method: str = 'random',
    render: bool = True,
    verbose: int = 1,
    all_true: bool = False,
    pause: float = 0.0,
    **kwargs
):
    import time
    np.random.seed(seed)
    torch.manual_seed(seed)

    sample_args = {'sample_size': size, 'graph_size': n}
    generator_args = {'coords_sampling_dist': coords_sampling_dist,
                      'n_components': 3,
                      }
    mode_args = {
        'accept_rule': acceptance,
        'epsilon': 0.1,
        'random_shuffle': random,
        'ls_ops': ls_op if isinstance(ls_op, list) else [ls_op],
        'num_nodes_per_iter': num_nodes_per_iter,
    } if mode_args is None else mode_args
    env = TSPEnv(
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
        for o1, o2 in zip(obs_old.values(), obs.values()):
            assert isinstance(o1, np.ndarray) and isinstance(o2, np.ndarray)
            assert np.all(np.array(o1.shape) == np.array(o2.shape))
        obs_old = obs
        i += 1

    if verbose > 0:
        print(f"final reward: {rewards}")


def _test(
    size: int = 1,
    n: int = 20,
    seed: int = 1234,
    n_steps: int = 100,
    verbose: bool = 0,
):
    """Test close to all possible TSP env configurations which differ significantly."""
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
    init_methods = ['random', 'nn']
    MODE_COMBS = itertools.product(ACCEPTANCE_MODES, OPERATOR_MODES, POSITION_MODES)

    seeds = [(seed * (i*7)) for i in range(5)]
    for s in seeds:
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
                                # print(cfg_str)
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
                                        for a in ACTIONS_NAMES[:-1]:    # without cross exchange
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
