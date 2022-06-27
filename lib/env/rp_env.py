#
from abc import abstractmethod
from copy import deepcopy
from warnings import warn
from typing import Optional, Tuple, List, Dict, Union, Any

import math
import numpy as np
import gym

from lib.env.base_env import BaseEnv
from lib.env.modes import *
from lib.routing.local_search.rules import *
from lib.routing import RPGenerator, RPDataset, ConstructionHeuristic, RPInstance
from lib.env.utils import inverse_key_lookup, Viewer, parse_solutions

EPS = np.finfo(np.float32).eps


class VRPBaseEnv(BaseEnv):
    """
    Gym environment to solve Routing Problems based on Local Search (LS).
    The underlying local search is performed by the VRPH C++ library with
    a custom wrapper and further custom adaptions and extensions.

    Args:
        construction_args: arguments for construction heuristic producing initial solution
        generator_args: arguments for data generator (sampling distribution, ...)
        sampling_args: arguments for data sampling procedure (size of sample, graph size, ...)
        num_steps: number of environment steps
        acceptance_mode: mode for LS move acceptance
        operator_mode: mode for selection of LS operator(s)
        position_mode: mode for selection of position (area/set/neighborhood) to apply LS
        mode_args: additional kwargs for modes
        solver_args: additional kwargs for LS solver

    """
    vrph_action_map = {k: v for k, v in zip(ACTIONS_NAMES + PERTURBATIONS_NAMES, ACTIONS + PERTURBATIONS)}
    vrph_acceptance_map = {k: v for k, v in zip(ACCEPTANCE_CRITERION_NAMES, ACCEPTANCE_CRITERION)}
    RULE_USE_NBH = VRPH_USE_NEIGHBOR_LIST
    RULE_META = VRPH_SAVINGS_ONLY + VRPH_FREE  # meta cfg for VRPH rule
    RULE_RND = VRPH_RANDOMIZED

    def __init__(self,
                 construction_args: Optional[Dict] = None,
                 generator_args: Optional[Dict] = None,
                 sampling_args: Optional[Dict] = None,
                 num_steps: int = 100,
                 acceptance_mode: str = 'SELECT',
                 operator_mode: str = 'ALL',
                 position_mode: str = 'ALL',
                 mode_args: Optional[Dict] = None,
                 solver_args: Optional[Dict] = None,
                 **kwargs):
        super(VRPBaseEnv, self).__init__(
            construction_args=construction_args,
            generator_args=generator_args,
            sampling_args=sampling_args,
            num_steps=num_steps,
            **kwargs,
        )
        # initialize construction operator to create initial solutions
        self.construction_op = ConstructionHeuristic(problem=self.PROBLEM, **self.construction_args)

        if acceptance_mode.upper() not in ACCEPTANCE_MODES:
            raise ValueError(f"unknown acceptance mode: '{acceptance_mode}'")
        self.acceptance_mode = acceptance_mode.upper()
        if operator_mode.upper() not in OPERATOR_MODES:
            raise ValueError(f"unknown operator mode: '{operator_mode}'")
        self.operator_mode = operator_mode.upper()
        if position_mode.upper() not in POSITION_MODES:
            raise ValueError(f"unknown position mode: '{position_mode}'")
        self.position_mode = position_mode.upper()
        self._check_modes()

        self.mode_args = mode_args if mode_args is not None else {}
        self.solver_args = solver_args if solver_args is not None else {}
        self.restart_at_step = self.mode_args.get("restart_at_step", 0)
        self._restart_args = self.mode_args.get('restart_args', {})

        self.create_nbh = False
        self.run_until_query = False
        self.max_num_vec = 0

        self._node_knn_nbh = None
        self._node_idx_set = None

    def _check_modes(self):
        # 'ACCEPT' and 'ACCEPT_EPSILON' only make sense
        # with operator and/or position selection
        if 'ACCEPT' in self.acceptance_mode:
            if not ('SELECT' in self.operator_mode or 'SELECT' in self.position_mode):
                warn(f"acceptance_mode '{self.acceptance_mode}' should use "
                     f"selection in operator_mode or position_mode")

    def _get_observation_space(self, **kwargs) -> gym.spaces.Space:
        N = self.instance.graph_size
        # original feature dim + cumulative demands and distances in fw and bw direction
        F = self.instance.node_features.shape[-1] + 4
        K = self.max_num_vec
        dict_space = {
            'coords': gym.spaces.Box(low=0, high=1, shape=(N, 2), dtype=self.float_prec),
            'node_features': gym.spaces.Box(low=0, high=1, shape=(N, F), dtype=self.float_prec),
            'current_sol': gym.spaces.Box(low=-1, high=N, shape=(2, N+K), dtype=np.int32),
            'best_sol': gym.spaces.Box(low=-1, high=N, shape=(2, N+K), dtype=np.int32),
            'meta_features': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=[8], dtype=self.float_prec),
        }
        return gym.spaces.Dict(dict_space)

    def _get_observation(self) -> Dict:
        return {
            'coords': self.instance.coords,
            'node_features': self._get_node_features(),
            'current_sol': self.current_sol,
            'best_sol': self.best_sol,
            'current_sol_seq': self.current_sol_seq,
            'best_sol_seq': self.best_sol_seq,
            'meta_features': np.array([
                self.previous_accept,
                self.previous_reward,
                self.current_cost,
                self.best_cost,
                self._num_steps_no_imp,
                self._num_restarts,
                self.previous_op,
                self._current_step,
            ], dtype=self.float_prec),
        }

    def _get_node_features(self):
        return np.concatenate((
            self.instance.node_features,
            np.array(self.solver.get_dynamic_node_data(),
                     dtype=self.instance.node_features.dtype)
        ), axis=-1)

    def _get_action_space(self, **kwargs) -> Union[gym.spaces.Space, Any]:
        self.acceptance_rule = self.vrph_acceptance_map[self.mode_args['accept_rule'].upper()]
        self.rnd_shuffle = self.mode_args.get('random_shuffle', False)

        # ACCEPTANCE
        self.epsilon = 0
        num_acceptance_a = 1
        if 'EPSILON' in self.acceptance_mode:
            self.epsilon = self.mode_args['epsilon']
        if 'SELECT' in self.acceptance_mode:
            num_acceptance_a = 2

        # POSITION
        num_nodes = self.mode_args.get('num_nodes_per_iter', 1)
        num_position_a = 1
        if self.position_mode == 'ALL':
            num_nodes = -1
        elif self.position_mode == 'RANDOM':
            assert num_nodes >= 0, f"position_mode == 'RANDOM' needs 'num_nodes_per_iter' >= 1"
        elif self.position_mode == 'SELECT_NODE':
            num_nodes = self.instance.graph_size
            num_position_a = num_nodes
        elif self.position_mode == 'SELECT_NBH':
            raise NotImplementedError
        else:
            raise ValueError(f"unknown position_mode '{self.position_mode}'")

        # OPERATORS
        self.ls_set = None
        if self.operator_mode == 'SET':
            ls_ops = self.mode_args['ls_ops']  # local search operator(s) to use
            if isinstance(ls_ops, str):
                ls_ops = [ls_ops]
            else:
                ls_ops = list(ls_ops)
            assert isinstance(ls_ops, list) and isinstance(ls_ops[0], str)
            ls_ops = [op.upper() for op in ls_ops]
        elif self.operator_mode == 'SELECT_LS':
            ls_ops = deepcopy(ACTIONS_NAMES)
        elif self.operator_mode in ['ALL', 'SELECT_LS+']:
            ls_ops = list(self.vrph_action_map.keys())
        else:
            raise ValueError(f"unknown operator_mode '{self.operator_mode}'")

        if self.PROBLEM.upper() == 'TSP':
            ls_ops = [op for op in ls_ops if op != 'CROSS_EXCHANGE']    # does not work for TSP
            assert len(ls_ops) > 0, f"'CROSS_EXCHANGE' ls_ops does not work for TSP!"

        if np.any([op == 'PERTURB_TARGET' for op in ls_ops]):
            # the PERTURB_TARGET operator requires the method to either provide
            # a list of at least 20% of nodes (or min 4, which ever is larger)
            # or to specify a custom search neighborhood
            # otherwise we automatically remove it from the operator list
            self.perturb_min_size = max(math.ceil(self.instance.graph_size / 5), 4)
            nodes_per_iter = self.mode_args.get('num_nodes_per_iter', 1)
            if not (
                    (nodes_per_iter is not None and self.perturb_min_size <= nodes_per_iter)
                    or num_nodes == -1
                    or self.position_mode == 'SELECT_NBH'
            ):
                warn(f"PERTURB_TARGET requires 'num_nodes_per_iter' to be "
                     f"at least 20% of graph size (or min 4 which ever is larger), "
                     f"but got {nodes_per_iter} < {self.perturb_min_size}. "
                     f"PERTURB_TARGET Operator will be removed.")
                ls_ops = [op for op in ls_ops if op != 'PERTURB_TARGET']

            else:
                self.create_nbh = True  # create local nbh for PERTURB_TARGET

        self.ls_set = [self.vrph_action_map[op] for op in ls_ops]
        num_operator_a = 1 if self.operator_mode in ['SET', 'ALL'] else len(self.ls_set)

        # finish
        self.num_nodes = num_nodes
        num_a = num_acceptance_a * num_position_a * num_operator_a
        if num_a == 1:
            warn(f"Current setup requires no actions but will just execute standard local search. "
                 f"Any provided actions will be ignored.")
            return None
        else:
            if (
                self.acceptance_mode == 'SELECT_EPSILON' and
                'SELECT' not in self.operator_mode and
                'SELECT' not in self.position_mode
            ):
                # if using 'SELECT_EPSILON' without needing any model decisions regarding the
                # selection of operators or positions, we can just continue executing the LS until
                # an improvement <= epsilon occurs, in which case we query the model
                # for an acceptance decision
                self.run_until_query = True

            return gym.spaces.Discrete(num_a)

    def _get_action(self, action: Union[np.ndarray, List, int]) -> Tuple[bool, int, int, List]:

        if isinstance(action, list) or isinstance(action, np.ndarray):
            assert len(action) <= 1
            action = action[0]

        # combine rules for VRPH to create corresponding action
        rule = self.acceptance_rule
        if self.rnd_shuffle:    # random order of nodes in moves
            rule = rule + self.RULE_RND

        # ACCEPTANCE
        accept_move = True      # 'ACCEPT' always
        if self.acceptance_mode == 'ACCEPT_EPSILON':
            accept_move = self.previous_reward > self.epsilon
        elif self.acceptance_mode == 'SELECT':
            accept_move = (action % 2 == 0)  # accept [0, 2, 4, ...] / reject [1, 3, 5, ...]
            action = action // 2
        elif self.acceptance_mode == 'SELECT_EPSILON':
            # accept if improvement > epsilon
            # else use provided decision
            if self.previous_reward <= self.epsilon:
                accept_move = (action % 2 == 0)  # accept [0, 2, 4, ...] / reject [1, 3, 5, ...]
            action = action // 2

        # OPERATORS
        if self.operator_mode in ['SET', 'ALL']:
            ls_op = sum(self.ls_set)  # all specified local search operators
        elif self.operator_mode in ['SELECT_LS', 'SELECT_LS+']:
            num_ls_ops = len(self.ls_set)
            ls_idx = action % num_ls_ops
            ls_op = self.ls_set[int(ls_idx)]
        else:
            raise ValueError

        # POSITION
        if self.position_mode == 'ALL':
            nodes = self._rnd.permutation(self._node_idx_set)[:-1]
            # for some reason VRPH does not work with num_nodes=graph_size,
            # so we remove the last random node in each iter
        elif self.position_mode == 'RANDOM':
            nodes = self._rnd.choice(
                self._node_idx_set,
                size=self.num_nodes,
                replace=False,
                shuffle=True
            )
        else:
            raise ValueError

        return (
            accept_move,
            ls_op,
            rule + self.RULE_META,
            nodes
        )

    def _step(self, action: Union[np.ndarray, List, int], **kwargs) -> Tuple[float, bool, Dict]:

        # if specified and no improvement for the last 'restart_at_step' steps,
        # we restart the solution procedure from a different point
        if 0 < self.restart_at_step <= self._num_steps_no_imp:
            self._restart()

        self.previous_cost = self.current_cost
        if self.run_until_query:
            prev_cost = self.current_cost   # original cost at start of iteration
            accept_move, ls_ops = self._solver_step(action, **kwargs)
            while True:
                rew = self._compute_reward(self.current_cost, self.previous_cost)
                # if improvement is smaller than epsilon,
                # leave the loop to query the model for an acceptance decision
                # (stop in any case if full budget of steps was used up)
                if rew <= self.epsilon or self._current_step >= self.num_steps-1:
                    break
                else:
                    # otherwise continue with next iteration of local search
                    # skip rejection check, since we only queried for the first step
                    self._solver_step(action, check_reject=False, **kwargs)
                    self.previous_cost = self.current_cost
                    self._current_step += 1
            self.previous_cost = prev_cost      # set to original value for correct reward calculation
        else:
            accept_move, ls_ops = self._solver_step(action, **kwargs)

        # get tour from solver assignments
        self.current_sol = self._parse_assignment(self.solver.get_routes())
        self.previous_accept = accept_move
        self.previous_op = float(ls_ops)

        if (
            self.operator_mode == "ALL"
            or (self.operator_mode == "SET" and len(self.ls_set) > 1)
            or ls_ops == -1
        ):
            ls = "MULTIPLE"
        else:
            ls = inverse_key_lookup(ls_ops, self.vrph_action_map)
        if self.debug > 1:
            # debug printing
            print(
                f'{self._current_step : <5}'
                f'{ls : ^20}'
                f'{inverse_key_lookup(self.acceptance_rule, self.vrph_acceptance_map) : ^20}'
                f'{accept_move: ^6}'
                f'{self.current_cost : ^20}'
                f'{self.best_cost : ^20}'
            )

        # compute rewards
        reward = self._compute_reward(self.current_cost, self.best_cost, clamp=self.clamp)
        done = self._current_step >= self.num_steps-1
        if ls in self._ls_op_cnt.keys():
            self._ls_op_cnt[ls] += 1
        else:
            self._ls_op_cnt[ls] = 1
        info = {'step': self._current_step, 'ls_op': ls}

        return reward, done, info

    def _solver_step(self,
                     action: Union[np.ndarray, List, int],
                     check_reject: bool = True,
                     **kwargs) -> Tuple[bool, int]:
        """Execute one step of the VRPH solver routine."""
        # unpack and format actions for VRPH solver
        accept_move, ls_ops, rule, nodes = self._get_action(action)

        if check_reject and not accept_move and self._current_step > 0:
            # can only reject after fist move was executed
            assert self.solver.reject_move()  # returns True if successfully rejected
            # make sure search does not get stuck in case of rejecting 'best move' which does not change
            if (
                self.operator_mode in ["ALL", "SET"]    # no operator selection
                and self.position_mode == "ALL"     # no random subset of positions
                and self.mode_args['accept_rule'].upper() in ["BEST_ACCEPT", "LI_ACCEPT"]
            ):
                # replace selection of best with first feasible
                rule -= self.acceptance_rule
                rule += VRPH_FIRST_ACCEPT

        self.current_cost = self._solve(
            ls_ops, 
            rule, 
            nodes, 
            iters=self.solver_args.get('num_iters', 1), 
            **kwargs
        )
        return accept_move, ls_ops

    def _solve(self,
               local_operators: int,
               rule: int,
               nodes: List[int],
               iters: int,
               err_max: float = 1e-5):
        """Execute specified operations in VRPH."""
        try:
            return self.solver.detailed_solve(
                local_operators,  # local_operators
                rule,  # rule
                nodes,  # node
                0,  # steps=0 to use all provided nodes
                iters,  # iters
                err_max,  # err_max
                False  # converge
            )
        except RuntimeError as e:
            msg = f"\nVRPH(C++) failed with: '{e}'\n"
            warn(msg)
            raise e

    def _restart(self):
        """Restart the problem solver...

            - from  new initial solution,
            - from  best solution found so far or
            - after major perturbation of current solution
        """
        mode = self.mode_args['restart_mode'].lower()
        if mode == "initial":
            # create new initial solution
            self.construction_op.construct(self.instance, vrph_model=self.solver)
            # self.current_sol_seq is set in call to self._parse_assignment()!
            self.current_sol = self._parse_assignment(self.solver.get_routes())
            self.current_cost = self.solver.get_total_route_length()
        elif mode == "best":
            # set current solution to best solution found so far
            self.current_cost = self.best_cost
            self.current_sol = self.best_sol.copy()
            self.current_sol_seq = self.best_sol_seq.copy()
            sol = parse_solutions(self.current_sol_seq)
            self.solver.use_initial_solution(sol)
            assert abs(self.current_cost - self.solver.get_total_route_length()) < EPS
        elif mode == "perturb":
            # apply major perturbations to current solution
            ops = sum([PERTURB_LI, PERTURB_OSMAN])
            rule = VRPH_LI_ACCEPT + self.RULE_RND
            nodes = self._rnd.permutation(self._node_idx_set)[:-1]
            new_cost = self._solve(
                local_operators=ops,
                rule=rule,
                nodes=nodes,
                iters=self._restart_args.get("iters", 3)
            )
            self.current_cost = new_cost
            self.current_sol = self._parse_assignment(self.solver.get_routes())
        elif mode == "perturb_kopt":
            # apply k-opt perturbations to current solution
            ops = sum([TWO_OPT, THREE_OPT])
            rule = VRPH_FIRST_ACCEPT + self.RULE_RND + self.RULE_META
            nodes = self._rnd.permutation(self._node_idx_set)[:-1]
            new_cost = self._solve(
                local_operators=ops,
                rule=rule,
                nodes=nodes,
                iters=self._restart_args.get("iters", 2)
            )
            self.current_cost = new_cost
            self.current_sol = self._parse_assignment(self.solver.get_routes())
        else:
            raise ValueError(f"unknown restart mode: '{mode}'")

        self._num_steps_no_imp = 0
        self._num_restarts += 1

    def _get_edges_to_render(self) -> Union[List, np.ndarray]:
        return self.current_sol

    def _init_instance(self, instance: RPInstance, **kwargs):
        """Unpack all relevant data, create initial solution and init VRPH model."""
        self.instance = instance
        N = instance.graph_size
        self._node_idx_set = np.arange(N)

    def _init_generator(self, problem: str, **kwargs):
        problem = problem.lower()
        assert problem in RPGenerator.RPS
        self.data_generator = RPDataset(
            problem=None if self.fixed_dataset and self.data_file_path is not None else problem,
            fname=self.data_file_path,
            seed=self._seed,
            float_prec=self.float_prec,
            verbose=self.debug > 1,
            **self.generator_args
        )

    @abstractmethod
    def _parse_assignment(self, vrph_route: Union[np.ndarray, List, Any]) -> np.ndarray:
        """Parse tour assignment of VRPH solver."""
        raise NotImplementedError

    def render(self, mode='human', as_gif: bool = True, **kwargs):
        assert self.enable_render, f"Need to specify <enable_render=True> on init."
        if as_gif and not self._render_step <= 150:
            warn(f"Can only render max ~150 steps as GIF but got render_step={self._render_step}. "
                 f"Following steps will not be rendered anymore!")
            return

        if self.viewer is None:
            # create new viewer object
            self.viewer = Viewer(
                locs=self.instance.coords.copy(),
                save_dir=self.plot_save_dir,
                gif_naming=f"render_ep{self._render_cnt}",
                as_gif=as_gif,
                **kwargs
            )
            # render iter=0 after reset
            self.viewer.update(
                buffer=self.render_buffer,
                cost=self.current_cost,
                n_iters=self._current_step,
                **kwargs
            )
            self._render_step += 1

        # update buffer and render new tour
        self.render_buffer['edges'] = self._get_edges_to_render()
        self.viewer.update(
            buffer=self.render_buffer,
            cost=self.current_cost,
            n_iters=self._current_step,
            **kwargs
        )
        self._render_step += 1
        return self.viewer.render_rgb()
