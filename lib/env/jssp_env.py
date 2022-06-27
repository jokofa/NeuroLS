#
from copy import deepcopy
from warnings import warn
from typing import Optional, Tuple, List, Dict, Union, Any

import numpy as np
import gym

from lib.env.base_env import BaseEnv
from lib.env.modes import *
from lib.scheduling import JSSPDataset, JSSPInstance
from lib.scheduling.jssp_ls import JSSPSolver, LS_MOVES
from lib.env.utils import Viewer2

EPS = np.finfo(np.float32).eps


class JSSPEnv(BaseEnv):
    """
    Gym environment to solve Job Shop Scheduling Problems based on Local Search (LS).

    Args:
        construction_args: arguments for PDR producing initial solution
        generator_args: arguments for data generator (sampling distribution, ...)
        sampling_args: arguments for data sampling procedure (size of sample, num_jobs, num_machines, ...)
        num_steps: number of environment steps
        acceptance_mode: mode for LS move acceptance
        operator_mode: mode for selection of LS operator(s)
        position_mode: mode for selection of position (area/set/neighborhood) to apply LS
        mode_args: additional kwargs for modes
        solver_args: additional kwargs for LS solver

    """
    PROBLEM = "jssp"

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
        super(JSSPEnv, self).__init__(
            construction_args=construction_args,
            generator_args=generator_args,
            sampling_args=sampling_args,
            num_steps=num_steps,
            **kwargs,
        )

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

        self.graph = None
        self.current_sol_w = None
        self.best_sol_w = None
        self.nbh_edges = None
        self.nbh_weights = None

        self.run_until_query = False
        self.max_num_vec = 0

        self._node_idx_set = None
        self._restart_mode = self.mode_args['restart_mode'].lower()
        self._restart_args = self.mode_args.get('restart_args', {})

        self.solver = JSSPSolver(
            pdr_method=self.construction_args['method'],
            search_criterion=self.mode_args['search_criterion'],
            selection_criterion=self.mode_args['selection_criterion'],
            num_rnd=self.mode_args.get('num_nodes_per_iter', 1),
            shuffle=self.mode_args.get('random_shuffle', False),
            verbose=self.debug > 1,
            **self.solver_args
        )

        # setup
        self._init_spaces()

    def _check_modes(self):
        # 'ACCEPT' and 'ACCEPT_EPSILON' only make sense
        # with operator and/or position selection
        if 'ACCEPT' in self.acceptance_mode:
            if not ('SELECT' in self.operator_mode or 'SELECT' in self.position_mode):
                warn(f"acceptance_mode '{self.acceptance_mode}' should use "
                     f"selection in operator_mode or position_mode")

    def _get_observation_space(self, **kwargs) -> gym.spaces.Space:
        N = self.instance.graph_size
        F = 6   # node features: (ops_idx, job_idx, mch_idx, duration, dist_from_src, dist_to_snk) 
        K = self.instance.num_machines
        J = self.instance.num_jobs
        e_max = N + 2*J
        dict_space = {
            'node_features': gym.spaces.Box(low=float('inf'), high=float('inf'), shape=(N, F), dtype=self.float_prec),
            'current_sol': gym.spaces.Box(low=-1, high=N, shape=(2, N+K), dtype=np.int32),
            'best_sol': gym.spaces.Box(low=-1, high=N, shape=(2, N+K), dtype=np.int32),
            'current_sol_w': gym.spaces.Box(low=float('inf'), high=float('inf'), shape=(N+K,), dtype=self.float_prec),
            'best_sol_w': gym.spaces.Box(low=float('inf'), high=float('inf'), shape=(N+K,), dtype=self.float_prec),
            'current_sol_seq': gym.spaces.Box(low=-1, high=N, shape=(K, J), dtype=np.int32),
            'best_sol_seq': gym.spaces.Box(low=-1, high=N, shape=(K, J), dtype=np.int32),
            'nbh_edges': gym.spaces.Box(low=-1, high=N, shape=(2, e_max), dtype=np.int32),
            'nbh_weights': gym.spaces.Box(low=0, high=float('inf'), shape=(e_max,), dtype=self.float_prec),
            'meta_features': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=[8], dtype=self.float_prec),
        }
        return gym.spaces.Dict(dict_space)

    def _get_observation(self) -> Dict:
        return {
            'node_features': self.graph.get_node_features(),
            'current_sol': self.current_sol,  # current machine graph edges
            'best_sol': self.best_sol,  # best machine graph edges
            'current_sol_w': self.current_sol_w,    # current machine graph weights
            'best_sol_w': self.best_sol_w,  # best machine graph weights
            'current_sol_seq': self.current_sol_seq,    # current machine sequences
            'best_sol_seq': self.best_sol_seq,  # best machine sequences
            'nbh_edges': self.nbh_edges,    # job graph
            'nbh_weights': self.nbh_weights,    # job graph weights
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

    def _get_action_space(self, **kwargs) -> Union[gym.spaces.Space, Any]:

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
            ls_ops = deepcopy(LS_MOVES)
        elif self.operator_mode in ['ALL', 'SELECT_LS+']:
            ls_ops = deepcopy(LS_MOVES) + ["PERTURB"]
        else:
            raise ValueError(f"unknown operator_mode '{self.operator_mode}'")

        self.ls_set = ls_ops
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

    def _get_action(self, action: Union[np.ndarray, List, int]) -> Tuple[bool, List, Union[List, str]]:

        if isinstance(action, list) or isinstance(action, np.ndarray):
            assert len(action) <= 1
            action = action[0]

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
            ls_op = deepcopy(self.ls_set)  # all specified local search operators
            self.previous_op = 0.0
        elif self.operator_mode in ['SELECT_LS', 'SELECT_LS+']:
            num_ls_ops = len(self.ls_set)
            ls_idx = action % num_ls_ops
            ls_op = self.ls_set[int(ls_idx)]
            self.previous_op = float(10 ** ls_idx)
        else:
            raise ValueError

        # POSITION
        if self.position_mode in ['ALL', 'RANDOM']:
            pos = self.position_mode
        else:
            raise ValueError

        return (
            accept_move,
            ls_op,
            pos
        )

    def _step(self, action: Union[np.ndarray, List, int], **kwargs) -> Tuple[float, bool, Dict]:

        # if specified and no improvement for the last 'restart_at_step' steps,
        # we restart the solution procedure from a different point
        if 0 < self.restart_at_step <= self._num_steps_no_imp:
            self._restart()

        self.previous_cost = self.current_cost
        self.solver.checkpoint_solution()   # checkpoint current solution to reject move if not accepted
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
        self.current_sol, self.current_sol_w = self.graph.get_mch_graph()
        self.current_sol_seq = self.graph.get_mch_seq()
        self.previous_accept = accept_move

        if (
            self.operator_mode == "ALL"
            or (self.operator_mode == "SET" and len(self.ls_set) > 1)
            or ls_ops == -1
        ):
            ls = "MULTIPLE"
        else:
            ls = ls_ops[0] if isinstance(ls_ops, list) else ls_ops
        if self.debug > 1:
            # debug printing
            print(
                f"{self._current_step : <5}"
                f"{ls : ^20}"
                f"{self.mode_args['search_criterion'] : ^20}"
                f"{accept_move: ^6}"
                f"{self.current_cost : ^20}"
                f"{self.best_cost : ^20}"
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
                     **kwargs) -> Tuple[bool, List]:
        """Execute one step of the LS routine."""
        # unpack and format actions for VRPH solver
        accept_move, ls_ops, pos = self._get_action(action)

        if check_reject and not accept_move and self._current_step > 0:
            # can only reject after fist move was executed
            self.solver.reject_move()

        self.graph, self.current_cost = self.solver.solve(
            ls_ops=ls_ops,
            position=pos
        )
        return accept_move, ls_ops

    def _restart(self):
        """Restart the problem solver...

            - from  new initial solution,
            - from  best solution found so far or
            - after major perturbation of current solution
        """
        if self._restart_mode == "initial":
            # create new initial solution
            self.graph, self.current_cost = self.solver.construct(randomize=True)
            self.current_sol, self.current_sol_w = self.graph.get_mch_graph()
        elif self._restart_mode == "best":
            # set current solution to best solution found so far
            self.current_cost = self.best_cost
            self.current_sol = self.best_sol.copy()
            assert self._restart_ckpt is not None
            self.solver.graph.load_state_dict(self._restart_ckpt)
        elif self._restart_mode == "perturb":
            # apply major perturbations to current solution
            self.graph, self.current_cost = self.solver.perturb(**self._restart_args)
            self.current_sol, self.current_sol_w = self.graph.get_mch_graph()
        else:
            raise ValueError(f"unknown restart mode: '{self._restart_mode}'")

        self._num_steps_no_imp = 0
        self._num_restarts += 1

    def _init_instance(self, instance: JSSPInstance, **kwargs):
        """Unpack all relevant data, load it and create initial solution."""
        self.instance = instance
        N = instance.graph_size
        self._node_idx_set = np.arange(N)
        self.solver.load_problem(instance)
        self.graph, cost = self.solver.construct(**kwargs)

        sol_e, sol_w = self.graph.get_mch_graph()
        nbh_e, nbh_w = self.graph.get_job_graph()
        seq = self.graph.get_mch_seq()
        self.current_sol = sol_e.copy()
        self.current_sol_w = sol_w.copy()
        self.best_sol = sol_e.copy()
        self.best_sol_w = sol_w.copy()
        self.current_sol_seq = seq.copy()
        self.best_sol_seq = seq.copy()
        self.nbh_edges = nbh_e
        self.nbh_weights = nbh_w
        self.current_cost = cost
        self.best_cost = cost

        return sol_e

    def _init_generator(self, problem: str, **kwargs):
        assert problem.upper() == "JSSP"
        self.data_generator = JSSPDataset(
            problem=None if self.fixed_dataset and self.data_file_path is not None else problem,
            fname=self.data_file_path,
            seed=self._seed,
            float_prec=self.float_prec,
            verbose=self.debug > 1,
            **self.generator_args
        )

    def render(self, mode='human', as_gif: bool = True, **kwargs):
        assert self.enable_render, f"Need to specify <enable_render=True> on init."
        if as_gif and not self._render_step <= 150:
            warn(f"Can only render max ~150 steps as GIF but got render_step={self._render_step}. "
                 f"Following steps will not be rendered anymore!")
            return

        if self.viewer is None:
            # create new viewer object
            self.viewer = Viewer2(
                save_dir=self.plot_save_dir,
                gif_naming=f"render_ep{self._render_cnt}",
                as_gif=as_gif,
                **kwargs
            )

        # render new DAG
        self.viewer.update(
            graph=self.graph,
            cost=self.current_cost,
            n_iters=self._current_step,
            **kwargs
        )
        self._render_step += 1
        return self.viewer.render_rgb()


#
# ============= #
# ### TEST #### #
# ============= #
def _test_single(
    size: int = 1,
    n_j: int = 8,
    n_m: int = 4,
    seed: int = 0,
    n_steps: int = 50,
    acceptance_mode: str = 'SELECT_EPSILON',
    operator_mode: str = 'SET',
    position_mode: str = 'ALL',
    mode_args: dict = None,
    ls_op: List[str] = 'CEI',
    acceptance: str = 'BEST',
    random: bool = True,
    num_nodes_per_iter: int = 1,
    init_method: str = 'FIFO',
    render: bool = False,
    verbose: int = 1,
    all_true: bool = False,
    pause: float = 0.0,
    **kwargs
):
    import time
    np.random.seed(seed)

    sample_args = {'sample_size': size, 'num_jobs': n_j, 'num_machines': n_m}
    generator_args = {}
    mode_args = {
        'search_criterion': acceptance,
        'selection_criterion': 'sampling',
        'epsilon': 0.1,
        'random_shuffle': random,
        'ls_ops': ls_op if isinstance(ls_op, list) else [ls_op],
        'num_nodes_per_iter': num_nodes_per_iter,
        'restart_at_step': 8,
        'restart_mode': 'perturb', #'initial',
        'restart_args': {'iters': 3},
    } if mode_args is None else mode_args
    env = JSSPEnv(
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
    info = {}
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
        print(f"info: {info}")
