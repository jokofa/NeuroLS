#
import faulthandler
from abc import abstractmethod
from warnings import warn
from typing import Optional, Tuple, List, Dict, Union, Any
from timeit import default_timer
import multiprocessing as mp
import logging

import numpy as np
import gym
from torch.utils.data import DataLoader

from lib.routing import RPInstance
from lib.scheduling import JSSPInstance
from lib.env.utils import DatasetChunk

logger = logging.getLogger(__name__)


class BaseEnv(gym.Env):
    """
    Gym environment to solve Combinatorial Optimization Problems based on Local Search (LS).

    Args:
        construction_args: arguments for construction heuristic producing initial solution
        generator_args: arguments for data generator (sampling distribution, ...)
        sampling_args: arguments for data sampling procedure (size of sample, graph size, ...)
        num_steps: number of environment steps
        stand_alone: flag if env is used as stand alone instance or in a vector env
        fixed_dataset: flag if the data is sampled from a fixed dataset
                       instead of being generated on the fly
        data_file_path: file path to fixed dataset
        enable_render: flag if rendering should be possible
        plot_save_dir: directory to save rendered plots/gifs
        debug: flag if additional debug information should be printed
        clamp: float specifying clamp value of reward
        float_prec: floating point precision
    """
    metadata = {'render.modes': ['human']}
    PROBLEM = "None"

    def __init__(self,
                 construction_args: Optional[Dict] = None,
                 generator_args: Optional[Dict] = None,
                 sampling_args: Optional[Dict] = None,
                 num_steps: int = 1000,
                 stand_alone: bool = True,   # if the env is used stand-alone or in a vec env
                 fixed_dataset: bool = False,
                 data_file_path: Optional[str] = None,
                 enable_render: bool = False,
                 plot_save_dir: Optional[str] = None,
                 debug: Union[bool, int] = False,
                 clamp: Optional[float] = None,
                 float_prec: np.dtype = np.float32,
                 report_on_improvement: bool = False,
                 ):
        super(BaseEnv, self).__init__()

        self.construction_args = construction_args.copy() if construction_args is not None else {}
        self.generator_args = generator_args.copy() if generator_args is not None else {}
        self.sampling_args = sampling_args.copy() if sampling_args is not None else {}
        self.num_steps = num_steps
        self.stand_alone = stand_alone
        self.fixed_dataset = fixed_dataset
        self.data_file_path = data_file_path
        self.enable_render = enable_render
        self.plot_save_dir = plot_save_dir
        self.debug = 2 if isinstance(debug, bool) and debug else int(debug)
        self.clamp = clamp
        self.float_prec = float_prec
        self.report_on_improvement = report_on_improvement
        self._distributed_dataset = fixed_dataset and not stand_alone

        self.data_generator = None
        self.solver = None
        self.construction_op = None
        self.action_space = None
        self.observation_space = None
        self.instance = None
        self.graph = None

        self.viewer = None
        self.render_buffer = None

        self.best_cost = None
        self.best_sol = None
        self.current_cost = None
        self.current_sol = None
        self.current_sol_seq = None
        self.best_sol_seq = None
        self.previous_accept = None
        self.previous_reward = None
        self.previous_cost = None
        self.previous_op = None
        self.ls_set = None

        self._seed = None
        self._rnd = np.random.default_rng(1)
        self._is_reset = False
        self._tinit = None
        self._dataset_chunk = None

        self._current_step = None
        self._num_steps_no_imp = None
        self._num_restarts = None
        self._restart_mode = None
        self._restart_ckpt = None
        self._ls_op_cnt = None
        self._render_step = None
        self._render_cnt = 0

    def seed(self, seed: Union[Optional[int], Optional[DatasetChunk]] = None):
        """Seed all pseudo random generators."""
        ###
        # just slightly hacky exploit of tianshou subproc worker here to distribute dataset chunks
        # without reimplementing large parts of the subproc env / worker classes
        if (
            seed is not None and isinstance(seed, DatasetChunk) and
            (isinstance(seed[0], RPInstance) or isinstance(seed[0], JSSPInstance))
        ):
            return self._set_data_chunk(seed)
        ###
        else:
            if (self._seed is None) or (seed is not None and self._seed != seed):
                self._seed = seed
                self._rnd = np.random.default_rng(seed)
                if hasattr(self.solver, "seed"):
                    self.solver.seed(seed)
                if self.data_generator is None:
                    raise RuntimeError(f"Must call _init_spaces() "
                                       f"in sub module during initialization")
                self.data_generator.seed(seed+1)
                if self.construction_op is not None:
                    self.construction_op.seed(seed+2)
                if self.action_space is not None:
                    self.action_space.seed(seed+3)
        return [seed]

    def reset(self):
        """Reset the simulator and return the initial state."""
        if self._seed is None:
            warn(f"No seed for random state defined. Generated data will not be reproducible!")
        self._current_step = 0
        self._num_steps_no_imp = 0
        self._num_restarts = 0
        self._ls_op_cnt = {}
        self._tinit = default_timer()

        # get data instance, init and set up model
        # also sets initial costs
        self._init_instance(self._get_instance())

        self.previous_accept = False
        self.previous_reward = 0.0
        self.previous_cost = None
        self.previous_op = 0

        if self.report_on_improvement:
            self._report_solution()
        if self.enable_render:
            if self.viewer is not None:
                self.viewer.save()
                self.viewer.close()
                self.viewer = None
            self.render_buffer = {'edges': self._get_edges_to_render()}
            # with epsilon acceptance we do not render every step
            # so we need to keep track of the number of steps we really rendered
            self._render_step = 0
            self._render_cnt += 1

        self._is_reset = True
        return self._get_observation()

    def step(self, action: Union[np.ndarray, int]):
        """Take an action and do one step in the environment.

        Args:
            action: discrete action id

        Returns:
            - State,
            - reward,
            - done,
            - info dict
        """
        assert self._is_reset
        self._current_step += 1

        if self.debug > 1:
            faulthandler.enable()
        # step dynamics
        reward, done, info = self._step(action)
        if self.debug > 1:
            faulthandler.disable()

        # update buffers for best solution and corresponding cost
        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self.best_sol = self.current_sol.copy()
            if self.graph is not None and \
                    self._restart_mode is not None and \
                    self._restart_mode == "best":
                self._restart_ckpt = self.graph.state_dict()
            if self.current_sol_seq is not None:
                self.best_sol_seq = self.current_sol_seq.copy()
            self._num_steps_no_imp = 0
            if self.report_on_improvement:
                self._report_solution()
        else:
            self._num_steps_no_imp += 1

        if done:
            tm = default_timer() - self._tinit
            # set flag requiring reset
            self._is_reset = False
            # add some additional info
            info['best_cost_final'] = self.best_cost
            info['solution'] = self.best_sol_seq
            info['time_elapsed'] = tm
            info['instance'] = (np.array(self.instance, dtype=object)
                                if self.fixed_dataset or self.stand_alone else None)
            info['num_restarts'] = self._num_restarts
            info['ls_op_cnt'] = self._ls_op_cnt

        return self._get_observation(), reward, done, info

    def render(self, mode: str = 'human', **kwargs):
        """Environment rendering functionality."""
        raise NotImplementedError

    def close(self):
        if self.solver is not None:
            del self.solver
        if self.viewer is not None:
            self.viewer.close()

    def _set_data_chunk(self, chunk: Union[List, DatasetChunk]):
        """Store chunk of fixed dataset provided by distributed data handler."""
        assert self._distributed_dataset, \
            f"can only work on data chunks when fixed_dataset=True and stand_alone=False"
        self._dataset_chunk = chunk
        try:
            self._get_data_chunk()
            self.data_samples = None
        except Exception as e:
            warn(f"Problem on data chunk distribution. Encountered: {e}")
            return False
        return True

    def _create_data(self,
                     problem: str,
                     sample_size: int,
                     shuffle_samples: bool = True,
                     init: bool = False,
                     **kwargs):
        """Create data instances with generator and wrap into dataloader for sampling and batching."""
        if self._distributed_dataset and not init:
            raise RuntimeError(f"only stand-alone env can load fixed dataset.")
        # create generator if does not exist yet
        if self.data_generator is None:
            self._init_generator(problem, **kwargs)
        # sample some data instances
        # or automatically load instances if file path is provided
        data = self.data_generator.sample(
            sample_size=sample_size,
            **kwargs
        )
        # slightly exploiting pytorch dataloader as sampling iterator
        dl = DataLoader(
            data,
            batch_size=1,
            collate_fn=lambda x: x,     # identity -> returning simple list of instances
            shuffle=shuffle_samples
        )
        self.data_samples = iter(dl)

    def _get_data_chunk(self):
        """Just (re-)create dataloader iterator over provided dataset."""
        assert self._distributed_dataset, \
            f"can only work on data chunks when fixed_dataset=True and stand_alone=False"
        assert self._dataset_chunk is not None, f"need to distribute data chunks first."
        dl = DataLoader(
            self._dataset_chunk,
            batch_size=1,
            collate_fn=lambda x: x,  # identity -> returning simple list of instances
            shuffle=False
        )
        self.data_samples = iter(dl)

    def _get_instance(self) -> Union[RPInstance, JSSPInstance]:
        """Get a data instance from the samples buffer."""
        try:
            sample = self.data_samples.next()
        except (StopIteration, AttributeError):
            if self._distributed_dataset:
                # reset iterator on data chunk
                self._get_data_chunk()
            else:
                # sample new data instances
                self._create_data(self.PROBLEM, **self.sampling_args)
            sample = self.data_samples.next()

        assert len(sample) == 1
        return sample[0]

    def _init_spaces(self):
        """
        Called to infer action and observation space
        dimensions from data sample.
        """
        # sample small amount of data to infer format
        sampling_args = self.sampling_args.copy()
        sampling_args['sample_size'] = 1
        self._create_data(self.PROBLEM, init=True, **sampling_args)
        sample = self.data_samples.next()
        # init and decode once. Also serves as a fast debug test for the env
        self._init_instance(sample[0])
        # infer dimensions and create spaces
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        # clean up
        self.data_samples = None
        self._render_cnt = 0

    def _report_solution(self):
        pid = mp.current_process().pid
        # format solution
        sol = self.best_sol_seq.copy()
        sol = sol[sol.sum(-1) != 0]
        p_sol = []
        for s in sol:
            _s = []
            for i in range(1, len(s)):
                if s[i] == 0:
                    _s.append(s[i - 1])
                    _s.append(0)
                    break
                else:
                    _s.append(s[i - 1])
            p_sol.append(_s)

        msg = f"process {pid} ({self.PROBLEM.upper()}) new best solution found at " \
              f"iter={self._current_step} (time={default_timer()-self._tinit :.6f}) " \
              f"with cost={self.best_cost :.8f}:  {p_sol}"
        logger.warning(msg)

    @abstractmethod
    def _init_generator(self, problem: str, **kwargs):
        """Init the problem specific data generator object."""
        raise NotImplementedError

    @abstractmethod
    def _init_instance(self, instance: Union[RPInstance, JSSPInstance], **kwargs):
        """Unpack all relevant data, create initial solution and init solver."""
        raise NotImplementedError

    @abstractmethod
    def _step(self, action: Union[np.ndarray, int], **kwargs) -> Tuple[float, bool, Dict]:
        """Execute one step of internal environment dynamics."""
        raise NotImplementedError

    @abstractmethod
    def _get_observation_space(self, **kwargs) -> gym.spaces.Space:
        """Construct and return observation space for respective problem env."""
        raise NotImplementedError

    @abstractmethod
    def _get_action_space(self, **kwargs) -> gym.spaces.Space:
        """Construct and return action space for respective problem env."""
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self, **kwargs) -> Dict:
        """
        Convenience function to assemble the current
        state observation of the environment.
        Is called by reset() and step().
        """
        raise NotImplementedError

    @abstractmethod
    def _get_action(self, action: Union[np.ndarray, List, int]) -> Tuple:
        """Infer the solver configuration from the provided action."""
        raise NotImplementedError

    def _get_edges_to_render(self) -> Union[List, np.ndarray, Any]:
        """Access and optionally preprocess the edges for rendering"""
        return

    def _compute_reward(self,
                        cost_new: Union[np.ndarray, float],
                        cost: Union[np.ndarray, float],
                        clamp: Optional[float] = None,
                        ) -> Union[np.ndarray, float]:
        """Take new cost and current/best cost and computes the reward."""
        rew = cost - cost_new
        self.previous_reward = rew
        if clamp is not None:
            rew = max(rew, clamp)
        return rew
