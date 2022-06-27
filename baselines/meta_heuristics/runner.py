#
import os
import logging
from typing import Dict, List, Union
from omegaconf import DictConfig

import random
import numpy as np
import hydra
import torch
from tianshou.data import VectorReplayBuffer

from lib.env import VecEnv
from lib.routing import eval_rp, RP_TYPES, RPSolution
from lib.scheduling import eval_jssp, JSSP_TYPES, JSSPSolution
from lib.utils.tianshou_utils import MonitorCallback, tester, TestCollector
from baselines.meta_heuristics.sa import SimulatedAnnealing
from baselines.meta_heuristics.ils import IteratedLocalSearch
from baselines.meta_heuristics.vns import VariableNeighborhoodSearch

logger = logging.getLogger(__name__)


#
class Runner:
    def __init__(self, cfg: DictConfig):
        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)
        # debug level
        if self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        self.policy_id = self.cfg.policy.upper()

    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self._build_env()
        self._build_policy()
        self.seed_all(self.cfg.global_seed)
        self._build_collectors()
        self._build_callbacks()

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)

    def _build_env(self):
        """Create and wrap the problem environments."""
        env_cfg = self.cfg.env_cfg.copy()
        env_kwargs = self.cfg.env_kwargs.copy()

        if self.policy_id in ["SA", "ILS"]:
            assert env_kwargs.get('acceptance_mode') == 'SELECT_EPSILON', \
                f"{self.policy_id} must use acceptance_mode='SELECT_EPSILON'"
            if (
                    'SELECT' in env_kwargs.get('operator_mode') or
                    'SELECT' in env_kwargs.get('position_mode')
            ):
                raise ValueError(f"{self.policy_id} does not work with 'select' "
                                 f"modes for operator and position")
        elif self.policy_id == "VNS":
            assert self.cfg.env_kwargs.get("acceptance_mode") == "ACCEPT_EPSILON", \
                f"{self.policy_id} must use acceptance_mode='ACCEPT_EPSILON'"
            assert self.cfg.env_kwargs.get("operator_mode") == "SELECT_LS", \
                f"{self.policy_id} must use operator_mode='SELECT_LS'"
        else:
            raise ValueError(f"unknown policy {self.policy_id}")

        self.render = self.cfg.get('render', False)
        if self.render:
            assert self.cfg.batch_size == 1, f"can only render for test_batch_size=1"
            env_kwargs['enable_render'] = True
            env_kwargs['plot_save_dir'] = self.cfg.log_path

        env_kwargs['debug'] = self.debug
        self.env = VecEnv(
            num_envs=self.cfg.batch_size,
            problem=self.cfg.problem,
            env_kwargs=env_kwargs,
            **env_cfg
        )
        self.env.seed(self.cfg.global_seed)

    def _build_policy(self):
        """Initialize policy."""
        policy_cfg = self.cfg.policy_cfg.copy()
        restart_at_step = policy_cfg.get("restart_at_step", 0)
        assert (
                self.cfg.env_kwargs.get("mode_args").get("restart_at_step", 0) ==
                restart_at_step
        ), f"'restart_at_step' must be same for env and policy."

        if self.policy_id == "SA":
            assert self.cfg.env_kwargs.get("mode_args").get("restart_mode") == "initial"
            self.policy = SimulatedAnnealing(env=self.env, **policy_cfg)
        elif self.policy_id == "ILS":
            assert 0 < restart_at_step <= 20
            assert self.cfg.env_kwargs.get("mode_args").get("restart_mode") in ["perturb", "perturb_kopt"]
            self.policy = IteratedLocalSearch(env=self.env, **policy_cfg)
        elif self.policy_id == "VNS":
            assert 0 < restart_at_step <= 4
            assert self.cfg.env_kwargs.get("mode_args").get("restart_mode") in ["perturb", "perturb_kopt"]
            self.policy = VariableNeighborhoodSearch(env=self.env, **policy_cfg)
        else:
            raise ValueError(f"unknown policy {self.policy_id}")

    def _build_collectors(self):
        """Create necessary collectors."""
        size = self.cfg.dataset_size + 3*self.cfg.batch_size
        buf = VectorReplayBuffer(size, self.cfg.batch_size)
        self.collector = TestCollector(
            policy=self.policy,
            env=self.env,
            buffer=buf,
        )

    def _build_callbacks(self):
        """Create necessary callbacks."""
        self.monitor = MonitorCallback(
            tb_log_path=self.cfg.tb_log_path,
            metric_key=self.cfg.eval_metric_key,
            **self.cfg.monitor_cfg
        )

    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.env.seed(seed)
        self.policy.seed(seed+1)

    def save_results(self, result: Dict):
        pth = os.path.join(self.cfg.log_path, "results.pkl")
        torch.save(result, pth)

    def _eval(self, solutions: List[Union[RPSolution, JSSPSolution]]):
        if self.cfg.problem.upper() in RP_TYPES:
            return eval_rp(
                solutions,
                problem=self.cfg.problem,
                strict_feasibility=self.cfg.get("strict_max_num", True)
            )
        elif self.cfg.problem.upper() in JSSP_TYPES:
            return eval_jssp(solutions, problem=self.cfg.problem)
        else:
            raise NotImplementedError(f"evaluation for {self.cfg.problem} not implemented.")

    def run(self):
        self.setup()
        logger.info("running inference...")
        result, solutions = tester(
            problem=self.cfg.problem,
            policy=self.policy,
            test_collector=self.collector,
            episode_per_test=self.cfg.dataset_size,
            monitor=self.monitor,
            render=0.0001 if self.render else 0,  # rendering is deactivated for render=0
            num_render_eps=self.cfg.get('num_render_eps', 1)
        )
        logger.info(f"finished.")
        logger.info(result)
        solutions, summary = self._eval(solutions)
        self.save_results({
            "solutions": solutions,
            "summary": summary
        })
        logger.info(list(summary.values()))
        logger.info(summary)


def update_path(cfg: DictConfig, fixed_dataset: bool = True):
    """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
    cwd = hydra.utils.get_original_cwd()
    if fixed_dataset:
        if cfg.env_cfg.data_file_path is not None:
            cfg.env_cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.env_cfg.data_file_path)
            )
    return cfg
