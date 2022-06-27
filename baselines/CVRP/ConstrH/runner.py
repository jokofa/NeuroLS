#
import os
import logging
from typing import Dict
from omegaconf import DictConfig

import random
import numpy as np
import hydra
import torch

from lib.routing import RPDataset, eval_rp
from baselines.CVRP.ConstrH.constr import ParallelSolver

logger = logging.getLogger(__name__)


class Runner:
    """
    Wraps all setup, training and testing functionality
    of the respective experiments configured by cfg.
    """

    def __init__(self, cfg: DictConfig):
        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)
        # debug level
        if self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0

    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self.seed_all(self.cfg.global_seed)
        self._get_data()
        self._build_policy()

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # val log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)

    def _get_data(self):
        """Load dataset."""
        ds = RPDataset(
            fname=self.cfg.data_file_path,
            seed=self.cfg.global_seed,
            verbose=self.debug > 1,
            **self.cfg.generator_args
        )
        # load dataset
        self.data = ds.sample(sample_size=self.cfg.dataset_size)

    def _build_policy(self):
        """Load and prepare data and initialize GORT routing models."""
        policy_cfg = self.cfg.policy_cfg.copy()
        self.policy = ParallelSolver(
            problem=self.cfg.problem,
            solver_args=policy_cfg,
            num_workers=self.cfg.batch_size
        )

    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_results(self, result: Dict):
        pth = os.path.join(self.cfg.log_path, "results.pkl")
        torch.save(result, pth)

    def run(self):
        self.setup()
        logger.info("running inference...")
        solutions = self.policy.solve(self.data)
        logger.info(f"finished.")
        solutions, summary = eval_rp(
            solutions, strict_feasibility=self.cfg.get("strict_max_num", True)
        )
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
        if cfg.data_file_path is not None:
            cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.data_file_path)
            )
    return cfg
