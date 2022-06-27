#
import os
import shutil
import logging
from warnings import warn
from typing import Optional, Dict, Union
from omegaconf import DictConfig, OmegaConf

import random
import numpy as np
import hydra
import torch

from lib.routing import RPDataset, RPInstance, RPSolution, eval_rp
from baselines.CVRP.DACT.DACT.problems.problem_vrp import CVRP
from baselines.CVRP.DACT.DACT.agent.ppo import PPO
from baselines.CVRP.DACT.dact import train_model, eval_model

logger = logging.getLogger(__name__)


class Runner:
    """
    Wraps all setup, training and testing functionality
    of the respective experiments configured by cfg.
    """
    def __init__(self, cfg: DictConfig):

        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)
        OmegaConf.set_struct(self.cfg, False)

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        if self.debug > 1:
            torch.autograd.set_detect_anomaly(True)

        # set device
        if torch.cuda.is_available() and not cfg.cuda:
            warn(f"Cuda GPU is available but not used! Specify <cuda=True> in config file.")
        self.device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")

        # raise error on strange CUDA warnings which are not caught
        if (self.cfg.run_type == "train") and cfg.cuda and not torch.cuda.is_available():
            e = "..."
            try:
                torch.zeros(10, device=torch.device("cuda"))
            except Exception as e:
                pass
            raise RuntimeError(f"specified training run on GPU but running on CPU! ({str(e)})")

    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self.seed_all(self.cfg.global_seed)
        self._build_env()
        self._build_policy()

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # checkpoint save dir
        self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
        # val log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)

    def _build_env(self):
        """Load dataset and create environment (problem state)."""
        ds = RPDataset(
            fname=self.cfg.data_file_path,
            seed=self.cfg.global_seed,
            verbose=self.debug > 1,
            **self.cfg.generator_args
        )
        # load dataset
        self.data = ds.sample(sample_size=self.cfg.dataset_size)

        # init problem state
        cfg = self.cfg.env_cfg.copy()
        self.env = CVRP(
            p_size=self.cfg.graph_size,
            step_method=cfg.step_method,
            init_val_met=cfg.init_val_met,
            with_assert=self.debug > 1,
            P=cfg.perturb_eps,
            DUMMY_RATE=cfg.dummy_rate
        )

    def _build_policy(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        cfg = self.cfg.copy()
        cfg.use_cuda = self.cfg.cuda and torch.cuda.is_available()
        cfg.distributed = False
        cfg.no_saving = True
        cfg.device = self.device.type
        cfg.no_progress_bar = False
        self.policy = PPO(self.env.NAME, self.env.size, cfg)

    def save_results(self, result: Dict):
        pth = os.path.join(self.cfg.log_path, "results.pkl")
        torch.save(result, pth)

    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train(self, **kwargs):
        """Train the specified model."""

        raise NotImplementedError
        #agent.start_training(problem, opts.val_dataset, tb_logger)

        logger.info(f"start training...")
        results, solutions = train_model(
            ...
        )
        logger.info(f"training finished.")
        logger.info(results)
        solutions, summary = eval_rp(solutions, problem=self.cfg.problem)
        self.save_results({
            "solutions": solutions,
            "summary": summary
        })
        logger.info(summary)

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""
        if not self.cfg.problem.upper() == "CVRP":
            raise NotImplementedError

        self.cfg.eval_only = True
        self.setup()
        self.policy.load(self.cfg.checkpoint_load_path)

        # run test inference
        logger.info("running test inference...")
        _, solutions = eval_model(
            data=self.data,
            problem=self.env,
            agent=self.policy,
            opts=self.cfg.copy(),
            dummy_rate=self.cfg.env_cfg.dummy_rate,
            device=self.device,
            batch_size=self.cfg.batch_size,
        )
        logger.info(f"finished.")

        solutions, summary = eval_rp(
            solutions,
            problem=self.cfg.problem,
            strict_feasibility=self.cfg.get("strict_max_num", True)
        )
        self.save_results({
            "solutions": solutions,
            "summary": summary,
        })
        logger.info(list(summary.values()))
        logger.info(summary)

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        self.setup()
        self.policy.load(self.cfg.checkpoint_load_path)
        epoch_resume = int(os.path.splitext(os.path.split(self.cfg.checkpoint_load_path)[-1])[0].split("-")[1])
        logger.info(f"Resuming after {epoch_resume}")
        self.policy.opts.epoch_start = epoch_resume + 1

        # remove the unnecessary new directory hydra creates
        new_hydra_dir = os.getcwd()
        if "resume" in new_hydra_dir:
            remove_dir_tree("resume", pth=new_hydra_dir)

        self.train(**kwargs)

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug']:
            self.setup()
            self.train()
        elif self.cfg.run_type == 'resume':
            self.resume()
        elif self.cfg.run_type in ['val', 'test']:
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['train', 'resume', 'val', 'test', 'debug']")


def update_path(cfg: DictConfig):
    """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
    cwd = hydra.utils.get_original_cwd()

    if cfg.data_file_path is not None:
        cfg.data_file_path = os.path.normpath(
            os.path.join(cwd, cfg.data_file_path)
        )
    # if cfg.val_env_cfg.data_file_path is not None:
    #     cfg.val_env_cfg.data_file_path = os.path.normpath(
    #         os.path.join(cwd, cfg.val_env_cfg.data_file_path)
    #     )
    # if cfg.tester_cfg.test_env_cfg.data_file_path is not None:
    #     cfg.tester_cfg.test_env_cfg.data_file_path = os.path.normpath(
    #         os.path.join(cwd, cfg.tester_cfg.test_env_cfg.data_file_path)
    #     )
    if cfg.checkpoint_load_path is not None:
        cfg.checkpoint_load_path = os.path.normpath(
                os.path.join(cwd, cfg.checkpoint_load_path)
            )
    return cfg


def remove_dir_tree(root: str, pth: Optional[str] = None):
    """Remove the full directory tree of the root directory if it exists."""
    if not os.path.isdir(root) and pth is not None:
        # select root directory from path by dir name
        i = pth.index(root)
        root = pth[:i+len(root)]
    if os.path.isdir(root):
        shutil.rmtree(root)
