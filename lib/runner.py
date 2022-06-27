#
import os
import logging
import shutil
from warnings import warn
from typing import Optional, Dict, Union, List
from omegaconf import DictConfig, OmegaConf as oc
from copy import deepcopy

import math
import random
import numpy as np
import hydra
import torch
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import DQNPolicy, IQNPolicy

from lib.utils import recursive_str_lookup
from lib.env import VecEnv, RPGraphVecEnv
from lib.routing import eval_rp, RP_TYPES, RPSolution
from lib.scheduling import eval_jssp, JSSP_TYPES, JSSPSolution
from lib.networks import Model
from lib.utils.tianshou_utils import (
    CheckpointCallback,
    MonitorCallback,
    offpolicy_trainer,
    tester,
    TestCollector
)

logger = logging.getLogger(__name__)


#
class Runner:
    """
    Wraps all setup, training and testing functionality
    of the respective experiments configured by cfg.
    """
    def __init__(self, cfg: DictConfig):

        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)

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
        self._build_env()
        self._build_model()
        self._build_policy()
        self.seed_all(self.cfg.global_seed)
        self._build_collectors()
        self._build_callbacks()

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # checkpoint save dir
        self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
        # val log dir
        self.cfg.val_log_path = os.path.join(self._cwd, self.cfg.val_log_path)
        os.makedirs(self.cfg.val_log_path, exist_ok=True)

    def _build_env(self):
        """Create and wrap the problem environments."""
        self.policy_type = self.cfg.policy.upper()
        env_cfg = self.cfg.env_cfg.copy()
        env_kwargs = self.cfg.env_kwargs.copy()
        env_kwargs['debug'] = self.debug
        clamp = self.cfg.policy_cfg.pop("clamp_reward", False)
        env_kwargs['clamp'] = clamp
        self.env = self._get_env_cl()(
            num_envs=self.cfg.train_batch_size,
            problem=self.cfg.problem,
            env_kwargs=env_kwargs,
            **env_cfg
        )
        self.env.seed(self.cfg.global_seed)
        # overwrite cfg for validation env
        val_env_cfg = deepcopy(self.cfg.env_cfg)
        val_env_cfg.update(self.cfg.get('val_env_cfg', {}))
        val_env_kwargs = deepcopy(env_kwargs)
        val_env_kwargs.update(self.cfg.get('val_env_kwargs', {}))
        render = self.cfg.get('render_val', False)
        if render:
            assert self.cfg.val_batch_size == 1, f"can only render for test_batch_size=1"
            val_env_kwargs['enable_render'] = True
            val_env_kwargs['plot_save_dir'] = self.cfg.val_log_path

        self._val_env_kwargs = val_env_kwargs.copy()
        self.val_env = self._get_env_cl()(
            num_envs=self.cfg.val_batch_size,
            problem=self.cfg.problem,
            env_kwargs=val_env_kwargs,
            dataset_size=self.cfg.val_dataset_size,
            **val_env_cfg
        )
        self.val_env.seed(self.cfg.global_seed+1)

    def _get_env_cl(self):
        # dont need additional graph support when using feed forward encoder
        if (
                self.cfg.problem.upper() == "JSSP" or
                "FF" in "".join(recursive_str_lookup(self.cfg.model_cfg.encoder_args))
        ):
            return VecEnv
        return RPGraphVecEnv

    def _build_model(self):
        """Initialize the model and the corresponding learning algorithm."""
        # make sure decoder works with specified env mode
        a_mode = self.cfg.env_kwargs.acceptance_mode
        o_mode = self.cfg.env_kwargs.operator_mode
        p_mode = self.cfg.env_kwargs.position_mode

        env_modes = {
            'acceptance': a_mode,
            'operator': o_mode,
            'position': p_mode,
        }
        self.model = Model(
            problem=self.cfg.problem,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            env_modes=env_modes,
            device=self.device,
            policy_type=self.policy_type,
            **self.cfg.model_cfg
        )
        logger.info(self.model)

    def _build_policy(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        policy_cfg = self.cfg.policy_cfg.copy()
        lr = self.cfg.optimizer_cfg.get('lr')
        self.optim = self.model.get_optimizer(**self.cfg.optimizer_cfg)
        # policy
        self.train_fn = None
        self.test_fn = None
        self.exploration_noise = policy_cfg.pop("exploration_noise", False)

        if self.policy_type in ["DQN", "IQN"]:
            eps_train = policy_cfg.pop("epsilon", 1.0)
            eps_test = policy_cfg.pop("epsilon_test", 0.0)
            eps_final = policy_cfg.pop("epsilon_final", 0.01)
            frac_epoch_final = policy_cfg.pop("frac_epoch_final", 0.7)
            max_epoch = self.cfg.trainer_cfg.max_epoch
            epoch_final = math.ceil(frac_epoch_final * max_epoch)

            if self.policy_type == "DQN":
                policy = DQNPolicy(model=self.model, optim=self.optim, **policy_cfg)
            elif self.policy_type == "IQN":
                policy = IQNPolicy(model=self.model, optim=self.optim, **policy_cfg)

            def train_fn(num_epoch: int, env_step: int):
                """A hook called at the beginning of training in each epoch."""
                # linear epsilon decay in the first epoch_final epochs
                if num_epoch <= epoch_final:
                    #eps = eps_train - env_step / 1e6 * (eps_train - eps_final)
                    eps = eps_train - num_epoch / epoch_final * (eps_train - eps_final)
                elif num_epoch == epoch_final:
                    eps = eps_final
                    # in final (late intermediate) epoch once reduce lr
                    for pg in policy.optim.param_groups:
                        pg['lr'] = lr * 0.1
                else:
                    eps = eps_final
                policy.set_eps(eps)

            def test_fn(num_epoch: int, env_step: int):
                """A hook called at the beginning of testing in each epoch."""
                policy.set_eps(eps_test)

            self.policy = policy
            self.train_fn = train_fn
            self.test_fn = test_fn
            
        else:
            raise ValueError(f"unknown policy: '{self.cfg.policy}'")

        # replay buffer
        replay_buffer_cfg = self.cfg.replay_buffer_cfg.copy()
        self.prioritized = replay_buffer_cfg.pop("prioritized", False)
        if self.prioritized:
            self.rp_buffer = PrioritizedVectorReplayBuffer(buffer_num=self.cfg.train_batch_size,
                                                           **replay_buffer_cfg)
        else:
            self.rp_buffer = VectorReplayBuffer(buffer_num=self.cfg.train_batch_size,
                                                **replay_buffer_cfg)

    def _build_collectors(self):
        """Create necessary collectors."""
        self.train_collector = Collector(
            policy=self.policy,
            env=self.env,
            buffer=self.rp_buffer,
            exploration_noise=self.exploration_noise
        )

        size = self.cfg.val_dataset_size + 2 * self.cfg.val_batch_size  # * test_env_kwargs.num_steps
        buf = VectorReplayBuffer(size, self.cfg.val_batch_size)
        # create collector
        self.val_collector = TestCollector(
            policy=self.policy,
            env=self.val_env,
            buffer=buf,
        )

    def _build_callbacks(self):
        """Create necessary callbacks."""
        self.callbacks = {}
        self.callbacks["save_checkpoint_fn"] = CheckpointCallback(
            exp=self,
            save_dir=self.cfg.checkpoint_save_path,
            metric_key=self.cfg.eval_metric_key,
            **self.cfg.checkpoint_cfg
        )
        self.callbacks["monitor"] = MonitorCallback(
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
        self.val_env.seed(seed+1)

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

    def train(self, **kwargs):
        """Train the specified model."""
        logger.info(f"start training...")
        results, solutions = offpolicy_trainer(
            problem=self.cfg.problem,
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.val_collector,
            episode_per_test=self.cfg.val_dataset_size,
            batch_size=self.cfg.update_batch_size,
            train_fn=self.train_fn,
            test_fn=self.test_fn,
            verbose=self.debug,
            render_val=self.cfg.render_val,
            **self.callbacks,
            **self.cfg.trainer_cfg,
            **kwargs
        )
        logger.info(f"training finished.")
        logger.info(results)
        solutions, summary = self._eval(solutions)
        self.callbacks['monitor'].save_results({
            "solutions": solutions,
            "summary": summary
        }, 'val_results')
        logger.info(summary)

    def test(self, test_cfg: Optional[Union[DictConfig, Dict]] = None, **kwargs):
        """Test (evaluate) the provided trained model."""

        # load model checkpoint
        ckpt_pth = self.cfg.get('checkpoint_load_path')
        assert ckpt_pth is not None
        logger.info(f"loading model checkpoint: {ckpt_pth}")
        state_dict = torch.load(ckpt_pth, map_location=self.device)

        # get tester_cfg from current (test) cfg file
        tester_cfg = self.cfg.tester_cfg.copy()
        tb_log_path = os.path.join(os.getcwd(), self.cfg.tb_log_path)

        # get checkpoint cfg and update
        self.cfg.update(state_dict["cfg"])
        # update cfg with additionally provided args
        if test_cfg is not None:
            test_cfg = oc.to_container(test_cfg, resolve=True) if isinstance(test_cfg, DictConfig) else test_cfg
            tester_cfg.update(test_cfg.get('tester_cfg', {}))
            self.cfg.update(test_cfg)

        # create test env
        test_env_cfg = self.cfg.env_cfg.copy()
        test_env_cfg.update(tester_cfg.get('test_env_cfg', {}))
        test_env_kwargs = self.cfg.env_kwargs.copy()
        test_env_kwargs.update(tester_cfg.get('env_kwargs', {}))
        clamp = self.cfg.policy_cfg.pop("clamp_reward", False)
        test_env_kwargs['clamp'] = clamp
        render = tester_cfg.get('render', False)
        if render:
            assert tester_cfg.test_batch_size == 1, f"can only render for test_batch_size=1"
            test_env_kwargs['enable_render'] = True
            test_env_kwargs['plot_save_dir'] = self.cfg.test_log_path

        self.env = self._get_env_cl()(
            num_envs=tester_cfg.test_batch_size,
            problem=self.cfg.problem,
            env_kwargs=test_env_kwargs,
            dataset_size=tester_cfg.get('test_dataset_size'),
            **test_env_cfg
        )
        self.env.seed(self.cfg.global_seed + 2)

        # load checkpoint model
        self.policy_type = self.cfg.policy.upper()
        self._build_model()
        try:
            self.model.load_state_dict(state_dict["model"])
        except RuntimeError as e:
            raise RuntimeError(
                f"modes specified in tester_cfg are different from modes specified during training "
                f"and action dimensionality is not compatible: \n   {e}")
        self._build_policy()

        size = tester_cfg.test_dataset_size + 3*tester_cfg.test_batch_size     # * test_env_kwargs.num_steps
        buf = VectorReplayBuffer(size, tester_cfg.test_batch_size)
        # create collector
        test_collector = TestCollector(
            policy=self.policy,
            env=self.env,
            buffer=buf,
        )

        # create callback
        monitor = MonitorCallback(
            tb_log_path=tb_log_path,
            metric_key=self.cfg.eval_metric_key,
            **self.cfg.monitor_cfg
        )

        # run test inference
        logger.info("running test inference...")
        results, solutions = tester(
            problem=self.cfg.problem,
            policy=self.policy,
            test_collector=test_collector,
            episode_per_test=tester_cfg.test_dataset_size,
            monitor=monitor,
            render=0.0001 if render else 0,  # rendering is deactivated for render=0
            num_render_eps=tester_cfg.get('num_render_eps', 1)
        )
        logger.info(f"finished.")
        logger.info(results)
        solutions, summary = self._eval(solutions)
        monitor.save_results({
            "solutions": solutions,
            "summary": summary
        }, 'test_results')
        logger.info(list(summary.values()))
        logger.info(summary)

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        ckpt_pth = self.cfg.get('checkpoint_load_path')
        assert ckpt_pth is not None
        state_dict = torch.load(ckpt_pth, map_location=self.device)
        self.load_state_dict(state_dict)

        # remove the unnecessary new directory hydra creates
        new_hydra_dir = os.getcwd()
        if "resume" in new_hydra_dir:
            remove_dir_tree("resume", pth=new_hydra_dir)

        logger.info(f"resuming training from: {ckpt_pth}")
        self.train(resume_from_log=True, **kwargs)

    def state_dict(self, *args, **kwargs) -> Dict:
        """Save states of all experiment components
        in PyTorch like state dictionary."""
        return {
            "cfg": self.cfg.copy(),
            "model": self.model.state_dict(*args, **kwargs),
            "optim": self.optim.state_dict(),
            #"rp_buffer": self.rp_buffer.__dict__.copy(),
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load a previously saved state_dict and
        reinitialize all required components.

        Examples:
            state_dict = torch.load(PATH)
            experiment.load_state_dict(state_dict)
        """
        self.cfg.update(state_dict["cfg"])
        self._dir_setup()
        self._build_env()
        self._build_model()
        self.model.load_state_dict(state_dict["model"])
        self._build_policy()
        self.optim.load_state_dict(state_dict['optim'])
        #self.rp_buffer.__dict__.update(state_dict["rp_buffer"])
        self._build_collectors()
        self._build_callbacks()

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug']:
            self.setup()
            self.train()
        elif self.cfg.run_type == 'resume':
            self.resume()
        elif self.cfg.run_type in ['test', 'val']:
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['train', 'resume', 'test', 'val', 'debug']")


def update_path(cfg: DictConfig, fixed_dataset: bool = True):
    """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
    cwd = hydra.utils.get_original_cwd()
    if fixed_dataset:
        if cfg.val_env_cfg.data_file_path is not None:
            cfg.val_env_cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.val_env_cfg.data_file_path)
            )
        if cfg.tester_cfg.test_env_cfg.data_file_path is not None:
            cfg.tester_cfg.test_env_cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.tester_cfg.test_env_cfg.data_file_path)
            )
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
