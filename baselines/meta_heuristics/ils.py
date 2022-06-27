#
import logging
from typing import Optional, Dict, Union, Any, List

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import BasePolicy

from lib.env import VecEnv

logger = logging.getLogger(__name__)
np.warnings.filterwarnings('ignore', category=RuntimeWarning)   # ignore overflow RuntimeWarning in 1st iteration


class IteratedLocalSearch(BasePolicy):
    """
    Policy executing an iterated LS strategy which can be used
    in the tianshou policy framework and is batched over a VectorEnv.
    """
    def __init__(self,
                 env: VecEnv,
                 use_sa: bool = False,
                 tau_init: float = 1.0,
                 tau_final: float = 0.0,
                 alpha: float = 0.8,
                 cooling_schedule: str = "lin",
                 acceptance_criterion: str = "metropolis",
                 num_max_steps: Optional[int] = None,
                 restart_at_step: int = 10,
                 **kwargs):
        super(IteratedLocalSearch, self).__init__()
        self._env = env
        self.use_sa = use_sa
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.alpha = alpha
        self.cooling_schedule = cooling_schedule.lower()
        self.acceptance_criterion = acceptance_criterion.lower()
        self.num_max_steps = num_max_steps
        assert restart_at_step > 0
        self.restart_at_step = restart_at_step
        self._rnd = None
        self._step_restart_correction = None

    def seed(self, seed: int):
        self._rnd = np.random.default_rng(seed)

    def _get_temp(self, step: np.ndarray) -> np.ndarray:
        """compute current temperature according to cooling schedule."""
        # "http://what-when-how.com/artificial-intelligence/
        # a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/"
        assert np.all(step >= 0)
        if self.cooling_schedule == "lin":
            return self.tau_init / (1 + step)
        elif self.cooling_schedule == "exp_mult":
            return self.tau_init * self.alpha**step
        elif self.cooling_schedule == "log_mult":
            return self.tau_init / (1 + self.alpha * np.log(step + 1))
        elif self.cooling_schedule == "lin_mult":
            return self.tau_init / (1 + self.alpha * step)
        elif self.cooling_schedule == "quad_mult":
            return self.tau_init / (1 + self.alpha * (step ** 2))
        else:
            assert self.num_max_steps is not None and self.tau_final is not None
            if self.cooling_schedule == "lin_add":
                return (self.tau_final + (self.tau_init - self.tau_final) *
                        ((self.num_max_steps - step)/self.num_max_steps))
            elif self.cooling_schedule == "quad_add":
                return (self.tau_final + (self.tau_init - self.tau_final) *
                        ((self.num_max_steps - step)/self.num_max_steps)**2)
            elif self.cooling_schedule == "exp_add":
                return (
                    self.tau_final + (self.tau_init - self.tau_final) *
                    (1 / (1 + np.exp(
                        (2*np.log(self.tau_init - self.tau_final)/self.num_max_steps) *
                        (step - 0.5*self.num_max_steps))))
                )
            else:
                raise ValueError(f"unknown cooling_schedule: '{self.cooling_schedule}'.")

    def _accept(self, tau: np.ndarray, e_old: np.ndarray, e_new: np.ndarray) -> np.ndarray:
        """Decide about acceptance."""
        delta = e_new - e_old
        if self.acceptance_criterion == "metropolis":
            # stochastic accepting rule used in Kirkpatrick et al. 1983
            eps = np.exp(delta / tau)
            return (delta > 0) | (self._rnd.random(delta.shape[0]) < eps)
        elif self.acceptance_criterion == "threshold":
            # simple threshold accepting rule by Dueck & Scheuer 1990
            return (delta > -tau)   # type: ignore
        else:
            raise ValueError(f"unknown acceptance_criterion: '{self.acceptance_criterion}'.")

    def _restart(self, step: np.ndarray):
        """
        <restart_at_step> represents the maximum allowable number of reductions in temperature
        if the value of the objective function has not improved.
        The algorithm restarts if restart_at_step is reached. When the algorithm restarts, the
        current temperature is reset to the initial temperature, and a new initial solution is generated
        randomly to initiate a new SA run. The algorithm is terminated once it reaches the
        maximum number of iterations num_max_steps.
        """
        if self.restart_at_step > 0:
            first_step = (step == 0)
            if np.any(first_step):     # first iter
                if self._step_restart_correction is None:
                    self._step_restart_correction = np.zeros_like(step)
                self._step_restart_correction[first_step] = 0

            steps_no_imp = np.array([w.get_env_attr("_num_steps_no_imp") for w in self._env.venv.workers])
            restart = steps_no_imp >= self.restart_at_step
            self._step_restart_correction[restart] = step[restart]
            step -= self._step_restart_correction
        return step

    @torch.no_grad()
    def forward(self,
                batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:

        if self.use_sa:
            # employ simulated annealing acceptance in ILS
            # get corresponding step, current and previous cost values from env
            pool = self._env.venv.workers
            step = np.array([w.get_env_attr("_current_step") for w in pool])
            e_old = np.array([w.get_env_attr("previous_cost") for w in pool])
            e_new = np.array([w.get_env_attr("current_cost") for w in pool]).astype(np.float)

            # correct for real batch size vs num_envs
            bs = batch['obs'].shape[0]
            # can happen in last batch which might be of bs < num_envs
            if bs != self._env.num_envs:
                msk = (step != 0)
                assert np.all(np.isnan(e_old[~msk].astype(np.float)))
                step = step[msk]
                e_old = e_old[msk]
                e_new = e_new[msk]
                if self.restart_at_step > 0:
                    self._step_restart_correction = self._step_restart_correction[msk]

            step = self._restart(step)
            tau = self._get_temp(step)
            first_step = (step == 0)
            if np.any(first_step):     # first iter
                e_old[first_step] = np.float(1e12)  # high cost value ^= inf
            e_old = e_old.astype(np.float)

            act = self._accept(tau, e_old, e_new)
            # accept = True, but env treats integer (float) value of 0 as accept and 1 as reject
            act = (~act).astype(np.float)[:, None]  # -> (BS, 1)

        else:
            # always accept and just perturb after restart_at_step steps without improvement
            bs = batch['obs'].shape[0]
            act = np.zeros(bs, dtype=np.float)
        return Batch(act=act, state=state)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        # no learning
        return {}
