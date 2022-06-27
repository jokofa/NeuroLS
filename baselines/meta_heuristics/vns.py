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


class VariableNeighborhoodSearch(BasePolicy):
    """
    Policy executing a variable neighborhood search strategy which can be used
    in the tianshou policy framework and is batched over a VectorEnv.
    """
    def __init__(self,
                 env: VecEnv,
                 restart_at_step: int = 4,
                 **kwargs):
        super(VariableNeighborhoodSearch, self).__init__()
        self._env = env
        assert restart_at_step > 0
        self.restart_at_step = restart_at_step
        self._rnd = None
        self._cur_nbh = None

    def seed(self, seed: int):
        self._rnd = np.random.default_rng(seed)

    @torch.no_grad()
    def forward(self,
                batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:

        # get corresponding step, current and previous cost values from env
        pool = self._env.venv.workers
        step = np.array([w.get_env_attr("_current_step") for w in pool])
        steps_no_imp = np.array([w.get_env_attr("_num_steps_no_imp") for w in pool])
        change_nbh = steps_no_imp >= self.restart_at_step

        # correct for real batch size vs num_envs
        bs = batch['obs'].shape[0]
        # can happen in last batch which might be of bs < num_envs
        if bs != self._env.num_envs:
            msk = (step != 0)
            step = step[msk]
            change_nbh = change_nbh[msk]

        first_step = (step == 0)
        if np.any(first_step):     # first iter
            if self._cur_nbh is None:
                self._cur_nbh = np.zeros(bs)
            self._cur_nbh[first_step] = 0

        # if no improvement can be found anymore,
        # use the next neighborhood, i.e. LS move, in action space
        max_act = self._env.action_space.n-1
        msk = (self._cur_nbh == max_act) & change_nbh
        self._cur_nbh[change_nbh] += 1
        self._cur_nbh[msk] = 0

        return Batch(act=self._cur_nbh, state=state)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        # no learning
        return {}
