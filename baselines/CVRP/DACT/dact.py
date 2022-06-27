#
import logging
import time
import itertools as it
from typing import Optional, Dict, Union, List, NamedTuple, Tuple, Any
from omegaconf import DictConfig

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.routing import RPInstance, RPSolution
from baselines.CVRP.DACT.DACT.problems.problem_vrp import CVRP
from baselines.CVRP.DACT.DACT.agent.ppo import PPO

logger = logging.getLogger(__name__)


class CVRPDataset(Dataset):
    def __init__(self,
                 data: List[RPInstance],
                 graph_size: int,
                 dummy_rate: Optional[float] = None
                 ):

        super(CVRPDataset, self).__init__()

        assert graph_size == data[0].graph_size-1

        self.size = int(np.ceil(graph_size * (1 + dummy_rate)))  # the number of real nodes plus dummy nodes in cvrp
        self.real_size = graph_size     # the number of real nodes in cvrp
        self.depot_reps = (self.size - self.real_size)
        self.data = [self.make_instance(d) for d in data]

    def make_instance(self, instance: RPInstance):
        depot = torch.from_numpy(instance.coords[0])
        loc = torch.from_numpy(instance.coords[1:])
        demand = torch.from_numpy(instance.node_features[1:, instance.constraint_idx[0]])

        return {
            'coordinates': torch.cat((depot.view(-1, 2).repeat(self.depot_reps, 1), loc), 0),
            'demand': torch.cat((torch.zeros(self.depot_reps), demand), 0)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def sol_to_list(sol: np.ndarray, depot_idx: int = 0) -> List[List]:
    lst, sol_lst = [], []
    for n in sol:
        if n == depot_idx:
            if len(lst) > 0:
                sol_lst.append(lst)
                lst = []
        else:
            lst.append(n)
    if len(lst) > 0:
        sol_lst.append(lst)
    return sol_lst


#
def train_model():
    raise NotImplementedError
    agent.start_training(problem, opts.val_dataset, tb_logger)


#
def eval_model(data: List[RPInstance],
               problem: CVRP,
               agent: PPO,
               opts: Union[DictConfig, NamedTuple],
               batch_size: int,
               dummy_rate: Optional[float] = None,
               device=torch.device("cpu")
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:

    # eval mode
    if device.type != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #opts = agent.opts
    agent.eval()
    problem.eval()
    #torch.manual_seed(opts.seed)
    #np.random.seed(opts.seed)
    logger.info(f'Inference with {opts.num_augments} augments...')

    val_dataset = CVRPDataset(data=data,
                              graph_size=opts.graph_size,
                              dummy_rate=dummy_rate)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

    sols, times = [], []

    for batch in val_dataloader:

        t_start = time.time()
        # bv, cost_hist, best_hist, r, best_sol_hist
        res = agent.rollout(
            problem, opts.num_augments, batch,
            do_sample=True,
            record=True,
            show_bar=True
        )
        t = time.time() - t_start
        t_per_inst = t / batch_size
        sols.append(res[-1].cpu().numpy())
        times.append([t_per_inst]*batch_size)

    #
    times = list(it.chain.from_iterable(times))
    # parse solutions
    num_dep = problem.dummy_size
    sols = np.concatenate(sols, axis=0)
    s_parsed = []
    for sol_ in sols:
        src = 0
        tour_lst, lst = [], []
        for i in range(len(sol_)):
            tgt = sol_[src]
            if tgt < num_dep:
                if len(lst) > 0:
                    tour_lst.append(lst)
                lst = []
            else:
                lst.append(tgt)
            src = tgt
        s_parsed.append([[e-(num_dep-1) for e in l] for l in tour_lst])

    # bs = sols.shape[0]
    # bidx = np.arange(bs)
    # from_idx = np.zeros(bs, dtype=int)
    # s_parsed = -np.ones_like(sols)
    # for i in range(1, sols.shape[-1]):
    #     to_idx = sols[bidx, from_idx]
    #     s_parsed[:, i] = np.maximum(to_idx-num_dep, -1)
    #     from_idx = to_idx
    # s_parsed += 1

    solutions = [
        RPSolution(
            solution=sol, #sol_to_list(sol.tolist()),
            run_time=t,
            problem=opts.problem,
            instance=inst
        )
        for sol, t, inst in zip(s_parsed, times, data)
    ]

    return {}, solutions
