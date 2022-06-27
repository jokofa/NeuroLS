#
import logging
import time
import itertools as it
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from lib.routing import RPInstance, RPSolution
import baselines.CVRP.POMO.POMO.cvrp.source.MODEL__Actor.grouped_actors as Actor
from baselines.CVRP.POMO.POMO.cvrp.source.utilities import augment_xy_data_by_8_fold

logger = logging.getLogger(__name__)


#
def train_model():
    raise NotImplementedError


def rollout(data: Tuple[Tensor, Tensor, Tensor],
            actor: Actor,
            bs: int,
            graph_size: int,
            augment: bool = False,
            single_trajectory: bool = False,
            device=torch.device("cpu")
            ):

    graph_size -= 1
    depot_xy, node_xy, node_demand = data
    if len(depot_xy.shape) == 2:
        depot_xy = depot_xy.unsqueeze(1)
    if len(node_demand.shape) == 2:
        node_demand = node_demand.unsqueeze(-1)

    t_start = time.time()

    if augment:
        # 8 fold Augmented
        depot_xy = augment_xy_data_by_8_fold(depot_xy)
        # aug_depot_xy.shape = (8*batch, 1, 2)
        node_xy = augment_xy_data_by_8_fold(node_xy)
        # aug_node_xy.shape = (8*batch, problem, 2)
        node_demand = node_demand.repeat(8, 1, 1)
        # aug_node_demand.shape = (8*batch, problem, 2)
        bs *= 8

    group_size = 1 if single_trajectory else graph_size

    with torch.no_grad():

        env = GROUP_ENVIRONMENT(depot_xy.to(device, dtype=torch.float),
                                node_xy.to(device, dtype=torch.float),
                                node_demand.to(device, dtype=torch.float),
                                graph_size,
                                device=device)
        state, reward, done = env.reset(group_size=group_size)
        actor.reset(state)

        # First Move is given
        first_action = torch.from_numpy(np.zeros((bs, group_size))).long().to(device)  # start from node_0-depot
        state, reward, done = env.step(first_action)

        # Second Move is given
        second_action = torch.from_numpy(np.arange(group_size)).long().to(device)[None, :].expand(bs, group_size)
        state, reward, done = env.step(second_action)

        while not done:
            action_probs = actor.get_action_probabilities(state)
            # shape = (batch, group, problem+1)
            action = action_probs.argmax(dim=2)
            # shape = (batch, group)
            action[state.finished] = 0  # stay at depot, if you are finished
            state, reward, done = env.step(action)

    if augment:
        if single_trajectory:
            raise NotImplementedError(f"single_trajectory with augments not implemented.")
        bs_ = bs // 8
        group_reward = reward.reshape(8, bs_, group_size).permute(1, 0, 2).reshape(bs_, -1)
        reward, best_idx = group_reward.max(dim=-1)
        sols = (
            state.selected_node_list.cpu()
            .reshape(8, bs_, group_size, -1)
            .permute(1, 0, 2, 3)
            .reshape(bs_, 8*group_size, -1)
        )[torch.arange(bs_), best_idx]
    else:
        if single_trajectory:
            reward = reward[:, 0]
            sols = state.selected_node_list[:, 0]
        else:
            reward, best_idx = reward.max(dim=-1)
            sols = state.selected_node_list.cpu()[torch.arange(bs), best_idx]

    reward *= -1    # since costs are negative
    reward = reward.cpu().numpy()
    sols = sols.cpu().numpy()
    t_total = time.time() - t_start
    t_per_inst = t_total / bs
    times = [t_per_inst] * bs
    return reward, sols, times


#
def eval_model(data: List[RPInstance],
               actor: Actor,
               batch_size: int,
               device=torch.device("cpu"),
               rollout_cfg: Optional[Dict] = None,
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:

    graph_size = data[0].graph_size

    # prepare data
    dataset = CVRPDataset(data)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    rews, sols, times = [], [], []
    for batch in dataloader:
        r, s, t = rollout(
            batch,
            actor,
            graph_size=graph_size,
            bs=batch_size,
            device=device,
            **rollout_cfg
        )
        rews.append(r)
        sols.append(s)
        times.append(t)

    # collate
    rews = np.concatenate(rews, axis=0)
    times = list(it.chain.from_iterable(times))
    # parse solutions
    s_parsed = []
    for sol_batch in sols:
        for sol in sol_batch:
            sol_lst, lst = [], []
            for e in sol:
                if e == 0:
                    if len(lst) > 0:
                        sol_lst.append(lst)
                    lst = []
                else:
                    lst.append(e)
            s_parsed.append(sol_lst)

    solutions = [
        RPSolution(
            solution=sol,
            run_time=t,
            instance=inst
        )
        for sol, t, inst in zip(s_parsed, times, data)
    ]
    res = {
        "reward_mean": np.mean(rews),
        "reward_std": np.std(rews),
    }

    return res, solutions


class CVRPDataset(Dataset):
    def __init__(self, data: List[RPInstance]):
        super(CVRPDataset, self).__init__()
        self.data = [self.make_instance(d) for d in data]

    def make_instance(self, instance: RPInstance):
        depot = torch.from_numpy(instance.coords[0])
        loc = torch.from_numpy(instance.coords[1:])
        demand = torch.from_numpy(instance.node_features[1:, instance.constraint_idx[0]])
        return (depot, loc, demand)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# state and env from POMO adapted to work without imported problem parameters
class GROUP_STATE:
    def __init__(self, group_size, data, PROBLEM_SIZE, device):
        # data.shape = (batch, problem+1, 3)

        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data
        self.PROBLEM_SIZE = PROBLEM_SIZE

        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch, group)
        self.selected_node_list = torch.from_numpy(np.zeros((self.batch_s, self.group_s, 0))).long().to(device)
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.at_the_depot = None
        # shape = (batch, group)
        self.loaded = torch.from_numpy(np.ones((self.batch_s, self.group_s))).to(device, dtype=torch.float)
        # shape = (batch, group)
        self.visited_ninf_flag = torch.from_numpy(np.zeros((self.batch_s, self.group_s, PROBLEM_SIZE+1))).to(device, dtype=torch.float)
        # shape = (batch, group, problem+1)
        self.ninf_mask = torch.from_numpy(np.zeros((self.batch_s, self.group_s, PROBLEM_SIZE+1))).to(device, dtype=torch.float)
        # shape = (batch, group, problem+1)
        self.finished = torch.from_numpy(np.zeros((self.batch_s, self.group_s))).bool().to(device)
        # shape = (batch, group)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        # Status
        ####################################
        self.at_the_depot = (selected_idx_mat == 0)
        demand_list = self.data[:, None, :, 2].expand(self.batch_s, self.group_s, -1)
        # shape = (batch, group, problem+1)
        gathering_index = selected_idx_mat[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape = (batch, group)
        self.loaded -= selected_demand
        self.loaded[self.at_the_depot] = 1 # refill loaded at the depot
        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(self.batch_s, self.group_s)
        self.visited_ninf_flag[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf
        self.finished = self.finished + (self.visited_ninf_flag == -np.inf).all(dim=2)
        # shape = (batch, group)

        # Status Edit
        ####################################
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # allow car to visit depot anytime
        round_error_epsilon = 0.000001
        demand_too_large = self.loaded[:, :, None] + round_error_epsilon < demand_list
        # shape = (batch, group, problem+1)
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.ninf_mask[demand_too_large] = -np.inf

        self.ninf_mask[self.finished[:, :, None].expand(self.batch_s, self.group_s, self.PROBLEM_SIZE+1)] = 0
        # do not mask finished episode


class GROUP_ENVIRONMENT:

    def __init__(self, depot_xy, node_xy, node_demand, PROBLEM_SIZE, device):
        # depot_xy.shape = (batch, 1, 2)
        # node_xy.shape = (batch, problem, 2)
        # node_demand.shape = (batch, problem, 1)

        self.batch_s = depot_xy.size(0)
        self.group_s = None
        self.group_state = None
        self.PROBLEM_SIZE=PROBLEM_SIZE
        self.device=device

        all_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape = (batch, problem+1, 2)
        depot_demand = torch.from_numpy(np.zeros((self.batch_s, 1, 1))).to(device, dtype=torch.float)
        all_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape = (batch, problem+1, 1)
        self.data = torch.cat((all_node_xy, all_node_demand), dim=2)
        # shape = (batch, problem+1, 3)

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = GROUP_STATE(group_size=group_size,
                                       data=self.data,
                                       PROBLEM_SIZE=self.PROBLEM_SIZE,
                                       device=self.device)

        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.finished.all()  # state.finished.shape = (batch, group)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_travel_distance(self):
        all_node_xy = self.data[:, None, :, 0:2].expand(self.batch_s, self.group_s, -1, 2)
        # shape = (batch, group, problem+1, 2)
        gathering_index = self.group_state.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape = (batch, group, selected_count, 2)
        ordered_seq = all_node_xy.gather(dim=2, index=gathering_index)
        # shape = (batch, group, selected_count, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # size = (batch, group, selected_count)

        travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return travel_distances
