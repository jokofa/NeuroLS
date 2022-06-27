#
import os
import logging
from warnings import warn
from typing import Dict, Optional, Tuple, Callable, Union, Any, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

import time
import tqdm
from collections import defaultdict

from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info
from tianshou.utils import tqdm_config, MovAvg
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv
from tianshou.data import (
    Collector,
    Batch,
    ReplayBuffer,
    to_numpy,
)

from lib.routing import RPInstance, RPSolution, RP_TYPES
from lib.scheduling import JSSPInstance, JSSPSolution, JSSP_TYPES
from lib.env.utils import parse_solutions

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """
    Callback to manage checkpoints.
    Is called by MonitorCallback.

    Args:
        exp: experiment object
        save_dir: directory for saved checkpoints
        fname: file name prefix (default='model')
        metric_key: key of metric (default='rew')
        compare_mode: mode of eval metric comparison, (rew -> 'max', cost -> 'min')
        top_k: number of checkpoints to keep
    """
    FILE_EXTS = ".ckpt"

    def __init__(self,
                 exp,
                 save_dir: str,
                 fname: str = "model",
                 metric_key: str = "rew",
                 compare_mode: str = "max",
                 top_k: int = 1,
                 save_last: bool = True,
                 **kwargs):
        self.exp = exp
        self.save_dir = save_dir
        self.prefix = fname
        self.metric_key = metric_key
        self.compare_mode = compare_mode
        self.top_k = top_k
        self.save_last = save_last

        os.makedirs(self.save_dir, exist_ok=True)
        v = -float("inf") if self.compare_mode == "max" else float("inf")
        self.top_k_checkpoints = [{'eval_metric': v, 'pth': None} for _ in range(top_k)]

    def __call__(self, epoch: int, eval_metric: Union[float, Any], is_last: bool = False):
        """Called to compare existing checkpoints and save top-k best models."""
        if is_last:
            if self.save_last:
                # save final checkpoint no matter how good
                m = f"{eval_metric: .6f}".lstrip()
                fname = f"ep{epoch}(last)_{self.prefix}_{self.metric_key}={m}{self.FILE_EXTS}"
                add_pth = os.path.join(self.save_dir, fname)
                # save
                logger.info(f"Saving last checkpoint to: {add_pth}")
                torch.save(self.exp.state_dict(), add_pth)

        else:
            # check metric
            if eval_metric is None:
                warn(f"Eval metric is None. No checkpoint saved.")
            else:
                is_better, idx = self._compare_metric(eval_metric)
                if is_better:
                    # delete worst checkpoint
                    del_pth = self.top_k_checkpoints.pop(-1)['pth']
                    if del_pth is not None and os.path.exists(del_pth):
                        os.remove(del_pth)
                    # add new checkpoint
                    m = f"{eval_metric: .6f}".lstrip()
                    fname = f"ep{epoch}_{self.prefix}_{str(self.metric_key)}={m}{self.FILE_EXTS}"
                    add_pth = os.path.join(self.save_dir, fname)
                    self.top_k_checkpoints.insert(idx, {'eval_metric': eval_metric, 'pth': add_pth})
                    # save
                    logger.info(f"Saving new checkpoint to: {add_pth}")
                    torch.save(self.exp.state_dict(), add_pth)

    def _compare_metric(self, eval_metric: float) -> Tuple[bool, int]:
        cur_best = np.array([cp['eval_metric'] for cp in self.top_k_checkpoints])
        if self.compare_mode == "max":
            check = cur_best < eval_metric
        elif self.compare_mode == "min":
            check = cur_best > eval_metric
        else:
            raise ValueError(f"unknown compare mode: '{self.compare_mode}'")
        is_better = np.any(check)
        # since list is ordered by metric, first nonzero position is target position for insertion
        idx = np.nonzero(check)[0][0] if is_better else None
        return is_better, idx


class MonitorCallback(TensorboardLogger):
    """
    Extends the TB logger to monitor eval metrics
    for calling the checkpoint callback.

    Args:
        tb_log_path: directory to save tensorboard events file
        metric_key: key of eval metric to compare
        train_interval: log training results every train_interval steps
        test_interval: log test results every test_interval epochs
        update_interval: log update results every update_interval steps
        save_interval: save model and results every save_interval epochs
    """
    def __init__(self,
                 tb_log_path: str,
                 metric_key: str = "rew",
                 train_interval: int = 1000,
                 test_interval: int = 1,
                 update_interval: int = 1000,
                 save_interval: int = 1,
                 ):
        # create TB summary writer
        writer = SummaryWriter(tb_log_path)

        super(MonitorCallback, self).__init__(
            writer=writer,
            train_interval=train_interval,
            test_interval=test_interval,
            update_interval=update_interval,
            save_interval=save_interval,
        )
        self.tb_log_path = tb_log_path
        self.metric_key = metric_key
        self.eval_metric = None
        assert os.path.exists(self.tb_log_path)

    def log_eval_data(self, collect_result: dict, step: int, mode: str = "test") -> None:
        assert collect_result["n/ep"] > 0
        rews, lens = collect_result["rews"], collect_result["lens"]
        rew, rew_std, len_, len_std = rews.mean(), rews.std(), lens.mean(), lens.std()
        final_cost = collect_result.get("final_costs", np.array([float("inf")])).mean()
        collect_result.update(rew=rew, rew_std=rew_std, len=len_, len_std=len_std, final_cost=final_cost)
        if step - self.last_log_test_step >= self.test_interval:
            log_data = {
                f"{mode}/env_step": step,
                f"{mode}/reward": collect_result["rew"],
                f"{mode}/length": collect_result["len"],
                f"{mode}/reward_std": collect_result["rew_std"],
                f"{mode}/length_std": collect_result["len_std"],
                f"{mode}/final_cost": collect_result["final_cost"],
            }
            self.write(f"{mode}/env_step", step, log_data)
            self.last_log_test_step = step

        self.eval_metric = collect_result.get(self.metric_key, None)

    def save_data(
            self,
            epoch: int,
            env_step: int,
            gradient_step: int,
            save_checkpoint_fn: Optional[Callable] = None,
            is_last: bool = False,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval or is_last:
            self.last_save_step = epoch
            # call CheckpointCallback
            save_checkpoint_fn(epoch, self.eval_metric, is_last)
            # write additional logs
            self.write("save/epoch", epoch, {"save/epoch": epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write(
                "save/gradient_step", gradient_step,
                {"save/gradient_step": gradient_step}
            )

    def save_results(self, result: Dict, naming: str = "results"):
        pth = os.path.join(self.tb_log_path, f"{naming}.pkl")
        torch.save(result, pth)


def add_append_to_dict(d, k, v):
    if isinstance(d[k], int) or isinstance(d[k], float):
        d[k] += v
    elif isinstance(d[k], np.ndarray):
        d[k] = np.append(d[k], v)
    else:
        raise TypeError(f"Dictionary values need to be of type int, float or numpy.ndarray.")
    return d


def parse_instances(instances: np.ndarray,
                    problem: str
                    ) -> List[Union[RPInstance, JSSPInstance]]:
    _w = RPInstance if problem.upper() in RP_TYPES else JSSPInstance
    return [_w(*inst.tolist()) for inst in instances]


class tqdm_no_stdout(tqdm.tqdm):
    def display(self, msg=None, pos=None):
        """Override to disable display in log."""
        return True


def test_episode(
    problem: str,
    policy: BasePolicy,
    collector: Collector,
    test_fn: Optional[Callable[[int, Optional[int]], None]],
    epoch: int,
    n_episode: int,
    tb_logger: Optional[MonitorCallback] = None,
    global_step: Optional[int] = None,
    render: float = 0,
    num_render_eps: int = 1,
) -> Tuple[Dict[str, Any], List[RPSolution]]:
    """A simple wrapper of testing policy in collector.

    >>
    Overwrites tianshou test_episode fn
    (https://tianshou.readthedocs.io/en/master/_modules/tianshou/trainer/utils.html#test_episode)
    to allow rendering
    <<

    """
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    add_result = None
    total_size = n_episode
    # execute rendering
    if render > 0:
        warn(f"Beware that rendering incurs a significant overhead and "
             f"therefore should not be used when evaluating for run time.")
        add_result = collector.collect(n_episode=num_render_eps, render=render)
        n_episode = max(n_episode-num_render_eps, 0)
    # execute remaining number of eval episodes
    result = collector.collect(n_episode=n_episode, render=0)
    # add result of rendered eps
    if add_result is not None:
        for k, v in add_result.items():
            result = add_append_to_dict(result, k, v)

    # retrieve solutions and additional info of all episodes
    # (the eval set buffer has exactly total_size = dataset_size * num_steps)
    ids = np.arange(collector.buffer.maxsize)
    dones = collector.buffer.get(ids, 'done')
    if sum(dones) < total_size:
        warn(f"specified {total_size} episodes but only finished {sum(dones)}")
    info_buf = collector.buffer.get(ids, 'info')
    costs = info_buf.get('best_cost_final')[dones]
    result['final_costs'] = costs
    sols = parse_solutions(info_buf.get('solution')[dones])  # type: ignore
    run_times = info_buf.get('time_elapsed')[dones]

    p = problem.upper()
    instances = parse_instances(info_buf.get('instance')[dones], problem=p)
    if p in RP_TYPES:
        solutions = [
            RPSolution(solution=s, run_time=r, instance=i)
            for s, r, i in zip(sols, run_times, instances)
        ]
    elif p in JSSP_TYPES:
        solutions = [
            JSSPSolution(solution=s, cost=c, run_time=r, instance=i)
            for s, c, r, i in zip(sols, costs, run_times, instances)
        ]
    else:
        raise ValueError()

    # calculate ls_op frequencies
    ls_ops = info_buf.get('ls_op_cnt')[dones]
    cnt_dict = {k: np.sum(ls_ops[k]) for k in ls_ops.keys()}
    n = sum([v for k, v in cnt_dict.items() if k is not None])
    result['ls_op_freqs'] = {k: (v/n) for k, v in cnt_dict.items() if k is not None}     # type: ignore

    if tb_logger and global_step is not None:
        tb_logger.log_eval_data(result, global_step, mode="val")
    return result, solutions


def offpolicy_trainer(
    problem: str,
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    update_per_step: Union[int, float] = 1,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    monitor: MonitorCallback = None,
    verbose: int = 2,
    render_val: bool = False,
):
    """A wrapper for off-policy trainer procedure.

    >>
    Overwrites the tianshou off-policy trainer
    (https://github.com/thu-ml/tianshou/blob/master/tianshou/trainer/offpolicy.py)
    to allow advanced custom monitoring, logging and checkpointing.
    <<

    The "step" in trainer means an environment step (a.k.a. transition).

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatly in each epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int/float update_per_step: the number of times the policy network would be
        updated per transition after (step_per_collect) transitions are collected,
        e.g., if update_per_step set to 0.3, and step_per_collect is 256, policy will
        be updated round(256 * 0.3 = 76.8) = 77 times after 256 transitions are
        collected by the collector. Default to 1.
    :param function train_fn: a hook called at the beginning of training in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param MonitorCallback monitor: Custom MonitorCallback
    :param bool verbose: whether to print the information. Default to True.
    :param bool render_val: flag to render episodes on validation set

    :return: See :func:`~tianshou.trainer.gather_info`.
    """

    start_epoch, env_step, gradient_step = 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, gradient_step = monitor.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_result, solutions = test_episode(problem, policy, test_collector, test_fn, start_epoch,
                                          episode_per_test, monitor, env_step,
                                          render=0.0001 if render_val else 0)
    best_epoch = start_epoch
    best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]

    epoch = None
    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        if verbose > 0:
            t = tqdm.tqdm(total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config)
        else:
            logger.info(f"Epoch #{epoch}")
            t = tqdm_no_stdout(total=step_per_epoch, **tqdm_config)
        with t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(n_step=step_per_collect)
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                monitor.log_train_data(result, env_step)
                last_rew = result['rew'] if 'rew' in result else last_rew
                last_len = result['len'] if 'len' in result else last_len
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": str(int(last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                # update
                for i in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    losses = policy.update(batch_size, train_collector.buffer)
                    for k in losses.keys():
                        stat[k].add(losses[k])
                        losses[k] = stat[k].get()
                        data[k] = f"{losses[k]:.3f}"
                    monitor.log_update_data(losses, gradient_step)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # val
        test_result, solutions = test_episode(problem, policy, test_collector, test_fn, epoch,
                                              episode_per_test, monitor, env_step,
                                              render=0.0001 if render_val else 0)
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if best_epoch < 0 or best_reward < rew:
            best_epoch, best_reward, best_reward_std = epoch, rew, rew_std
        monitor.log_eval_data(test_result, env_step, mode="val")
        monitor.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)
        if verbose > 1:
            print(f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                  f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}")

    monitor.save_data(epoch, env_step, gradient_step, save_checkpoint_fn, is_last=True)
    result = gather_info(start_time, train_collector, test_collector, best_reward, best_reward_std)
    result['ls_op_freqs'] = test_result['ls_op_freqs']
    return result, solutions


def tester(
    problem: str,
    policy: BasePolicy,
    test_collector: Collector,
    episode_per_test: int,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    monitor: MonitorCallback = None,
    render: float = 0,
    num_render_eps: int = 1,
):
    """

    Args:
        policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
        test_collector: the collector used for testing.
        episode_per_test: the number of testing episodes (= size of test set).
        test_fn: a hook called at the beginning of testing.
        monitor: Custom MonitorCallback
        render: number of sleep seconds between render steps (deactivates rendering for render=0)
        num_render_eps: number of episodes to render

    Returns:
        See :func:`~tianshou.trainer.gather_info`.

    """
    start_time = time.time()
    test_collector.reset_stat()
    test_result, solutions = test_episode(problem, policy, test_collector, test_fn,
                                          1, episode_per_test, monitor, 1,
                                          render=render, num_render_eps=num_render_eps)
    monitor.log_eval_data(test_result, 0, mode="test")
    result = gather_info(start_time, None, test_collector, test_result["rew"], test_result["rew_std"])
    result['ls_op_freqs'] = test_result['ls_op_freqs']
    return result, solutions


class TestCollector(Collector):
    """..."""

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseVectorEnv,
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

    def reset_env(self) -> None:
        super().reset_env()
        self._ready_env_ids = np.arange(self.env_num)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode with async env setting.

        This function doesn't collect exactly n_step or n_episode number of
        transitions. Instead, in order to support async setting, it may collect more
        than given n_step or n_episode transitions and save into buffer.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
        """
        # collect at least n_step or n_episode
        if n_step is not None:
            assert n_episode is None, (
                "Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
        elif n_episode is not None:
            assert n_episode > 0
        else:
            raise TypeError("Please specify at least one (either n_step or n_episode) "
                            "in AsyncCollector.collect().")

        ready_env_ids = self._ready_env_ids
        start_time = time.time()
        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        while True:
            whole_data = self.data
            self.data = self.data[ready_env_ids]
            assert len(whole_data) == self.env_num  # major difference
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                self.data.update(
                    act=[self._action_space[i].sample() for i in ready_env_ids])
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            obs_next, rew, done, info = self.env.step(
                action_remap, ready_env_ids)  # type: ignore

            # change self.data here because ready_env_ids has changed
            ready_env_ids = np.array([i["env_id"] for i in info])
            self.data = whole_data[ready_env_ids]
            self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            if step_count == 0:
                self.buffer.add(self.data, buffer_ids=ready_env_ids)

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)

                # add only last step output to buffer
                ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                    self.data[env_ind_global],
                    buffer_ids=env_ind_global
                )
                episode_lens.append(ep_len)
                episode_rews.append(ep_rew)
                episode_start_indices.append(ep_idx)

                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                obs_reset = self.env.reset(env_ind_global)
                if self.preprocess_fn:
                    obs_reset = self.preprocess_fn(obs=obs_reset).get("obs", obs_reset)
                self.data.obs_next[env_ind_local] = obs_reset
                for i in env_ind_local:
                    self._reset_state(i)

            try:
                whole_data.obs[ready_env_ids] = self.data.obs_next
                whole_data.rew[ready_env_ids] = self.data.rew
                whole_data.done[ready_env_ids] = self.data.done
                whole_data.info[ready_env_ids] = self.data.info
            except ValueError:
                _alloc_by_keys_diff(whole_data, self.data, self.env_num, False)
                self.data.obs = self.data.obs_next
                whole_data[ready_env_ids] = self.data  # lots of overhead
            self.data = whole_data

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        self._ready_env_ids = ready_env_ids

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if episode_count > 0:
            rews, lens, idxs = list(map(
                np.concatenate, [episode_rews, episode_lens, episode_start_indices]))
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
        }
