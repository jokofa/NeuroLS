#
import os
import logging
from warnings import warn
from typing import Optional, Dict, Union, List
from torch_geometric.typing import Adj
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ImageMagickWriter

import torch
from torch import Tensor
import torch.nn as nn
from torch_cluster import knn_graph

from torch.utils.data import Dataset
from tianshou.env import SubprocVectorEnv, ShmemVectorEnv
from lib.routing import RPGenerator
from lib.scheduling import JSSPGraph, JSSPGenerator

# disable matplotlib logger for non-essential cases
logging.getLogger('matplotlib').setLevel(logging.WARNING)
if "PYCHARM_HOSTED" in os.environ:
    matplotlib.use("TKAgg")     # for use with GUI/IDE


def time_out_handler(signum, frame):
    raise TimeoutError(f"time out on function call")


def inverse_key_lookup(v, d):
    """Get key by value (assumes dict values are unique!)."""
    return list(d.keys())[list(d.values()).index(v)]


def parse_solutions(solutions: np.ndarray) -> List[List]:
    """Convert array of padded tour sequences
    to nested list of tour indices."""
    if len(solutions.shape) == 2:
        return [
            [n for n in tour if n != 0]
            for tour in solutions if sum(tour) > 0
        ]
    elif len(solutions.shape) == 3:
        return [
            [
                [n for n in tour if n != 0]
                for tour in sol if sum(tour) > 0
            ] for sol in solutions
        ]
    else:
        raise RuntimeError


def negative_nbh_sampling(edge_index: Adj,
                          max_k: int,
                          num_neg_samples: int,
                          loop: bool = False) -> Adj:
    """Takes a sparse neighborhood adjacency matrix and
    adds <num_neg_samples> random edges for each node."""
    _, n, k = edge_index.size()
    # possible range of indices
    idx_range = torch.arange(max_k, device=edge_index.device)
    # get indices not yet in edge_index
    mask = ~(
        edge_index[0][:, :, None].expand(n, k, max_k)
        ==
        idx_range[None, None, :].expand(n, k, max_k)
    ).any(dim=1)
    # mask same node indices (self loops)
    if not loop:
        mask &= (edge_index[1, :, 0][:, None].expand(-1, max_k) != idx_range[None, :].expand(n, max_k))
    # get candidate indices
    candidates = idx_range[None, :].expand(n, -1)[mask].view(n, -1)
    # sample idx and create edge
    i = int(not loop)  # i = 1 when not considering self loops!
    return torch.cat(
        (candidates[:, torch.randperm(max_k-k-i)[:num_neg_samples]].unsqueeze(0),
         edge_index[1, :, 0][:, None].expand(-1, num_neg_samples).unsqueeze(0)),
        dim=0
    )


class GraphNeighborhoodSampler(nn.Module):
    def __init__(self,
                 graph_size: int,
                 k_frac: Union[int, float] = 0.3,
                 rnd_edge_ratio: float = 0.0,
                 num_workers: int = 4,
                 **kwargs):
        """Samples <k_frac> nearest neighbors +
        <rnd_edge_ratio> random nodes as initial graph.

        Args:
            graph_size: size of considered graph
            k_frac: number of neighbors considered
            rnd_edge_ratio: ratio of random edges among neighbors
                            to have connections beyond local neighborhood
            num_workers: number of workers
            **kwargs:
        """
        super(GraphNeighborhoodSampler, self).__init__()
        self.graph_size = graph_size
        self.k_frac = k_frac
        self.rnd_edge_ratio = rnd_edge_ratio
        self.num_workers = num_workers
        self.k, self.max_k, self.k_nn, self.num_rnd = None, None, None, None
        self._infer_k(graph_size)

    def _infer_k(self, n: int):
        self.max_k = n
        if isinstance(self.k_frac, float):
            assert 0.0 < self.k_frac < 1.0
            self.k = int(math.floor(self.k_frac*self.max_k))
        elif isinstance(self.k_frac, int):
            self.k = int(min(self.k_frac, self.max_k))
        else:
            raise ValueError
        # infer how many neighbors are nodes sampled randomly from graph
        assert 0.0 <= self.rnd_edge_ratio <= 1.0
        self.num_rnd = int(math.floor(self.k * self.rnd_edge_ratio))
        self.k_nn = self.k - self.num_rnd

    @torch.no_grad()
    def forward(self, coords: Tensor, loop: bool = True):
        n, d = coords.size()
        if n != self.graph_size:
            self._infer_k(n)
        # get k nearest neighbors
        coords = coords.view(-1, d)
        edge_idx = knn_graph(coords,
                             k=self.k_nn,
                             loop=loop,     # include self-loops flag
                             num_workers=self.num_workers)
        # sample additional edges to random nodes if specified
        if self.num_rnd > 0:
            edge_idx = edge_idx.view(2, -1, self.k_nn)
            rnd_edge_idx = negative_nbh_sampling(edge_index=edge_idx,
                                                 max_k=self.max_k,
                                                 num_neg_samples=self.num_rnd,
                                                 loop=False)
            edge_idx = torch.cat((edge_idx, rnd_edge_idx), dim=-1).view(2, -1)

        # calculate euclidean distances between neighbors as weights
        idx_coords = coords[edge_idx]
        edge_weights = torch.norm(idx_coords[0] - idx_coords[1], p=2, dim=-1)
        return edge_idx, edge_weights, self.k


class Viewer:
    """Renders routing environment by plotting changes of routing edges for each step."""
    def __init__(self,
                 locs: np.ndarray,
                 save_dir: Optional[str] = None,
                 as_gif: bool = True,
                 gif_naming: Optional[str] = None,
                 **kwargs):
        self.locs = locs
        self.save_dir = os.path.join(save_dir, "gifs") if save_dir is not None else None
        self.as_gif = as_gif
        if self.as_gif:
            matplotlib.use("Agg")   # for saving stream to file

        self.edges = None
        self.writer = None
        self.cmap = plt.get_cmap("tab20")

        plt.ion()
        # scale arrow sizes by plot scale, indicated by max distance from center
        max_dist_from_zero = np.max(np.abs(locs))
        self.hw = max_dist_from_zero * 0.02
        self.hl = self.hw * 1.25

        # create figure objects
        self.fig, self.ax = plt.subplots()

        self.plot_locs(self.locs)

        if save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        if not self.as_gif:
            plt.show(block=False)
        else:
            assert save_dir is not None, f"Must specify save_dir to create gif."
            metadata = dict(title='routing_env_render', artist='Matplotlib', comment='matplotlib2gif')
            self.writer = ImageMagickWriter(fps=2, metadata=metadata)
            if gif_naming is None:
                gif_naming = f"render.gif"
            if gif_naming[-4:] != ".gif":
                gif_naming += ".gif"
            outfile = os.path.join(self.save_dir, gif_naming)
            self.writer.setup(fig=self.fig, outfile=outfile)

    def plot_locs(self, locs: np.ndarray, add_idx: bool = True):
        # scatter plot of locations
        self.ax.scatter(locs[:, 0], locs[:, 1], c='k')
        self.ax.scatter(locs[0, 0], locs[0, 1], c='r', s=7 ** 2, marker='s')  # depot/start node
        if add_idx:
            # add node indices
            for i in range(locs.shape[0]):
                self.ax.annotate(i, (locs[i, 0], locs[i, 1]),
                                 xytext=(locs[i, 0]+0.012, locs[i, 1]+0.012),
                                 fontsize='medium', fontweight='roman')

    def update(self,
               buffer: Dict,
               cost: float,
               n_iters: Optional[int] = None,
               pause_sec: float = 0.5,
               new_locs: Optional[np.ndarray] = None,
               **kwargs):
        """Update current dynamic figure.

        Args:
            buffer: dictionary of data to plot
            cost: cost of current solution
            n_iters: current iteration
            pause_sec: float specifying seconds to wait before updating figure
            new_locs: optional new locations

        """
        # previous_action = buffer.get('previous_action')
        # old_edges = buffer.get('old_edges')
        # if previous_action is not None and old_edges is not None:
        #     previous_action = np.asarray(previous_action).astype(dtype=np.int)
        #     locs = self.locs[previous_action]
        #     for x, y in zip(locs[:, 0],  locs[:, 1]):
        #         self.ax.add_patch(
        #             CirclePolygon(xy=(x, y), radius=0.02, color='c', fill=True)
        #         )
        #     self._draw_edges(edges=old_edges, color="b")
        #     self._flush(pause_sec, **kwargs)

        if new_locs is not None:
            self.plot_locs(new_locs)

        self.ax.patches = []    # remove all previous patches
        edges = buffer['edges']

        if isinstance(edges, np.ndarray):   # TSP
            edges = [edges]
        if len(edges) > self.cmap.N:
            self.cmap = plt.get_cmap('jet', len(edges))
        for i, r in enumerate(edges):
            assert len(r.shape) == 2 and r.shape[0] == 2
            self._draw_edges(edges=r, color=self.cmap(i))

        iter_str = f"Iter: {n_iters}, " if n_iters is not None else ''
        self.ax.set_title(f"{iter_str}cost: {cost:.4f}")
        self.ax.set_aspect('equal', adjustable='box')
        self._flush(pause_sec, **kwargs)

    def _flush(self, pause_sec: float = 0.1, **kwargs):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.as_gif and self.writer is not None:
            self.writer.grab_frame(**kwargs)
        else:
            plt.pause(pause_sec)

    def _draw_edges(self, edges: np.ndarray, color: str = "b", **kwargs):
        coords = self.locs[edges]
        X = coords[0, :, 0]
        Y = coords[0, :, 1]
        dX = coords[1, :, 0] - X
        dY = coords[1, :, 1] - Y
        for x, y, dx, dy in zip(X, Y, dX, dY):
            self.ax.arrow(x, y, dx, dy,
                          color=color,
                          linestyle='-',
                          head_width=self.hw,
                          head_length=self.hl,
                          length_includes_head=True,
                          **kwargs)

    def render_rgb(self) -> np.ndarray:
        """Returns the current figure as RGB value numpy array."""
        return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)\
            .reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    def save(self, path: Optional[str] = None):
        """Save the current figure on specified path. If path is None, uses default save_dir."""
        outfile = path if path is not None else os.path.join(self.save_dir, "final.gif")
        self.writer.finish()
        self.writer.saving(fig=self.fig, outfile=outfile, dpi=120)

    def close(self):
        """Finish and clean up figure and writer processing."""
        plt.clf()
        plt.close('all')
        plt.ioff()


class Viewer2:
    """Renders JSSP environment by plotting DAG."""
    def __init__(self,
                 save_dir: Optional[str] = None,
                 as_gif: bool = True,
                 gif_naming: Optional[str] = None,
                 **kwargs):
        self.save_dir = os.path.join(save_dir, "gifs") if save_dir is not None else None
        self.as_gif = as_gif
        if self.as_gif:
            matplotlib.use("Agg")   # for saving stream to file
        self.writer = None

        plt.ion()
        # create figure objects
        self.fig, self.ax = plt.subplots()

        if save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        if not self.as_gif:
            plt.show(block=False)
        else:
            assert save_dir is not None, f"Must specify save_dir to create gif."
            metadata = dict(title='routing_env_render', artist='Matplotlib', comment='matplotlib2gif')
            self.writer = ImageMagickWriter(fps=2, metadata=metadata)
            if gif_naming is None:
                gif_naming = f"render.gif"
            if gif_naming[-4:] != ".gif":
                gif_naming += ".gif"
            outfile = os.path.join(self.save_dir, gif_naming)
            self.writer.setup(fig=self.fig, outfile=outfile)

    def update(self,
               graph: JSSPGraph,
               cost: float,
               n_iters: Optional[int] = None,
               pause_sec: float = 0.5,
               **kwargs):
        """Update current dynamic figure.

        Args:
            cost: cost of current solution
            n_iters: current iteration
            pause_sec: float specifying seconds to wait before updating figure

        """
        self.ax.clear()
        graph.plot(ax=self.ax)
        iter_str = f"Iter: {n_iters}, " if n_iters is not None else ''
        self.ax.set_title(f"{iter_str}cost: {cost:.4f}")
        self.ax.set_aspect('equal', adjustable='box')
        self._flush(pause_sec, **kwargs)

    def _flush(self, pause_sec: float = 0.1, **kwargs):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.as_gif and self.writer is not None:
            self.writer.grab_frame(**kwargs)
        else:
            plt.pause(pause_sec)

    def render_rgb(self) -> np.ndarray:
        """Returns the current figure as RGB value numpy array."""
        return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)\
            .reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    def save(self, path: Optional[str] = None):
        """Save the current figure on specified path. If path is None, uses default save_dir."""
        outfile = path if path is not None else os.path.join(self.save_dir, "final.gif")
        self.writer.finish()
        self.writer.saving(fig=self.fig, outfile=outfile, dpi=120)

    def close(self):
        """Finish and clean up figure and writer processing."""
        plt.clf()
        plt.close('all')
        plt.ioff()


class DatasetChunk(Dataset):
    """Chunk of distributed dataset."""
    def __init__(self,
                 data: Union[np.ndarray, List],
                 **kwargs):
        super(DatasetChunk, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class DistributedDataset:
    """Dataset wrapper to split and distribute a fixed dataset."""
    def __init__(self, problem: str, venv: Union[SubprocVectorEnv, ShmemVectorEnv]):
        super(DistributedDataset, self).__init__()
        self.problem = problem
        self.venv = venv
        self.ids = self.venv._wrap_id(None)
        self.data = None

    def load(self, fpath: str, limit: Optional[int] = None, **kwargs):
        """Load dataset from specified file path"""
        if self.problem.upper() == "JSSP":
            data = JSSPGenerator.load_dataset(fpath, limit=limit, **kwargs)
        else:
            data = RPGenerator.load_dataset(fpath, limit=limit, **kwargs)
        data = data.tolist() if isinstance(data, np.ndarray) else data
        assert isinstance(data, list)
        self.data = data

    def distribute(self):
        """Distribute the loaded dataset over the venv."""
        assert self.data is not None, f"need to load some dataset first."
        assert not self.venv.is_closed
        async_env = self.venv.is_async
        num_env = len(self.venv)
        num_inst = len(self.data)
        chunk_size = num_inst // num_env
        if async_env:
            if num_inst % num_env != 0:
                chunk_size += 1
            data = self.data
        else:
            if num_inst % num_env != 0:
                warn(f"For non asynchronous environments the dataset chunks need to be of exactly same size. "
                     f"Since len(data) % num_env != 0, the last {num_inst % num_env} instances will be dropped.")
            limit = num_env * chunk_size
            data = self.data[:limit]

        # chunk data
        chunks = [DatasetChunk(data[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]

        # distribute to workers
        if async_env:
            raise NotImplementedError
        else:
            assert num_env == len(self.ids)
            for i, j in enumerate(self.ids):
                self.venv.workers[j].parent_remote.send(["seed", chunks[i]])
            checks = [self.venv.workers[j].parent_remote.recv() for j in self.ids]
        if not np.all(checks):
            raise RuntimeError(f"Encountered error while distributing dataset.")


#
# ============= #
# ### TEST #### #
# ============= #
def _test(
    size: int = 1,
    n: int = 20,
    seed: int = 1,
    n_steps: int = 20,
    mode_args: dict = None,
    render: bool = True,
    **kwargs
):
    from lib.env import TSPEnv
    np.random.seed(seed)
    torch.manual_seed(seed)

    mode = 'ACCEPT_LS'
    sample_args = {'sample_size': size, 'graph_size': n}
    mode_args = {'ls_op': #'ONE_POINT_MOVE',
                 'TWO_OPT',
                 'acceptance_rule': 'BEST_ACCEPT',
                 'random_node_shuffle': True,
                 'num_iters': 1,
                 'num_nodes_per_iter': 1,
                 } if mode_args is None else mode_args
    env = TSPEnv(
        num_steps=n_steps,
        construction_args={'method': 'nn'},
        sampling_args=sample_args,
        mode=mode,
        mode_args=mode_args,
        debug=True,
        enable_render=True,
        **kwargs
    )

    # explicit viewer
    # ===============================================
    env.seed(seed)
    # reset and step
    obs_old = env.reset()

    viewer = Viewer(locs=env.instance.coords.copy(), as_gif=False)
    buffer = {}
    buffer['edges'] = env.current_sol
    viewer.update(buffer, cost=env.current_cost, n_iters=env._current_step)

    rewards = 0
    i = 0
    while i < n_steps:
        #a = env.action_space.sample()
        a=1
        obs, rew, done, info = env.step(a)

        buffer['edges'] = env.current_sol
        viewer.update(buffer, cost=env.current_cost, n_iters=env._current_step)

        print(f"reward: {rew}")
        rewards += rew
        for o1, o2 in zip(obs_old.values(), obs.values()):
            assert isinstance(o1, np.ndarray) and isinstance(o2, np.ndarray)
            assert np.all(np.array(o1.shape) == np.array(o2.shape))
        obs_old = obs
        i += 1

    if render:
        # standard render
        # ===============================================
        env.seed(seed)
        # reset and step
        obs_old = env.reset()
        # print(f"init obs: {obs_old}")
        rewards = 0
        i = 0
        while i < n_steps:
            # a = env.action_space.sample()
            a = 1
            obs, rew, done, info = env.step(a)
            env.render(as_gif=False)
            # print(f"obs: {obs}")
            print(f"reward: {rew}")
            rewards += rew
            for o1, o2 in zip(obs_old.values(), obs.values()):
                assert isinstance(o1, np.ndarray) and isinstance(o2, np.ndarray)
                assert np.all(np.array(o1.shape) == np.array(o2.shape))
            obs_old = obs
            i += 1
