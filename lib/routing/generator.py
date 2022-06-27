#
from warnings import warn
from typing import Union, Optional, Tuple, List
import os
import io
import pickle
import logging
import math

import numpy as np
from scipy.linalg import block_diag
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig, ListConfig

from lib.routing.formats import RPInstance
from lib.utils import format_ds_save_path

__all__ = [
    "RPGenerator",
    "RPDataset",
]
logger = logging.getLogger(__name__)


def load_benchmark_instance(filepath: str, specification: str = "standard"):
    """For loading and parsing benchmark instances in CVRPLIB format."""
    with io.open(filepath, 'rt', newline='') as f:
        cap = 1
        n, k = None, None
        coord_flag = True
        idx = 0
        for i, line in enumerate(f):
            data = line.strip().split()
            if i in [1, 4]:
                pass
            elif i == 0:
                assert data[0] == "NAME"
                n_str = data[-1]
                assert 'k' in n_str
                k = int(n_str.split('k')[-1])
            elif i == 2:
                assert data[0] == "TYPE"
                assert data[-1] == "CVRP"
            elif i == 3:
                assert data[0] == "DIMENSION"
                n = int(data[-1])
                node_features = np.zeros((n, 3), dtype=np.single)
            elif i == 5:
                assert data[0] == "CAPACITY"
                cap = int(data[-1])
            else:
                if data[0] == "DEPOT_SECTION":
                    break
                elif data[0] == "NODE_COORD_SECTION":
                    coord_flag = True
                    idx = 0
                elif data[0] == "DEMAND_SECTION":
                    coord_flag = False
                    idx = 0
                else:
                    if specification.lower() == "standard":
                        if coord_flag:
                            # read coordinates
                            assert len(data) == 3
                            node_features[idx, :2] = np.array(data[1:]).astype(np.single)
                            idx += 1
                        else:
                            # read demands
                            assert len(data) == 2
                            node_features[idx, -1] = np.array(data[-1]).astype(np.single)
                            idx += 1
                    else:
                        raise NotImplementedError(specification)

    # normalize coords and demands
    assert node_features[:, :2].max() <= 1000
    assert node_features[:, :2].min() >= 0
    node_features[:, :2] = node_features[:, :2]/1000
    node_features[:, -1] = node_features[:, -1]/cap

    # add additional indicators
    depot_1_hot = np.zeros(n, dtype=np.single)
    depot_1_hot[0] = 1
    customer_1_hot = np.ones(n, dtype=np.single)
    customer_1_hot[0] = 0

    return RPInstance(
        coords=node_features[:, :2],
        node_features=np.concatenate((
            depot_1_hot[:, None],
            customer_1_hot[:, None],
            node_features
        ), axis=-1),
        graph_size=n,
        constraint_idx=[-1],  # demand is at last position of node features
        vehicle_capacity=1.0,  # demands are normalized
        max_num_vehicles=k,
    )


def parse_from_cfg(x):
    if isinstance(x, DictConfig):
        return dict(x)
    elif isinstance(x, ListConfig):
        return list(x)
    else:
        return x


class DataSampler:
    """Sampler implementing different options to generate data for RPs."""
    def __init__(self,
                 n_components: int = 5,
                 n_dims: int = 2,
                 coords_sampling_dist: str = "uniform",
                 covariance_type: str = "diag",
                 mus: Optional[np.ndarray] = None,
                 sigmas: Optional[np.ndarray] = None,
                 mu_sampling_dist: str = "normal",
                 mu_sampling_params: Tuple = (0, 1),
                 sigma_sampling_dist: str = "uniform",
                 sigma_sampling_params: Tuple = (0.1, 0.3),
                 weights_sampling_dist: str = "random_int",
                 weights_sampling_params: Tuple = (1, 10),
                 random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
                 try_ensure_feasibility: bool = True,
                 verbose: bool = False,
                 ):
        """

        Args:
            n_components: number of mixture components
            n_dims: dimension of sampled features, e.g. 2 for Euclidean coordinates
            coords_sampling_dist: type of distribution to sample coordinates, one of ["uniform"]
            covariance_type: type of covariance matrix, one of ['diag', 'full']
            mus: user provided mean values for mixture components
            sigmas: user provided covariance values for mixture components
            mu_sampling_dist: type of distribution to sample initial mus, one of ['uniform', 'normal']
            mu_sampling_params: parameters for mu sampling distribution
            sigma_sampling_dist: type of distribution to sample initial sigmas, one of ['uniform', 'normal']
            sigma_sampling_params: parameters for sigma sampling distribution
            weights_sampling_dist: type of distribution to sample weights,
                                    one of ['random_int', 'uniform', 'gamma']
            weights_sampling_params: parameters for weight sampling distribution
            random_state: seed integer or numpy random (state) generator
            try_ensure_feasibility: flag to try to ensure the feasibility of the generated instances
            verbose: verbosity flag to print additional info and warnings
        """
        self.nc = n_components
        self.f = n_dims
        self.coords_sampling_dist = coords_sampling_dist.lower()
        self.covariance_type = covariance_type
        self.mu_sampling_dist = mu_sampling_dist.lower()
        self.mu_sampling_params = mu_sampling_params
        self.sigma_sampling_dist = sigma_sampling_dist.lower()
        self.sigma_sampling_params = sigma_sampling_params
        self.weights_sampling_dist = weights_sampling_dist.lower()
        self.weights_sampling_params = weights_sampling_params
        self.try_ensure_feasibility = try_ensure_feasibility
        self.verbose = verbose
        # set random generator
        if random_state is None or isinstance(random_state, int):
            self.rnd = np.random.default_rng(random_state)
        else:
            self.rnd = random_state

        if self.coords_sampling_dist in ["gm", "gaussian_mixture"]:
            # sample initial mu and sigma if not provided
            if mus is not None:
                assert (
                    (mus.shape[0] == self.nc and mus.shape[1] == self.f) or
                    (mus.shape[0] == self.nc * self.f)
                )
                self.mu = mus.reshape(self.nc * self.f)
            else:
                self.mu = self._sample_mu(mu_sampling_dist.lower(), mu_sampling_params)
            if sigmas is not None:
                assert (
                    (sigmas.shape[0] == self.nc and sigmas.shape[1] == (self.f if covariance_type == "diag" else self.f**2))
                    or (sigmas.shape[0] == (self.nc * self.f if covariance_type == "diag" else self.nc * self.f**2))
                )
                self.sigma = self._create_cov(sigmas, cov_type=covariance_type)
            else:
                covariance_type = covariance_type.lower()
                if covariance_type not in ["diag", "full"]:
                    raise ValueError(f"unknown covariance type: <{covariance_type}>")
                self.sigma = self._sample_sigma(sigma_sampling_dist.lower(), sigma_sampling_params, covariance_type)

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.rnd = np.random.default_rng(seed)

    def resample_gm(self):
        """Resample initial mus and sigmas."""
        self.mu = self._sample_mu(
            self.mu_sampling_dist,
            self.mu_sampling_params
        )
        self.sigma = self._sample_sigma(
            self.sigma_sampling_dist,
            self.sigma_sampling_params,
            self.covariance_type
        )

    def sample_coords(self,
                      n: int,
                      resample_mixture_components: bool = True,
                      **kwargs) -> np.ndarray:
        """
        Args:
            n: number of samples to draw
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance

        Returns:
            coords: (n, n_dims)
        """
        if self.coords_sampling_dist == "uniform":
            coords = self._sample_unf_coords(n, **kwargs)
        else:
            if resample_mixture_components:
                self.resample_gm()
            n_per_c = math.ceil(n / self.nc)
            coords = self._sample_gm_coords(n_per_c, n, **kwargs)

        return coords.astype(np.float32)

    def sample_weights(self,
                       n: int,
                       k: Union[int, Tuple[int, int]],
                       cap: Optional[Union[float, int, Tuple[int, int]]] = None,
                       max_cap_factor: Optional[float] = None,
                       ) -> np.ndarray:
        """
        Args:
            n: number of samples to draw
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle

        Returns:
            weights: (n, )
        """
        n_wo_depot = n-1
        # sample a weight for each point
        if self.weights_sampling_dist == "random_int":
            assert cap is not None, \
                f"weight sampling dist 'random_int' requires <cap> to be specified"

            # if isinstance(k, Tuple) or isinstance(k, List):
            #     k = self.sample_rnd_int(*k)
            # if isinstance(cap, Tuple) or isinstance(cap, List):
            #     cap = self.sample_rnd_int(*cap)
            #weights = self.rnd.integers(1, 10, size=(n_wo_depot,))

            weights = self.rnd.integers(1, (cap-1)//2, size=(n_wo_depot, ))
            # normalize weights by total max capacity of vehicles
            # in order to create a large variety of instances in terms of
            # required number of vehicles to serve all customers,
            # we sample the normalization constant dependent on the
            # max number of vehicles k
            if self.try_ensure_feasibility:
                if max_cap_factor is not None:
                    normalizer = np.ceil((weights.sum(axis=-1)) * max_cap_factor) / self.sample_rnd_int(k//4, k)
                else:
                    normalizer = np.ceil((weights.sum(axis=-1)) * 1.08) / self.sample_rnd_int(k//4, k)
            else:
                normalizer = cap + 1
        else:
            assert max_cap_factor is not None, \
                f"weight sampling dists 'uniform' and 'gamma' require <max_cap_factor> to be specified"
            if self.weights_sampling_dist == "uniform":
                weights = self._sample_uniform(n_wo_depot, *self.weights_sampling_params)
            elif self.weights_sampling_dist == "gamma":
                weights = self._sample_gamma(n_wo_depot, *self.weights_sampling_params)
            else:
                raise ValueError
            weights = weights.reshape(-1)
            if self.verbose:
                if np.any(weights.max(-1) / weights.min(-1) > 10):
                    warn(f"Largest weight is more than 10-times larger than smallest weight.")
            # normalize weights w.r.t. norm capacity of 1.0 per vehicle and specified max_cap_factor
            # using ceiling adds a slight variability in the total sum of weights,
            # such that not all instances are exactly limited to the max_cap_factor
            normalizer = np.ceil((weights.sum(axis=-1)) * max_cap_factor) / k

        weights = weights / normalizer

        if np.sum(weights) > k:
            if self.verbose:
                warn(f"generated instance is infeasible just by demands vs. "
                     f"total available capacity of specified number of vehicles.")
            if self.try_ensure_feasibility:
                raise RuntimeError

        weights = np.concatenate((np.array([0]), weights), axis=-1)     # add 0 weight for depot
        return weights.astype(np.float32)

    def sample_rnd_int(self, lower: int, upper: int) -> int:
        return self.rnd.integers(lower, upper, 1)[0]

    def sample(self,
               n: int,
               k: Union[int, Tuple[int, int]],
               cap: Optional[Union[float, int, Tuple[int, int]]] = None,
               max_cap_factor: Optional[float] = None,
               resample_mixture_components: bool = True,
               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            n: number of samples to draw
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance

        Returns:
            coords: (n, n_dims)
            weights: (n, )
        """
        coords = self.sample_coords(n=n, resample_mixture_components=resample_mixture_components, **kwargs)
        weights = self.sample_weights(n=n, k=k, cap=cap, max_cap_factor=max_cap_factor)
        return coords, weights

    def _sample_mu(self, dist: str, params: Tuple):
        size = self.nc * self.f
        if dist == "uniform":
            return self._sample_uniform(size, params[0], params[1])
        elif dist == "normal":
            return self._sample_normal(size, params[0], params[1])
        else:
            raise ValueError(f"unknown sampling distribution: <{dist}>")

    def _sample_sigma(self, dist: str, params: Tuple, cov_type: str):
        if cov_type == "full":
            size = self.nc * self.f**2
        else:
            size = self.nc * self.f
        if dist == "uniform":
            x = self._sample_uniform(size, params[0], params[1])
        elif dist == "normal":
            x = np.abs(self._sample_normal(size, params[0], params[1]))
        else:
            raise ValueError(f"unknown sampling distribution: <{dist}>")
        return self._create_cov(x, cov_type=cov_type)

    def _create_cov(self, x, cov_type: str):
        if cov_type == "full":
            # create block diagonal matrix to model covariance only
            # between features of each individual component
            x = x.reshape((self.nc, self.f, self.f))
            return block_diag(*x.tolist())
        else:
            return np.diag(x.reshape(-1))

    def _sample_uniform(self,
                        size: Union[int, Tuple[int, ...]],
                        low: Union[int, np.ndarray] = 0.0,
                        high: Union[int, np.ndarray] = 1.0):
        return self.rnd.uniform(size=size, low=low, high=high)

    def _sample_normal(self,
                       size: Union[int, Tuple[int, ...]],
                       mu: Union[int, np.ndarray],
                       sigma: Union[int, np.ndarray]):
        return self.rnd.normal(size=size, loc=mu, scale=sigma)

    def _sample_gamma(self,
                      size: Union[int, Tuple[int, ...]],
                      alpha: Union[int, np.ndarray],
                      beta: Union[int, np.ndarray]):
        return self.rnd.gamma(size=size, shape=alpha, scale=beta)

    def _sample_unf_coords(self, n: int, **kwargs) -> np.ndarray:
        """Sample coords uniform in [0, 1]."""
        return self.rnd.uniform(size=(n, self.f))

    def _sample_gm_coords(self, n_per_c: int, n: Optional[int] = None, **kwargs) -> np.ndarray:
        """Sample coordinates from k Gaussians."""
        coords = self.rnd.multivariate_normal(
            mean=self.mu,
            cov=self.sigma,
            size=n_per_c,
        ).reshape(-1, self.f)   # (k*n, f)
        if n is not None:
            coords = coords[:n]     # if k % n != 0, some of the components have 1 more sample than others
        # normalize coords in [0, 1]
        return self._normalize_coords(coords)

    @staticmethod
    def _normalize_coords(coords: np.ndarray):
        """Applies joint min-max normalization to x and y coordinates."""
        coords[:, 0] = coords[:, 0] - coords[:, 0].min()
        coords[:, 1] = coords[:, 1] - coords[:, 1].min()
        max_val = coords.max()  # joint max to preserve relative spatial distances
        coords[:, 0] = coords[:, 0] / max_val
        coords[:, 1] = coords[:, 1] / max_val
        return coords


class RPGenerator:
    """Wraps data generation, loading and
    saving functionalities for routing problems."""
    RPS = ['tsp', 'cvrp']  # routing problem variants currently implemented in generator

    def __init__(self,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 float_prec: np.dtype = np.float32,
                 **kwargs):
        self._seed = seed
        self._rnd = np.random.default_rng(seed)
        self.verbose = verbose
        self.float_prec = float_prec
        self.sampler = DataSampler(verbose=verbose, **kwargs)

    def generate(self,
                 problem: str,
                 sample_size: int = 1000,
                 graph_size: int = 100,
                 **kwargs):
        """Generate data with corresponding RP generator function."""
        try:
            generate = getattr(self, f"generate_{problem.lower()}_data")
        except AttributeError:
            raise ModuleNotFoundError(f"The corresponding generator for the problem <{problem}> does not exist.")
        return generate(size=sample_size, graph_size=graph_size, **kwargs)

    def seed(self, seed: Optional[int] = None):
        """Set generator seed."""
        if self._seed is None or (seed is not None and self._seed != seed):
            self._seed = seed
            self._rnd = np.random.default_rng(seed)
            self.sampler.seed(seed)

    @staticmethod
    def load_dataset(filename: Optional[str] = None,
                     offset: int = 0,
                     limit: Optional[int] = None,
                     **kwargs) -> List[RPInstance]:
        """Load data from file."""
        f_ext = os.path.splitext(filename)[1]
        filepath = os.path.normpath(os.path.expanduser(filename))
        if len(f_ext) == 0 or f_ext in ['.txt', '.vrp']:
            # benchmark instance
            logger.info(f"Loading benchmark instance from:  {filepath}")
            data = load_benchmark_instance(filepath, kwargs.get('specification', "standard"))
            assert isinstance(data, RPInstance)
            return [data]
        else:
            assert f_ext in ['.pkl', '.dat', '.pt']
            logger.info(f"Loading dataset from:  {filepath}")
            try:
                data = torch.load(filepath)
            except RuntimeError:
                # fall back to pickle loading
                assert os.path.splitext(filepath)[1] == '.pkl', "Can only load pickled datasets."
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

            if limit is not None and len(data) != (limit-offset):
                assert isinstance(data, List) or isinstance(data, np.ndarray), \
                    f"To apply limit the data has to be of type <List> or <np.ndarray>."
                if len(data) < limit:
                    warn(f"Provided data size limit={limit} but limit is larger than data size={len(data)}.")
                logger.info(f"Specified offset={offset} and limit={limit}. "
                            f"Loading reduced dataset of size={limit-offset}.")
                return data[offset:limit]
            else:
                return data

    @staticmethod
    def save_dataset(dataset: Union[List, np.ndarray],
                     filepath: str,
                     **kwargs):
        """Saves data set to file path"""
        filepath = format_ds_save_path(filepath, **kwargs)
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        logger.info(f"Saving dataset to:  {filepath}")
        try:
            torch.save(dataset, filepath)
        except RuntimeError:
            # fall back to pickle save
            assert os.path.splitext(filepath)[1] == '.pkl', "Can only save as pickle. Please add extension '.pkl'!"
            with open(filepath, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        return str(filepath)

    @staticmethod
    def _create_nodes(size: int,
                      graph_size: int,
                      features: List,
                      n_depots: int = 1):
        """Create node id and type vectors and concatenate with other features."""
        return np.dstack((
            np.broadcast_to(np.concatenate((  # add id and node type (depot / customer)
                np.array([1] * n_depots +
                         [0] * graph_size)[:, None],  # depot/customer type 1-hot
                np.array([0] * n_depots +
                         [1] * graph_size)[:, None],  # depot/customer type 1-hot
            ), axis=-1), (size, graph_size + n_depots, 2)),
            *features,
        ))

    @staticmethod
    def _distance_matrix(coords: np.ndarray, l_norm: Union[int, float] = 2):
        """Calculate distance matrix with specified norm. Default is l2 = Euclidean distance."""
        return np.linalg.norm(coords[:, :, None] - coords[:, None, :], ord=l_norm, axis=0)[:, :, :, None]

    def state_dict(self):
        """Converts the current generator state to a PyTorch style state dict."""
        return {'seed': self._seed, 'rnd': self._rnd}

    def load_state_dict(self, state_dict):
        """Load state from state dict."""
        self._seed = state_dict['seed']
        self._rnd = state_dict['rnd']

    # Standard TSP
    def generate_tsp_data(self,
                          size: int,
                          graph_size: int,
                          **kwargs) -> List[RPInstance]:
        """Generate data for TSP

        Args:
            size (int): size of dataset (number of problem instances)
            graph_size (int): size of problem instance graph (number of nodes)

        Returns:
            RPDataset
        """
        if self.verbose:
            print(f"Sampling {size} problems with graph of size {graph_size}.")
            if kwargs:
                print(f"Provided additional kwargs: {kwargs}")

        coords = np.stack([
            self.sampler.sample_coords(n=graph_size, **kwargs) for _ in range(size)
        ])
        # use dummy depot node as start node in TSP tour, therefore need to reduce graph size by 1
        node_features = self._create_nodes(size, graph_size-1, n_depots=1, features=[coords])

        # type cast
        coords = coords.astype(self.float_prec)
        node_features = node_features.astype(self.float_prec)

        return [
            RPInstance(
                coords=coords[i],
                node_features=node_features[i],
                graph_size=graph_size,
            )
            for i in range(size)
        ]

    # Standard CVRP
    def generate_cvrp_data(self,
                           size: int,
                           graph_size: int,
                           k: Union[int, Tuple[int, int]],
                           cap: Optional[Union[float, int, Tuple[int, int]]] = None,
                           max_cap_factor: Optional[float] = None,
                           n_depots: int = 1,
                           **kwargs) -> List[RPInstance]:
        """Generate data for CVRP

        Args:
            size (int): size of dataset (number of problem instances)
            graph_size (int): size of problem instance graph (number of nodes)
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            n_depots: number of depots (default = 1)

        Returns:
            RPDataset
        """
        if self.verbose:
            print(f"Sampling {size} problems with graph of size {graph_size}+{n_depots}.")
            if kwargs:
                print(f"Provided additional kwargs: {kwargs}")

        n = graph_size + n_depots
        coords = np.stack([
            self.sampler.sample_coords(n=n, **kwargs) for _ in range(size)
        ])
        demands = np.stack([
            self.sampler.sample_weights(n=n, k=k, cap=cap, max_cap_factor=max_cap_factor) for _ in range(size)
        ])
        node_features = self._create_nodes(size, graph_size, n_depots=n_depots, features=[coords, demands])

        # type cast
        coords = coords.astype(self.float_prec)
        node_features = node_features.astype(self.float_prec)

        return [
            RPInstance(
                coords=coords[i],
                node_features=node_features[i],
                graph_size=graph_size+n_depots,
                constraint_idx=[-1],    # demand is at last position of node features
                vehicle_capacity=1.0,   # demands are normalized
                max_num_vehicles=k,
            )
            for i in range(size)
        ]


class RPDataset(Dataset):
    """Routing problem dataset wrapper."""
    def __init__(self,
                 problem: str = None,
                 fname: str = None,
                 seed: int = None,
                 **kwargs):
        """

        Args:
            problem: name/id of routing problem
            fname: optional file name to load dataset
            seed: seed for random generator
            **kwargs:  additional kwargs for the generator
        """
        super(RPDataset, self).__init__()
        assert problem is not None or fname is not None
        if fname is not None:
            logger.info(f"provided dataset {fname}, so no new samples are generated.")
        self.problem = problem
        self.fname = fname
        self.gen = RPGenerator(seed=seed, **kwargs)

        self.size = None
        self.data = None

    def seed(self, seed: int):
        self.gen.seed(seed)

    def sample(self, sample_size: int = 1000, graph_size: int = 100, **kwargs):
        """Loads fixed dataset if filename was provided else
        samples a new dataset based on specified config."""
        if self.fname is not None:   # load data
            self.data = RPGenerator.load_dataset(self.fname, limit=sample_size, **kwargs)
        else:
            self.data = self.gen.generate(
                problem=self.problem,
                sample_size=sample_size,
                graph_size=graph_size,
                **kwargs
            )
        self.size = len(self.data)
        return self

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# ============= #
# ### TEST #### #
# ============= #
def _test(
    size: int = 10,
    n: int = 20,
    seed: int = 1,
):
    problems = ['tsp', 'cvrp']
    coord_samp = ['uniform', 'gm']
    weight_samp = ['random_int', 'uniform', 'gamma']
    k = 4
    cap = 9
    max_cap_factor = 1.1

    for p in problems:
        for csmp in coord_samp:
            for wsmp in weight_samp:
                ds = RPDataset(problem=p,
                               seed=seed,
                               coords_sampling_dist=csmp,
                               weights_sampling_dist=wsmp,
                               n_components=2 if p == "tsp" else 3,
                               )
                ds.sample(sample_size=size, graph_size=n, k=k, cap=cap, max_cap_factor=max_cap_factor)


def _test2():
    PTH = "data/CVRP/benchmark/uchoa/n1/X-n101-k25.vrp"
    inst = RPGenerator.load_dataset(PTH)
    print(inst)
