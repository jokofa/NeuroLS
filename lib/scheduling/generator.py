#
from warnings import warn
from typing import Union, Optional, List
import os
import io
import pickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from lib.scheduling.formats import JSSPInstance
from lib.utils import format_ds_save_path

__all__ = [
    "JSSPGenerator",
    "JSSPDataset",
]
logger = logging.getLogger(__name__)


def load_benchmark_instance(filepath: str, specification: str = "standard"):
    """For loading and parsing benchmark instances
    from files in JSSP standard or Taillard specification format."""
    with io.open(filepath, 'rt', newline='') as f:
        for i, line in enumerate(f):
            data = line.strip().split()
            if i == 0:
                n_j, n_m = int(data[0]), int(data[1])
                durations = np.zeros((n_j, n_m), dtype=np.single)
                sequences = np.zeros((n_j, n_m), dtype=int)
            else:
                if specification.lower() == "standard":
                    assert len(data) % 2 == 0
                    assert len(data) / 2 == n_m
                    for j in range(0, len(data), 2):
                        sequences[i-1, j//2] = int(data[j])
                        durations[i-1, j//2] = float(data[j+1])
                else:
                    raise NotImplementedError(specification)

    # normalize
    assert np.all(durations > 0)
    max_dur = float(durations.max())
    durations = durations / max_dur

    # make sure sequence machine idx at 1 (not 0!)
    if sequences.min() < 1 and sequences.max() < n_m:
        sequences += 1

    return JSSPInstance(
        num_jobs=n_j,
        num_machines=n_m,
        durations=durations,
        sequences=sequences,
        org_max_dur=max_dur
    )


class JSSPGenerator:
    """Wraps data generation, loading and
    saving functionalities for jop shop scheduling problem."""
    def __init__(self,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 float_prec: np.dtype = np.float32,
                 **kwargs):
        self._seed = seed
        self._rnd = np.random.default_rng(seed)
        self.verbose = verbose
        self.float_prec = float_prec

    def generate(self,
                 problem: str = "JSSP",
                 sample_size: int = 1000,
                 num_jobs: int = 10,
                 num_machines: int = 10,
                 **kwargs):
        """Generate data with corresponding RP generator function."""
        assert problem.upper() in ["JSP", "JSSP"]
        return [
            JSSPInstance(
                num_jobs, num_machines,
                *self.unf_instance_gen(num_jobs, num_machines, **kwargs)
            ) for _ in range(sample_size)
        ]

    def _permute_rows(self, x: np.ndarray):
        ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
        ix_j = self._rnd.random(x.shape).argsort(axis=1)
        return x[ix_i, ix_j]

    def unf_instance_gen(self, n_j: int, n_m: int, low: int = 1, high: int = 100):
        """from L2D code
        https://github.com/zcaicaros/L2D/blob/main/uniform_instance_gen.py
        """
        assert low > 0
        durations = self._rnd.integers(low=low, high=high, size=(n_j, n_m))
        machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
        machines = self._permute_rows(machines)
        max_dur = float(durations.max())
        durations = durations / max_dur
        return durations, machines, max_dur

    def seed(self, seed: Optional[int] = None):
        """Set generator seed."""
        if self._seed is None or (seed is not None and self._seed != seed):
            self._seed = seed
            self._rnd = np.random.default_rng(seed)

    @staticmethod
    def load_dataset(filename: Optional[str] = None,
                     offset: int = 0,
                     limit: Optional[int] = None,
                     **kwargs) -> List[JSSPInstance]:
        """Load data from file."""
        f_ext = os.path.splitext(filename)[1]
        filepath = os.path.normpath(os.path.expanduser(filename))
        if len(f_ext) == 0 or f_ext == ".txt":
            # benchmark instance
            logger.info(f"Loading benchmark instance from:  {filepath}")
            data = load_benchmark_instance(filepath, kwargs.get('specification', "standard"))
            assert isinstance(data, JSSPInstance)
            return [data]
        else:
            assert f_ext in ['.pkl', '.dat', '.pt']
            logger.info(f"Loading dataset from:  {filepath}")
            try:
                data = torch.load(filepath, **kwargs)
            except RuntimeError:
                # fall back to pickle loading
                assert os.path.splitext(filepath)[1] == '.pkl', "Can only load pickled datasets."
                with open(filepath, 'rb') as f:
                    data = pickle.load(f, **kwargs)

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

    def state_dict(self):
        """Converts the current generator state to a PyTorch style state dict."""
        return {'seed': self._seed, 'rnd': self._rnd}

    def load_state_dict(self, state_dict):
        """Load state from state dict."""
        self._seed = state_dict['seed']
        self._rnd = state_dict['rnd']


class JSSPDataset(Dataset):
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
        super(JSSPDataset, self).__init__()
        assert problem is not None or fname is not None
        if fname is not None:
            logger.info(f"provided dataset {fname}, so no new samples are generated.")
        self.problem = problem
        self.fname = fname
        self.gen = JSSPGenerator(seed=seed, **kwargs)

        self.size = None
        self.data = None

    def seed(self, seed: int):
        self.gen.seed(seed)

    def sample(self,
               sample_size: int = 1000,
               num_jobs: int = 10,
               num_machines: int = 10,
               **kwargs):
        """Loads fixed dataset if filename was provided else
        samples a new dataset based on specified config."""
        if self.fname is not None:   # load data
            self.data = JSSPGenerator.load_dataset(self.fname, limit=sample_size, **kwargs)
        else:
            self.data = self.gen.generate(
                problem=self.problem,
                sample_size=sample_size,
                num_jobs=num_jobs,
                num_machines=num_machines,
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
def _test():
    PTH = "data/JSSP/benchmark/ta41"
    instance = load_benchmark_instance(PTH)
    print(instance)
    instance = JSSPGenerator.load_dataset(PTH)
    print(instance)
    instance = JSSPGenerator().generate(sample_size=1, num_jobs=30, num_machines=20)[0]
    print(instance)
