#
from typing import Optional, Callable, Union, List
import numpy as np

from lib.routing.formats import RPInstance
from lib.routing.local_search import VRPH


def euclidean_dist_mat(instance: RPInstance) -> np.ndarray:
    return np.linalg.norm(instance.coords[:, None, :] - instance.coords[None, :, :], axis=-1)


def knn_nbh(instance: RPInstance, k: int) -> np.ndarray:
    """Compute the indices of the 'k' nearest neighbors of each node in the instance."""
    dist_mat = euclidean_dist_mat(instance)
    np.fill_diagonal(dist_mat, float('inf'))    # mask self
    return dist_mat.argsort(axis=-1)[:, :k]


class ConstructionHeuristic:
    """
    Wraps several construction heuristics
    for different routing problems.
    """
    def __init__(self,
                 problem: str = "TSP",
                 method: str = "random",
                 dist_mat_fn: Callable = euclidean_dist_mat,
                 **kwargs):
        self.problem = problem.lower()
        self.method = method.lower()
        self.dist_mat_fn = dist_mat_fn
        self.rnd = np.random.default_rng(1)

    def seed(self, seed: Optional[int] = None):
        self.rnd = np.random.default_rng(seed)

    def construct(self, instance: RPInstance, vrph_model: VRPH, **kwargs):
        """Construct an initial solution for the provided problem instance.

        Args:
            instance: the VRP instance to solve
            vrph_model: VRPH model of problem instance
        """
        try:
            solve = getattr(self, f"_{self.method}_{self.problem}")
        except AttributeError:
            raise ModuleNotFoundError(f"The corresponding construction method '{self.method}' "
                                      f"for the problem '{self.problem}' does not exist.")
        return solve(instance, vrph_model, **kwargs)

    @staticmethod
    def _push_solution(sol: Union[np.ndarray, List], vrph_model: VRPH):
        """Push the created initial tour to the VRPH model."""
        vrph_model.use_initial_solution([sol.tolist()] if isinstance(sol, np.ndarray) else sol)

    def _random_tsp(self, instance: RPInstance, vrph_model: VRPH, **kwargs):
        """Complete random construction."""
        N = instance.graph_size
        # the tour has to end with a 0 (necessary for TSP in VRPH)
        # -> shuffle 1 to N-1 and append 0 at the end
        tour = self.rnd.permutation(N-1)+1
        tour = np.append(tour, 0)
        self._push_solution(tour, vrph_model)

    def _nn_tsp(self, instance: RPInstance, vrph_model: VRPH, **kwargs):
        """Nearest neighbor construction."""
        N = instance.graph_size
        dist_mat = self.dist_mat_fn(instance)
        np.fill_diagonal(dist_mat, float('inf'))
        idx = instance.depot_idx[0]
        tour = np.empty(N, dtype=np.int)
        for i in range(instance.graph_size-1):
            old_idx = idx
            idx = np.argmin(dist_mat[idx])  # min distance == next neighbor
            tour[i] = idx
            dist_mat[:, old_idx] = float('inf')     # mask selected nodes with infinity distance
        # add zero idx node at the end (necessary for TSP in VRPH)
        tour[-1] = 0
        self._push_solution(tour, vrph_model)

    def _random_cvrp(self, instance: RPInstance, vrph_model: VRPH, **kwargs):
        """Complete random construction."""
        # random sequential construction starting a new tour when capacity is reached.
        N = instance.graph_size
        assert len(instance.depot_idx) == 1 and instance.depot_idx[0] == 0
        # shuffle node indices without depot
        n_idx = self.rnd.permutation(N - 1) + 1
        # add nodes in loop until capacity is reached, then start new tour
        tours = []
        tr = []
        cap = instance.vehicle_capacity
        demands = instance.node_features[:, instance.constraint_idx[0]]
        for n in n_idx:
            d = demands[n]
            if d > cap:
                tours.append(tr)
                # start new tour
                tr = []
                cap = instance.vehicle_capacity
            # add to current tour
            cap -= d
            tr.append(n)
        if len(tr) > 0:
            tours.append(tr)

        assert sum([len(t) for t in tours]) == (N-1)
        self._push_solution(tours, vrph_model)

    def _nn_cvrp(self, instance: RPInstance, vrph_model: VRPH, **kwargs):
        """Nearest neighbor construction."""
        raise NotImplementedError

    def _savings_cvrp(self, instance: RPInstance, vrph_model: VRPH, lambd=1.2, use_nbh: bool = False, **kwargs):
        """
        Uses the Clarke-Wright savings method to construct an initial solution for the problem.

        Args:
            instance: the VRP instance to solve
            vrph_model: VRPH model of problem instance
            lambd: CW savings lambda param
            use_nbh: flag to use provided neighbor_lists in VRPH

        Returns:
            None, the solution is automatically loaded into the VRP problem given
        """
        N = instance.graph_size
        # initialize the CW solver with the number of nodes (without depot) in the problem
        cw = VRPH.ClarkeWright(int(N-1))
        # call the solver with (problem model, lambda, use_neighbor_list)
        cw.Construct(vrph_model, float(lambd), use_nbh)

    def _sweep_cvrp(self, instance: RPInstance, vrph_model: VRPH, **kwargs):
        """
        Constructs an initial VRP solution by the simple sweep method.
        Start by picking a random node and then sweep counterclockwise
        and add nodes until we reach vehicle capacity or max route length.

        Args:
            instance: the VRP instance to solve
            vrph_model: VRPH model of problem instance

        Returns:
            None, the solution is automatically loaded into the VRP problem given
        """
        VRPH.Sweep().Construct(vrph_model)
