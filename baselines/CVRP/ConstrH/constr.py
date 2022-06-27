#
import os
import warnings
from typing import Optional, Callable, Union, List, Dict, Tuple
from timeit import default_timer

import math
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from lib.routing import ConstructionHeuristic, RPInstance, RPSolution
from lib.routing.local_search import VRP
from lib.routing.local_search.rules import VRPH_EXACT_2D, VRPH_FUNCTION
from lib.env.utils import parse_solutions


def _parse_assignment(vrph_route: Union[np.ndarray, List]):
    """Parse tour assignment of VRPH solver."""
    assert len(vrph_route) > 0
    max_seq_len = max([len(r) for r in vrph_route]) + 1
    # buffer array of padded tour sequences
    return np.array([
        r + [0] * (max_seq_len - len(r) - 1) for r in vrph_route
    ])


class ParallelSolver:
    """Parallelization wrapper for construction heuristics
     based on multi-processing pool."""
    def __init__(self,
                 problem: str,
                 solver_args: Optional[Dict] = None,
                 num_workers: int = 1,
                 ):
        self.problem = problem
        self.solver_args = solver_args if solver_args is not None else {}
        self.solver_args['problem'] = self.problem
        if num_workers > os.cpu_count():
            warnings.warn(f"num_workers > num logical cores! This can lead to "
                          f"decrease in performance if env is not IO bound.")
        self.num_workers = num_workers

    @staticmethod
    def _solve(params: Tuple):
        """
        params:
            solver_cl: RoutingSolver.__class__
            data: GORTInstance
            solver_args: Dict
        """
        solver_cl, instance, solver_args = params
        solver = solver_cl(
            problem=solver_args.get("problem"),
            method=solver_args.get("method")
        )
        solver.seed(solver_args.get("seed"))

        N = instance.graph_size

        vrph = VRP(N)
        VRP.load_problem(
            vrph,
            1,  # 1 for CVRP, 0 for TSP
            instance.coords.tolist(),  # coordinates
            instance.node_features[:, instance.constraint_idx[0]].tolist(),  # demands
            # [best_known_dist, capacity, max_route_len, normalize_flag, neighborhood_size]
            [float(-1), float(instance.vehicle_capacity), float(-1), float(1), float(0)],
            [[float(-1)]],  # no TW
            VRPH_EXACT_2D,  # edge type
            VRPH_FUNCTION,  # edge format
        )

        t = default_timer()
        solver.construct(instance, vrph_model=vrph)
        solution = _parse_assignment(vrph.get_routes())
        t = default_timer() - t
        return [solution, t]

    def solve(self, data: List[RPInstance]) -> List[RPSolution]:

        assert isinstance(data[0], RPInstance)

        if self.num_workers <= 1:
            results = list(tqdm(
                [self._solve((ConstructionHeuristic, d, self.solver_args)) for d in data],
                total=len(data)
            ))
        else:
            with Pool(self.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(
                        self._solve,
                        [(ConstructionHeuristic, d, self.solver_args) for d in data]
                    ),
                    total=len(data),
                ))

            failed = [str(i) for i, res in enumerate(results) if res is None]
            if len(failed) > 0:
                warnings.warn(f"Some instances failed: {failed}")

        return [
            RPSolution(
                solution=parse_solutions(r[0]),
                run_time=r[1],
                problem=self.problem,
                instance=d,
            )
            for d, r in zip(data, results)
        ]
