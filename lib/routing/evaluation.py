#
import warnings
from typing import List, Optional
import numpy as np

from lib.routing.formats import RPSolution

EPS = np.finfo(np.float16).eps
RP_TYPES = ["CVRP"]


def eval_rp(solutions: List[RPSolution],
            problem: Optional[str] = None,
            strict_feasibility: bool = True,
            **kwargs):
    """Wraps different evaluation schemes for several
    routing problems to ensure consistent evaluation."""
    results = []
    for sol in solutions:
        if problem is not None:
            if sol.problem is None:
                sol = sol.update(problem=problem)
            else:
                assert sol.problem.upper() == problem.upper()
        if sol.problem.upper() == "CVRP":
            res = eval_cvrp(sol, strict=strict_feasibility)
        else:
            raise ValueError(f"unknown problem: '{sol.problem}'")
        results.append(res)

    costs = [r.cost for r in results if r.cost != float("inf")]
    num_vehicles = [r.num_vehicles for r in results if r.num_vehicles != float("inf")]
    num_inf = sum([1 for r in results if (r.num_vehicles == float("inf") or r.cost == float("inf"))])

    summary = {
        "cost_mean": np.mean(costs) if len(costs) > 0 else float("inf"),
        "cost_std": np.std(costs) if len(costs) > 0 else float("inf"),
        "num_vehicles_mean": np.mean(num_vehicles),
        "num_vehicles_std": np.std(num_vehicles),
        "num_vehicles_median": np.median(num_vehicles),
        "run_time_mean": np.mean([r.run_time for r in results]),
        "run_time_total": np.sum([r.run_time for r in results]),
        "num_infeasible": num_inf,
    }

    return results, summary


def eval_cvrp(solution: RPSolution, strict: bool = True) -> RPSolution:
    """(Re-)Evaluate provided solutions for the CVRP."""
    data = solution.instance
    depot = data.depot_idx[0]
    coords = data.coords
    demands = data.node_features[:, data.constraint_idx[0]]
    routes = solution.solution

    # check feasibility of routes and calculate cost
    if routes is None or len(routes) == 0:
        k = float("inf")
        cost = float("inf")
    else:
        k = 0
        cost = 0.0
        for r in routes:
            if r and sum(r) > depot:    # not empty and not just depot idx
                if r[0] != depot:
                    r = [depot] + r
                if r[-1] != depot:
                    r.append(depot)
                transit = 0
                source = r[0]
                cum_d = 0
                for target in r[1:]:
                    transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                    cum_d += demands[target]
                    source = target
                if strict and cum_d > 1.0 + 2*EPS:
                    warnings.warn(f"solution infeasible. setting cost and k to 'inf'")
                    cost = float("inf")
                    k = float("inf")
                    break
                if strict and data.max_num_vehicles is not None and k > data.max_num_vehicles:
                    warnings.warn(f"solution infeasible. setting cost to 'inf'")
                    cost = float("inf")
                cost += transit
                k += 1

    return solution.update(cost=cost, num_vehicles=k)
