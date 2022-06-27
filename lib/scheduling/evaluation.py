#
from typing import List, Optional
import numpy as np

from lib.scheduling.formats import JSSPSolution

EPS = np.finfo(np.float32).eps
JSSP_TYPES = ["JSSP"]


def eval_jssp(solutions: List[JSSPSolution], problem: Optional[str] = None, **kwargs):
    """Wraps different evaluation schemes for
    scheduling problems to ensure consistent evaluation."""
    # only JSSP here which is already evaluated via longest path in DAG in JSSP env
    results = solutions
    costs = [r.cost*r.instance.org_max_dur for r in results if r.cost != float("inf")]
    num_inf = sum([1 for r in results if r.cost == float("inf")])

    summary = {
        "cost_mean": np.mean(costs),
        "cost_std": np.std(costs),
        "run_time_mean": np.mean([r.run_time for r in results]),
        "run_time_total": np.sum([r.run_time for r in results]),
        "num_infeasible": num_inf,
    }

    return results, summary
