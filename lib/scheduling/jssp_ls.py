#
import logging
from typing import Optional, List, Dict, Union, Any, Tuple, NamedTuple

import numpy as np
from scipy.special import softmax

from lib.scheduling.formats import JSSPInstance, INF
from lib.scheduling.jssp_graph import JSSPGraph, PathNbh
from lib.scheduling.jssp_pdr import PriorityDispatchingRule

logger = logging.getLogger(__name__)

LS_MOVES = [
    "CT",
    "CET",
    "ECET",
    "CEI"
]
SEARCH_CRITS = [
    "best",             # best possible move in neighborhood
    "first",            # first improving move
    "first_eps",        # first move improving more than epsilon
]
SELECTION_CRITS = [
    "greedy",       # greedily select best move
    "sampling"      # sample move according to potentials
]


class JSSPSolver:

    def __init__(self,
                 pdr_method: str,
                 search_criterion: str = "best",
                 selection_criterion: str = "sampling",
                 num_rnd: int = 3,
                 shuffle: bool = False,
                 eps: float = 3.0,
                 verbose: bool = False,
                 **kwargs):
        self.pdr = PriorityDispatchingRule(method=pdr_method)
        assert search_criterion.lower() in SEARCH_CRITS
        self.search_crit = search_criterion.lower()
        assert selection_criterion.lower() in SELECTION_CRITS
        self.select_crit = selection_criterion.lower()
        self.num_rnd = num_rnd
        self.shuffle = shuffle
        self.eps = eps
        self.verbose = verbose
        self._rnd = np.random.default_rng(1)

        self.instance = None
        self.graph = None
        self.cost = None
        self._ckpt = None

    def seed(self, seed: Optional[int] = None):
        self._rnd = np.random.default_rng(seed)
        self.pdr.seed(seed+1)

    def _get_state(self) -> Tuple[JSSPGraph, Union[np.ndarray, float, int]]:
        return self.graph, self.cost

    def load_problem(self, instance: JSSPInstance):
        self.instance = instance

    def construct(self,
                  instance: Optional[JSSPInstance] = None,
                  **kwargs):
        """Construct an initial solution."""
        if instance is not None:
            self.instance = instance
        _, self.cost, self.graph = self.pdr.dispatch(self.instance, **kwargs)
        return self._get_state()

    def solve(self, ls_ops: Union[str, List[str]], position: str = "ALL"):
        assert self.instance is not None and self.graph is not None
        ls_ops = [ls_ops.lower()] if isinstance(ls_ops, str) else [op.lower() for op in ls_ops]
        if "perturb" in ls_ops:
            if len(ls_ops) > 1:
                ls_ops.remove("perturb")
            else:
                ls_ops = None
            return self.perturb(ls_ops=ls_ops)
        self._search_blocks(ls_ops, position.upper(), num_rnd=self.num_rnd)
        _, self.cost = self.graph.longest_path_seq_val(set_dist_from_src=True)
        return self._get_state()

    def perturb(self, iters: int = 3, ls_ops: Optional[List[str]] = None):
        """Perturb ('shake') current solution."""
        if ls_ops is None:
            ls_ops = ["ct"]
        else:
            ls_ops = [ls_ops.lower()] if isinstance(ls_ops, str) else [op.lower() for op in ls_ops]
        for _ in range(iters):
            for op in ls_ops:
                assert op.upper() in LS_MOVES
                # get blocks, critical path, its weight values and its nbh
                cbl, cp, cv, nbh = self.graph.get_critical_blocks()
                ms = sum(cv) if isinstance(cv, list) else float(cv)
                valid_bl = np.array([len(b) > 1 for b in cbl])  # blocks must include at least 2 nodes
                n_valid = valid_bl.sum()
                assert n_valid > 0, f"no valid moves in current neighborhood"
                # select a random block
                idx = self._rnd.choice(valid_bl.nonzero()[0], size=1, replace=False)[0]
                move_eval = getattr(self, f"_{op}_eval")
                potentials, _ = move_eval([], cbl, cp, cv, ms, nbh, valid_bl, idx, no_eval=True)
                # select a random move from block
                args = potentials[self._rnd.integers(0, len(potentials), 1)[0]][1:]
                getattr(self, f"_{op}")(*args, nbh=nbh)
        _, self.cost = self.graph.longest_path_seq_val(set_dist_from_src=True)
        return self._get_state()

    def checkpoint_solution(self):
        """Save a checkpoint of the current solution
        to revert it in case the move is rejected."""
        self._ckpt = self.graph.state_dict()

    def _search_blocks(self, ls_ops: List[str], position: str, num_rnd: int = 3):
        """
        - a block is a (partial) conjunction of a machine clique
        - therefore LS is normally done on a per block (i.e. per machine) basis
        - a critical block is part of a critical path
        - any path from src to snk of maximum length corresponds to a critical path of the solution.
        """
        for op in ls_ops:
            assert op.upper() in LS_MOVES, f"unknown operator: {op.upper()}"
            # get blocks, critical path, its weight values and its nbh
            cbl, cp, cv, nbh = self.graph.get_critical_blocks()
            ms = sum(cv) if isinstance(cv, list) else float(cv)
            valid_bl = np.array([len(b) > 1 for b in cbl])   # blocks must include at least 2 nodes
            n_valid = valid_bl.sum()
            assert n_valid > 0, f"no valid moves in current neighborhood"

            if position == "ALL":
                # check all valid blocks
                #num_check = n_valid
                idx_check = valid_bl.nonzero()[0]
                if self.shuffle:    # shuffle order of evaluation
                    self._rnd.shuffle(idx_check)    # inplace
            elif position in ["RND", "RANDOM"]:
                # check num_rnd random blocks
                num_check = min(num_rnd, n_valid)
                idx_check = self._rnd.choice(valid_bl.nonzero()[0], size=num_check, replace=False)
            else:
                raise NotImplementedError(f"position={position}")

            # evaluate potentials for possible moves
            move_eval = getattr(self, f"_{op}_eval")
            potentials = []
            for i in idx_check:
                potentials, imp = move_eval(potentials, cbl, cp, cv, ms, nbh, valid_bl, i)
                if self._accept_improvement(imp):
                    break

            self.execute_move(op, potentials, nbh)

    def _accept_improvement(self, imp: Union[int, float]):
        if self.search_crit == "best":
            return False
        elif self.search_crit == "first":
            return imp > 0
        elif self.search_crit == "first_eps":
            return imp > self.eps
        else:
            raise ValueError(f"unknown search criterion {self.search_crit}")

    def execute_move(self,
                     op: str,
                     potentials: List[List],
                     nbh: PathNbh,
                     **kwargs):
        pots = np.array([v[0] for v in potentials])
        if self.select_crit == "sampling":
            # sample according to potentials
            best_idx = self._rnd.multinomial(1, softmax(pots)).argmax()
        elif self.select_crit == "greedy":
            # greedy selection of move with highest potential
            best_idx = pots.argmax()
        else:
            raise ValueError(f"unknwon selection_criterion: {self.select_crit}")

        best_pot, *args = potentials[best_idx]
        if self.verbose:
            print(f"potential: {best_pot}")
        getattr(self, f"_{op}")(*args, nbh=nbh, **kwargs)

    def reject_move(self):
        """Reject the last move (sequence of moves)."""
        assert self._ckpt is not None, \
            f"need to checkpoint solutions before rejecting."
        self.graph.load_state_dict(self._ckpt)

    @staticmethod
    def _get_val(d: Dict, k: str, default_val: Optional[Any] = 0):
        if len(d) == 0:
            return default_val
        return next(iter(d.values())).get(k, default_val)

    @staticmethod
    def _get_node(d: Dict):
        return next(iter(d.keys()))

    def eval_transpose_potential(self,
                                 u: int,
                                 v: int,
                                 nbh: PathNbh
                                 ) -> Union[int, float]:
        """

        Args:
            u: node in critical block before v
            v: node in critical block after u
            nbh: nbh tuple

        from:
            Taillard, E. D. (1994).
            Parallel taboo search techniques for the job shop scheduling problem.
            ORSA Journal of Computing, 62, 108–117.
        """
        # processing times
        p_u = nbh.pth[u].get("weight")
        p_v = nbh.pth[v].get("weight")
        # dist from source
        r_v = max(
            self._get_val(nbh.pm[u], "dist_from_src") + self._get_val(nbh.pm[u], "weight"),
            self._get_val(nbh.pj[v], "dist_from_src") + self._get_val(nbh.pj[v], "weight")
        )
        r_u = max(
            r_v + p_v,
            self._get_val(nbh.pj[u], "dist_from_src") + self._get_val(nbh.pj[u], "weight")
        )
        # dist to sink
        t_u = max(
            self._get_val(nbh.sm[v], "dist_to_snk"),
            self._get_val(nbh.sj[u], "dist_to_snk")
        ) + p_u
        t_v = max(
            t_u,
            self._get_val(nbh.sj[v], "dist_to_snk")
        ) + p_v
        return max(r_u + t_u, r_v + t_v)

    def fw_insert_feasible(self,
                           u: int,
                           v: int,
                           nbh: PathNbh,
                           cp: Optional[np.ndarray] = None
                           ) -> bool:
        """Check feasibility of forward insertion (u directly after v).

        Args:
            u: node in critical block somewhere before v
            v: node in critical block somewhere after u
            nbh: nbh tuple
            cp: critical path

        from:
            Balas, E., & Vazacopoulos, A. (1998).
            Guided local search with shifting bottleneck for job shop scheduling.
            Management science, 44(2), 262-275.

            Zhang, C., Li, P., Guan, Z., & Rao, Y. (2007).
            A tabu search algorithm with a new neighborhood structure
            for the job shop scheduling problem.
            Computers & Operations Research, 34(11), 3229-3242.
        """
        if self.select_crit == "greedy":
            # to be sure that the move can improve the makespan,
            # we need to check if sj[v] lies on critical path
            if self._get_node(nbh.sj[v]) not in cp:
                return False
        return (    # t_v >= t_sj_u
            nbh.pth[v].get("dist_to_snk")
            >=
            self._get_val(nbh.sj[u], "dist_to_snk")
        )

    def bw_insert_feasible(self,
                           u: int,
                           v: int,
                           nbh: PathNbh,
                           cp: Optional[np.ndarray] = None
                           ) -> bool:
        """Check feasibility of backward insertion (v directly before u).

        Args:
            u: node in critical block somewhere before v
            v: node in critical block somewhere after u
            nbh: nbh tuple
            cp: critical path

        from:
            Balas, E., & Vazacopoulos, A. (1998).
            Guided local search with shifting bottleneck for job shop scheduling.
            Management science, 44(2), 262-275.

            Zhang, C., Li, P., Guan, Z., & Rao, Y. (2007).
            A tabu search algorithm with a new neighborhood structure
            for the job shop scheduling problem.
            Computers & Operations Research, 34(11), 3229-3242.
        """
        if self.select_crit == "greedy":
            # to be sure that the move can improve the makespan,
            # we need to check if pj[u] lies on critical path
            if self._get_node(nbh.pj[u]) not in cp:
                return False
        return (
            (   # r_u + p_u
                nbh.pth[u].get("dist_from_src") +
                nbh.pth[u].get("weight")
            )
            >=
            (   # r_pj_v + p_pj_v
                self._get_val(nbh.pj[v], "dist_from_src") +
                self._get_val(nbh.pj[v], "weight")
            )
        )

    def eval_fw_insert_potential(self,
                                 u: int,
                                 v: int,
                                 nbh: PathNbh,
                                 bl_range: np.ndarray
                                 ) -> Union[int, float]:
        """
        from:
            Murovec, B. (2015).
            Job-shop local-search move evaluation without
            direct consideration of the criterion’s value.
            European Journal of Operational Research, 241(2), 320-329.
        """
        # bl range is critical block between u and v (both inclusive)
        e = self._get_val(nbh.pm[u], "dist_from_src") + self._get_val(nbh.pm[u], "weight")
        for a in bl_range:
            e = max(
                e,
                self._get_val(nbh.pj[a], "dist_from_src") + self._get_val(nbh.pj[a], "weight")
            ) + nbh.pth[a].get("weight")
        assert a == v
        e = max(
            e,
            self._get_val(nbh.pj[u], "dist_from_src") + self._get_val(nbh.pj[u], "weight")
        ) + nbh.pth[u].get("weight")
        t = max(
            self._get_val(nbh.sj[u], "dist_to_snk") + self._get_val(nbh.sj[u], "weight"),
            self._get_val(nbh.sm[v], "dist_to_snk") + self._get_val(nbh.sm[v], "weight"),
        )
        return e + t

    def eval_bw_insert_potential(self,
                                 u: int,
                                 v: int,
                                 nbh: PathNbh,
                                 bl_range: np.ndarray
                                 ) -> Union[int, float]:
        """
        from:
            Murovec, B. (2015).
            Job-shop local-search move evaluation without
            direct consideration of the criterion’s value.
            European Journal of Operational Research, 241(2), 320-329.
        """
        # bl range is reversed critical block between u and v (both inclusive)
        s = self._get_val(nbh.sm[v], "dist_to_snk") + self._get_val(nbh.sm[v], "weight")
        for a in bl_range:
            s = max(
                s,
                self._get_val(nbh.sj[a], "dist_to_snk") + self._get_val(nbh.sj[a], "weight")
            ) + nbh.pth[a].get("weight")
        assert a == u
        s = max(
            s,
            self._get_val(nbh.sj[v], "dist_to_snk") + self._get_val(nbh.sj[v], "weight")
        ) + nbh.pth[v].get("weight")
        r = max(
            self._get_val(nbh.pj[v], "dist_from_src") + self._get_val(nbh.pj[v], "weight"),
            self._get_val(nbh.pm[u], "dist_from_src") + self._get_val(nbh.pm[u], "weight"),
        )
        return s + r

    def _ct_eval(self,
                 potentials: List,
                 cbl: np.ndarray,
                 cp: np.ndarray,
                 cv: Union[int, float, List],
                 ms: Union[int, float],
                 nbh: PathNbh,
                 valid_bl: np.ndarray,
                 idx: int,
                 no_eval: bool = False,
                 **kwargs) -> Tuple[List, Union[int, float]]:
        bl = cbl[idx]
        if len(bl) == 2:    # just reverse single edge
            u, v = bl
            if no_eval:
                return [[0, u, v]], 0
            else:
                pot = ms - self.eval_transpose_potential(u, v, nbh)
                potentials.append([pot, u, v])
        else:   # eval reversal of all adjacent edge pairs
            if no_eval:
                # just gather args of possible moves
                return [[0, u, v] for u, v in zip(bl[:-1], bl[1:])], 0
            else:
                pots = [
                    [ms - self.eval_transpose_potential(u, v, nbh), u, v]
                    for u, v in zip(bl[:-1], bl[1:])
                ]
                pot = max([p[0] for p in pots])
            potentials += pots

        return potentials, pot

    def _ct(self, *args, nbh: PathNbh, **kwargs):
        """Critical Transpose.

        - Reverses an arc in a critical block.
        - Always feasible

        from:
            Van Laarhoven, P. J., Aarts, E. H., & Lenstra, J. K. (1992).
            Job shop scheduling by simulated annealing.
            Operations research, 40(1), 113-125.
        """
        u, v, *_ = args
        if self.verbose:
            print(f"CT: transpose {u}->{v}")
        self.graph.transpose_edge(u, v, nbh, **kwargs)

    def _cet_eval(self,
                  potentials: List,
                  cbl: np.ndarray,
                  cp: np.ndarray,
                  cv: Union[int, float, List],
                  ms: Union[int, float],
                  nbh: PathNbh,
                  valid_bl: np.ndarray,
                  idx: int,
                  **kwargs) -> Tuple[List, Union[int, float]]:
        bl = cbl[idx]
        if len(bl) == 2:    # just reverse single edge
            u, v = bl
            pot = ms - self.eval_transpose_potential(u, v, nbh)
            potentials.append([pot, u, v])
        else:   # try to reverse first or last edge in block
            u, v = bl[:2]
            pot1 = ms - self.eval_transpose_potential(u, v, nbh)
            potentials.append([pot1, u, v])
            u, v = bl[-2:]
            pot2 = ms - self.eval_transpose_potential(u, v, nbh)
            potentials.append([pot2, u, v])
            pot = max(pot1, pot2)

        return potentials, pot

    def _cet(self, *args, nbh: PathNbh, **kwargs):
        """Critical End Transpose.

        - Reverses critical arcs only at the beginning or
          at the end of a critical block.
        - Always feasible

        from:
            Nowicki E, Smutnicki. C.
            A fast taboo search algorithm for the job shop problem.
            Manag Sci 1996;42(6):797–813.
        """
        u, v, *_ = args
        if self.verbose:
            print(f"CET: transpose {u}->{v}")
        self.graph.transpose_edge(u, v, nbh, **kwargs)

    def _ecet_eval(self,
                  potentials: List,
                  cbl: np.ndarray,
                  cp: np.ndarray,
                  cv: Union[int, float, List],
                  ms: Union[int, float],
                  nbh: PathNbh,
                  valid_bl: np.ndarray,
                  idx: int,
                  **kwargs) -> Tuple[List, Union[int, float]]:
        bl = cbl[idx]
        if len(bl) == 2:    # just reverse single edge (same as CET)
            u, v = bl
            pot = ms - self.eval_transpose_potential(u, v, nbh)
            potentials.append([pot, u, v, 0, 0, None])
        else:   # try to reverse first or last edge in block
            u1, v1 = bl[:2]
            pot1 = ms - self.eval_transpose_potential(u1, v1, nbh)
            potentials.append([pot1, u1, v1, 0, 0, None])
            u2, v2 = bl[-2:]
            pot2 = ms - self.eval_transpose_potential(u2, v2, nbh)
            potentials.append([pot2, u2, v2, 0, 0, None])
            # or both
            pot3 = -float("inf")
            if len(bl) > 3:
                pot3 = pot1 + pot2
                potentials.append([pot3, u1, v1, u2, v2, bl])
            pot = max(pot1, pot2, pot3)

        return potentials, pot

    def _ecet(self, *args, nbh: PathNbh, **kwargs):
        """Extended Critical End Transpose.

        - Reverses critical arcs at the beginning or
          at the end of a critical block or both.
        - Always feasible

        from:
            Kuhpfahl, J., & Bierwirth, C. (2016).
            A study on local search neighborhoods for the job shop
            scheduling problem with total weighted tardiness objective.
            Computers & Operations Research, 66, 44-57.
        """
        u1, v1, u2, v2, bl, *_ = args
        # self.graph.transpose_edge(u1, v1, nbh, **kwargs)
        db_str = ""
        if u2 == 0 and v2 == 0:
            self.graph.transpose_edge(u1, v1, nbh, **kwargs)
        else:
            db_str = f" and {u2}->{v2}"
            self.graph.multi_transpose([(u1, v1), (u2, v2)], nbh, bl, **kwargs)
        if self.verbose:
            print(f"ECET: transpose {u1}->{v1}{db_str}")

    def _cei_eval(self,
                  potentials: List,
                  cbl: np.ndarray,
                  cp: np.ndarray,
                  cv: Union[int, float, List],
                  ms: Union[int, float],
                  nbh: PathNbh,
                  valid_bl: np.ndarray,
                  idx: int,
                  **kwargs) -> Tuple[List, Union[int, float]]:
        bl = cbl[idx]
        if len(bl) == 2:  # just single edge
            u, v = bl
            pot = ms - self.eval_transpose_potential(u, v, nbh)
            potentials.append([pot, u, v, True])
        else:
            cp = cp if self.select_crit == "greedy" else None
            # eval fw moves (u right after v)
            v = bl[-1]
            pots = [
                [ms - self.eval_fw_insert_potential(u, v, nbh, bl[i:]), u, v, True]
                for i, u in enumerate(bl[:-1])
                if self.fw_insert_feasible(u, v, nbh, cp=cp)
            ]
            # eval bw moves (v right before u)
            u = bl[0]
            rev_bl = np.flip(bl)
            pots += [
                [ms - self.eval_bw_insert_potential(u, v, nbh, rev_bl[i:]), u, v, False]
                for i, v in enumerate(rev_bl[:-1])
                if self.bw_insert_feasible(u, v, nbh, cp=cp)
            ]

            pot = max([p[0] for p in pots])
            potentials += pots

        return potentials, pot

    def _cei(self, *args, nbh: PathNbh, **kwargs):
        """Critical End Insertion.

        - Move an operation to the beginning or
          to the end of the critical block
          via forward or backward insertion (interchange)
        - Requires explicit feasibility check

        from:
            Balas, E., & Vazacopoulos, A. (1998).
            Guided local search with shifting bottleneck for job shop scheduling.
            Management science, 44(2), 262-275.
        """
        u, v, fw, *_ = args
        if fw:
            if self.verbose:
                print(f"CEI: fw insert {v}->{u}")
            self.graph.fw_insert(u, v, nbh, **kwargs)
        else:
            if self.verbose:
                print(f"CEI: bw insert {v}->{u}")
            self.graph.bw_insert(u, v, nbh, **kwargs)


# ============= #
# ### TEST #### #
# ============= #
def _test():
    import time
    from .generator import JSSPGenerator

    seed = 1234
    size = 10
    n_j = 10
    n_m = 10
    iters = 100
    verb = False
    plot = False
    reject = False
    perturb = True

    ls_ops = "ECET"
    pos = "ALL"

    gen = JSSPGenerator(seed=seed)
    instances = gen.generate("JSSP", size, num_jobs=n_j, num_machines=n_m)
    inst = instances[0]

    solver = JSSPSolver(pdr_method="FIFO",
                        search_criterion="best",
                        selection_criterion="sampling",
                        num_rnd=2,
                        verbose=verb)
    t = time.time()
    s, cost_min = solver.construct(inst)
    print(cost_min)

    for i in range(iters):
        if perturb and i % 5 == 0:  # just for testing
            solver.perturb()
        if reject:
            solver.checkpoint_solution()
        s, cost = solver.solve(ls_ops, position=pos)
        print(cost)
        if cost < cost_min:
            cost_min = cost
        if reject and i % 4 == 0:  # just for testing
            solver.reject_move()
        if plot:
            solver.graph.plot()
            time.sleep(0.5)

    t = time.time() - t
    #print(solver.graph.get_node_features())
    print(f"\nbest cost found ({t:.4f}s): {cost_min}")
