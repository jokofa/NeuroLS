#
import os
import warnings
from typing import Optional, Dict, NamedTuple, List, Tuple
from timeit import default_timer

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from lib.scheduling.formats import JSSPInstance, JSSPSolution, INF
from lib.scheduling.jssp_graph import JSSPGraph

PDR_IDS = ["RND", "FIFO", "SPT", "MWKR", "MOPNR", "FDD", "FDD_MWKR"]


class PDRState(NamedTuple):
    """Typed tuple to define components 
    of PDR rollout states."""
    mch_free: np.ndarray
    job_av: np.ndarray
    r_ij: np.ndarray
    i_q: np.ndarray
    p_qj: np.ndarray
    p_qj_remaining: np.ndarray
    ops_remaining: np.ndarray
    cum_p_qj: np.ndarray


class PDRRollout:

    def __init__(self,
                 instance: JSSPInstance,
                 debug: bool = False,
                 **kwargs):

        self.instance = instance

        num_jobs, num_machines, durations, sequences, _ = instance
        self.seq = sequences - 1   # make idx start at 0
        self.dur = durations
        self.graph = None
        self.debug = debug

        self.cur_time = None
        self.mch_free = None
        self.mch_pos = None
        self.mch_job = None
        self.job_pos = None
        self.op_time_left = None
        self.schedule = None
        self.cum_dur = None
        self.dur_remaining = None

        self._mch_range = np.arange(num_machines)
        self._job_range = np.arange(num_jobs)

    def finished(self):
        return np.all(self.schedule >= 0)

    def reset(self) -> PDRState:
        self.graph = JSSPGraph(self.instance, init_disjunctions=False)
        self.cur_time = 0
        self.mch_free = np.ones(self.graph.num_machines, dtype=np.bool)
        self.mch_pos = np.zeros(self.graph.num_machines, dtype=int)
        self.mch_job = -np.ones(self.graph.num_machines, dtype=int)
        self.job_pos = np.zeros(self.graph.num_jobs, dtype=int)
        self.op_time_left = np.ones((self.graph.num_machines, self.graph.num_jobs), dtype=int) * INF
        # set arrival time on first operation of each job to 0
        self.op_time_left[self.seq[:, 0], self._job_range] = 0
        self.schedule = -np.ones((self.graph.num_machines, self.graph.num_jobs), dtype=int)
        self.cum_dur = self.dur.cumsum(-1)
        self.dur_remaining = np.fliplr(np.fliplr(self.dur).cumsum(-1))

        return self._get_state()

    def step(self, j: int, i: int) -> PDRState:
        if self.mch_free[i]:
            # schedule job j on machine i
            pos = self.mch_pos[i]
            pred_j = None if pos == 0 else self.schedule[i, pos-1]
            assert pred_j is None or pred_j >= 0
            # the JSSP graph does all necessary feasibility checks
            dur = self.graph.schedule_j_on_i(j, i+1, pred_j)    # machine idx of graph starts at 1

            if dur == INF:  # infeasible move
                raise RuntimeError
            else:
                self.schedule[i, pos] = j
                self.mch_job[i] = j
                self.mch_pos[i] += 1
                self.mch_free[i] = False
                # set job duration on current machine
                self.op_time_left[i, j] = dur

        if not self.finished():
            job_av = True
            free = np.any(self.mch_free)
            if free:
                # check if there is any job available for one of the free machines
                job_av = np.any(self.op_time_left[self.mch_free] < INF)

            while not (free and job_av):

                # fast forward time until at least one machine is free again
                try:
                    max_step = self.op_time_left[(0 < self.op_time_left) & (self.op_time_left < INF)].min()
                except ValueError:
                    max_step = 0
                inf_msk = self.op_time_left < INF
                self.op_time_left[inf_msk] = self.op_time_left[inf_msk] - max_step
                # check finished jobs and update machine status
                fin = (self.op_time_left[self._mch_range, self.mch_job] <= 0)
                # set arrival time of job with finished operation for next machine
                fin_job = self.mch_job[fin]
                if self.debug:
                    assert np.all(fin_job >= 0)
                # if not np.all(fin_job >= 0):
                #     print(fin_job >= 0)

                self.mch_job[fin] = -1
                self.job_pos[fin_job] += 1
                # set time on finished machine to inf
                self.op_time_left[fin, fin_job] = INF
                fin_job = fin_job[self.job_pos[fin_job] < self.graph.num_machines]
                if len(fin_job) > 0:
                    fin_job_nxt_mch = self.seq[fin_job, self.job_pos[fin_job]]
                    self.op_time_left[fin_job_nxt_mch, fin_job] = 0

                # update machine status
                self.mch_free = (self.mch_job < 0) & np.any(self.schedule < 0, -1)
                self.cur_time += max_step

                free = np.any(self.mch_free)
                job_av = np.any(self.op_time_left[self.mch_free] < INF)

        if self.debug:
            assert np.all(((0 <= self.op_time_left) & (self.op_time_left < INF)).sum(0) <= 1)

        return self._get_state()

    def _get_state(self) -> PDRState:

        num_mch = self.graph.num_machines
        lim_pos = self.job_pos.copy()
        msk = lim_pos == num_mch
        lim_pos[msk] = lim_pos[msk]-1
        mch_at_q = self.seq[self._job_range, lim_pos]
        dur_at_q = self.dur[self._job_range, lim_pos]
        dur_at_q_rem = self.dur_remaining[self._job_range, lim_pos]
        cum_dur_at_q = self.cum_dur[self._job_range, lim_pos]
        mch_at_q[msk] = INF
        dur_at_q[msk] = INF
        dur_at_q_rem[msk] = -INF
        cum_dur_at_q[msk] = INF

        return PDRState(
            mch_free=self.mch_free,
            job_av=np.any(self.op_time_left[self.mch_free] < INF, -1),
            r_ij=self.op_time_left,
            i_q=mch_at_q,
            p_qj=dur_at_q,
            p_qj_remaining=dur_at_q_rem,
            ops_remaining=(num_mch-self.job_pos),
            cum_p_qj=cum_dur_at_q,
        )

    def result(self) -> Tuple[np.ndarray, np.ndarray, JSSPGraph]:
        _, make_span = self.graph.longest_path_seq_val()
        if self.debug:
            v = self.cur_time + self.op_time_left[(0 < self.op_time_left) & (self.op_time_left < INF)].max()
            assert make_span == v
        return self.schedule, make_span, self.graph


class PriorityDispatchingRule:
    """Implementing a selection of the PDRs presented in:

    Sels, V., Gheysen, N., & Vanhoucke, M. (2012).
    A comparison of priority rules for the job shop scheduling problem
    under different flow time-and tardiness-related objective functions.
    International Journal of Production Research, 50(15), 4255-4270.

    """
    def __init__(self, method: str = "FIFO"):
        self.method = method.lower()
        self._rnds = np.random.default_rng(1)

    def seed(self, seed: Optional[int] = None):
        self._rnds = np.random.default_rng(seed)

    def dispatch(self,
                 instance: JSSPInstance,
                 randomize: bool = False,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray, JSSPGraph]:
        """Apply PDR to provided problem instance.

        Args:
            instance: the JSSP instance to solve
            randomize: randomize PDR rollout by sampling from best three operations
                       instead of greedily selecting locally best, proposed by:
                            Lourenço, H. R. (1995).
                            Job-shop scheduling: Computational study of local search
                            and large-step optimization methods.
                            European Journal of Operational Research, 83(2), 347-364.
        """
        try:
            pdr = getattr(self, f"_{self.method}")
        except AttributeError:
            raise ModuleNotFoundError(f"The corresponding method '{self.method}' does not exist.")

        sim = PDRRollout(instance, **kwargs)
        state = sim.reset()
        while not sim.finished():
            i, j = pdr(state, randomize, **kwargs)
            state = sim.step(j, i)
        return sim.result()

    def _rnd(self, state: PDRState, randomize: bool = False, **kwargs):
        """Random schedule as baseline."""
        r_ij = state.r_ij
        # select rnd machine which is free
        i = self._rnds.choice(state.mch_free.nonzero()[0][state.job_av], 1)[0]
        # select rnd job to schedule
        j = self._rnds.choice((r_ij[i] != INF).nonzero()[0], 1)[0]
        return i, j

    def _fifo(self, state: PDRState, randomize: bool = False, **kwargs):
        """First in first out."""
        r_ij = state.r_ij
        # select first machine which is free
        i = state.mch_free.nonzero()[0][state.job_av][0]
        # select job to schedule
        # (first ready to be scheduled on machine)
        v = r_ij[i]
        if randomize and (v != INF).sum() >= 3:
            j = self._rnds.choice(np.argpartition(v, 2)[:3], 1)[0]
        else:
            j = v.argmin()
        return i, j

    def _spt(self, state: PDRState, randomize: bool = False, **kwargs):
        """Shortest processing time first."""
        i_q, p_qj, = state.i_q, state.p_qj
        # select first machine which is free
        i = state.mch_free.nonzero()[0][state.job_av][0]
        # select job to schedule
        # (ready job with shortest processing time on selected machine)
        slct = i_q == i
        v = p_qj[slct]
        if randomize and len(v) >= 3:
            j = self._rnds.choice(np.argpartition(v, 2)[:3], 1)[0]
        else:
            j = v.argmin()
        j = slct.nonzero()[0][j]
        return i, j

    def _mwkr(self, state: PDRState, randomize: bool = False, **kwargs):
        """Most Work Remaining."""
        i_q, p_qj_rem, = state.i_q, state.p_qj_remaining
        # select first machine which is free
        i = state.mch_free.nonzero()[0][state.job_av][0]
        # select job to schedule
        # (ready job with most work (=longest cumulative processing time) remaining)
        # O(j) = num_machines in our case
        slct = i_q == i
        v = p_qj_rem[slct]
        if randomize and len(v) >= 3:
            j = self._rnds.choice(np.argpartition(v, -2)[-3:], 1)[0]
        else:
            j = v.argmax()
        j = slct.nonzero()[0][j]
        return i, j

    def _mopnr(self, state: PDRState, randomize: bool = False, **kwargs):
        """Most Operations Remaining."""
        i_q, ops_rem, = state.i_q, state.ops_remaining
        # select first machine which is free
        i = state.mch_free.nonzero()[0][state.job_av][0]
        # select job to schedule
        # (ready job with highest number of operations remaining)
        # O(j) = num_machines in our case
        slct = i_q == i
        v = ops_rem[slct]
        if randomize and len(v) >= 3:
            j = self._rnds.choice(np.argpartition(v, -2)[-3:], 1)[0]
        else:
            j = v.argmax()
        j = slct.nonzero()[0][j]
        return i, j

    def _fdd(self, state: PDRState, randomize: bool = False, **kwargs):
        """Flow Due Date."""
        i_q, cum_p_qj, = state.i_q, state.cum_p_qj
        # select first machine which is free
        i = state.mch_free.nonzero()[0][state.job_av][0]
        # select job to schedule
        # (ready job with smallest flow due date (=cumulative processing time until q))
        slct = i_q == i
        v = cum_p_qj[slct]
        if randomize and len(v) >= 3:
            j = self._rnds.choice(np.argpartition(v, 2)[:3], 1)[0]
        else:
            j = v.argmin()
        j = slct.nonzero()[0][j]
        return i, j

    def _fdd_mwkr(self, state: PDRState, randomize: bool = False, **kwargs):
        """Ratio of Flow Due Date to Most Work Remaining (minimize!)."""
        i_q, p_qj_rem, cum_p_qj, = state.i_q, state.p_qj_remaining, state.cum_p_qj
        # select first machine which is free
        i = state.mch_free.nonzero()[0][state.job_av][0]
        # select job to schedule
        # (ready job with smallest ratio of FDD and MWKR )
        slct = i_q == i
        v = (cum_p_qj[slct]/p_qj_rem[slct])
        if randomize and len(v) >= 3:
            j = self._rnds.choice(np.argpartition(v, 2)[:3], 1)[0]
        else:
            j = v.argmin()
        j = slct.nonzero()[0][j]
        return i, j


class ParallelSolver:
    """Parallelization wrapper for PDR based on multi-processing pool."""
    def __init__(self,
                 solver_args: Optional[Dict] = None,
                 num_workers: int = 1,
                 ):
        self.solver_args = solver_args if solver_args is not None else {}
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
        solver_cl, data, solver_args = params
        solver = solver_cl(solver_args.get("method"))
        solver.seed(solver_args.get("seed"))
        t = default_timer()
        solution, cost, _ = solver.dispatch(data)
        t = default_timer() - t
        return [solution, cost, t]

    def solve(self, data: List[JSSPInstance]) -> List[JSSPSolution]:

        assert isinstance(data[0], JSSPInstance)

        if self.num_workers <= 1:
            results = list(tqdm(
                [self._solve((PriorityDispatchingRule, d, self.solver_args)) for d in data],
                total=len(data)
            ))
        else:
            with Pool(self.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(
                        self._solve,
                        [(PriorityDispatchingRule, d, self.solver_args) for d in data]
                    ),
                    total=len(data),
                ))

            failed = [str(i) for i, res in enumerate(results) if res is None]
            if len(failed) > 0:
                warnings.warn(f"Some instances failed: {failed}")

        return [
            JSSPSolution(
                solution=r[0],
                cost=r[1],
                run_time=r[2],
                instance=d,
            )
            for d, r in zip(data, results)
        ]


# ============= #
# ### TEST #### #
# ============= #
def _test():
    import time
    from .generator import JSSPGenerator

    seed = 1234
    randomize = False
    size = 10
    n_j = 15
    n_m = 15
    gen = JSSPGenerator(seed=seed)
    instances = gen.generate("JSSP", size, num_jobs=n_j, num_machines=n_m)

    summary = {id: {"cost": [], "time": []} for id in PDR_IDS}
    for inst in instances:
        for mthd in PDR_IDS:

            pdr = PriorityDispatchingRule(mthd)
            pdr.seed(seed)
            t = time.time()
            solution, cost, graph = pdr.dispatch(inst, randomize=randomize, debug=True)
            t = time.time() - t
            # print(solution)
            # print(cost)
            # print(f"took: {t}s")
            # print(graph.longest_path_idx())
            summary[mthd]["cost"].append(cost)
            summary[mthd]["time"].append(t)

    print("\n_______________________________________")
    for mthd, res in summary.items():
        print(f"{mthd:>8}: cost={np.mean(res['cost']):.3f}, time={np.mean(res['time']):.8f}")
