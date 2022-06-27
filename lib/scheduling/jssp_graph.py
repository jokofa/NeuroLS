#
from typing import Optional, Union, List, Iterable, Tuple, NamedTuple, Dict
from collections import deque
from warnings import warn
from copy import deepcopy
import itertools as it

import matplotlib.axes
import numpy as np
import networkx as nx
from networkx.utils import pairwise
from networkx.exception import NetworkXError
import matplotlib.pyplot as plt
from matplotlib import cm

from lib.scheduling.formats import JSSPInstance, INF


class PathNbh(NamedTuple):
    pth: Dict[int, Dict]
    pj: Dict[int, Dict]
    pm: Dict[int, Dict]
    sj: Dict[int, Dict]
    sm: Dict[int, Dict]


class JSSPGraph:
    """(Disjunctive) Graph object for the JSSP.

    Graph G(V, C, D, W):
    # V: set of nodes representing operations + 2 dummy nodes for start and end
    # C: set of directed edges (conjunctive) representing precedence relations
         on the order of operations for each job,
         also connecting to the start and end dummy nodes
    # D: set of pairs of directed edges (disjunctive) between operations to be performed on the same machine
    # W: set of weights given by the processing times

    """
    INT = int
    JOB = 0
    MCH = 1

    def __init__(self,
                 instance: JSSPInstance,
                 init_disjunctions: bool = False,
                 **kwargs):

        num_jobs, num_machines, durations, sequences, _ = instance

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.graph_size = num_jobs * num_machines + 2

        self._src_idx = 0
        self._snk_idx = self.graph_size-1
        self.has_disjunctions = False

        # conjunctive edge adjacency matrix
        c = np.eye(self.graph_size, k=1, dtype=np.bool)
        from_src = np.arange(start=1, stop=self.graph_size-1, step=num_machines)
        to_snk = from_src + (num_machines-1)
        c[0, from_src] = 1
        c[to_snk] = 0
        c[to_snk, -1] = 1

        # disjunctive edge adjacency matrix
        assert sequences.shape == (num_jobs, num_machines)
        assert sequences.min() == 1 and sequences.max() == num_machines
        self.sequences = sequences
        d = None
        if init_disjunctions:
            d = np.zeros((self.graph_size, self.graph_size), dtype=np.bool)
            # create machine cliques
            for mch_idx in range(1, num_machines+1):
                # idx of all operations which need to be scheduled on machine with mch_idx
                idx = (sequences == mch_idx).nonzero()[1] + from_src
                rw, cl = np.repeat(idx, num_jobs), np.tile(idx, num_jobs)
                d[rw, cl] = 1
            np.fill_diagonal(d, 0)  # no self loops
            self.has_disjunctions = True

        # weights
        z = np.zeros(1, dtype=durations.dtype)
        self.w_node = np.concatenate([z, durations.reshape(-1), z]).astype(np.single)

        self.job_start_node_idx = from_src
        self.mch_src_idx = np.zeros(num_machines, dtype=self.INT)

        o = -np.ones(1, dtype=self.INT)
        self.nodes = np.stack([
            np.arange(self.graph_size),     # node idx
            np.concatenate([o, np.arange(self.num_jobs).repeat(self.num_machines), o]),   # job idx
            np.concatenate([o, self.sequences.reshape(-1), o]),     # mch idx
            self.w_node     # processing time
        ], axis=-1).astype(np.single)

        # create graph representation
        weights = np.repeat(self.w_node, self.graph_size).reshape(self.graph_size, self.graph_size)
        self._job_graph_e = self._to_coo(c)
        self._job_graph_w = weights[c].astype(np.single)
        edge_list = self._job_graph_e.tolist()
        self.graph = nx.DiGraph([
            (u, v, {'weight': float(w), 'set': self.JOB}) for u, v, w in
            zip(edge_list[0], edge_list[1], self._job_graph_w.tolist())
        ])
        if d is not None:
            edge_list = self._to_coo(d).tolist()
            self.graph.add_edges_from([
                (u, v, {'weight': float(w), 'set': self.MCH})
                for u, v, w in zip(edge_list[0], edge_list[1], weights[d])
            ])
        # add processing times as node weights
        nx.set_node_attributes(self.graph, {
            i: {'weight': float(w)}
            for i, w in enumerate(self.w_node.tolist())     # type: ignore
        })
        assert nx.is_directed(self.graph)
        assert nx.is_weighted(self.graph)
        self._recompute_dists = True

    def _ops_idx(self, j: int, i: int) -> int:
        """Return the idx of operation O_ji of job j on machine i."""
        assert i > 0
        pos = (self.sequences[j] == i).nonzero()[0]
        return int(self.job_start_node_idx[j] + pos)

    def insert_no_cycle(self, u: int, v: int, i: int) -> bool:
        """Check if the insertion of an edge between nodes u and v
        on machine i leads to the creation of a cycle."""
        if self.mch_src_idx[i-1] == self._src_idx:
            # if it is the first operation scheduled on machine i, no problem
            return True
        else:
            # use DFS to check if there is a path from v to u.
            # if not, it won't create a cycle!
            return u not in set(nx.dfs_preorder_nodes(self.graph, source=v))

    def is_dag(self) -> bool:
        """Check if the graph is a DAG"""
        if self.has_disjunctions:
            return False
        return nx.is_directed_acyclic_graph(self.graph)

    def _longest_path(self,
                      w_key: str = 'weight',
                      topo_order=None,
                      set_dist_from_src: bool = False,
                      **kwargs):
        if topo_order is None:
            # return the node sequence of the longest path
            # based on the default topological order
            topo_order = nx.topological_sort(self.graph)

        dist = {}  # stores {v : (length, u)}
        for v in topo_order:
            us = [
                (dist[u][0] + data.get(w_key, 1), u)
                for u, data in self.graph.pred[v].items()
            ]
            # Use the best predecessor if there is one and its distance is
            # non-negative, otherwise terminate.
            maxu = max(us, key=lambda x: x[0]) if us else (0, v)
            dist[v] = maxu if maxu[0] >= 0 else (0, v)

        if set_dist_from_src or self._recompute_dists:
            for n, v in dist.items():
                self.graph.nodes[n]["dist_from_src"] = float(v[0])

        u = None
        v = max(dist, key=lambda x: dist[x][0])
        path = []
        while u != v:
            path.append(v)
            u = v
            v = dist[v][1]

        path.reverse()
        return path

    def longest_path_seq(self, w_key: str = 'weight', **kwargs) -> List:
        if self.has_disjunctions:
            return []
        assert self.is_dag()
        return self._longest_path(w_key, **kwargs)

    def longest_path_seq_val(self,
                             w_key: str = 'weight',
                             ts: Optional[Iterable] = None,
                             stepwise_val: bool = False,
                             **kwargs) -> Tuple[List, Union[int, float, List]]:
        """Returns longest path AND its length,
        without re-computing longest path sequence."""
        if self.has_disjunctions:
            return [], float("inf")
        assert self.is_dag()
        path = self._longest_path(w_key, topo_order=ts, **kwargs)
        if stepwise_val:
            return path, [self.graph[u][v].get(w_key, 1) for (u, v) in pairwise(path)]
        else:
            path_length = 0
            for (u, v) in pairwise(path):
                path_length += self.graph[u][v].get(w_key, 1)
            return path, path_length

    def critical_paths(self, max_num: int = 3
                       ) -> Tuple[List[List], List]:
        if self.has_disjunctions:
            raise NotImplementedError
        seqs, vals, ts_unq = deque(maxlen=max_num), deque(maxlen=max_num), deque(maxlen=max_num)
        i = 0
        for ts in nx.all_topological_sorts(self.graph):
            if ts not in ts_unq:
                ts_unq.append(ts)
                seq, val = self.longest_path_seq_val(ts=ts)
                if seq not in seqs:
                    seqs.append(seq)
                    vals.append(val)
                    assert vals[0] == val
                    i += 1
                    if i == max_num-1:
                        break
        return list(seqs), list(vals)

    def calc_dist_from_src(self,
                           w_key: str = 'weight',
                           topo_order=None,
                           **kwargs):
        self._longest_path(w_key, topo_order, set_dist_from_src=True, **kwargs)

    def calc_dist_to_snk(self,
                         w_key: str = 'weight',
                         **kwargs):
        dist = {}
        for v in nx.topological_sort(self.graph.reverse()):     # topological sort from sink to source
            us = [dist[n] for n in self.graph.succ[v].keys()]
            maxu = max(us) if us else 0
            dist[v] = self.graph.nodes[v].get(w_key, 0) + (maxu if maxu >= 0 else 0)

        for n, v in dist.items():
            self.graph.nodes[n]["dist_to_snk"] = v

    def get_critical_blocks(self) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           Union[int, float, List],
                                           PathNbh]:
        """Split critical path into blocks of operations on same machine.

        Let P be a critical path in G. A sequence of
        successive nodes in P is called a block on P in G if the
        following properties are satisfied:

        - The sequence contains at least two nodes
        - All operations represented by the nodes in the sequence are assigned to the same machine.
        - Enlarging the sequence by one operation yields a sequence which does not fulfill the second property.
        """
        crit_path, crit_vals = self.longest_path_seq_val(stepwise_val=True)
        op_on_mch = self.nodes[crit_path, 2].astype(int)    # 2 is column of machine idx
        # make sure node attributes include actual dist to sink.
        # dist from source was already updated during call to longest_path_seq_val()
        if self._recompute_dists:
            self.calc_dist_to_snk()
        nodes = self.graph.nodes(data=True)

        return (
            np.array([  # critical blocks
                np.array(list(l))[:, 0] for _, l in
                it.groupby(zip(crit_path, op_on_mch), key=lambda x: x[1])
            ], dtype=object),
            np.array(crit_path),    # critical path
            crit_vals,  # weights of nodes in critical path
            PathNbh(  # nodes in critical path and their machine and job predecessors and successors
                pth={n: nodes[n] for n in crit_path},
                pj={n: {
                    idx: nodes[idx] for idx in self.graph.predecessors(n)
                    if self.graph[idx][n].get("set") == self.JOB
                } for n in crit_path},
                pm={n: {
                    idx: nodes[idx] for idx in self.graph.predecessors(n)
                    if self.graph[idx][n].get("set") == self.MCH
                } for n in crit_path},
                sj={n: {
                    idx: nodes[idx] for idx in self.graph.successors(n)
                    if self.graph[n][idx].get("set") == self.JOB
                } for n in crit_path},
                sm={n: {
                    idx: nodes[idx] for idx in self.graph.successors(n)
                    if self.graph[n][idx].get("set") == self.MCH
                } for n in crit_path}
            )
        )

    def _to_coo(self, adj: np.ndarray) -> np.ndarray:
        """Convert dense adj matrix to coo sparse format."""
        return np.stack(adj.nonzero())

    def get_job_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return job graph adj + weights in coo format."""
        # since it does not change, we can cache it
        return self._job_graph_e, self._job_graph_w

    def get_mch_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return machine graph adj + weights in coo format."""
        # changes with each valid LS move, so we need to re-collate
        d = np.array([
            (u, v, e['weight'])
            for u, v, e in self.graph.edges(data=True)
            if e['set'] == self.MCH
        ], dtype=np.single)
        return d[:, :2].T.astype(int), d[:, -1]

    def get_mch_seq(self) -> np.ndarray:
        """Return the idx sequence of jobs on each machine."""
        seqs = []
        for h in self.mch_src_idx:
            assert len(list(self.graph.predecessors(h))) <= 1, \
                f"idx {h} is not src of machine graph."
            u = h
            s = [u]
            vs = list(self.graph.successors(u))
            while len(vs) >= 2:     # as long as there is a job and a mch successor
                v = [v for v in vs if self.graph.get_edge_data(u, v).get('set') == self.MCH][0]
                s.append(v)
                u = v
                vs = list(self.graph.successors(u))
            assert len(s) == self.num_jobs
            seqs.append(s)
        return np.array(seqs)

    def get_node_features(self) -> np.ndarray:
        # normally called after move execution, so need to recompute distances
        # dist_from_src already recomputed when calculating makespan
        self.calc_dist_to_snk()
        self._recompute_dists = False
        # static self.nodes and dynamic dist_from_src and dist_to_snk
        dyn = np.array([
            [i, n["dist_from_src"], n["dist_to_snk"]]
            for i, n in self.graph.nodes(data=True)
        ], dtype=np.single)
        return np.concatenate((
            self.nodes,
            dyn[np.argsort(dyn[:, 0]), 1:]  # reorder by node idx
        ), axis=-1)

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None, _show: bool = True):
        """Plot the current graph."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            _show = False
        # infer node position grid
        x_pos = np.linspace(0.0, 1.0, self.num_machines)
        y_pos = np.linspace(1.0, 0.0, self.num_jobs)
        d = 1.5*(x_pos[1]-x_pos[0])
        n_size = d*500
        color_map = cm.get_cmap('viridis', self.num_machines)

        node_pos = {self._src_idx: (0.0-d, 0.5), self._snk_idx: (1.0+d, 0.5)}
        node_labels = {self._src_idx: f"src", self._snk_idx: f"snk"}
        grey = (0.0, 0.0, 0.0, 0.3)
        node_colors = [grey]
        x_idx = 0
        y_idx = -1
        for n in range(0, self.graph_size-2):
            if n % self.num_machines == 0:
                x_idx = 0
                y_idx += 1
            node_pos[n+1] = (x_pos[x_idx], y_pos[y_idx])
            x_idx += 1
            node_labels[n+1] = str(self.nodes[n+1, 3])
            node_colors.append(color_map(int(self.nodes[n+1, 2]-1)))
        node_colors.append(grey)
        nx.draw_networkx_nodes(self.graph,
                               pos=node_pos,
                               nodelist=list(range(self.graph_size)),
                               node_size=n_size,
                               node_color=node_colors,
                               ax=ax)
        nx.draw_networkx_labels(self.graph,
                                pos=node_pos,
                                labels=node_labels,
                                font_size=d*17,
                                ax=ax)

        edges_data = self.graph.edges(data=True)
        job_edges = [(u, v) for u, v, e in edges_data if e['set'] == self.JOB]
        nx.draw_networkx_edges(self.graph,
                               pos=node_pos,
                               edgelist=job_edges,
                               edge_color="k",
                               style="solid",
                               node_size=n_size,
                               alpha=0.25,
                               ax=ax)

        mch_edges = [(u, v) for u, v, e in edges_data if e['set'] == self.MCH]
        mch_of_edge = [int(self.nodes[e[0], 2]) for e in mch_edges]
        mch_edge_colors = [color_map(e-1) for e in mch_of_edge]
        nx.draw_networkx_edges(self.graph,
                               pos=node_pos,
                               edgelist=mch_edges,
                               edge_color=mch_edge_colors,
                               style="dashed",
                               connectionstyle=f"arc3, rad={0.5*d}",
                               node_size=n_size,
                               ax=ax)
        if _show:
            plt.show()
            plt.clf()

    def schedule_j_on_i(self, j: int, i: int, pred_j: int):
        """Helper function for PDRs."""
        if self.has_disjunctions:
            raise NotImplementedError
        assert i > 0
        j_idx = self._ops_idx(j, i)
        if pred_j is None:  # first job on machine
            assert self.mch_src_idx[i-1] == self._src_idx
            self.mch_src_idx[i-1] = j_idx
        else:
            pred_j_idx = self._ops_idx(pred_j, i)
            if self.insert_no_cycle(pred_j_idx, j_idx, i):
                self.graph.add_edge(pred_j_idx, j_idx, weight=float(self.w_node[pred_j_idx]), set=self.MCH)
            else:
                warn("encountered cycle. continue...")
                return INF
        self._recompute_dists = True
        return self.w_node[j_idx]

    def state_dict(self) -> nx.DiGraph:
        """Return state just as deepcopy of the nx graph dict."""
        return deepcopy(self.graph)

    def load_state_dict(self, state: nx.DiGraph):
        """Load a previously saved graph state."""
        assert isinstance(state, nx.DiGraph)
        self.graph = state

    def transpose_edge(self,
                       u: int,
                       v: int,
                       nbh: Optional[PathNbh] = None,
                       **kwargs):
        if nbh is None:
            raise NotImplementedError
        # get machine predecessor and successor
        pm_u = nbh.pm[u]
        pm_idx, pm_attr = next(iter(pm_u.items())) if len(pm_u) > 0 else (None, {})
        sm_v = nbh.sm[v]
        sm_idx, sm_attr = next(iter(sm_v.items())) if len(sm_v) > 0 else (None, {})
        # reverse u->v
        self.graph.remove_edge(u, v)
        self.graph.add_edge(v, u, weight=nbh.pth[v].get("weight"), set=self.MCH)
        # change pred
        if pm_idx is not None:
            try:
                self.graph.remove_edge(pm_idx, u)
            except NetworkXError as nxe:
                warn(str(nxe))
            self.graph.add_edge(pm_idx, v, weight=pm_attr.get("weight", 0), set=self.MCH)
        else:
            # u was src node of one of the machine graphs
            mch_idx = (u == self.mch_src_idx)
            assert np.any(mch_idx)
            mch_idx = mch_idx.nonzero()[0]
            self.mch_src_idx[mch_idx] = v

        # change succ
        if sm_idx is not None:
            try:
                self.graph.remove_edge(v, sm_idx)
            except NetworkXError as nxe:
                warn(str(nxe))
            self.graph.add_edge(u, sm_idx, weight=nbh.pth[u].get("weight"), set=self.MCH)
        self._recompute_dists = True

    def multi_transpose(self,
                        node_pairs: List[Tuple[int, int]],
                        nbh: Optional[PathNbh] = None,
                        bl: Optional[np.ndarray] = None,
                        **kwargs):
        """Transpose multiple edges at the same time."""
        if nbh is None:
            raise NotImplementedError
        if len(node_pairs) > 2:  # for ICT move...
            raise NotImplementedError

        (u1, v1), (u2, v2) = node_pairs
        if len(bl) > 5:
            # no special attention necessary when more than 5 nodes present
            self.transpose_edge(u1, v1, nbh)
            self.transpose_edge(u2, v2, nbh)
        else:
            # remove intra block edges
            for u, v in pairwise(bl):
                try:
                    self.graph.remove_edge(u, v)
                except NetworkXError as nxe:
                    warn(str(nxe))
            # transpose
            self.graph.add_edge(v1, u1, weight=nbh.pth[v1].get("weight"), set=self.MCH)
            self.graph.add_edge(v2, u2, weight=nbh.pth[v2].get("weight"), set=self.MCH)
            # adapt edge into block
            pm_u = nbh.pm[u1]
            pm_u_idx, pm_u_attr = next(iter(pm_u.items())) if len(pm_u) > 0 else (None, {})
            if pm_u_idx is not None:
                try:
                    self.graph.remove_edge(pm_u_idx, u1)
                except NetworkXError as nxe:
                    warn(str(nxe))
                self.graph.add_edge(pm_u_idx, v1, weight=pm_u_attr.get("weight"), set=self.MCH)
            else:
                # u1 was src node of one of the machine graphs
                mch_idx = (u1 == self.mch_src_idx)
                assert np.any(mch_idx)
                mch_idx = mch_idx.nonzero()[0]
                self.mch_src_idx[mch_idx] = v1

            # adapt edge out of block
            sm_v = nbh.sm[v2]
            sm_v_idx, sm_v_attr = next(iter(sm_v.items())) if len(sm_v) > 0 else (None, {})
            if sm_v_idx is not None:
                try:
                    self.graph.remove_edge(v2, sm_v_idx)
                except NetworkXError as nxe:
                    warn(str(nxe))
                self.graph.add_edge(u2, sm_v_idx, weight=nbh.pth[u2].get("weight"), set=self.MCH)
            # fix intra block edges
            if len(bl) == 4:
                self.graph.add_edge(u1, v2, weight=nbh.pth[u1].get("weight"), set=self.MCH)
            elif len(bl) == 5:
                m = bl[2]
                self.graph.add_edge(u1, m, weight=nbh.pth[u1].get("weight"), set=self.MCH)
                self.graph.add_edge(m, v2, weight=nbh.pth[m].get("weight"), set=self.MCH)
            else:
                raise RuntimeError()
        self._recompute_dists = True

    def fw_insert(self,
                  u: int,
                  v: int,
                  nbh: Optional[PathNbh] = None,
                  **kwargs):
        # insert u after v
        if nbh is None:
            raise NotImplementedError
        # get machine predecessor and successor
        pm_u = nbh.pm[u]
        pm_u_idx, pm_u_attr = next(iter(pm_u.items())) if len(pm_u) > 0 else (None, {})
        sm_u = nbh.sm[u]
        sm_u_idx, sm_u_attr = next(iter(sm_u.items())) if len(sm_u) > 0 else (None, {})
        sm_v = nbh.sm[v]
        sm_v_idx, sm_v_attr = next(iter(sm_v.items())) if len(sm_v) > 0 else (None, {})
        if pm_u_idx is not None:
            # remove pm_u->u
            self.graph.remove_edge(pm_u_idx, u)
            # insert pm_u->sm_u
            self.graph.add_edge(pm_u_idx, sm_u_idx, weight=pm_u_attr.get("weight", 0), set=self.MCH)
        else:
            # u was src node of one of the machine graphs
            mch_idx = (u == self.mch_src_idx)
            assert np.any(mch_idx)
            mch_idx = mch_idx.nonzero()[0]
            self.mch_src_idx[mch_idx] = sm_u_idx

        # remove u->sm_u
        self.graph.remove_edge(u, sm_u_idx)
        if sm_v_idx is not None:
            # remove v->sm_v
            self.graph.remove_edge(v, sm_v_idx)
            # insert u->sm_v
            self.graph.add_edge(u, sm_v_idx, weight=nbh.pth[u].get("weight"), set=self.MCH)
        # insert v->u and u->sm_v
        self.graph.add_edge(v, u, weight=nbh.pth[v].get("weight"), set=self.MCH)
        self._recompute_dists = True

    def bw_insert(self,
                  u: int,
                  v: int,
                  nbh: Optional[PathNbh] = None,
                  **kwargs):
        # insert v before u
        if nbh is None:
            raise NotImplementedError
        # get machine predecessor and successor
        pm_u = nbh.pm[u]
        pm_u_idx, pm_u_attr = next(iter(pm_u.items())) if len(pm_u) > 0 else (None, {})
        pm_v = nbh.pm[v]
        pm_v_idx, pm_v_attr = next(iter(pm_v.items())) if len(pm_v) > 0 else (None, {})
        sm_v = nbh.sm[v]
        sm_v_idx, sm_v_attr = next(iter(sm_v.items())) if len(sm_v) > 0 else (None, {})
        if sm_v_idx is not None:
            # remove v->sm_v
            self.graph.remove_edge(v, sm_v_idx)
            # insert pm_v->sm_v
            self.graph.add_edge(pm_v_idx, sm_v_idx, weight=pm_v_attr.get("weight", 0), set=self.MCH)
        # remove pm_v->v
        self.graph.remove_edge(pm_v_idx, v)
        if pm_u_idx is not None:
            # remove pm_u->u
            self.graph.remove_edge(pm_u_idx, u)
            # insert pm_u->v
            self.graph.add_edge(pm_u_idx, v, weight=pm_u_attr.get("weight", 0), set=self.MCH)
        else:
            # u was src node of one of the machine graphs
            mch_idx = (u == self.mch_src_idx)
            assert np.any(mch_idx)
            mch_idx = mch_idx.nonzero()[0]
            self.mch_src_idx[mch_idx] = v

        # insert v->u
        self.graph.add_edge(v, u, weight=nbh.pth[v].get("weight"), set=self.MCH)
        self._recompute_dists = True


# ============= #
# ### TEST #### #
# ============= #
def _test():
    from .jssp_pdr import PriorityDispatchingRule
    from .generator import JSSPGenerator

    seed = 1234
    size = 4
    n_j = 10
    n_m = 8
    gen = JSSPGenerator(seed=seed)
    instances = gen.generate("JSSP", size, num_jobs=n_j, num_machines=n_m)
    inst = instances[0]

    pdr = PriorityDispatchingRule()
    pdr.seed(seed)
    solution, cost, g = pdr.dispatch(inst, debug=True)
    g.calc_dist_from_src()
    g.calc_dist_to_snk()

    print(g.get_critical_blocks())
    print(nx.get_node_attributes(g.graph, "weight"))
    print(nx.get_node_attributes(g.graph, "dist_from_src"))
    print(nx.get_node_attributes(g.graph, "dist_to_snk"))
    print(g.get_node_features())
