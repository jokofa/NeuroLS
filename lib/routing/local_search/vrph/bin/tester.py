# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 04:16:18 2021

@author: inbox
"""

import os, sys
import numpy as np
from scipy.spatial import distance_matrix
import pickle
import importlib
from pathlib import Path

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from data_utils import TSP_Generator, load_vrph
from definitions import *
import VRPH

importlib.reload(VRPH)


n = 100
gen = TSP_Generator(n)
data = gen.sample_instance()


model = VRPH.VRP(n-1)

VRPH.VRP.load_problem(
    model,
    0,                          # 0 for TSP, 1 for CVRP
    data['customers'].tolist(),   # coordinates
    [float(-1)],                # demands - none for TSP
    # [best_known_dist, capacity, max_route_length, normalize, neighborhood_size]
    [float(-1), float(-1), float(-1), float(-1), float(-1)],
    9,              # edge type
    -1,              # edge format
)

#model.read_TSPLIB_file("att48.tsp")


rnd = np.random.default_rng(133)
tour = rnd.permutation(n-1)+1
tour = np.append(tour, 0)

model.use_initial_solution([(tour).tolist()])
#model.call_clarke(1.1, False)
model.show_routes()
print(model.get_dummy_index())

accept_types =[VRPH_LI_ACCEPT, VRPH_FIRST_ACCEPT, VRPH_BEST_ACCEPT]
acceptors=["VRPH_LI_ACCEPT", "VRPH_FIRST_ACCEPT", "VRPH_BEST_ACCEPT"]

move_types = [TWO_OPT, THREE_OPT, ONE_POINT_MOVE, TWO_POINT_MOVE, OR_OPT, CROSS_EXCHANGE, THREE_POINT_MOVE]
movers =["TWO_OPT", "THREE_OPT", "ONE_POINT_MOVE", "TWO_POINT_MOVE", "OR_OPT", "CROSS_EXCHANGE", "THREE_POINT_MOVE"]

random = [False, True]
num_nodes = [1, 2, 3, -1]

for i_num, i  in enumerate(accept_types):
    for k in num_nodes:
        for l in random:
            for j_num, j in enumerate(move_types):
                heuristics = j
                if l:
                    rules = i + VRPH_FREE + VRPH_SAVINGS_ONLY+ VRPH_RANDOMIZED
                else:
                    rules = i + VRPH_FREE + VRPH_SAVINGS_ONLY
                print("\n Now testing "+ movers[j_num]+ " with " + acceptors[i_num] + " and num_nodes " + str(k))
                new_cost = model.detailed_solve(
                    heuristics,  # local_operators
                    rules,  # rule
                    [k],       # node (-1 for consider all nodes)
                    0,       # steps
                    100,  # iters
                    1e-5,  # err_max
                    False  # converge
                )
                print("test_done \n")
                model.accept_solution(True);
        #new_cost = model.Single_solve(heuristics, rules,  0.001,  1, False)

print("finished test")
model.show_routes()

