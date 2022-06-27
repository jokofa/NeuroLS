
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:01:51 2021

@author: inbox
"""
import sys 
import os

import VRPH as VRPH
from VRPH import VRP

import numpy as np
from definitions import *
import time



details = [""]

rules = VRPH_BEST_ACCEPT + VRPH_SAVINGS_ONLY + VRPH_DOWNHILL

start = time.time()
results = []

for i in range(0,1):
    #filename = "../data/tsp10/tsp10_num_"+ str(i) +"_seed1357.vrp"
    filename = "../../att48.vrp"

    n = VRPH.VRPGetDimension(filename)
    model = VRP(n)
    initial_sol = []

    for i in range(1,5):
        part = list(range(10*i - 9,10*i+1))
        initial_sol.append(part)

    initial_sol.append(list(range(41, 48)))

    model.read_TSPLIB_file(filename)
    model.use_initial_solution(initial_sol) 

    print("Current Length: " + str(model.get_total_route_length()))
    print("Best Length: " + str(model.get_best_total_route_length()))

    model.show_routes()

    for i in range (100):

        #print(model.print_stats())
        print("Current Length: " + str(model.get_total_route_length()))
        print("Best Length: " + str(model.get_best_total_route_length()))

        heuristics = TWO_OPT 

        model.Single_solve(heuristics, rules, 0.000001, 2, False)


        print("Current Length: " + str(model.get_total_route_length()))
        print("Best Length: " + str(model.get_best_total_route_length()))

        heuristics = THREE_OPT 

        model.Single_solve(heuristics, rules, 0.000001, 2, False)

        #print(model.print_stats())

        print("Current Length: " + str(model.get_total_route_length()))
        print("Best Length: " + str(model.get_best_total_route_length()))


#end = time.time()
#print(end - start)