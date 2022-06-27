import gym
from gym import spaces
import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

#import VRPH as VRPH
#from VRPH import VRP

from data_utils import TSP_Generator
from construction_heuristics import random_tsp_tour
from definitions import *
from VRPH_utils import load_tsp
from stable_baselines3.common.env_checker import check_env

ACTIONS = [
  ONE_POINT_MOVE,
 # TWO_POINT_MOVE,
  #TWO_OPT,
  THREE_POINT_MOVE,
  THREE_OPT]
  

ACCEPTANCE_CRITERION = [
  VRPH_FIRST_ACCEPT,
  VRPH_LI_ACCEPT,
  VRPH_BEST_ACCEPT]

ACTIONS_NAMES = [
  'ONE_POINT_MOVE',
  #'TWO_POINT_MOVE',
  #'TWO_OPT',
  'THREE_POINT_MOVE',
  'THREE_OPT']

ACCEPTANCE_CRITERION_NAMES = [
  'FIRST_ACCEPT',
  'LI_ACCEPT',
  'BEST_ACCEPT']

class TSP_Local_Search_Base_Environment(gym.Env):
  """Custom Environment that follows the gym interface for Routing Problems with Local Search.
  
    The underlying local search is performed by a wrapper around the VRPH C++ library"""

  metadata = {'render.modes': ['human']}

  def __init__(self, data_generator, construction_heuristic, max_steps=100):
    super(TSP_Local_Search_Base_Environment, self).__init__()

    self.generator = data_generator
    self.construction_heuristic = construction_heuristic
    self.problem_type = 'TSP'
    self.max_steps = max_steps
    self.model = None

  def step(self, action):
    node_index = action[0] # currently not used, needs to be built into the C++ interface yet
    heuristic = ACTIONS[action[1]]
    rule = ACCEPTANCE_CRITERION[action[2]] + VRPH_SAVINGS_ONLY + VRPH_DOWNHILL
    self.model.Single_solve(heuristic, rule, 0.000001, 2, False)

    # calc reward and set new current cost
    new_cost = self.model.get_best_total_route_length()
    reward = self.current_cost - new_cost
    self.current_cost = new_cost
    self.tentative_cost = self.model.get_total_route_length()

    ## get new observation
    ## currently show_route only prints, this will be replaced by retrieving the actual object.
    #observation = {
    #  'customers': self.instance['customers']
    #  #'solution': None
    #  }

    r = []
    returned_route = np.array(self.model.get_routes())
    for i in range(1, self.num_customers+1):
        r.append(np.argwhere(returned_route==i))

    observation = {
      'customers': self.instance['customers'],
      'solution': np.array(r).reshape(self.num_customers,2)
      #'solution': None
      }
    
    # debug printing
    #self.model.show_route(1)
    print(f'{self.num_step : <20}{ACTIONS_NAMES[action[1]] : ^20}{ACCEPTANCE_CRITERION_NAMES[action[2]] : ^20}{self.current_cost : ^20}{self.tentative_cost : ^20}')
    
    #step count
    done = self.num_step >= self.max_steps
    info = {'step':self.num_step}
    self.num_step += 1

    return observation, reward, done, info

  def reset(self):
    if self.model:
      del(self.model)
      # I sometimes had problems with these C bound objects that they dont get garbage collected properly when the reference is not explicitly deleted 
      # (for example with the SCIP, GUROBI and ORTOOLS solver)
      # this leads to memory accumulation over time when often creating these objects
      # This is a preemptive measure in this case

    self.instance = self.generator.sample_instance()
    self.num_customers = self.instance['customers'].shape[0]
    self.model = load_tsp(self.instance['customers'])
    initial_solution = self.construction_heuristic(self.instance['customers'])

    #remove later
    initial_solution = [(np.array(initial_solution)+1).tolist()]

    self.model.use_initial_solution(initial_solution)

    self.current_cost = self.model.get_best_total_route_length()
    self.tentative_cost = self.model.get_total_route_length()
    self.num_step = 1

    #debug printing
    print(f"{'Step' : <20}{'Action' : ^20}{'Rule' : ^20}{'Cost' : >10}{'Tentative Cost' : ^20}")
    repl = '-'
    print(f'{self.num_step : <20}{repl : ^20}{repl : ^20}{self.current_cost : ^20}{self.tentative_cost : ^20}')



    self.observation_space = spaces.Dict({
      #'customers': spaces.Box(low=0, high=1, shape=(1, 2), dtype=np.float32),
      #'solution': spaces.MultiDiscrete([self.num_customers for i in range(self.num_customers)])
      'customers': spaces.Box(low=0, high=1, shape=(self.num_customers, 2), dtype=np.float32),
      'solution': spaces.Box(low = -1, high = self.num_customers, shape=(self.num_customers, 2))
    })

    self.action_space = spaces.MultiDiscrete([self.num_customers, len(ACTIONS), len(ACCEPTANCE_CRITERION)])

    r = []
    returned_route = np.array(self.model.get_routes())
    for i in range(1, self.num_customers+1):
        r.append(np.argwhere(returned_route==i))

    observation = {
      'customers': self.instance['customers'],
      'solution': np.array(r).reshape(self.num_customers,2)
      #'solution': None
      }

    return observation  # reward, done, info can't be included

  def render(self, mode='human'):
    pass

  def close(self):
    #see reset policy
    del(self.model)


## TEST EXAMPLE
#generator = TSP_Generator(100)
#env = TSP_Local_Search_Base_Environment(generator, random_tsp_tour)

#done = False
#observation = env.reset()

#check_env(env)
#while not done:
#     #env.render()
#     #print(observation)
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)

#env.close()
