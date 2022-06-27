from environment import TSP_Local_Search_Base_Environment
import gym
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from data_utils import TSP_Generator
from construction_heuristics import random_tsp_tour


generator = TSP_Generator(100)
env = TSP_Local_Search_Base_Environment(generator, random_tsp_tour)
observation = env.reset()

model = PPO("MultiInputPolicy", env, verbose=1)

model.learn(total_timesteps=1000)
model.save("sac_pendulum")