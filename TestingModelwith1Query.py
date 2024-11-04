#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:41:05 2024

@author: sohail
"""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation
from tentrisEnv import TentrisEnv

# Path to the SQLite database
db_path = "/home/sohail/CLionProjects/tentris/query_data.db"

# Load the trained PPO model
model = PPO.load("ppo_tentris")

# Initialize the environment
env = DummyVecEnv([lambda: FlattenObservation(TentrisEnv(db_path, reset_on_init=False))])

# Reset the environment to start a new episode (i.e., load a new query)
obs = env.reset()

# Initialize variables to track the episode
done = False
total_reward = 0

# Loop until the query plan is fully generated
while not done:
    # Let the model select an action based on the current observation
    action, _states = model.predict(obs)

    # Step the environment using the selected action
    obs, reward, done, _ = env.step(action)

    # Accumulate the total reward
    total_reward += reward

    # Print the step details
    #print(f"Action: {action}, Reward: {reward}, Done: {done}")

# Episode is finished, print the total reward
print(f"Total Reward: {total_reward}")
