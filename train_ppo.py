#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:38:15 2024

@author: sohail
"""

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tentrisEnv import TentrisEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation

# Initialize the custom environment with the SQLite database path
db_path = "/home/sohail/CLionProjects/tentris/query_data.db"  # Path to the SQLite database

#env = DummyVecEnv([lambda: FlattenObservation(TentrisEnv(db_path))])
env = DummyVecEnv([lambda: FlattenObservation(TentrisEnv(db_path, reset_on_init=False))])

# Ensure the environment follows Gym API by checking it
#check_env(env)

# Initialize the PPO agent with the environment
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tentris_tensorboard/")

# Train the PPO model
model.learn(total_timesteps=10000)  # Adjust total_timesteps as needed

# Save the trained model
model.save("ppo_tentris")

# Optionally load the model later
# model = PPO.load("ppo_tentris", env=env)

# Test the trained model
# obs = env.reset()
# for _ in range(1000):  # Adjust number of steps for testing
#     print(f"Observation at start: {obs}")
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(f"Action taken: {action}, Rewards: {rewards}, Done: {dones}")
#     env.render()  # Optional