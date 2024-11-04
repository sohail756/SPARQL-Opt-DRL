#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:10:07 2023

@author: sohail
"""
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Sequence, Box, Discrete, MultiBinary
from variableNamesEmbedding import VariableEmbedding
from query_features_loader import QueryFeaturesLoader
from gymnasium.spaces.utils import flatten, unflatten
import sqlite3


gym.register(
    id='TentrisEnv-v0',  # Your custom environment name and version
    entry_point='tentrisEnv:TentrisEnv',  # Where your env is located
    #max_episode_steps=1000,  # Optional: max number of steps per episode
    #reward_threshold=100.0,  # Optional: reward threshold for "solving"
    apply_api_compatibility=True,  # If using different gym versions
)


class TentrisEnv(gym.Env):
    metadata = {
        'render_modes': ['None'], }
    
    def __init__(self, db_path="/home/sohail/CLionProjects/tentris/query_data.db", embedding_dim=10, max_vars=100, reset_on_init=True):
        super().__init__()
        # Define the observation space
        #embedding_dim = 10  # Dimension of the embedding for variable names
        #max_vars = 100  # Maximum number of variables in the query
        self.reset_on_init = reset_on_init
        self.max_vars = max_vars
        self.observation_space = Dict({
            # QueryVars: Now represented as fixed-length sequences of embeddings (max_vars)
            "QueryVars": Box(low=-float('inf'), high=float('inf'), shape=(self.max_vars, embedding_dim), dtype=float),
        
            # ProjVars, JoinVars: Binary values with a fixed self.max_vars length
            "ProjVars": Box(low=0, high=1, shape=(self.max_vars,), dtype=int),  # Binary, fixed length
            "JoinVars": Box(low=0, high=1, shape=(self.max_vars,), dtype=int),  # Binary, fixed length
            
            # LonelyNonResultLabels: Binary flag with a fixed self.max_vars length
            "NonJoinVars": Box(low=0, high=1, shape=(self.max_vars,), dtype=int),
        
            # MinCardinalityInTP: Continuous value with a fixed self.max_vars length
            "MinCardinalityInTP": Box(low=0, high=float('inf'), shape=(self.max_vars,), dtype=float),
        
            # SelectModifier: 3 possible discrete values
            "SelectModifier": Discrete(3),
        
            # NoTPs: Single integer value
            "NoTPs": Box(low=0, high=float('inf'), shape=(), dtype=int),
        
            # TP_sizes: Continuous values with a fixed length
            "TPSizes": Box(low=0, high=float('inf'), shape=(self.max_vars,), dtype=float),
            
            # A binary mask indicating available (unmasked) actions
            "action_mask": MultiBinary(self.max_vars)
        })
        
        # Define the action space
        self.action_space = Discrete(self.max_vars)  # Action space for selecting a variable

        
        # Initialize the query features loader with the path to the CSV
        self.query_loader = QueryFeaturesLoader(db_path)
        self.current_obs = None  # This will store the current observation

        # Only reset if the flag is set to True
        if self.reset_on_init:
            self.reset()
    
    
    def reset(self, seed=None, options=None):
        self.partial_query_plan = []
        self.current_step = 0
        
        query_features = self.query_loader.get_query_features()
        #self.query_loader.close()
        if query_features is None:
            print("No more queries available.")
            return self.observation_space.sample(), {}
    
        
        
        self.current_row_id = query_features['row_id']
        
        self.query_runtime = query_features['DRLQueryRuntime']
        
        self.query_string = query_features['QueryString']
        
        self.queryplan_db = query_features['QueryPlan']
        
        # Log loaded query details to ensure proper initialization
        print(f"Loaded Query: {self.query_string}")
        
        # Process variable names and binary flags
        self.query_vars = query_features['QueryVars'].split()
        print(f"The Query Variables are: {self.query_vars}")
        proj_vars = query_features['ProjVars'].split()
        print(f"The Projection Variables are: {proj_vars}")
        self.join_vars = query_features['JoinVars'].split() if query_features['JoinVars'] else []
        print(f"The Join Variables are: {self.join_vars}")
        self.NonJoinVars = query_features['NonJoinVars'].split() if query_features['NonJoinVars'] else []
        print(f"The NonJoin Variables are: {self.NonJoinVars}")
    
        # Load the VartoLabelMap
        self.varto_label_map = {}
        if query_features['VartoLabelMap']:
            self.varto_label_map = eval(query_features['VartoLabelMap'])
        
        proj_flags = np.array([1 if var in proj_vars else 0 for var in self.query_vars], dtype=int)
        join_flags = np.array([1 if var in self.join_vars else 0 for var in self.query_vars], dtype=int)
        nonjoin_flags = np.array([1 if var in self.NonJoinVars else 0 for var in self.query_vars], dtype=int)
    
        # Ensure flags have the correct length
        if len(proj_flags) < self.max_vars:
            proj_flags = np.pad(proj_flags, (0, self.max_vars - len(proj_flags)), constant_values=0)
        if len(join_flags) < self.max_vars:
            join_flags = np.pad(join_flags, (0, self.max_vars - len(join_flags)), constant_values=0)
        if len(nonjoin_flags) < self.max_vars:
            nonjoin_flags = np.pad(nonjoin_flags, (0, self.max_vars - len(nonjoin_flags)), constant_values=0)
    
        # Embedding model
        embedding_dim = 10
        hash_space_size = 10000
        variable_embedder = VariableEmbedding(embedding_dim, hash_space_size)
        query_var_embeddings = np.array([variable_embedder(var).detach().numpy() for var in self.query_vars])
                
        if query_var_embeddings.shape[0] < self.max_vars:
            padding = np.zeros((self.max_vars - query_var_embeddings.shape[0], embedding_dim))
            query_var_embeddings = np.vstack((query_var_embeddings, padding))
        elif query_var_embeddings.shape[0] > self.max_vars:
            query_var_embeddings = query_var_embeddings[:self.max_vars]
    
        print(f"The embeddings for Variable names are: {query_var_embeddings}")    
    
        # MinCardinalityInTP
        min_cardinality = np.array(query_features['MinCardinalityInTP'].split(), dtype=float)
        if len(min_cardinality) < self.max_vars:
            min_cardinality = np.pad(min_cardinality, (0, self.max_vars - len(min_cardinality)), constant_values=0)
    
        # TPSizes
        TPSizes = np.array(query_features['TPSizes'].split(), dtype=float)
        if len(TPSizes) < self.max_vars:
            TPSizes = np.pad(TPSizes, (0, self.max_vars - len(TPSizes)), constant_values=0)
    
        # Action mask
        self.action_mask = np.array(join_flags, dtype=np.int32)
        if len(self.action_mask) < self.max_vars:
            self.action_mask = np.pad(self.action_mask, (0, self.max_vars - len(self.action_mask)), constant_values=0)
    
        # Construct the observation
        self.current_obs = {
            'QueryVars': query_var_embeddings,
            'ProjVars': proj_flags,
            'JoinVars': join_flags,
            'NonJoinVars': nonjoin_flags,
            'MinCardinalityInTP': min_cardinality,
            'SelectModifier': int(query_features['SelectModifier']),
            'NoTPs': int(query_features['NoTPs']),
            'TPSizes': TPSizes,
            'action_mask': self.action_mask
        }
    
        # Print statement to confirm successful reset
        print("Environment reset complete.")
        return self.current_obs, {}

        
    
    def step(self, action):
        # Check if the action is allowed
        if self.action_mask[action] == 1:
            selected_var = self.query_vars[action]
            self.partial_query_plan.append(selected_var)
    
            # Update the action mask
            self.action_mask[action] = 0
            self.current_obs['action_mask'] = self.action_mask
            
            # Log the current state
            print(f"Step: {self.current_step}, Selected Variable: {selected_var}")
            print(f"Partial Query Plan: {self.partial_query_plan}")
    
            # Check if the query plan is complete
            if len(self.partial_query_plan) == len(self.join_vars):
                if len(self.NonJoinVars) > 0:
                    self.partial_query_plan.extend(self.NonJoinVars)
                # Ensure the length of the query plan matches the total number of query variables
                assert len(self.partial_query_plan) == len(self.query_vars), \
                "The generated query plan does not match the total number of query variables!"    
                print(f"Full Query Plan: {self.partial_query_plan}")
                # Replace variable names in the query plan with their corresponding labels from VartoLabelMap
                labeled_query_plan = [self.varto_label_map.get(var, var) for var in self.partial_query_plan]
                
                # The query plan is complete, so save it to the database
                query_plan = " ".join(labeled_query_plan)
                row_id = self.current_row_id  # Use the current_row_id from the loaded query
                print(f"Full Query Plan: {query_plan}")
                
                # print(f"Db Query Plan: {self.queryplan_db}, Generated Query Plan: {query_plan}")
                
                # # Compare the generated query plan with the one in the database
                # if query_plan == self.queryplan_db:
                #     print("The generated query plan matches the query plan in the database.")
                # else:
                #     print("The generated query plan does NOT match the query plan in the database.")
                
                # print(f"Query Plan Complete. Saving to database. Row ID: {self.current_row_id}")

                #Connect to the SQLite database and update the QueryPlan column for the corresponding row
                conn = sqlite3.connect("/home/sohail/CLionProjects/tentris/query_data.db")
                cursor = conn.cursor()
                
                #Update the QueryPlan in the row corresponding to the current ROWID
                cursor.execute("UPDATE TestQueryData SET QueryPlan = ? WHERE ROWID = ?", (query_plan, row_id))
                conn.commit()
                conn.close()
                
                terminated = True
                reward = -self.query_runtime
            else:
                reward = 0
                terminated = False
    
            truncated = False
            self.current_step += 1
    
            return self.current_obs, reward, terminated, truncated, {}
        else:
            return self.current_obs, -1, False, False, {}

    
    
    def close(self):
        # Make sure to close the database connection when done
        self.query_loader.close()