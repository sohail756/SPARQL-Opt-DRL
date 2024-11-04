#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:40:07 2024

@author: sohail
"""

import torch
import torch.nn as nn
import hashlib

# Define the embedding model
class VariableEmbedding(nn.Module):
    def __init__(self, embedding_dim, hash_space_size):
        super(VariableEmbedding, self).__init__()
        self.embedding = nn.Embedding(hash_space_size, embedding_dim)
        self.hash_space_size = hash_space_size
    
    def forward(self, variable_name):
        # Hash the variable name to an integer within the hash space
        variable_hash = int(hashlib.md5(variable_name.encode()).hexdigest(), 16) % self.hash_space_size
        variable_hash_tensor = torch.tensor([variable_hash])
        # Generate the embedding (1, embedding_dim) and return as (embedding_dim,)
        embedding = self.embedding(variable_hash_tensor).squeeze(0)
        return embedding
"""
# Set parameters
embedding_dim = 10  # Dimension of the embedding vector
hash_space_size = 10000  # The size of the hash space, adjust based on your needs

# Instantiate the embedding model
variable_embedder = VariableEmbedding(embedding_dim, hash_space_size)

# Example variable name
variable_name = "x1"

# Get the embedding for the variable name
variable_embedding = variable_embedder(variable_name)
print(variable_embedding)
"""
