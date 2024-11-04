import numpy as np
import gym
from gym import spaces
from gymnasium.spaces.utils import flatten
from variableNamesEmbedding import VariableEmbedding
from query_features_loader import QueryFeaturesLoader

class TentrisSimpleEnv(gym.Env):
    metadata = {'render_modes': ['None']}
    
    def __init__(self, file_path="cleaned_watdiv-query_features.csv", max_vars=100, embedding_dim=10):
        super().__init__()

        self.max_vars = max_vars
        self.embedding_dim = embedding_dim  # Embedding dimension for variable names

        # Define subspaces
        embedding_space = spaces.Box(
            low=-1, high=1, shape=(self.max_vars * self.embedding_dim,), dtype=np.float32
        )
        binary_vars_space = spaces.Box(
            low=0, high=1, shape=(self.max_vars * 3,), dtype=np.float32  # ProjVars, JoinVars, LonelyNonResultVars
        )
        no_tps_space = spaces.Box(
            low=0, high=1000, shape=(1,), dtype=np.float32  # No_TPs
        )
        tp_sizes_space = spaces.Box(
            low=0, high=10000, shape=(self.max_vars,), dtype=np.float32  # TP_sizes
        )
        min_cardinality_space = spaces.Box(
            low=0, high=10000, shape=(self.max_vars,), dtype=np.float32  # min_cardinality_of_var_in_TP
        )
        
        # Combine subspaces into a Tuple
        self.observation_space = spaces.Tuple((
            embedding_space,
            binary_vars_space,
            no_tps_space,
            tp_sizes_space,
            min_cardinality_space
        ))
        

        # Action mask for valid actions (join variables)
        self.action_mask = spaces.MultiBinary(self.max_vars)

        # Action space will be based on join variables, allowing only valid selections
        self.action_space = spaces.Discrete(self.max_vars)

        # Initialize the query features loader
        self.query_loader = QueryFeaturesLoader(file_path)
        self.current_obs = None
        self.reset()

    def reset(self):
        # Reset internal state
        self.partial_query_plan = []
        self.current_step = 0

        # Load query features
        query_features = self.query_loader.get_query_features()
        if query_features is None:
            print("No more queries available.")
            return self.observation_space.sample(), {}

        self.query_runtime = query_features['QueryRuntime']
        query_vars = query_features['QueryVars'].split()
        proj_vars = query_features['ProjVars'].split()
        join_vars = query_features['JoinVars'].split() if query_features['JoinVars'] else []
        lonely_vars = query_features['LonelyNonResultVars'].split() if query_features['LonelyNonResultVars'] else []

        # Create binary flags for ProjVars, JoinVars, and LonelyNonResultVars
        proj_flags = np.array([1 if var in proj_vars else 0 for var in query_vars], dtype=int)
        join_flags = np.array([1 if var in join_vars else 0 for var in query_vars], dtype=int)
        lonely_flags = np.array([1 if var in lonely_vars else 0 for var in query_vars], dtype=int)

        # Save variable names
        self.join_vars = join_vars
        self.query_vars = query_vars

        # Embed the variable names
        variable_embedder = VariableEmbedding(self.embedding_dim, 10000)
        query_var_embeddings = np.array([variable_embedder(var).detach().numpy() for var in query_vars])

        # Action mask reflects valid join variables
        self.action_mask = np.array(join_flags, dtype=np.int32) if join_flags.any() else np.zeros(len(query_vars), dtype=np.int32)
        self.action_space = spaces.Discrete(len(self.join_vars)) if len(self.join_vars) > 0 else spaces.Discrete(1)

        # Construct the observation
        flattened_embeddings = query_var_embeddings.flatten()  # Flatten the embeddings for all query variables

        self.current_obs = np.concatenate([
            flattened_embeddings,               # Variable name embeddings
            proj_flags,                         # Binary flags for ProjVars
            join_flags,                         # Binary flags for JoinVars
            lonely_flags,                       # Binary flags for LonelyNonResultVars
            np.array([int(query_features['No_TPs'])]),  # No. of triple patterns
            np.array(query_features['TP_sizes'].split(), dtype=float),  # TP sizes
            np.array(query_features['min_cardinality_of_var_in_TP'].split(), dtype=float)  # Min cardinality of var in TP
        ])

        return self._get_obs(), {}

    def step(self, action):
        if self.action_mask[action] == 1:
            selected_var = self.query_vars[action]
            self.partial_query_plan.append(selected_var)
            self.action_mask[action] = 0
            self.current_obs[-self.max_vars:] = self.action_mask  # Update the mask in the observation

            if len(self.partial_query_plan) == len(self.join_vars):
                reward = -self.query_runtime
                terminated = True
            else:
                reward = 0
                terminated = False

            self.current_step += 1
            return self._get_obs(), reward, terminated, False, {}
        else:
            return self._get_obs(), -1, False, False, {}

    def _get_obs(self):
        return flatten(self.observation_space, self.current_obs)
