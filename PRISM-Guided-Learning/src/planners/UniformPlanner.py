from verification.PrismVerifier import PrismVerifier
from verification.SimplifiedVerifier import SimplifiedVerifier
from verification.PrismModelGenerator import PrismModelGenerator
from environment.GridWorld import GridWorld

import logging
import os
import numpy as np
from itertools import product
import pandas as pd


from config.Settings import PRISM_PATH, LOGGING_PATH
from utils.Logging import setup_logger

# os.makedirs(LOGGING_PATH, exist_ok=True)

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     handlers=[
#                         logging.FileHandler(f"{LOGGING_PATH}/debug.log"),
#                         logging.StreamHandler()
#                     ])

# with open(f"{LOGGING_PATH}/debug.log", "w") as f:
#     f.write("")  # Clear the log file at the start of each run

# logger = logging.getLogger(__name__)

# all q-values initialized to 1
class BaselinePlanner:
    def __init__(self, size, goals, static_obstacles, moving_obstacles):
        self.size = size
        self.env = GridWorld(size, goals, static_obstacles, moving_obstacles)
        self.model_generator = PrismModelGenerator(self.env)
        self.prism_verifier = PrismVerifier(self.get_prism_path())
        self.simplified_verifier = SimplifiedVerifier(self.prism_verifier, self.env)
        self.action_space = 4
        self.num_goals = len(goals)
        self.q_table = self.initialize_q_table()
        
        self.logger = setup_logger("baseline_planner")
                
        self.logger.info("Baseline Planner initialized.")
        
    def get_prism_path(self):
        """Get PRISM executable path"""
        prism_path = PRISM_PATH  # Update this path for your system
        if os.path.exists(prism_path) and os.access(prism_path, os.X_OK):
            return prism_path
        else:
            raise FileNotFoundError(f"PRISM executable not found or not executable at {prism_path}")
        
    def initialize_q_table(self):
        """Initialize Q-table with variable number of goals"""
        q_table = {}
        
        # Generate all combinations of goal states
        goal_combinations = list(product([False, True], repeat=self.num_goals))
        
        for x in range(self.size):
            for y in range(self.size):
                for goal_combo in goal_combinations:
                    state = (x, y) + goal_combo
                    q_table[state] = np.ones(self.action_space) * 1.0
        
        self.logger.info(f"Q-table initialized with {len(q_table)} states.")
        return q_table
            
    def step(self):
        # after updated q-table
        model_str = self.model_generator.generate_prism_model(self.q_table)
        ltl_score = self.simplified_verifier.verify_policy(model_str)
        
        self.logger.info(f"LTL Score (BASELINE): {ltl_score}")
        return ltl_score