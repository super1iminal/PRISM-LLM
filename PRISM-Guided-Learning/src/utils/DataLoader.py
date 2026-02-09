import ast
import pandas as pd

from typing import List, Tuple
from environment.GridWorld import GridWorld

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_data(self):
        self.data: List[Tuple[GridWorld, int]] = []
        df = pd.read_csv(self.data_path, index_col=False)
        for _, row in df.iterrows():
            row = row.to_dict()
            size = int(row["n"])
            goals = ast.literal_eval(row["goals"] if isinstance(row["goals"], str) else row["goals"])
            obstacles = ast.literal_eval(row["static"]) if isinstance(row["static"], str) else row["static"]

            # Parse optional moving obstacle column (backward-compatible)
            if "moving" in row and pd.notna(row["moving"]):
                moving = ast.literal_eval(row["moving"]) if isinstance(row["moving"], str) else row["moving"]
            else:
                moving = []

            expected = int(row["BFS_steps"])
            self.data.append((GridWorld(size, goals, obstacles, moving), expected))
            print(f"Loaded gridworld of size {size} with goals {goals} and obstacles {obstacles}.")
        
    def __iter__(self):
        for item in self.data:
            yield item