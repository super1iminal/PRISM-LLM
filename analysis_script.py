import pandas as pd
import numpy as np

VP = "C:/Users/asher/Documents/Coding/PRISM-LLM/PRISM-LLM/PRISM-Guided-Learning/out/results/100_20260210_22-18-20/LLM_VANILLA_PLUS_GPT5_MINI_results.parquet"
FM = "C:/Users/asher/Documents/Coding/PRISM-LLM/PRISM-LLM/PRISM-Guided-Learning/out/results/100_20260210_22-18-20/LLM_FEEDBACK_MINUS_GPT5_MINI_results.parquet"
vp = pd.read_parquet(VP)
fm = pd.read_parquet(FM)
SEP = "=" * 100
prob_cols = [c for c in vp.columns if c.startswith("prob_")]
print(SEP)
print("COLUMN NAMES")
print(SEP)
print(vp.columns.tolist())
print()
print("Probability columns:", prob_cols)
print()
