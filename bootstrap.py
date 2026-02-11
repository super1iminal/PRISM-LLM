
import pathlib

script = chr(10).join([
    "import pandas as pd",
    "import numpy as np",
    "",
    "VP = " + repr(r'C:/Users/asher/Documents/Coding/PRISM-LLM/PRISM-LLM/PRISM-Guided-Learning/out/results/100_20260210_22-18-20/LLM_VANILLA_PLUS_GPT5_MINI_results.parquet'),
    "FM = " + repr(r'C:/Users/asher/Documents/Coding/PRISM-LLM/PRISM-LLM/PRISM-Guided-Learning/out/results/100_20260210_22-18-20/LLM_FEEDBACK_MINUS_GPT5_MINI_results.parquet'),
    "vp = pd.read_parquet(VP)",
    "fm = pd.read_parquet(FM)",
    "print('test ok')",
])

pathlib.Path(r'C:/Users/asher/Documents/Coding/PRISM-LLM/PRISM-LLM/analysis_script.py').write_text(script)
print('done')
