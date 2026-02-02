PRISM_PATH = "C:\\Program Files\\prism-4.10\\bin\\prism.bat"  # Update this path for your system
LOGGING_PATH = "PRISM-Guided-Learning/out/logs/"
EVAL_PATH = "PRISM-Guided-Learning/out/eval_results/"
RESULTS_PATH = "PRISM-Guided-Learning/out/results/"

# RL Termination Settings
RL_MAX_EPISODES = 500
RL_CONVERGENCE_EPSILON = 0.001  # Min change in probabilities to continue
RL_PROBABILITY_THRESHOLD = 0.9  # Target threshold for each probability
RL_MIN_EPISODES_BEFORE_CONVERGENCE = 50  # Don't check convergence until this many episodes