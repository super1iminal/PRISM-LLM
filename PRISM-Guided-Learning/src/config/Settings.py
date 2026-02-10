PRISM_PATH = "/u/asherarya/prism-4.10-src/prism/bin/prism"  # Update this path for your system
LOGGING_PATH = "PRISM-Guided-Learning/out/logs/"
EVAL_PATH = "PRISM-Guided-Learning/out/eval_results/"
RESULTS_PATH = "PRISM-Guided-Learning/out/results/"

# RL Termination Settings
RL_MAX_EPISODES = 500
RL_CONVERGENCE_EPSILON = 0.001  # Min change in probabilities to continue
GOAL_REACHABILITY_THRESHOLD = 0.8
SEQUENCE_ORDERING_THRESHOLD = 0.8
OBSTACLE_AVOIDANCE_THRESHOLD = 0.5

def get_threshold_for_key(key: str) -> float:
    """Map a probability key to its per-requirement threshold."""
    if key.startswith('goal'):
        return GOAL_REACHABILITY_THRESHOLD
    elif key.startswith('seq_') or key == 'complete_sequence':
        return SEQUENCE_ORDERING_THRESHOLD
    elif key.startswith('avoid_moving'):
        return OBSTACLE_AVOIDANCE_THRESHOLD
    return GOAL_REACHABILITY_THRESHOLD  # default fallback

RL_MIN_EPISODES_BEFORE_CONVERGENCE = 50  # Don't check convergence until this many episodes
