import numpy as np
import subprocess
import tempfile
import os
from typing import Dict, Tuple, List, Any
from fractions import Fraction
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import traceback
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("debug.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

class GridWorldEnv:
    def __init__(self, size: int):
        self.size = size
        self.goals = {
            1: (5, 8),
            2: (4, 2),
            3: (9, 9)
        }
        
        self.static_obstacles = [(6, 6), (3, 4)]
        
        
        self.moving_obstacle_positions = [
            (7, 4), (7, 3), (7, 2), (6, 2), (5, 2),
            (6, 2), (7, 2), (7, 3), (7, 4), (7, 5),
            (7, 6), (7, 7), (7, 8), (7, 9), (8, 9)
        ]
        self.current_moving_index = 0
        self.current_moving_obstacle = self.moving_obstacle_positions[0]
        
        self.max_steps = 200
        self.steps = 0
        self.current_pos = (0, 0)
        self.visited_goals = set()
        self.reset()

    def reset(self) -> Tuple[int, int, bool, bool, bool]:
        self.current_pos = (0, 0)
        self.visited_goals = set()
        self.steps = 0
        self.current_moving_index = 0
        self.current_moving_obstacle = self.moving_obstacle_positions[self.current_moving_index]
        return self.get_state()

    def get_state(self) -> Tuple[int, int, bool, bool, bool]:
        x, y = self.current_pos
        g1_reached = 1 in self.visited_goals
        g2_reached = 2 in self.visited_goals
        g3_reached = 3 in self.visited_goals
        return (x, y, g1_reached, g2_reached, g3_reached)

    def _update_moving_obstacle(self):
        self.current_moving_index = (self.current_moving_index + 1) % len(self.moving_obstacle_positions)
        self.current_moving_obstacle = self.moving_obstacle_positions[self.current_moving_index]

    def step(self, action: int) -> Tuple[Tuple[int, int, bool, bool, bool], float, bool]:
        self.steps += 1
        
        self._update_moving_obstacle()
            
        x, y = self.current_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]

        new_x = max(0, min(self.size - 1, x + dx))
        new_y = max(0, min(self.size - 1, y + dy))
        
        
        if ((new_x, new_y) in self.static_obstacles or
            (new_x, new_y) == self.current_moving_obstacle):
            new_x, new_y = x, y
            reward = -10.0
        else:
            self.current_pos = (new_x, new_y)
            reward = -0.1

        if self.steps >= self.max_steps:
            return self.get_state(), reward, True

        done = False
        
        current_goal = len(self.visited_goals) + 1
        if self.current_pos == self.goals.get(current_goal):
            self.visited_goals.add(current_goal)
            if current_goal == 3:
                reward = 100.0
                done = True
            else:
                reward = 10.0
        elif self.current_pos in self.goals.values():
            reward = -5.0

        if not done and len(self.visited_goals) < 3:
            next_goal = self.goals[len(self.visited_goals) + 1]
            distance = abs(new_x - next_goal[0]) + abs(new_y - next_goal[1])
            reward -= 0.01 * distance

        return self.get_state(), reward, done
        
class PrismModelGenerator:
    def __init__(self, size: int):
        self.size = size
        self.logger = logging.getLogger(__name__)
        self.static_obstacles = [(6, 6), (3, 4)]
        
        self.moving_obstacle_positions = [
            (7, 4), (7, 3), (7, 2), (6, 2), (5, 2),
            (6, 2), (7, 2), (7, 3), (7, 4), (7, 5),
            (7, 6), (7, 7), (7, 8), (7, 9), (8, 9)
        ]
        self.goals = {
            1: (5, 8),
            2: (4, 2),
            3: (9, 9)
        }
        self.debug_transitions = set()

    def _generate_transitions(self, x: int, y: int, g1: bool, g2: bool, g3: bool,
                           q_values: np.ndarray) -> list:
        
        q_exp = np.exp((q_values - np.max(q_values)) / 0.1)
        probs = q_exp / np.sum(q_exp)
        
        int_probs = np.round(probs * 1000).astype(int)
        diff = 1000 - int_probs.sum()
        int_probs[np.argmax(probs)] += diff
        
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        transitions = []
        
        guard = (f"  [] (x={x} & y={y}"
                f" & g1={str(g1).lower()}"
                f" & g2={str(g2).lower()}"
                f" & g3={str(g3).lower()}) ->")
        
        updates = []
        
        for i, (dx, dy) in enumerate(moves):
            if int_probs[i] > 0:
                new_x = max(0, min(self.size - 1, x + dx))
                new_y = max(0, min(self.size - 1, y + dy))
                
                for obs_idx, current_obs_pos in enumerate(self.moving_obstacle_positions):
                    next_obs_idx = (obs_idx + 1) % len(self.moving_obstacle_positions)
                    next_obs_pos = self.moving_obstacle_positions[next_obs_idx]
                    
                    if (new_x, new_y) == current_obs_pos:
                        transition_prob = 0.9
                        stay_prob = 0.1
                        updates.append(f"{(int_probs[i]/1000) * stay_prob}:(x'={x})"
                                    f" & (y'={y})"
                                    f" & (g1'={str(g1).lower()})"
                                    f" & (g2'={str(g2).lower()})"
                                    f" & (g3'={str(g3).lower()})")
                        
                        
                        updates.append(f"{(int_probs[i]/1000) * transition_prob}:(x'={next_obs_pos[0]})"
                                    f" & (y'={next_obs_pos[1]})"
                                    f" & (g1'={str(g1).lower()})"
                                    f" & (g2'={str(g2).lower()})"
                                    f" & (g3'={str(g3).lower()})")
                        break
                else:
                    
                    if (new_x, new_y) in self.static_obstacles:
                        updates.append(f"{int_probs[i]/1000}:(x'={x})"
                                    f" & (y'={y})"
                                    f" & (g1'={str(g1).lower()})"
                                    f" & (g2'={str(g2).lower()})"
                                    f" & (g3'={str(g3).lower()})")
                    else:
                        
                        new_g1 = g1 or ((new_x, new_y) == self.goals[1])
                        new_g2 = g2 or (g1 and (new_x, new_y) == self.goals[2])
                        new_g3 = g3 or (g2 and (new_x, new_y) == self.goals[3])
                        
                        updates.append(f"{int_probs[i]/1000}:(x'={new_x})"
                                    f" & (y'={new_y})"
                                    f" & (g1'={str(new_g1).lower()})"
                                    f" & (g2'={str(new_g2).lower()})"
                                    f" & (g3'={str(new_g3).lower()})")
        
        if updates:
            transition = f"{guard} " + " + ".join(updates) + ";"
            transitions.append(transition)
        
        return transitions

    def generate_prism_model(self, policy: Dict[Tuple[int, int, bool, bool, bool], int],
                           q_table: Dict[Tuple[int, int, bool, bool, bool], np.ndarray]) -> str:
        
        self.debug_transitions = set()
        
        model = ["dtmc", "", f"const int N = {self.size};", ""]
        
        model.append("module gridworld")
        model.append(f"  x : [0..{self.size-1}] init 0;")
        model.append(f"  y : [0..{self.size-1}] init 0;")
        model.append("  g1 : bool init false;")
        model.append("  g2 : bool init false;")
        model.append("  g3 : bool init false;")
        model.append("")
        
        total_states = 0
        
        for x in range(self.size):
            for y in range(self.size):
                for g1 in [False, True]:
                    for g2 in [False, True]:
                        for g3 in [False, True]:
                            state = (x, y, g1, g2, g3)
                            if state in q_table:
                                total_states += 1
                                transitions = self._generate_transitions(
                                    x, y, g1, g2, g3, q_table[state]
                                )
                                model.extend(transitions)
        
        model.append("endmodule")
        model.append("")
        model.append("// Labels for properties")
        model.append(f'label "at_goal1" = x={self.goals[1][0]} & y={self.goals[1][1]};')
        model.append(f'label "at_goal2" = x={self.goals[2][0]} & y={self.goals[2][1]};')
        model.append(f'label "at_goal3" = x={self.goals[3][0]} & y={self.goals[3][1]};')
        
        moving_obstacle_label = " | ".join([f"(x={x} & y={y})" 
                                          for x, y in self.moving_obstacle_positions])
        static_obstacle_label = " | ".join([f"(x={x} & y={y})" 
                                          for x, y in self.static_obstacles])
        
        model.append(f'label "at_obstacle" = ({moving_obstacle_label}) | ({static_obstacle_label});')
        
        model_str = "\n".join(model)
        return model_str
class PrismVerifier:
    def __init__(self, prism_bin_path: str):
        self.prism_bin_path = prism_bin_path
        self.logger = logging.getLogger(__name__)
        self.temp_files = [] 

    def verify_property(self, model_str: str, property_str: str) -> List[float]:
        model_path = None
        prop_path = None
        try:
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.nm', delete=False) as model_file:
                model_file.write(model_str)
                model_path = model_file.name
                self.temp_files.append(model_path)

            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.props', delete=False) as prop_file:
                prop_file.write(property_str)
                prop_path = prop_file.name
                self.temp_files.append(prop_path)

            
            self.logger.debug("PRISM Model:")
            self.logger.debug(model_str)
            self.logger.debug("Properties:")
            self.logger.debug(property_str)

            cmd = [
                self.prism_bin_path,
                model_path,
                prop_path,
                "-explicit",
                "-javamaxmem", "4g",
                "-maxiters", "1000000",  
                "-power", 
                "-verbose",
                "-exportstates", "states.txt"  
            ]

            self.logger.debug(f"Running PRISM command: {' '.join(cmd)}")
            result = subprocess.run(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True)

            self.logger.debug("PRISM stdout:")
            self.logger.debug(result.stdout)
            self.logger.debug("PRISM stderr:")
            self.logger.debug(result.stderr)

            probabilities = []
            for line in result.stdout.split('\n'):
                if "Result:" in line:
                    try:
                        value_str = line.split(':')[1].strip().split()[0]
                        prob = float(value_str)
                        probabilities.append(prob)
                        self.logger.debug(f"Parsed probability: {prob}")
                    except (IndexError, ValueError) as e:
                        self.logger.error(f"Failed to parse line: {line}")
                        self.logger.error(f"Error: {str(e)}")
                        continue

            if not probabilities:
                self.logger.error("No probabilities found in PRISM output")
                return [0.0] * 8  

            return probabilities

        except Exception as e:
            self.logger.error(f"Verification error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [0.0] * 8

        finally:
            
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    self.logger.error(f"Failed to delete temporary file {temp_file}: {str(e)}")
            self.temp_files = []


class SimplifiedVerifier:
    def __init__(self, prism_verifier: PrismVerifier):
        self.prism_verifier = prism_verifier
        self.logger = logging.getLogger(__name__)
        self.ltl_probabilities = []
        self.verification_history = []  

    def verify_policy(self, model_str: str) -> float:
        property_str = (
            "P=? [ F \"at_goal1\" ];\n"
            "P=? [ F \"at_goal2\" ];\n"
            "P=? [ F \"at_goal3\" ];\n"
            "P=? [ !\"at_goal2\" U \"at_goal1\" ];\n"
            "P=? [ !\"at_goal3\" U (\"at_goal1\" & (!\"at_goal3\" U \"at_goal2\")) ];\n"
            "P=? [ (!\"at_goal2\" U \"at_goal1\") & (!\"at_goal3\" U (\"at_goal1\" & (!\"at_goal3\" U \"at_goal2\"))) ];\n"
            "P=? [ G<=30 !\"at_obstacle\" ];\n"
        )
        
        probabilities = self.prism_verifier.verify_property(model_str, property_str)
        self.ltl_probabilities.append(probabilities)
        
        score = self._calculate_score(probabilities)
        self.logger.info(f"Goal Sequence Probabilities: {probabilities}")
        self.logger.info(f"Combined Score: {score:.4f}")
        return score

    def _calculate_score(self, results: List[float]) -> float:
        if len(results) >= 7:
            goal1_reach = results[0]
            goal2_reach = results[1] 
            goal3_reach = results[2]
            seq1 = results[3]  # G1 before G2
            seq2 = results[4]  # G1G2 before G3
            seq3 = results[5]  # Complete sequence
            avoid_obstacle = results[6]  # Obstacle avoidance
            
            score = (0.1 * goal1_reach + 
                    0.1 * goal2_reach + 
                    0.1 * goal3_reach + 
                    0.15 * seq1 +
                    0.15 * seq2 + 
                    0.2 * seq3 +
                    0.2 * avoid_obstacle)  
            
            return score
        return 0.0

    def save_probabilities_to_file(self, agent, save_dir: str = "results"):
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, 'ltl_probabilities_and_rewards.txt')
        
        try:
            with open(output_path, 'w') as f:
                
                f.write("Episode,Reach_G1,Reach_G2,Reach_G3,G1_before_G2,G1G2_before_G3,"
                    "Complete_Sequence,Avoid_Obstacle,Episode_Reward,Path_Exists,Score\n")
                
                for episode, probs in enumerate(self.ltl_probabilities):
                    prob_str = ','.join(f"{p:.4f}" for p in probs)
                    reward = agent.episode_rewards[episode] if episode < len(agent.episode_rewards) else 0.0
                    score = self._calculate_score(probs)
                    f.write(f"{episode},{prob_str},{reward:.4f},1.0,{score:.4f}\n")  # Added Path_Exists=1.0
                    
                self.logger.info(f"Saved LTL probabilities and rewards to {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving probabilities to file: {str(e)}")
class LTLGuidedQLearningWithObstacle:
    def __init__(self, size: int, prism_verifier: PrismVerifier):
        self.size = size
        self.env = GridWorldEnv(size)
        self.model_generator = PrismModelGenerator(size)
        self.verifier = SimplifiedVerifier(prism_verifier)
        self.logger = logging.getLogger(__name__)
        
        self.q_table = {}
        self.action_space = 4
        self._initialize_q_table()
        
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.997
        self.alpha = 0.2
        self.alpha_min = 0.05
        self.gamma = 0.99
        self.K = 5
        
        self.prism_probs = {
            'goal1': 0.0,
            'goal2': 0.0,
            'goal3': 0.0,
            'seq1': 0.0,
            'seq2': 0.0,
            'seq3': 0.0,
            'avoid_obstacle': 0.0,
            'path_exists': 0.0
        }
        
        self.goals = {
            1: (5, 8),
            2: (4, 2),
            3: (9, 9)
        }
        self.static_obstacles = [(6, 6), (3, 4)]
        
        # Performance tracking
        self.episode_rewards = []
        self.ltl_scores = []
        self.best_policy = None
        self.best_ltl_score = 0.0
        self.training_stats = []

    def _initialize_q_table(self):
        for x in range(self.size):
            for y in range(self.size):
                for g1 in [False, True]:
                    for g2 in [False, True]:
                        for g3 in [False, True]:
                            state = (x, y, g1, g2, g3)
                            self.q_table[state] = np.ones(self.action_space) * 1.0

    def _update_prism_probabilities(self, probabilities: List[float]):
        if len(probabilities) >= 8:
            self.prism_probs = {
                'goal1': probabilities[0],
                'goal2': probabilities[1],
                'goal3': probabilities[2],
                'seq1': probabilities[3],
                'seq2': probabilities[4],
                'seq3': probabilities[5],
                'avoid_obstacle': probabilities[6],
                'path_exists': probabilities[7]
            }

    def _calculate_sequence_weight(self, state: tuple, next_state: tuple) -> float:
        _, _, g1, g2, g3 = state
        _, _, next_g1, next_g2, next_g3 = next_state
        
        weight = 1.0
        
        if not g1 and next_g1:
            weight *= (1.5 + self.prism_probs['seq1'])
        if g1 and not g2 and next_g2:
            weight *= (1.5 + self.prism_probs['seq2'])
        if g2 and not g3 and next_g3:
            weight *= (1.5 + self.prism_probs['seq3'])
            
        if self.prism_probs['path_exists'] > 0:
            weight *= (1 + self.prism_probs['path_exists'])
            
        return weight

    def _calculate_counterfactual_reward(self, state: tuple, next_state: tuple, k: int) -> float:
        x, y, g1, g2, g3 = state
        next_x, next_y, next_g1, next_g2, next_g3 = next_state
        
        base_reward = -0.05 * (1 + 0.05 * self.size/6)

        sequence_weight = self._calculate_sequence_weight(state, next_state)
        k_bonus = 1.0 + (k / self.K) * sequence_weight * (1 + self.prism_probs['path_exists'])
        obstacle_prob = self.prism_probs.get('avoid_obstacle', 0.0)
        
        moving_positions = [
            (7, 4), (7, 3), (7, 2), (6, 2), (5, 2),
            (6, 2), (7, 2), (7, 3), (7, 4), (7, 5),
            (7, 6), (7, 7), (7, 8), (7, 9), (8, 9)
        ]
        current_obs_idx = self.env.current_moving_index
        next_obs_idx = (current_obs_idx + 1) % len(moving_positions)
        next_obs_pos = moving_positions[next_obs_idx]
        
        if ((next_x, next_y) in [(6, 6), (3, 4)] or
            (next_x, next_y) == self.env.current_moving_obstacle):
            obstacle_penalty = -10.0 * (1.0 + 0.2 * obstacle_prob)
            base_reward += obstacle_penalty
        else:
            dist_to_moving = abs(next_x - next_obs_pos[0]) + abs(next_y - next_obs_pos[1])
            dist_to_static1 = abs(next_x - 6) + abs(next_y - 6)
            dist_to_static2 = abs(next_x - 3) + abs(next_y - 4)
            min_dist = min(dist_to_moving, dist_to_static1, dist_to_static2)
            
            if min_dist < 3:
                avoidance_bonus = 1.0 * obstacle_prob * (1/max(1, min_dist))
                base_reward += avoidance_bonus
        current_goal = None
        if not g1:
            current_goal = (5, 8)
        elif not g2:
            current_goal = (4, 2)
        elif not g3:
            current_goal = (9, 9)
                
        if current_goal:
            dist_to_goal = abs(next_x - current_goal[0]) + abs(next_y - current_goal[1])
            progress_reward = -0.005 * dist_to_goal
            base_reward += progress_reward

        if not g1 and next_x == self.goals[1][0] and next_y == self.goals[1][1]:
            base_reward += 100.0 * k_bonus * (1 + self.prism_probs['goal1'])
        elif g1 and not g2 and next_x == self.goals[2][0] and next_y == self.goals[2][1]:
            base_reward += 100.0 * k_bonus * (1 + self.prism_probs['goal2'])
        elif g2 and not g3 and next_x == self.goals[3][0] and next_y == self.goals[3][1]:
            base_reward += 1000.0 * k_bonus * (1 + self.prism_probs['goal3'])
                        
        return base_reward

    def _get_counterfactual_states(self, state: tuple, next_state: tuple) -> List[Tuple]:
        x, y, g1, g2, g3 = state
        next_x, next_y, next_g1, next_g2, next_g3 = next_state
        
        counterfactuals = []
        
        if np.random.random() > 0.3:
            return counterfactuals
        
        max_counterfactuals = 2
        
        if not g1 and not next_g1:

            goal1_x, goal1_y = self.goals[1]
            dist_to_goal1 = abs(next_x - goal1_x) + abs(next_y - goal1_y)
            
            if dist_to_goal1 <= 2:
                cf_state = (x, y, g1, g2, g3)
                cf_next_state = (next_x, next_y, True, g2, g3)
                counterfactuals.append((cf_state, cf_next_state))
        
        elif g1 and not g2 and not next_g2:

            goal2_x, goal2_y = self.goals[2]
            dist_to_goal2 = abs(next_x - goal2_x) + abs(next_y - goal2_y)
            
            if dist_to_goal2 <= 2:
                cf_state = (x, y, g1, g2, g3)
                cf_next_state = (next_x, next_y, g1, True, g3)
                counterfactuals.append((cf_state, cf_next_state))
        
        elif g1 and g2 and not g3 and not next_g3:

            goal3_x, goal3_y = self.goals[3]
            dist_to_goal3 = abs(next_x - goal3_x) + abs(next_y - goal3_y)
            

            if dist_to_goal3 <= 2:
                cf_state = (x, y, g1, g2, g3)
                cf_next_state = (next_x, next_y, g1, g2, True)
                counterfactuals.append((cf_state, cf_next_state))
        
        
        return counterfactuals[:max_counterfactuals]

    def _step_with_counterfactuals(self, state: tuple, action: int, k: int) -> Tuple[tuple, list, bool, int]:
        next_state, reward, done = self.env.step(action)
        experiences = [(state, action, next_state, reward)]
        
        counterfactuals = self._get_counterfactual_states(state, next_state)
        for cf_state, cf_next_state in counterfactuals:
            cf_reward = self._calculate_counterfactual_reward(cf_state, cf_next_state, k)
            if cf_state != state or cf_next_state != next_state:
                experiences.append((cf_state, action, cf_next_state, cf_reward))

        next_k = min(k + 1, self.K - 1) if reward > 0 else k
        
        return next_state, experiences, done, next_k

    def _get_action(self, state: tuple) -> int:
        if np.random.random() < self.epsilon:
            x, y, g1, g2, g3 = state
            if not g1:
                target = (5, 8)
            elif not g2:
                target = (4, 2)
            elif not g3:
                target = (9, 9)
            else:
                return np.random.randint(self.action_space)
            dx = target[0] - x
            dy = target[1] - y
            
            if abs(dx) > abs(dy):
                if dx > 0:
                    preferred_action = 1
                else:
                    preferred_action = 3
            else:
                if dy > 0:
                    preferred_action = 2
                else:
                    preferred_action = 0

            if np.random.random() < 0.7:
                return preferred_action
            else:
                return np.random.randint(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def _adjust_parameters(self, ltl_score: float):
        if ltl_score > self.best_ltl_score:
            self.best_ltl_score = ltl_score
            self.best_policy = self.get_current_policy()
            
            self.epsilon = max(self.epsilon_min,
                             self.epsilon * 0.997)
            
            self.alpha = max(self.alpha_min,
                           self.alpha * 0.998)
        else:
            self.epsilon = min(0.9, self.epsilon * 1.02)
            
        self.training_stats.append({
            'ltl_score': ltl_score,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'best_score': self.best_ltl_score
        })

    def get_current_policy(self) -> Dict[Tuple[int, int, bool, bool, bool], int]:
        return {state: int(np.argmax(q_values)) 
                for state, q_values in self.q_table.items()}

    def save_training_stats(self, save_dir: str = "results"):
        os.makedirs(save_dir, exist_ok=True)
        stats_file = os.path.join(save_dir, 'training_statistics.txt')
        
        try:
            with open(stats_file, 'w') as f:
                f.write("Episode,LTL_Score,Epsilon,Alpha,Best_Score\n")
                for episode, stats in enumerate(self.training_stats):
                    f.write(f"{episode},"
                           f"{stats['ltl_score']:.4f},"
                           f"{stats['epsilon']:.4f},"
                           f"{stats['alpha']:.4f},"
                           f"{stats['best_score']:.4f}\n")
                           
            self.logger.info(f"Training statistics saved to {stats_file}")
        except Exception as e:
            self.logger.error(f"Failed to save training statistics: {str(e)}")

    def train(self, episodes: int, verify_interval: int = 100):
        try:
            start_time = time.time()
            best_episode = 0
            no_improvement_count = 0
            
            for episode in range(episodes):
                state = self.env.reset()
                total_reward = 0
                done = False
                k = 0
                steps = 0
                
                if episode % verify_interval == 0:
                    try:
                        model_str = self.model_generator.generate_prism_model(
                            self.get_current_policy(), self.q_table)
                        ltl_score = self.verifier.verify_policy(model_str)
                        
                        if self.verifier.ltl_probabilities:
                            latest_probs = self.verifier.ltl_probabilities[-1]
                            self._update_prism_probabilities(latest_probs)
                        
                        self.ltl_scores.append(ltl_score)
                        self._adjust_parameters(ltl_score)
                        
                        self.logger.info(
                            f"Episode {episode}: LTL Score = {ltl_score:.4f}, "
                            f"Epsilon = {self.epsilon:.3f}, "
                            f"Alpha = {self.alpha:.3f}, "
                            f"Best Score = {self.best_ltl_score:.4f}, "
                            f"Time = {(time.time() - start_time)/60:.1f}m"
                        )
                        
                        if ltl_score > self.best_ltl_score:
                            best_episode = episode
                            
                    except Exception as e:
                        self.logger.error(f"Verification error in episode {episode}: {str(e)}")
                        continue
                
                while not done:
                    action = self._get_action(state)
                    next_state, experiences, done, next_k = self._step_with_counterfactuals(
                        state, action, k)

                    for exp_state, exp_action, exp_next_state, exp_reward in experiences:
                        current_q = self.q_table[exp_state][exp_action]
                        next_max_q = np.max(self.q_table[exp_next_state])
                        new_q = current_q + self.alpha * (
                            exp_reward + self.gamma * next_max_q - current_q
                        )
                        self.q_table[exp_state][exp_action] = new_q
                    
                    state = next_state
                    k = next_k
                    total_reward += experiences[0][3]
                    steps += 1
                    
                    if steps >= self.env.max_steps:
                        done = True
                
                self.episode_rewards.append(total_reward)
                
                if episode % 10 == 0:
                    self.logger.info(f"Episode {episode}: Steps = {steps}, "
                                f"Total Reward = {total_reward:.2f}")
            
            try:
                model_str = self.model_generator.generate_prism_model(
                    self.get_current_policy(), self.q_table)
                final_score = self.verifier.verify_policy(model_str)
                self.logger.info(f"\nFinal LTL Score: {final_score:.4f}")
            except Exception as e:
                self.logger.error(f"Final verification error: {str(e)}")
            
            self.verifier.save_probabilities_to_file(self)
            self.save_training_stats()
            
            training_time = time.time() - start_time
            self.logger.info(f"\nTraining completed in {training_time/60:.1f} minutes")
            self.logger.info(f"Best LTL Score: {self.best_ltl_score:.4f} "
                        f"(Episode {best_episode})")
            
            return self.best_policy, self.best_ltl_score
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

def plot_training_progress(self, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    episodes = range(len(self.episode_rewards))
    plt.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, label='Rewards')
    
    window = min(10, len(self.episode_rewards))
    if len(self.episode_rewards) >= window:
        smoothed = np.convolve(self.episode_rewards,
                            np.ones(window)/window,
                            mode='valid')
        plt.plot(range(window-1, len(self.episode_rewards)),
                smoothed, 'r-', label=f'Moving Average (w={window})')
    
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    verify_episodes = range(0, len(self.training_stats))
    
    plt.plot(verify_episodes,
            [s['ltl_score'] for s in self.training_stats],
            'g-', label='LTL Score')
    plt.plot(verify_episodes,
            [s['epsilon'] for s in self.training_stats],
            'b--', label='Epsilon')
    plt.plot(verify_episodes,
            [s['alpha'] for s in self.training_stats],
            'r--', label='Alpha')
    
    plt.title('LTL Scores and Parameters')
    plt.xlabel('Verification Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()

def get_prism_path():
    prism_path = "PRISMPath"
    if os.path.exists(prism_path) and os.access(prism_path, os.X_OK):
        return prism_path
    raise FileNotFoundError(f"PRISM executable not found at {prism_path}")

def main():
    try:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/training.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        try:
            prism_path = get_prism_path()
            logger.info(f"Successfully initialized PRISM at: {prism_path}")
        except FileNotFoundError as e:
            logger.error(f"PRISM initialization failed: {e}")
            return
        
        verify_interval = 1
        total_episodes = 200
        grid_size = 10
        
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info("Initializing verifier and agent...")
        prism_verifier = PrismVerifier(prism_path)
        agent = LTLGuidedQLearningWithObstacle(size=grid_size, 
                                              prism_verifier=prism_verifier)
        
        logger.info(f"Starting training on {grid_size}x{grid_size} grid")
        logger.info(f"Goals: G1(5,8), G2(4,2), G3(9,9)")
        logger.info(f"Static Obstacles: (6,6), (3,4)")
        logger.info(f"Moving Obstacle Path: (7,4)->(7,3)->(7,2)->(6,2)->(5,2)->(6,2)->(7,2)->(7,3)->(7,4)->(7,5)->(7,6)->(7,7)->(7,8)->(7,9)->(8,9)")
        logger.info(f"Training for {total_episodes} episodes")
        
        start_time = time.time()
        best_policy, best_ltl_score = agent.train(
            episodes=total_episodes, 
            verify_interval=verify_interval
        )
        training_time = time.time() - start_time
        
        agent.save_training_stats(results_dir)
        agent.verifier.save_probabilities_to_file(agent, results_dir)
        
        logger.info("\nTraining Summary:")
        logger.info(f"Training time: {training_time/60:.1f} minutes")
        logger.info(f"Best LTL Score: {best_ltl_score:.4f}")
        logger.info(f"Results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()