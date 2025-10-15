from . import PrismVerifier
import logging
from typing import Dict, Tuple, List, Any
import os

class SimplifiedVerifier:
    def __init__(self, prism_verifier: PrismVerifier):
        self.prism_verifier = prism_verifier
        self.logger = logging.getLogger(__name__)
        self.ltl_probabilities = []
        self.verification_history = []  

    # use generate_prism_model from PrismModelGenerator for this
    def verify_policy(self, model_str: str) -> float:
        property_str = (
            "P=? [ F \"at_goal1\" ];\n"
            "P=? [ F \"at_goal2\" ];\n"
            "P=? [ F \"at_goal3\" ];\n"
            "P=? [ !\"at_goal2\" U \"at_goal1\" ];\n"
            "P=? [ !\"at_goal3\" U (\"at_goal1\" & (!\"at_goal3\" U \"at_goal2\")) ];\n"
            "P=? [ (!\"at_goal2\" U \"at_goal1\") & (!\"at_goal3\" U (\"at_goal1\" & (!\"at_goal3\" U \"at_goal2\"))) ];\n"
            "P=? [ G<=30 !\"at_obstacle\" ];\n"  # Increased steps for larger grid
        )
        
        probabilities = self.prism_verifier.verify_property(model_str, property_str)
        self.ltl_probabilities.append(probabilities)
        
        score = self._calculate_score(probabilities)
        self.logger.info(f"Goal Sequence Probabilities: {probabilities}")
        self.logger.info(f"Combined Score: {score:.4f}")
        return score

    def _calculate_score(self, results: List[float]) -> float:
        """Calculate policy score with balanced weights"""
        if len(results) >= 7:
            goal1_reach = results[0]
            goal2_reach = results[1] 
            goal3_reach = results[2]
            seq1 = results[3]  # G1 before G2
            seq2 = results[4]  # G1G2 before G3
            seq3 = results[5]  # Complete sequence
            avoid_obstacle = results[6]  # Obstacle avoidance
            
            # Balanced weights that sum to 1
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
                # Write header with corrected order
                f.write("Episode,Reach_G1,Reach_G2,Reach_G3,G1_before_G2,G1G2_before_G3,"
                    "Complete_Sequence,Avoid_Obstacle,Episode_Reward,Path_Exists,Score\n")
                
                # Write data for each episode with corrected order
                for episode, probs in enumerate(self.ltl_probabilities):
                    prob_str = ','.join(f"{p:.4f}" for p in probs)
                    reward = agent.episode_rewards[episode] if episode < len(agent.episode_rewards) else 0.0
                    score = self._calculate_score(probs)
                    f.write(f"{episode},{prob_str},{reward:.4f},1.0,{score:.4f}\n")  # Added Path_Exists=1.0
                    
                self.logger.info(f"Saved LTL probabilities and rewards to {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving probabilities to file: {str(e)}")