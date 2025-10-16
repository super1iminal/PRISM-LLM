# PRISM-Guided-Learning

This repository contains implementation of Robot Mission Adaptation with Quantitative Guarantees for Heterogeneous Requirements

## Overview

This project implements three methods :

1. ** Standard Q-Learning (qlearningwithoutpmc.py) **
   Baseline : Q-Learning implementation without PMC integration
2. ** Q Learning with PMC (q learning withPMC.py) **
   Q-Learning with Probabilistic Model Checking

3. ** Counterfactual Learning with PMC (counterfactural learning withPMC.py) **
   Enhanced Q-learning using counterfactual reasoning and PMC

All three implementations address the problem of navigating in a grid model with goals. obstacles and a moving person.

## Requirements:

-Python 3.6+ (I use 3.13)

-Numpy

-Matpotlib

-Seaborn

-langchain and langchain-openai

-an openai api key

-PRISM Model Checker (for PMC integration)

## Installation

1.Download the repo

2.Install required Python Packages

3.Install PRISM (https://www.prismmodelchecker.org/download.php)

4.Put the closest directory to bin prism-xxx-xxx in the repo's root and rename it to prism-4.9

5.Update the PRISM path in the code (in the 'get_prism_path()' functions and in `/prism-4.9/bin/prism` executable)

6.Set your OPENAI_API_KEY environment variable to your openai key (I do it in .bashrc, not sure how to do it on windows)

# Problem Description

[Visual example of 2x2 grid with 1 goal and 1 obstacle](https://excalidraw.com/#json=MjBL7CyFVRjrzrOSkrB2H,USHNMvlESIBBMSa9X4yFmQ)
[Github](https://github.com/super1iminal/PRISM-LLM?tab=readme-ov-file)

## Input

The input to the problem, i.e., the domain, is an n by n grid of squares, a set of goals that must be accomplished in order, and a set of obstacles that block movement.

The combination of position and goal states (accomplished/not accomplished) make up the "state".

## Output

The output to the problem is a set of four values for each state. These values indicate affinity for each direction. They are called Q-values.

## Evaluation

Currently, PRISM outputs a set of probabilities corresponding to:

1. The probability of the goals being achieved individually.
2. The probability of each sequence of goals being achieved (i.e, goal 1; goal 1 -> 2; goal 1, 2 -> 3, ...).
3. The probability of hitting a moving obstacle.

This assumes that the starting point is (0, 0). This score is weighted arbitrarily to produce an "ltl score". For now, this is used as a simple evaluation metric.

### Transition Probabilities

PRISM uses softmax with a temperature $\tau$ to turn Q-values (affinity for a direction) into transition probabilities.

$$
P(a|s) = \frac{\exp(Q(s, a)/\tau)}{\sum_{a'}\exp(Q(s,a')/\tau)}
$$

### Basic LLM testing

I have refactored the code to work with any number of obstacles and any number of goals. When testing the LLM in a 2x2 grid with 1 obstacle and 1 goal, the LLM achieves the best path given the current evaluation metrics.

## Next steps

1. Set transition probability to 1 in the direction of the highest Q-value and 0 everywhere else.
2. Make the LLM only plan along a single path, not a set of Q-values for each state, and modify PRISM accordingly.
3. Allow the LLM to plan in multiple directions, not only a single direction.
4. Expand to 3x3 grid
5. Add another goal
6. Add more static obstacles
7. Introduce moving obstacles

## Possible improvements

- Calculate the average probability of reaching a goal from any arbitrary position.
- Use PRISM's reward model to calculate a more interesting result (e.g., lower reward for high probability of hitting obstacles, etc.). Currently, the PRISM output does not really interpret the probability of hitting obstacles well at all.
