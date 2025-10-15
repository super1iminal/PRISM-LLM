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
