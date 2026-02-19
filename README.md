# PRISM-Guided-Learning

This repository contains implementation of LLM Path Planning with Quantitative Guarantees

## Requirements

- included a .yml with required libraries
- an openai api key
- a gemini api key
- PRISM Model Checker (for PMC integration)

## Installation

1.Download the repo

2.Install required Python Packages

3.Install PRISM (https://www.prismmodelchecker.org/download.php)

4.Update the PRISM path in the code (in the 'get_prism_path()' functions and in `/prism-4.9/bin/prism` executable)

5.Set your OPENAI_API_KEY environment variable to your openai key (I do it in .bashrc, not sure how to do it on windows)

## Structure

- Final results are in `PRISM-Guided-Learning/out/results/100-balanced-paper-results`

- Source files are in `PRISM-Guided-Learning/src`

- Instructions for running the code are above

