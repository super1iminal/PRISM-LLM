import datetime
from enum import Enum
import os
from time import time
from utils.DataLoader import DataLoader
from utils.Logging import setup_logger
from planners.VanillaLLMPlanner import VanillaLLMPlanner
from planners.UniformPlanner import BaselinePlanner
from planners.RLCounterfactual import LTLGuidedQLearningWithObstacle
from planners.FeedbackLLMPlanner import FeedbackLLMPlanner
from config.Settings import EVAL_PATH, RESULTS_PATH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


class EvalModel(Enum):
    LLM_VANILLA = 1
    LLM_FEEDBACK = 2
    RL = 3
    RANDOM = 4
    UNIFORM = 5


def main():
    logger = setup_logger("eval", console_output=False, log_path=EVAL_PATH) 
    
    dataloader = DataLoader("PRISM-Guided-Learning/data/grid_1_sample.csv")
    dataloader.load_data()
    
    models = [EvalModel.LLM_FEEDBACK]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_PATH, f"{len(dataloader.data)}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    model_dfs = {}
    
    for modelType in models:
        model = get_model(modelType)
        
        start_time = time()
        results = model.evaluate(dataloader, parallel=True)
        end_time = time()
        delta_time = end_time - start_time
        logger.info(f"Results for {modelType.name} (in {delta_time:.2f} seconds):")
        for idx, result in enumerate(results):
            logger.info(f"  Gridworld {idx+1}: LTL Score = {result['LTL_Score']:.4f}")
        logger.info("\n"*2)
        
        df = output_results(results, dataloader, modelType.name, run_dir)
        model_dfs[modelType.name] = df
        
        plot_single_model_results(df, modelType.name, run_dir)
    
    plot_all_models_comparison(model_dfs, run_dir)
    run_statistical_tests(model_dfs, run_dir)


def get_model(model_type: EvalModel):
    if model_type == EvalModel.LLM_VANILLA:
        return VanillaLLMPlanner()
    elif model_type == EvalModel.LLM_FEEDBACK:
        return FeedbackLLMPlanner(10)
    elif model_type == EvalModel.RL:
        return LTLGuidedQLearningWithObstacle()
    else:
        raise ValueError("Unknown model type")


def output_results(results, dataloader, model_name, run_dir):
    rows = []
    
    for idx, result in enumerate(results):
        size = dataloader.data[idx][0].size
        goals = len(dataloader.data[idx][0].goals)
        obstacles = len(dataloader.data[idx][0].static_obstacles)
        complexity = dataloader.data[idx][1]  # BFS steps
        
        ltl_score = result['LTL_Score']
        prism_probabilities = result['Prism_Probabilities']
        evaluation_time = result['Evaluation_Time']
        
        rows.append({
            'ltl_score': ltl_score,
            'prism_probability': prism_probabilities,
            'evaluation_time': evaluation_time,
            'size': size,
            'goals': goals,
            'obstacles': obstacles,
            'complexity': complexity
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(f"{run_dir}/results_{model_name}.csv", index=False)
    
    return df


def plot_single_model_results(df, model_name, run_dir):
    """
    Generate plots for a single model:
    - LTL score vs size, goals, obstacles, complexity
    - Evaluation time vs size, goals, obstacles, complexity
    Both scatter and line plot versions are generated.
    """
    x_vars = [
        ('size', 'Grid Size'),
        ('goals', 'Number of Goals'),
        ('obstacles', 'Number of Obstacles'),
        ('complexity', 'Expected Path Length (BFS Steps)')
    ]
    
    for plot_type in ['scatter', 'line']:
        # LTL Score plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_name} - LTL Score Analysis ({plot_type.capitalize()})', fontsize=14, fontweight='bold')
        
        for ax, (x_col, x_label) in zip(axes.flatten(), x_vars):
            df_sorted = df.sort_values(by=x_col)
            if plot_type == 'scatter':
                ax.scatter(df_sorted[x_col], df_sorted['ltl_score'], alpha=0.7, edgecolors='black', linewidth=0.5)
            else:
                ax.plot(df_sorted[x_col], df_sorted['ltl_score'], marker='o', alpha=0.7, linewidth=1.5, markersize=5)
            ax.set_xlabel(x_label)
            ax.set_ylabel('LTL Score')
            ax.set_title(f'LTL Score vs {x_label}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'{model_name}_ltl_score_{plot_type}.png'), dpi=150)
        plt.close()
        
        # Evaluation Time plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_name} - Evaluation Time Analysis ({plot_type.capitalize()})', fontsize=14, fontweight='bold')
        
        for ax, (x_col, x_label) in zip(axes.flatten(), x_vars):
            df_sorted = df.sort_values(by=x_col)
            if plot_type == 'scatter':
                ax.scatter(df_sorted[x_col], df_sorted['evaluation_time'], alpha=0.7, edgecolors='black', linewidth=0.5)
            else:
                ax.plot(df_sorted[x_col], df_sorted['evaluation_time'], marker='o', alpha=0.7, linewidth=1.5, markersize=5)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Evaluation Time (s)')
            ax.set_title(f'Evaluation Time vs {x_label}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'{model_name}_eval_time_{plot_type}.png'), dpi=150)
        plt.close()


def plot_all_models_comparison(model_dfs, run_dir):
    """
    Generate comparison plots with all models overlayed:
    - LTL score vs size, goals, obstacles, complexity
    - Evaluation time vs size, goals, obstacles, complexity
    Both scatter and line plot versions are generated.
    """
    if not model_dfs:
        return
    
    x_vars = [
        ('size', 'Grid Size'),
        ('goals', 'Number of Goals'),
        ('obstacles', 'Number of Obstacles'),
        ('complexity', 'Expected Path Length (BFS Steps)')
    ]
    
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']
    
    for plot_type in ['scatter', 'line']:
        # LTL Score comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(f'Model Comparison - LTL Score ({plot_type.capitalize()})', fontsize=14, fontweight='bold')
        
        for ax, (x_col, x_label) in zip(axes.flatten(), x_vars):
            for idx, (model_name, df) in enumerate(model_dfs.items()):
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                df_sorted = df.sort_values(by=x_col)
                if plot_type == 'scatter':
                    ax.scatter(df_sorted[x_col], df_sorted['ltl_score'], 
                              alpha=0.7, label=model_name, color=color, marker=marker,
                              edgecolors='black', linewidth=0.5)
                else:
                    ax.plot(df_sorted[x_col], df_sorted['ltl_score'], 
                           marker=marker, alpha=0.7, label=model_name, color=color,
                           linewidth=1.5, markersize=5)
            ax.set_xlabel(x_label)
            ax.set_ylabel('LTL Score')
            ax.set_title(f'LTL Score vs {x_label}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'comparison_ltl_score_{plot_type}.png'), dpi=150)
        plt.close()
        
        # Evaluation Time comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(f'Model Comparison - Evaluation Time ({plot_type.capitalize()})', fontsize=14, fontweight='bold')
        
        for ax, (x_col, x_label) in zip(axes.flatten(), x_vars):
            for idx, (model_name, df) in enumerate(model_dfs.items()):
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                df_sorted = df.sort_values(by=x_col)
                if plot_type == 'scatter':
                    ax.scatter(df_sorted[x_col], df_sorted['evaluation_time'], 
                              alpha=0.7, label=model_name, color=color, marker=marker,
                              edgecolors='black', linewidth=0.5)
                else:
                    ax.plot(df_sorted[x_col], df_sorted['evaluation_time'], 
                           marker=marker, alpha=0.7, label=model_name, color=color,
                           linewidth=1.5, markersize=5)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Evaluation Time (s)')
            ax.set_title(f'Evaluation Time vs {x_label}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'comparison_eval_time_{plot_type}.png'), dpi=150)
        plt.close()
    
    # Also create box plots for overall comparison
    plot_box_comparison(model_dfs, run_dir)


def plot_box_comparison(model_dfs, run_dir):
    """
    Generate box plots comparing LTL scores and evaluation times across models.
    """
    model_names = list(model_dfs.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Comparison - Distribution Summary', fontsize=14, fontweight='bold')
    
    # LTL Score box plot
    ltl_data = [model_dfs[name]['ltl_score'].values for name in model_names]
    axes[0].boxplot(ltl_data, labels=model_names)
    axes[0].set_ylabel('LTL Score')
    axes[0].set_title('LTL Score Distribution by Model')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Evaluation Time box plot
    time_data = [model_dfs[name]['evaluation_time'].values for name in model_names]
    axes[1].boxplot(time_data, labels=model_names)
    axes[1].set_ylabel('Evaluation Time (s)')
    axes[1].set_title('Evaluation Time Distribution by Model')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'comparison_box_plots.png'), dpi=150)
    plt.close()


def run_statistical_tests(model_dfs, run_dir):
    """
    Run Wilcoxon Signed-Rank tests for research questions:
    - RQ1: LLM_VANILLA vs LLM_FEEDBACK (effect of feedback)
    - RQ2: LLM_FEEDBACK vs RL (feedback LLM vs RL performance)
    
    Outputs results to CSV.
    """
    comparisons = [
        ("RQ1", "LLM_VANILLA", "LLM_FEEDBACK", "Effect of feedback on LLM"),
        ("RQ2", "LLM_FEEDBACK", "RL", "Feedback LLM vs RL"),
    ]
    
    results = []
    
    for rq, model_a, model_b, description in comparisons:
        # Check both models were evaluated
        if model_a not in model_dfs:
            print(f"Skipping {rq}: {model_a} was not evaluated")
            continue
        if model_b not in model_dfs:
            print(f"Skipping {rq}: {model_b} was not evaluated")
            continue
            
        df_a = model_dfs[model_a]
        df_b = model_dfs[model_b]
        
        # Check same number of samples
        if len(df_a) != len(df_b):
            print(f"Skipping {rq}: mismatched sample sizes ({len(df_a)} vs {len(df_b)})")
            continue
        
        for metric in ['ltl_score', 'evaluation_time']:
            try:
                stat, p_value = wilcoxon(df_a[metric], df_b[metric])
            except ValueError as e:
                # Handle case where all differences are zero
                print(f"Warning {rq} {metric}: {e}")
                stat, p_value = np.nan, np.nan
            
            results.append({
                'research_question': rq,
                'description': description,
                'metric': metric,
                'model_a': model_a,
                'model_b': model_b,
                'n_samples': len(df_a),
                'median_a': np.median(df_a[metric]),
                'median_b': np.median(df_b[metric]),
                'mean_a': np.mean(df_a[metric]),
                'mean_b': np.mean(df_b[metric]),
                'std_a': np.std(df_a[metric]),
                'std_b': np.std(df_b[metric]),
                'W_statistic': stat,
                'p_value': p_value,
                'significant_0.05': p_value < 0.05 if not np.isnan(p_value) else False
            })
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(run_dir, 'statistical_tests_wsr.csv'), index=False)
        
        # Print summary to console
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS (Wilcoxon Signed-Rank Test)")
        print("="*70)
        for _, row in results_df.iterrows():
            sig = "*" if row['significant_0.05'] else ""
            print(f"\n{row['research_question']}: {row['description']}")
            print(f"  Metric: {row['metric']}")
            print(f"  {row['model_a']}: median={row['median_a']:.4f}, mean={row['mean_a']:.4f} ± {row['std_a']:.4f}")
            print(f"  {row['model_b']}: median={row['median_b']:.4f}, mean={row['mean_b']:.4f} ± {row['std_b']:.4f}")
            print(f"  W={row['W_statistic']:.1f}, p={row['p_value']:.4f} {sig}")
        print("\n" + "="*70)
        
        return results_df
    
    return None


if __name__ == "__main__":
    main()