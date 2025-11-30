import os
import pandas as pd
import matplotlib.pyplot as plt

from config.Settings import RESULTS_PATH

def load_results(run_dir):
    """
    Load results CSVs from a given directory.
    Returns a dictionary mapping model names to DataFrames.
    """
    model_dfs = {}
    
    for filename in os.listdir(run_dir):
        if filename.startswith("results_") and filename.endswith(".csv"):
            model_name = filename.replace("results_", "").replace(".csv", "")
            filepath = os.path.join(run_dir, filename)
            df = pd.read_csv(filepath)
            
            # Add obstacle density column (proportion of grid that is obstacles)
            df['obstacle_density'] = df['obstacles'] / (df['size'] ** 2)
            
            model_dfs[model_name] = df
            print(f"Loaded {model_name}: {len(df)} samples")
    
    return model_dfs


def plot_single_model(model_dfs, model_name, run_dir):
    """
    Generate scatter plots for a single model:
    - LTL score vs size, obstacle_density, obstacles, complexity
    - Evaluation time vs size, obstacle_density, obstacles, complexity
    """
    if model_name not in model_dfs:
        print(f"Skipping single model plot: {model_name} not found")
        return
    
    df = model_dfs[model_name]
    
    x_vars = [
        ('size', 'Grid Size'),
        ('obstacle_density', 'Obstacle Density (Obstacles / Size²)'),
        ('obstacles', 'Number of Obstacles'),
        ('complexity', 'Expected Path Length (BFS Steps)')
    ]
    
    # LTL Score plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name} - LTL Score Analysis', fontsize=14, fontweight='bold')
    
    for ax, (x_col, x_label) in zip(axes.flatten(), x_vars):
        df_sorted = df.sort_values(by=x_col)
        ax.scatter(df_sorted[x_col], df_sorted['ltl_score'],
                   alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel('LTL Score')
        ax.set_title(f'LTL Score vs {x_label}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'{model_name}_ltl_score_scatter.png'), dpi=150)
    plt.close()
    print(f"Saved: {model_name}_ltl_score_scatter.png")
    
    # Evaluation Time plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name} - Evaluation Time Analysis', fontsize=14, fontweight='bold')
    
    for ax, (x_col, x_label) in zip(axes.flatten(), x_vars):
        df_sorted = df.sort_values(by=x_col)
        ax.scatter(df_sorted[x_col], df_sorted['evaluation_time'],
                   alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Evaluation Time (s)')
        ax.set_title(f'Evaluation Time vs {x_label}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'{model_name}_eval_time_scatter.png'), dpi=150)
    plt.close()
    print(f"Saved: {model_name}_eval_time_scatter.png")


def plot_pairwise_comparison(model_dfs, model_a, model_b, run_dir):
    """
    Generate scatter plots comparing two models:
    - LTL score vs size, obstacle_density, obstacles, complexity
    - Evaluation time vs size, obstacle_density, obstacles, complexity
    """
    if model_a not in model_dfs:
        print(f"Skipping comparison: {model_a} not found")
        return
    if model_b not in model_dfs:
        print(f"Skipping comparison: {model_b} not found")
        return
    
    df_a = model_dfs[model_a]
    df_b = model_dfs[model_b]
    
    x_vars = [
        ('size', 'Grid Size'),
        ('obstacle_density', 'Obstacle Density (Obstacles / Size²)'),
        ('obstacles', 'Number of Obstacles'),
        ('complexity', 'Expected Path Length (BFS Steps)')
    ]
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for model_a, Orange for model_b
    markers = ['o', 's']
    
    comparison_name = f"{model_a}_vs_{model_b}"
    
    # LTL Score comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'{model_a} vs {model_b} - LTL Score Comparison', fontsize=14, fontweight='bold')
    
    for ax, (x_col, x_label) in zip(axes.flatten(), x_vars):
        df_a_sorted = df_a.sort_values(by=x_col)
        df_b_sorted = df_b.sort_values(by=x_col)
        
        ax.scatter(df_a_sorted[x_col], df_a_sorted['ltl_score'],
                   alpha=0.7, label=model_a, color=colors[0], marker=markers[0],
                   edgecolors='black', linewidth=0.5)
        ax.scatter(df_b_sorted[x_col], df_b_sorted['ltl_score'],
                   alpha=0.7, label=model_b, color=colors[1], marker=markers[1],
                   edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('LTL Score')
        ax.set_title(f'LTL Score vs {x_label}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'{comparison_name}_ltl_score_scatter.png'), dpi=150)
    plt.close()
    print(f"Saved: {comparison_name}_ltl_score_scatter.png")
    
    # Evaluation Time comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'{model_a} vs {model_b} - Evaluation Time Comparison', fontsize=14, fontweight='bold')
    
    for ax, (x_col, x_label) in zip(axes.flatten(), x_vars):
        df_a_sorted = df_a.sort_values(by=x_col)
        df_b_sorted = df_b.sort_values(by=x_col)
        
        ax.scatter(df_a_sorted[x_col], df_a_sorted['evaluation_time'],
                   alpha=0.7, label=model_a, color=colors[0], marker=markers[0],
                   edgecolors='black', linewidth=0.5)
        ax.scatter(df_b_sorted[x_col], df_b_sorted['evaluation_time'],
                   alpha=0.7, label=model_b, color=colors[1], marker=markers[1],
                   edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Evaluation Time (s)')
        ax.set_title(f'Evaluation Time vs {x_label}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'{comparison_name}_eval_time_scatter.png'), dpi=150)
    plt.close()
    print(f"Saved: {comparison_name}_eval_time_scatter.png")


def main():
    # Specify the directory containing results CSVs
    run_dir = RESULTS_PATH + "100_20251128_21-54-39"
    
    # Load all results
    model_dfs = load_results(run_dir)
    
    if not model_dfs:
        print(f"No results found in {run_dir}")
        return
    
    # Generate single model scatter plots for LLM_FEEDBACK
    plot_single_model(model_dfs, "LLM_FEEDBACK", run_dir)
    
    # Generate pairwise comparison plots
    # RQ1: Effect of feedback on LLM
    plot_pairwise_comparison(model_dfs, "LLM_VANILLA", "LLM_FEEDBACK", run_dir)
    
    # RQ2: Feedback LLM vs RL
    plot_pairwise_comparison(model_dfs, "LLM_FEEDBACK", "RL", run_dir)


if __name__ == "__main__":
    main()