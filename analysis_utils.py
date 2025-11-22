"""
Analysis utilities for Economic Networks Model

Provides visualization and analysis functions for model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
from collections import defaultdict


def plot_time_series(history: Dict, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot time series of key network metrics.
    
    Args:
        history: Model history dictionary
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Network Evolution Over Time', fontsize=16)
    
    # Number of firms and links
    ax = axes[0, 0]
    ax.plot(history['period'], history['num_firms'], label='Firms', linewidth=2)
    ax.plot(history['period'], history['num_links'], label='Links', linewidth=2)
    ax.set_xlabel('Period')
    ax.set_ylabel('Count')
    ax.set_title('Firms and Links')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average degree
    ax = axes[0, 1]
    ax.plot(history['period'], history['avg_degree'], linewidth=2, color='green')
    ax.set_xlabel('Period')
    ax.set_ylabel('Average Degree')
    ax.set_title('Network Connectivity')
    ax.grid(True, alpha=0.3)
    
    # Average profit
    ax = axes[0, 2]
    ax.plot(history['period'], history['avg_profit'], linewidth=2, color='orange')
    ax.set_xlabel('Period')
    ax.set_ylabel('Average Profit')
    ax.set_title('Average Profit')
    ax.grid(True, alpha=0.3)
    
    # Giant component size
    ax = axes[1, 0]
    ax.plot(history['period'], history['giant_component_size'], 
            linewidth=2, color='red')
    ax.set_xlabel('Period')
    ax.set_ylabel('Fraction in Giant Component')
    ax.set_title('Giant Component Size')
    ax.grid(True, alpha=0.3)
    
    # Density
    ax = axes[1, 1]
    ax.plot(history['period'], history['density'], linewidth=2, color='purple')
    ax.set_xlabel('Period')
    ax.set_ylabel('Density')
    ax.set_title('Network Density')
    ax.grid(True, alpha=0.3)
    
    # Clustering
    ax = axes[1, 2]
    ax.plot(history['period'], history['clustering'], linewidth=2, color='brown')
    ax.set_xlabel('Period')
    ax.set_ylabel('Clustering Coefficient')
    ax.set_title('Clustering Coefficient')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_degree_distribution(degree_dist: Dict[int, int], 
                             log_scale: bool = True,
                             figsize: Tuple[int, int] = (10, 6)):
    """
    Plot degree distribution and fit power law.
    
    Args:
        degree_dist: Dictionary mapping degree to frequency
        log_scale: Whether to use log-log scale
        figsize: Figure size
    """
    degrees = np.array(sorted(degree_dist.keys()))
    frequencies = np.array([degree_dist[d] for d in degrees])
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Regular scale
    ax = axes[0]
    ax.bar(degrees, frequencies, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Number of Firms')
    ax.set_title('Degree Distribution')
    ax.grid(True, alpha=0.3)
    
    # Log-log scale with power law fit
    ax = axes[1]
    if log_scale and len(degrees) > 2:
        # Filter out zero degree and zero frequency
        mask = (degrees > 0) & (frequencies > 0)
        log_degrees = np.log(degrees[mask])
        log_frequencies = np.log(frequencies[mask])
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_degrees, log_frequencies
        )
        
        # Plot
        ax.scatter(log_degrees, log_frequencies, s=100, alpha=0.7, label='Data')
        ax.plot(log_degrees, slope * log_degrees + intercept, 
                'r--', linewidth=2, 
                label=f'Power law fit: slope={slope:.2f}\n$R^2$={r_value**2:.3f}')
        ax.set_xlabel('Log(Degree)')
        ax.set_ylabel('Log(Number of Firms)')
        ax.set_title('Degree Distribution (Log-Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_experiments(results_list: List[Dict], 
                       labels: List[str],
                       metric: str = 'avg_degree',
                       figsize: Tuple[int, int] = (12, 6)):
    """
    Compare multiple experimental results.
    
    Args:
        results_list: List of experiment result dictionaries
        labels: Labels for each experiment
        metric: Which metric to compare
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot of final values
    ax = axes[0]
    final_values = []
    for results in results_list:
        values = [run['final_metrics'][metric] for run in results['runs']]
        final_values.append(values)
    
    bp = ax.boxplot(final_values, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Time series comparison (mean trajectory)
    ax = axes[1]
    for results, label in zip(results_list, labels):
        # Average across runs
        all_trajectories = [run['history'][metric] for run in results['runs']]
        mean_trajectory = np.mean(all_trajectories, axis=0)
        periods = results['runs'][0]['history']['period']
        
        ax.plot(periods, mean_trajectory, linewidth=2, label=label, alpha=0.8)
    
    ax.set_xlabel('Period')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def statistical_comparison(results1: Dict, results2: Dict, 
                           metric: str = 'avg_degree') -> Dict:
    """
    Perform statistical comparison between two experiments.
    
    Args:
        results1: First experiment results
        results2: Second experiment results
        metric: Metric to compare
        
    Returns:
        Dictionary with statistical test results
    """
    values1 = [run['final_metrics'][metric] for run in results1['runs']]
    values2 = [run['final_metrics'][metric] for run in results2['runs']]
    
    # Descriptive statistics
    stats_dict = {
        'metric': metric,
        'experiment1': {
            'mean': np.mean(values1),
            'std': np.std(values1),
            'median': np.median(values1),
            'n': len(values1)
        },
        'experiment2': {
            'mean': np.mean(values2),
            'std': np.std(values2),
            'median': np.median(values2),
            'n': len(values2)
        }
    }
    
    # T-test (assumes unequal variances)
    t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
    stats_dict['t_test'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05
    }
    
    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_mw = stats.mannwhitneyu(values1, values2, alternative='two-sided')
    stats_dict['mann_whitney'] = {
        'u_statistic': u_stat,
        'p_value': p_value_mw,
        'significant_at_0.05': p_value_mw < 0.05
    }
    
    return stats_dict


def create_summary_table(results_list: List[Dict], 
                        labels: List[str],
                        metrics: List[str] = None) -> pd.DataFrame:
    """
    Create summary statistics table for multiple experiments.
    
    Args:
        results_list: List of experiment results
        labels: Labels for each experiment
        metrics: List of metrics to include (None = all)
        
    Returns:
        Pandas DataFrame with summary statistics
    """
    if metrics is None:
        # Get all available metrics from first result
        metrics = list(results_list[0]['runs'][0]['final_metrics'].keys())
    
    summary_data = []
    
    for results, label in zip(results_list, labels):
        row = {'Experiment': label}
        
        for metric in metrics:
            values = [run['final_metrics'][metric] for run in results['runs']]
            row[f'{metric}_mean'] = np.mean(values)
            row[f'{metric}_std'] = np.std(values)
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    return df


def plot_network_snapshot(model, figsize: Tuple[int, int] = (12, 10),
                          node_size_by_degree: bool = True,
                          color_by_profit: bool = True):
    """
    Plot a snapshot of the network structure.
    
    Args:
        model: EconomicNetworkModel instance
        figsize: Figure size
        node_size_by_degree: Scale node size by degree
        color_by_profit: Color nodes by profit level
    """
    import networkx as nx
    
    fig, ax = plt.subplots(figsize=figsize)
    
    G = model.G
    
    # Calculate node sizes
    if node_size_by_degree:
        degrees = dict(G.degree())
        node_sizes = [100 + 50 * degrees.get(node, 0) for node in G.nodes()]
    else:
        node_sizes = 200
    
    # Calculate node colors
    if color_by_profit:
        profits = [model._calculate_profit(node, G) for node in G.nodes()]
        node_colors = profits
        cmap = 'RdYlGn'
    else:
        node_colors = 'lightblue'
        cmap = None
    
    # Use spring layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          cmap=cmap, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    
    ax.set_title(f'Network Snapshot (Period {model.current_period})', fontsize=16)
    ax.axis('off')
    
    # Add colorbar if coloring by profit
    if color_by_profit:
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                   norm=plt.Normalize(vmin=min(profits), 
                                                     vmax=max(profits)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Profit', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig


def export_results_to_csv(results: Dict, filename: str):
    """
    Export experiment results to CSV file.
    
    Args:
        results: Experiment results dictionary
        filename: Output filename
    """
    # Collect all run data
    all_data = []
    
    for run_idx, run in enumerate(results['runs']):
        history = run['history']
        n_periods = len(history['period'])
        
        for t in range(n_periods):
            row = {
                'run': run_idx,
                'delta': results['delta'],
                'exit_threshold': results['exit_threshold'],
                'period': history['period'][t],
                'num_firms': history['num_firms'][t],
                'num_links': history['num_links'][t],
                'avg_degree': history['avg_degree'][t],
                'avg_profit': history['avg_profit'][t],
                'giant_component_size': history['giant_component_size'][t],
                'density': history['density'][t],
                'clustering': history['clustering'][t]
            }
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")


if __name__ == "__main__":
    print("Analysis utilities module for Economic Networks Model")
    print("Import this module to use visualization and analysis functions")
