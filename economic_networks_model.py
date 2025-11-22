"""
Economic Networks Model
Based on "The Evolution of Economic Networks" by Pedro P. Romero (2010)

This module implements an agent-based computational model of economic networks
where firms make strategic decisions based on profits and information generated
through their immediate social network.

Author: Implementation based on Romero (2010)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class Firm:
    """
    Represents a firm (node) in the economic network.
    
    Attributes:
        firm_id: Unique identifier for the firm
        w: Market valuation of firm's innovation contribution
        neighbors: Set of firm IDs that this firm is directly connected to
        profit_history: List of profits over time
        isolated_periods: Number of consecutive periods firm has been isolated
        negative_profit_periods: Number of consecutive periods with negative profits
        age: Number of periods firm has been in the market
    """
    firm_id: int
    w: float
    neighbors: set = field(default_factory=set)
    profit_history: List[float] = field(default_factory=list)
    isolated_periods: int = 0
    negative_profit_periods: int = 0
    age: int = 0
    
    def __hash__(self):
        return hash(self.firm_id)
    
    def __eq__(self, other):
        return self.firm_id == other.firm_id


class EconomicNetworkModel:
    """
    Agent-based model of an evolving economic network.
    
    The model implements:
    - Endogenous link formation and deletion
    - Endogenous firm entry and exit
    - Profit-based strategic decisions
    - Local information-based interactions
    """
    
    def __init__(self, 
                 delta: float = 0.95,
                 exit_threshold: int = 4,
                 cost_mean: float = 0.5,
                 cost_std: float = 0.1,
                 w_mean: float = 1.0,
                 w_std: float = 0.2,
                 random_seed: Optional[int] = None):
        """
        Initialize the economic network model.
        
        Args:
            delta: Externality/transferability parameter (0 < delta < 1)
            exit_threshold: Number of consecutive bad periods before exit
            cost_mean: Mean of link formation costs
            cost_std: Standard deviation of link formation costs
            w_mean: Mean of firm innovation valuations
            w_std: Standard deviation of firm innovation valuations
            random_seed: Random seed for reproducibility
        """
        self.delta = delta
        self.exit_threshold = exit_threshold
        self.cost_mean = cost_mean
        self.cost_std = cost_std
        self.w_mean = w_mean
        self.w_std = w_std
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Network state
        self.firms: Dict[int, Firm] = {}
        self.G = nx.Graph()  # NetworkX graph for analysis
        self.link_costs: Dict[Tuple[int, int], float] = {}
        
        # Simulation state
        self.current_period = 0
        self.next_firm_id = 0
        
        # Data collection
        self.history = {
            'period': [],
            'num_firms': [],
            'num_links': [],
            'avg_degree': [],
            'avg_profit': [],
            'giant_component_size': [],
            'density': [],
            'clustering': []
        }
        
    def _generate_firm_valuation(self) -> float:
        """Generate market valuation for a new firm."""
        return max(0.1, np.random.normal(self.w_mean, self.w_std))
    
    def _generate_link_cost(self) -> float:
        """Generate cost for forming a link."""
        return max(0.01, np.random.normal(self.cost_mean, self.cost_std))
    
    def _get_link_cost(self, firm_i: int, firm_j: int) -> float:
        """Get or create link cost between two firms."""
        key = tuple(sorted([firm_i, firm_j]))
        if key not in self.link_costs:
            self.link_costs[key] = self._generate_link_cost()
        return self.link_costs[key]
    
    def _calculate_network_value(self, firm_id: int, G_temp: nx.Graph) -> float:
        """
        Calculate accumulated value for a firm from the network.
        V_i(G_t) = w_i + Σ_{j≠i} δ^{d_ij(g_t)} w_j
        
        Args:
            firm_id: ID of the firm
            G_temp: NetworkX graph (potentially modified for what-if analysis)
            
        Returns:
            Total accumulated value
        """
        if firm_id not in self.firms:
            return 0.0
            
        firm = self.firms[firm_id]
        value = firm.w  # Own innovation value
        
        # Add value from network connections
        if firm_id in G_temp:
            # Get shortest path lengths to all other nodes
            try:
                path_lengths = nx.single_source_shortest_path_length(G_temp, firm_id)
            except:
                path_lengths = {firm_id: 0}
            
            for other_id, distance in path_lengths.items():
                if other_id != firm_id and other_id in self.firms:
                    other_firm = self.firms[other_id]
                    value += (self.delta ** distance) * other_firm.w
        
        return value
    
    def _calculate_profit(self, firm_id: int, G_temp: nx.Graph) -> float:
        """
        Calculate profit for a firm.
        π_i(G_t) = V_i(G_t) - Σ_{j∈neighbors} c_ij
        
        Args:
            firm_id: ID of the firm
            G_temp: NetworkX graph (potentially modified for what-if analysis)
            
        Returns:
            Profit value
        """
        value = self._calculate_network_value(firm_id, G_temp)
        
        # Subtract costs of direct links
        costs = 0.0
        if firm_id in G_temp:
            for neighbor_id in G_temp.neighbors(firm_id):
                costs += self._get_link_cost(firm_id, neighbor_id)
        
        return value - costs
    
    def add_firm(self) -> int:
        """
        Add a new firm to the market.
        
        Returns:
            ID of the newly added firm
        """
        firm_id = self.next_firm_id
        self.next_firm_id += 1
        
        w = self._generate_firm_valuation()
        firm = Firm(firm_id=firm_id, w=w)
        self.firms[firm_id] = firm
        self.G.add_node(firm_id)
        
        return firm_id
    
    def remove_firm(self, firm_id: int):
        """
        Remove a firm from the market and delete all its links.
        
        Args:
            firm_id: ID of the firm to remove
        """
        if firm_id not in self.firms:
            return
        
        # Remove all links
        neighbors = list(self.firms[firm_id].neighbors)
        for neighbor_id in neighbors:
            self._delete_link(firm_id, neighbor_id)
        
        # Remove from network
        if firm_id in self.G:
            self.G.remove_node(firm_id)
        
        # Remove firm
        del self.firms[firm_id]
    
    def _create_link(self, firm_i: int, firm_j: int):
        """Create a link between two firms."""
        if firm_i not in self.firms or firm_j not in self.firms:
            return
        
        self.firms[firm_i].neighbors.add(firm_j)
        self.firms[firm_j].neighbors.add(firm_i)
        self.G.add_edge(firm_i, firm_j)
    
    def _delete_link(self, firm_i: int, firm_j: int):
        """Delete a link between two firms."""
        if firm_i in self.firms and firm_j in self.firms[firm_i].neighbors:
            self.firms[firm_i].neighbors.discard(firm_j)
            self.firms[firm_j].neighbors.discard(firm_i)
        
        if self.G.has_edge(firm_i, firm_j):
            self.G.remove_edge(firm_i, firm_j)
    
    def propose_link(self, new_firm_id: int, target_firm_id: int) -> bool:
        """
        Propose a link between two firms using myopic pairwise stability.
        
        A link is formed if:
        1. Both firms don't get worse off
        2. At least one firm is strictly better off
        
        Args:
            new_firm_id: ID of firm proposing the link
            target_firm_id: ID of firm receiving the proposal
            
        Returns:
            True if link is formed, False otherwise
        """
        if new_firm_id not in self.firms or target_firm_id not in self.firms:
            return False
        
        if new_firm_id == target_firm_id:
            return False
        
        # Already connected
        if target_firm_id in self.firms[new_firm_id].neighbors:
            return False
        
        # Calculate current profits
        profit_i_before = self._calculate_profit(new_firm_id, self.G)
        profit_j_before = self._calculate_profit(target_firm_id, self.G)
        
        # Create temporary graph with new link
        G_temp = self.G.copy()
        G_temp.add_edge(new_firm_id, target_firm_id)
        
        # Calculate profits after link formation
        profit_i_after = self._calculate_profit(new_firm_id, G_temp)
        profit_j_after = self._calculate_profit(target_firm_id, G_temp)
        
        # Check myopic pairwise stability conditions
        both_not_worse = (profit_i_after >= profit_i_before) and (profit_j_after >= profit_j_before)
        at_least_one_better = (profit_i_after > profit_i_before) or (profit_j_after > profit_j_before)
        
        if both_not_worse and at_least_one_better:
            self._create_link(new_firm_id, target_firm_id)
            return True
        
        return False
    
    def check_exit_conditions(self):
        """
        Check exit conditions for all firms and remove those that should exit.
        
        A firm exits if:
        - Negative profits for more than exit_threshold consecutive periods, OR
        - Isolated (degree = 0) for more than exit_threshold consecutive periods
        """
        firms_to_remove = []
        
        for firm_id, firm in self.firms.items():
            # Check if isolated
            is_isolated = len(firm.neighbors) == 0
            
            # Check profit
            current_profit = self._calculate_profit(firm_id, self.G)
            has_negative_profit = current_profit < 0
            
            # Update counters
            if is_isolated:
                firm.isolated_periods += 1
            else:
                firm.isolated_periods = 0
            
            if has_negative_profit:
                firm.negative_profit_periods += 1
            else:
                firm.negative_profit_periods = 0
            
            # Check exit conditions
            if (firm.isolated_periods >= self.exit_threshold or 
                firm.negative_profit_periods >= self.exit_threshold):
                firms_to_remove.append(firm_id)
        
        # Remove firms
        for firm_id in firms_to_remove:
            self.remove_firm(firm_id)
    
    def step(self):
        """
        Execute one time step of the simulation.
        
        Each step consists of:
        1. Add one new firm
        2. New firm proposes link to random existing firm
        3. Check exit conditions
        4. Update firm ages and record data
        """
        self.current_period += 1
        
        # Step 1: Add new firm
        new_firm_id = self.add_firm()
        
        # Step 2: Propose link to random existing firm (if any)
        existing_firms = [fid for fid in self.firms.keys() if fid != new_firm_id]
        if existing_firms:
            # Choose from neighborhood (if connected) or random
            if len(existing_firms) > 0:
                target_id = np.random.choice(existing_firms)
                self.propose_link(new_firm_id, target_id)
        
        # Step 3: Check exit conditions
        self.check_exit_conditions()
        
        # Step 4: Update firm ages and calculate profits
        for firm_id, firm in self.firms.items():
            firm.age += 1
            profit = self._calculate_profit(firm_id, self.G)
            firm.profit_history.append(profit)
        
        # Step 5: Record statistics
        self._record_statistics()
    
    def _record_statistics(self):
        """Record current network statistics."""
        num_firms = len(self.firms)
        num_links = self.G.number_of_edges()
        
        # Average degree
        if num_firms > 0:
            avg_degree = (2 * num_links) / num_firms
        else:
            avg_degree = 0
        
        # Average profit
        if num_firms > 0:
            avg_profit = np.mean([self._calculate_profit(fid, self.G) 
                                 for fid in self.firms.keys()])
        else:
            avg_profit = 0
        
        # Giant component size
        if num_firms > 0 and num_links > 0:
            largest_cc = max(nx.connected_components(self.G), key=len)
            giant_size = len(largest_cc) / num_firms
        else:
            giant_size = 0
        
        # Density
        if num_firms > 1:
            max_links = num_firms * (num_firms - 1) / 2
            density = num_links / max_links if max_links > 0 else 0
        else:
            density = 0
        
        # Clustering coefficient
        if num_firms > 2 and num_links > 0:
            try:
                clustering = nx.average_clustering(self.G)
            except:
                clustering = 0
        else:
            clustering = 0
        
        # Record
        self.history['period'].append(self.current_period)
        self.history['num_firms'].append(num_firms)
        self.history['num_links'].append(num_links)
        self.history['avg_degree'].append(avg_degree)
        self.history['avg_profit'].append(avg_profit)
        self.history['giant_component_size'].append(giant_size)
        self.history['density'].append(density)
        self.history['clustering'].append(clustering)
    
    def run_simulation(self, num_periods: int, verbose: bool = False):
        """
        Run the simulation for a specified number of periods.
        
        Args:
            num_periods: Number of time periods to simulate
            verbose: If True, print progress information
        """
        for t in range(num_periods):
            self.step()
            
            if verbose and (t + 1) % 100 == 0:
                print(f"Period {t + 1}/{num_periods}: "
                      f"{len(self.firms)} firms, "
                      f"{self.G.number_of_edges()} links, "
                      f"avg degree: {self.history['avg_degree'][-1]:.2f}")
    
    def get_degree_distribution(self) -> Dict[int, int]:
        """
        Get the degree distribution of the network.
        
        Returns:
            Dictionary mapping degree to frequency
        """
        degrees = dict(self.G.degree())
        degree_dist = defaultdict(int)
        for degree in degrees.values():
            degree_dist[degree] += 1
        return dict(degree_dist)
    
    def get_network_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive network metrics.
        
        Returns:
            Dictionary of network metrics
        """
        metrics = {}
        
        num_firms = len(self.firms)
        num_links = self.G.number_of_edges()
        
        metrics['num_firms'] = num_firms
        metrics['num_links'] = num_links
        
        if num_firms > 0:
            metrics['avg_degree'] = (2 * num_links) / num_firms
            metrics['avg_profit'] = np.mean([self._calculate_profit(fid, self.G) 
                                            for fid in self.firms.keys()])
        else:
            metrics['avg_degree'] = 0
            metrics['avg_profit'] = 0
        
        if num_firms > 0 and num_links > 0:
            # Giant component
            largest_cc = max(nx.connected_components(self.G), key=len)
            metrics['giant_component_size'] = len(largest_cc)
            metrics['giant_component_fraction'] = len(largest_cc) / num_firms
            
            # Clustering
            try:
                metrics['avg_clustering'] = nx.average_clustering(self.G)
            except:
                metrics['avg_clustering'] = 0
            
            # Average path length (on largest component)
            if len(largest_cc) > 1:
                subgraph = self.G.subgraph(largest_cc)
                try:
                    metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
                except:
                    metrics['avg_path_length'] = 0
            else:
                metrics['avg_path_length'] = 0
        else:
            metrics['giant_component_size'] = 0
            metrics['giant_component_fraction'] = 0
            metrics['avg_clustering'] = 0
            metrics['avg_path_length'] = 0
        
        # Density
        if num_firms > 1:
            max_links = num_firms * (num_firms - 1) / 2
            metrics['density'] = num_links / max_links if max_links > 0 else 0
        else:
            metrics['density'] = 0
        
        return metrics
    
    def reset(self):
        """Reset the model to initial state."""
        self.firms = {}
        self.G = nx.Graph()
        self.link_costs = {}
        self.current_period = 0
        self.next_firm_id = 0
        self.history = {
            'period': [],
            'num_firms': [],
            'num_links': [],
            'avg_degree': [],
            'avg_profit': [],
            'giant_component_size': [],
            'density': [],
            'clustering': []
        }


def run_experiment(delta: float, 
                   exit_threshold: int,
                   num_periods: int = 1000,
                   num_runs: int = 1,
                   random_seed: Optional[int] = None,
                   verbose: bool = False) -> Dict:
    """
    Run an experiment with specified parameters.
    
    Args:
        delta: Externality parameter
        exit_threshold: Exit threshold (m periods)
        num_periods: Number of periods per run
        num_runs: Number of runs to average over
        random_seed: Base random seed
        verbose: Print progress
        
    Returns:
        Dictionary with aggregated results
    """
    results = {
        'delta': delta,
        'exit_threshold': exit_threshold,
        'runs': []
    }
    
    for run in range(num_runs):
        seed = random_seed + run if random_seed is not None else None
        
        model = EconomicNetworkModel(
            delta=delta,
            exit_threshold=exit_threshold,
            random_seed=seed
        )
        
        if verbose:
            print(f"\nRun {run + 1}/{num_runs}")
            print(f"Parameters: δ={delta}, exit_threshold={exit_threshold}")
        
        model.run_simulation(num_periods, verbose=verbose)
        
        final_metrics = model.get_network_metrics()
        degree_dist = model.get_degree_distribution()
        
        run_result = {
            'final_metrics': final_metrics,
            'degree_distribution': degree_dist,
            'history': model.history
        }
        
        results['runs'].append(run_result)
        
        if verbose:
            print(f"Final: {final_metrics['num_firms']} firms, "
                  f"avg degree: {final_metrics['avg_degree']:.2f}, "
                  f"avg profit: {final_metrics['avg_profit']:.2f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Economic Networks Model - Example Run")
    print("=" * 50)
    
    # Single run with default parameters
    model = EconomicNetworkModel(delta=0.95, exit_threshold=4, random_seed=42)
    model.run_simulation(num_periods=1000, verbose=True)
    
    print("\nFinal Network Metrics:")
    print("-" * 50)
    metrics = model.get_network_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nDegree Distribution:")
    print("-" * 50)
    degree_dist = model.get_degree_distribution()
    for degree, count in sorted(degree_dist.items()):
        print(f"Degree {degree}: {count} firms")
