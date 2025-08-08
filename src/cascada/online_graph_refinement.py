"""
Component C1: Online Graph Refinement (OGR)
Implements kernelised conditional independence testing for dynamic causal graph learning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scipy.stats import chi2
import networkx as nx
from sklearn.gaussian_process.kernels import RBF


class OnlineGraphRefinement(nn.Module):
    """
    Online Graph Refinement module that dynamically updates causal graph structure
    using kernelised conditional independence tests on latent variables.
    
    Key innovation: Unlike existing continual learning methods that freeze the initial
    causal graph, OGR allows the causal structure to evolve as new data arrives.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        kernel_bandwidth: float = 1.0,
        alpha_cit: float = 0.01,
        min_samples: int = 50,
        max_edges: int = 100
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_bandwidth = kernel_bandwidth
        self.alpha_cit = alpha_cit  # Significance level for conditional independence tests
        self.min_samples = min_samples
        self.max_edges = max_edges
        
        # Initialize RBF kernel for conditional independence testing
        self.kernel = RBF(length_scale=kernel_bandwidth)
        
        # Graph structure storage
        self.adjacency_matrix = torch.zeros(latent_dim, latent_dim)
        self.graph = nx.DiGraph()
        
        # Statistical buffers for online testing
        self.latent_buffer = []
        self.test_statistics_history = []
        
    def forward(
        self, 
        z_stable: torch.Tensor, 
        z_context: torch.Tensor,
        current_graph: nx.DiGraph
    ) -> nx.DiGraph:
        """
        Refine the causal graph structure based on new latent representations.
        
        Args:
            z_stable: Stable latent variables [batch_size, stable_dim]
            z_context: Context-specific latent variables [batch_size, context_dim] 
            current_graph: Current causal graph structure
            
        Returns:
            refined_graph: Updated causal graph with added/removed edges
        """
        batch_size = z_stable.size(0)
        
        # Combine stable and context representations
        z_combined = torch.cat([z_stable, z_context], dim=1)
        
        # Add to buffer for statistical testing
        self.latent_buffer.append(z_combined.detach().cpu().numpy())
        
        # Only perform refinement if we have sufficient samples
        if len(self.latent_buffer) < self.min_samples:
            return current_graph
            
        # Stack recent samples for testing
        recent_samples = np.vstack(self.latent_buffer[-self.min_samples:])
        
        # Create refined graph copy
        refined_graph = current_graph.copy()
        
        # Test for new edges to add
        new_edges = self._test_for_new_edges(recent_samples, refined_graph)
        refined_graph.add_edges_from(new_edges)
        
        # Test for edges to remove
        edges_to_remove = self._test_for_edge_removal(recent_samples, refined_graph)
        refined_graph.remove_edges_from(edges_to_remove)
        
        # Update internal adjacency matrix
        self._update_adjacency_matrix(refined_graph)
        
        return refined_graph
    
    def _test_for_new_edges(
        self, 
        samples: np.ndarray, 
        graph: nx.DiGraph
    ) -> List[Tuple[int, int]]:
        """
        Test for potential new causal edges using conditional independence testing.
        """
        new_edges = []
        n_vars = samples.shape[1]
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j or graph.has_edge(i, j):
                    continue
                    
                # Test conditional independence X_i ⊥ X_j | Pa(X_j)
                parents_j = list(graph.predecessors(j))
                
                if self._conditional_independence_test(samples, i, j, parents_j):
                    continue  # They are conditionally independent
                else:
                    # Not conditionally independent, add edge
                    new_edges.append((i, j))
                    
                    # Prevent too many edges
                    if len(new_edges) + graph.number_of_edges() >= self.max_edges:
                        break
            
            if len(new_edges) + graph.number_of_edges() >= self.max_edges:
                break
                
        return new_edges
    
    def _test_for_edge_removal(
        self,
        samples: np.ndarray,
        graph: nx.DiGraph
    ) -> List[Tuple[int, int]]:
        """
        Test existing edges for removal based on conditional independence.
        """
        edges_to_remove = []
        
        for edge in list(graph.edges()):
            i, j = edge
            
            # Get parents of j excluding i
            parents_j = [p for p in graph.predecessors(j) if p != i]
            
            # Test if X_i ⊥ X_j | Pa(X_j) \ {X_i}
            if self._conditional_independence_test(samples, i, j, parents_j):
                edges_to_remove.append(edge)
                
        return edges_to_remove
    
    def _conditional_independence_test(
        self,
        samples: np.ndarray,
        i: int,
        j: int, 
        conditioning_set: List[int]
    ) -> bool:
        """
        Kernelised conditional independence test using HSIC (Hilbert-Schmidt Independence Criterion).
        
        Returns True if X_i ⊥ X_j | Z, False otherwise.
        """
        n_samples = samples.shape[0]
        
        if n_samples < 10:  # Need minimum samples for reliable testing
            return False
            
        X_i = samples[:, i].reshape(-1, 1)
        X_j = samples[:, j].reshape(-1, 1)
        
        if len(conditioning_set) == 0:
            # Marginal independence test
            return self._marginal_independence_test(X_i, X_j)
        
        Z = samples[:, conditioning_set]
        
        # Compute kernel matrices
        K_i = self.kernel(X_i)
        K_j = self.kernel(X_j)
        K_z = self.kernel(Z)
        
        # Compute conditional HSIC test statistic
        test_stat = self._compute_conditional_hsic(K_i, K_j, K_z)
        
        # Chi-square test with appropriate degrees of freedom
        df = 1  # Simplified; actual df depends on kernel properties
        p_value = 1 - chi2.cdf(test_stat, df)
        
        return p_value > self.alpha_cit
    
    def _marginal_independence_test(self, X_i: np.ndarray, X_j: np.ndarray) -> bool:
        """Marginal independence test using HSIC."""
        K_i = self.kernel(X_i)
        K_j = self.kernel(X_j)
        
        n = K_i.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
        
        # HSIC test statistic
        test_stat = np.trace(K_i @ H @ K_j @ H) / (n - 1)**2
        
        # Approximate null distribution (gamma approximation)
        # Simplified version - in practice would use more sophisticated approximation
        threshold = 2 * self.alpha_cit / n
        
        return test_stat < threshold
    
    def _compute_conditional_hsic(
        self, 
        K_i: np.ndarray, 
        K_j: np.ndarray, 
        K_z: np.ndarray
    ) -> float:
        """
        Compute conditional HSIC test statistic.
        
        This implements the conditional independence test where we test
        whether X_i and X_j are independent given Z.
        """
        n = K_i.shape[0]
        
        # Regularization parameter
        reg_param = 1e-3
        
        try:
            # Compute conditional kernels using the inversion trick
            K_z_reg = K_z + reg_param * np.eye(n)
            K_z_inv = np.linalg.inv(K_z_reg)
            
            # Project out the effect of Z
            P_z = np.eye(n) - K_z @ K_z_inv
            K_i_cond = P_z @ K_i @ P_z
            K_j_cond = P_z @ K_j @ P_z
            
            # Compute conditional HSIC
            test_stat = np.trace(K_i_cond @ K_j_cond) / (n - 1)**2
            
        except np.linalg.LinAlgError:
            # Fallback to marginal test if conditioning fails
            H = np.eye(n) - np.ones((n, n)) / n
            test_stat = np.trace(K_i @ H @ K_j @ H) / (n - 1)**2
            
        return test_stat
    
    def _update_adjacency_matrix(self, graph: nx.DiGraph):
        """Update internal adjacency matrix representation."""
        n_nodes = self.adjacency_matrix.size(0)
        self.adjacency_matrix.zero_()
        
        for edge in graph.edges():
            i, j = edge
            if i < n_nodes and j < n_nodes:
                self.adjacency_matrix[i, j] = 1.0
    
    def get_edge_strengths(self, samples: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        Compute strength scores for all potential edges based on dependency measures.
        """
        n_vars = samples.shape[1]
        edge_strengths = {}
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    X_i = samples[:, i].reshape(-1, 1)
                    X_j = samples[:, j].reshape(-1, 1)
                    
                    K_i = self.kernel(X_i)
                    K_j = self.kernel(X_j)
                    
                    # Use HSIC as dependency strength measure
                    n = K_i.shape[0]
                    H = np.eye(n) - np.ones((n, n)) / n
                    strength = np.trace(K_i @ H @ K_j @ H) / (n - 1)**2
                    
                    edge_strengths[(i, j)] = float(strength)
        
        return edge_strengths
    
    def reset_buffer(self):
        """Clear the latent variable buffer."""
        self.latent_buffer = []
        self.test_statistics_history = []
        
    def get_statistics(self) -> Dict:
        """Return diagnostic statistics about the graph refinement process."""
        return {
            'buffer_size': len(self.latent_buffer),
            'current_edges': int(self.adjacency_matrix.sum()),
            'test_history_length': len(self.test_statistics_history),
            'alpha_cit': self.alpha_cit
        }