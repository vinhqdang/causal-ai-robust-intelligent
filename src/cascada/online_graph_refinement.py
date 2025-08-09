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
from scipy.linalg import expm


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
        max_edges: int = 100,
        notears_lambda: float = 1.0,
        acyclicity_tolerance: float = 1e-8
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_bandwidth = kernel_bandwidth
        self.alpha_cit = alpha_cit  # Significance level for conditional independence tests
        self.min_samples = min_samples
        self.max_edges = max_edges
        self.notears_lambda = notears_lambda  # Weight for NOTEARS acyclicity penalty
        self.acyclicity_tolerance = acyclicity_tolerance  # Tolerance for acyclicity check
        
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
        
        # Enforce acyclicity using NOTEARS penalty
        refined_graph = self._enforce_acyclicity(refined_graph, recent_samples)
        
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
            'alpha_cit': self.alpha_cit,
            'is_acyclic': self._is_acyclic(nx.from_numpy_array(self.adjacency_matrix.numpy(), create_using=nx.DiGraph))
        }
    
    def _enforce_acyclicity(self, graph: nx.DiGraph, samples: np.ndarray) -> nx.DiGraph:
        """
        Enforce acyclicity using NOTEARS-style optimization to remove cycles.
        
        This implements a greedy cycle-breaking approach guided by NOTEARS penalty.
        """
        if self._is_acyclic(graph):
            return graph
            
        # Convert to adjacency matrix for NOTEARS computation
        adj_matrix = nx.to_numpy_array(graph)
        n_nodes = adj_matrix.shape[0]
        
        # Iteratively remove edges that contribute most to cyclicity
        refined_adj = adj_matrix.copy()
        max_iterations = 50  # Prevent infinite loops
        
        for _ in range(max_iterations):
            if self._is_acyclic_matrix(refined_adj):
                break
                
            # Compute NOTEARS penalty for current graph
            acyclicity_penalty = self._compute_notears_penalty(refined_adj)
            
            if acyclicity_penalty < self.acyclicity_tolerance:
                break
                
            # Find edge that reduces cyclicity most when removed
            best_edge = self._find_best_edge_to_remove(refined_adj, samples)
            if best_edge is not None:
                i, j = best_edge
                refined_adj[i, j] = 0
            else:
                break
        
        # Convert back to NetworkX graph
        refined_graph = nx.from_numpy_array(refined_adj, create_using=nx.DiGraph)
        
        # Remove edges with zero weight (artifacts from conversion)
        edges_to_remove = [(u, v) for u, v, d in refined_graph.edges(data=True) if d.get('weight', 1) == 0]
        refined_graph.remove_edges_from(edges_to_remove)
        
        return refined_graph
    
    def _is_acyclic(self, graph: nx.DiGraph) -> bool:
        """Check if the graph is acyclic."""
        try:
            return nx.is_directed_acyclic_graph(graph)
        except:
            return False
    
    def _is_acyclic_matrix(self, adj_matrix: np.ndarray) -> bool:
        """Check if adjacency matrix represents acyclic graph using NOTEARS constraint."""
        penalty = self._compute_notears_penalty(adj_matrix)
        return penalty < self.acyclicity_tolerance
    
    def _compute_notears_penalty(self, adj_matrix: np.ndarray) -> float:
        """
        Compute NOTEARS acyclicity penalty: tr(exp(W ⊙ W)) - d
        where W is the weighted adjacency matrix and d is the number of nodes.
        """
        n_nodes = adj_matrix.shape[0]
        
        # Element-wise square of adjacency matrix
        W_squared = adj_matrix * adj_matrix
        
        try:
            # Compute matrix exponential
            exp_W = expm(W_squared)
            
            # NOTEARS penalty: tr(exp(W ⊙ W)) - d
            penalty = np.trace(exp_W) - n_nodes
            
        except (np.linalg.LinAlgError, OverflowError):
            # Fallback to simple cycle detection if matrix exponential fails
            temp_graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            penalty = 1000.0 if not nx.is_directed_acyclic_graph(temp_graph) else 0.0
        
        return max(0.0, penalty)  # Penalty should be non-negative
    
    def _find_best_edge_to_remove(self, adj_matrix: np.ndarray, samples: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the edge that, when removed, reduces the acyclicity penalty most
        while maintaining reasonable causal strength.
        """
        current_penalty = self._compute_notears_penalty(adj_matrix)
        best_edge = None
        best_penalty_reduction = 0
        
        # Get current edges
        edges = [(i, j) for i in range(adj_matrix.shape[0]) 
                for j in range(adj_matrix.shape[1]) if adj_matrix[i, j] > 0]
        
        for i, j in edges:
            # Temporarily remove edge
            test_adj = adj_matrix.copy()
            test_adj[i, j] = 0
            
            # Compute penalty reduction
            new_penalty = self._compute_notears_penalty(test_adj)
            penalty_reduction = current_penalty - new_penalty
            
            # Prefer edges with high penalty reduction and low causal strength
            causal_strength = self._compute_causal_strength(samples, i, j) if samples is not None else 0.5
            
            # Score combines penalty reduction with inverse causal strength
            score = penalty_reduction - 0.1 * causal_strength
            
            if score > best_penalty_reduction:
                best_penalty_reduction = score
                best_edge = (i, j)
        
        return best_edge
    
    def _compute_causal_strength(self, samples: np.ndarray, i: int, j: int) -> float:
        """
        Compute causal strength between variables i and j using HSIC.
        Higher values indicate stronger causal relationships.
        """
        if samples is None or i >= samples.shape[1] or j >= samples.shape[1]:
            return 0.5  # Default moderate strength
        
        X_i = samples[:, i].reshape(-1, 1)
        X_j = samples[:, j].reshape(-1, 1)
        
        try:
            K_i = self.kernel(X_i)
            K_j = self.kernel(X_j)
            
            n = K_i.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            strength = np.trace(K_i @ H @ K_j @ H) / (n - 1)**2
            
            return max(0.0, min(1.0, float(strength)))  # Clamp to [0, 1]
            
        except:
            return 0.5  # Default if computation fails