"""
Component C3: Path-Integrated CPI (PI-CPI)
Implements path-integrated conditional parameter importance for causal path regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any
import networkx as nx
import numpy as np
from collections import defaultdict


class PathIntegratedCPI(nn.Module):
    """
    Path-Integrated Conditional Parameter Importance (PI-CPI) module.
    
    Key innovation: Unlike traditional CPI that operates per-edge, PI-CPI integrates 
    influence along entire causal paths using:
    
    PI-CPI_j = Σ_{paths p} ∫₀¹ ∂_{θⱼ} τₚ(θ₀ + α Δθⱼ) dα
    
    This prevents the "over-freeze" trap where important parameters are frozen
    due to single-edge analysis, missing their importance in multi-hop causal paths.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_path_length: int = 5,
        integration_steps: int = 20,
        max_paths_per_edge: int = 20,
        influence_threshold: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.model = model
        self.max_path_length = max_path_length
        self.integration_steps = integration_steps
        self.max_paths_per_edge = max_paths_per_edge
        self.influence_threshold = influence_threshold
        self.device = device
        
        # Cache for computed path influences
        self.path_influences_cache = {}
        self.parameter_influences_cache = {}
        
        # Track parameter names and shapes
        self.param_names = []
        self.param_shapes = []
        self._build_parameter_index()
        
    def _build_parameter_index(self):
        """Build index of model parameters for efficient access."""
        self.param_names = []
        self.param_shapes = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes.append(param.shape)
    
    def compute_path_integrated_cpi(
        self,
        causal_graph: nx.DiGraph,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Compute path-integrated conditional parameter importance.
        
        Args:
            causal_graph: Current causal graph structure
            input_batch: Input data for influence computation
            target_batch: Target data (if supervised)
            loss_fn: Loss function for computing gradients
            
        Returns:
            pi_cpi_scores: Dictionary mapping parameter names to PI-CPI scores
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        # Use importance sampling to get representative causal paths
        important_paths = self._importance_sample_paths(causal_graph)
        
        # Compute path-integrated influences with importance weighting
        pi_cpi_scores = defaultdict(float)
        
        for path, path_weight in important_paths:
            path_influences = self._compute_path_influence(
                path, input_batch, target_batch, loss_fn
            )
            
            # Aggregate influences across parameters for this path with importance weighting
            for param_name, influence in path_influences.items():
                # Apply importance sampling bias correction
                weighted_influence = influence * path_weight
                pi_cpi_scores[param_name] += weighted_influence
                
        return dict(pi_cpi_scores)
    
    def _importance_sample_paths(self, graph: nx.DiGraph) -> List[Tuple[List[int], float]]:
        """
        Use importance sampling to efficiently sample representative causal paths.
        
        This replaces full enumeration (O(E!)) with bounded sampling (O(E * max_paths_per_edge)).
        Returns paths with their importance weights for bias correction.
        """
        sampled_paths = []
        edges = list(graph.edges())
        
        if not edges:
            return []
        
        # Sample paths for each edge to ensure coverage
        for edge in edges:
            source, target = edge
            
            # Sample paths that go through this edge
            edge_paths = self._sample_paths_through_edge(graph, source, target)
            sampled_paths.extend(edge_paths)
        
        # Remove duplicates while preserving weights
        unique_paths = {}
        for path, weight in sampled_paths:
            path_tuple = tuple(path)
            if path_tuple not in unique_paths:
                unique_paths[path_tuple] = weight
            else:
                # Combine weights for duplicate paths
                unique_paths[path_tuple] += weight
        
        # Convert back to list format
        return [(list(path), weight) for path, weight in unique_paths.items()]
    
    def _sample_paths_through_edge(self, graph: nx.DiGraph, source: int, target: int) -> List[Tuple[List[int], float]]:
        """
        Sample paths that go through a specific edge using importance sampling.
        """
        paths_with_weights = []
        max_samples = min(self.max_paths_per_edge, 100)  # Cap for efficiency
        
        # Direct edge path (always include)
        direct_path = [source, target]
        direct_weight = self._compute_path_importance_weight(graph, direct_path)
        paths_with_weights.append((direct_path, direct_weight))
        
        # Sample extended paths using random walks
        for _ in range(max_samples - 1):
            path = self._sample_extended_path(graph, source, target)
            if path and len(path) <= self.max_path_length:
                weight = self._compute_path_importance_weight(graph, path)
                paths_with_weights.append((path, weight))
        
        return paths_with_weights
    
    def _sample_extended_path(self, graph: nx.DiGraph, source: int, target: int) -> Optional[List[int]]:
        """
        Sample an extended path from source to target using biased random walk.
        """
        max_attempts = 10
        
        for _ in range(max_attempts):
            path = [source]
            current = source
            
            # Random walk with bias toward target
            for _ in range(self.max_path_length - 1):
                successors = list(graph.successors(current))
                if not successors:
                    break
                
                if target in successors:
                    # High probability to go to target if available
                    if np.random.random() < 0.7:
                        path.append(target)
                        return path
                
                # Otherwise, choose successor based on structural importance
                weights = []
                for successor in successors:
                    # Weight by node importance (degree centrality)
                    weight = graph.degree(successor) + 1
                    
                    # Bonus if successor leads toward target
                    if nx.has_path(graph, successor, target):
                        try:
                            shortest_path_length = nx.shortest_path_length(graph, successor, target)
                            weight *= 2.0 / (shortest_path_length + 1)
                        except:
                            pass
                    
                    weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                    current = np.random.choice(successors, p=weights)
                    
                    if current in path:  # Avoid cycles
                        break
                    path.append(current)
                    
                    if current == target:
                        return path
                else:
                    break
        
        return None
    
    def _compute_path_importance_weight(self, graph: nx.DiGraph, path: List[int]) -> float:
        """
        Compute importance weight for a path based on structural properties.
        This is used for importance sampling bias correction.
        """
        if len(path) < 2:
            return 1.0
        
        weight = 1.0
        
        # Length penalty (prefer shorter paths)
        weight *= 1.0 / len(path)
        
        # Node centrality bonus
        for node in path:
            centrality = graph.degree(node) / max(len(graph.nodes()), 1)
            weight *= (1.0 + centrality)
        
        # Edge strength bonus (simplified)
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            if graph.has_edge(source, target):
                # Assume uniform edge weights for simplicity
                weight *= 1.1
        
        # Connectivity bonus for bridging paths
        if len(path) > 2:
            for node in path[1:-1]:  # Intermediate nodes
                in_degree = graph.in_degree(node)
                out_degree = graph.out_degree(node)
                bridge_score = min(in_degree, out_degree) / max(max(in_degree, out_degree), 1)
                weight *= (1.0 + 0.5 * bridge_score)
        
        return weight
    
    def _compute_path_influence(
        self,
        path: List[int],
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Compute influence of parameters along a specific causal path using integration.
        
        This implements the core PI-CPI computation:
        ∫₀¹ ∂_{θⱼ} τₚ(θ₀ + α Δθⱼ) dα
        """
        path_key = tuple(path)
        
        # Check cache first
        if path_key in self.path_influences_cache:
            return self.path_influences_cache[path_key]
        
        # Get baseline parameters
        baseline_params = {
            name: param.clone().detach() 
            for name, param in self.model.named_parameters() 
            if param.requires_grad
        }
        
        # Compute parameter perturbation directions using random sampling
        perturbation_directions = self._get_perturbation_directions()
        
        # Initialize influence accumulator
        path_influences = defaultdict(float)
        
        # Numerical integration over α ∈ [0, 1]
        for i in range(self.integration_steps):
            alpha = i / (self.integration_steps - 1)  # α ∈ [0, 1]
            
            # Compute path influence at this interpolation point
            step_influences = self._compute_path_influence_at_alpha(
                path, alpha, baseline_params, perturbation_directions,
                input_batch, target_batch, loss_fn
            )
            
            # Accumulate influences (trapezoidal integration)
            weight = 1.0 / (self.integration_steps - 1)
            if i == 0 or i == self.integration_steps - 1:
                weight *= 0.5  # Trapezoidal rule endpoints
                
            for param_name, influence in step_influences.items():
                path_influences[param_name] += weight * influence
        
        # Cache result
        path_influences_dict = dict(path_influences)
        self.path_influences_cache[path_key] = path_influences_dict
        
        return path_influences_dict
    
    def _get_perturbation_directions(self) -> Dict[str, torch.Tensor]:
        """
        Generate perturbation directions for parameter influence computation.
        """
        directions = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Use random Gaussian directions normalized by parameter scale
                direction = torch.randn_like(param)
                param_scale = param.abs().mean() + 1e-8
                direction = direction / direction.norm() * param_scale * 0.01
                directions[name] = direction
                
        return directions
    
    def _compute_path_influence_at_alpha(
        self,
        path: List[int],
        alpha: float,
        baseline_params: Dict[str, torch.Tensor],
        directions: Dict[str, torch.Tensor],
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Compute parameter influences at a specific α interpolation point.
        """
        influences = {}
        
        # Set model parameters to θ₀ + α * Δθ
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in directions:
                    param.data = baseline_params[name] + alpha * directions[name]
        
        # Enable gradients for influence computation
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad_(True)
        
        # Forward pass
        self.model.train()
        outputs = self.model(input_batch)
        
        # Compute path-specific output
        path_output = self._extract_path_output(outputs, path)
        
        # Compute loss
        if target_batch is not None:
            path_target = self._extract_path_target(target_batch, path)
            loss = loss_fn(path_output, path_target)
        else:
            # Unsupervised case: use output magnitude as proxy
            loss = path_output.abs().mean()
        
        # Compute gradients
        gradients = torch.autograd.grad(
            loss, 
            [param for param in self.model.parameters() if param.requires_grad],
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )
        
        # Extract influence magnitudes
        param_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param_idx < len(gradients) and gradients[param_idx] is not None:
                    # Influence is gradient magnitude weighted by perturbation direction
                    grad = gradients[param_idx]
                    direction = directions[name]
                    
                    # Directional influence: |∇θ · direction|
                    influence = torch.abs(torch.sum(grad * direction)).item()
                    influences[name] = influence
                else:
                    influences[name] = 0.0
                param_idx += 1
        
        # Restore baseline parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = baseline_params[name]
        
        return influences
    
    def _extract_path_output(
        self, 
        full_output: torch.Tensor, 
        path: List[int]
    ) -> torch.Tensor:
        """
        Extract output corresponding to a specific causal path.
        
        This is a simplified version - in practice would depend on model architecture.
        """
        # For simplicity, take output dimensions corresponding to path nodes
        if len(full_output.shape) >= 2:
            # Extract features corresponding to path nodes
            path_indices = [i % full_output.size(-1) for i in path]
            return full_output[..., path_indices]
        else:
            return full_output
    
    def _extract_path_target(
        self,
        full_target: torch.Tensor,
        path: List[int] 
    ) -> torch.Tensor:
        """Extract target corresponding to a specific causal path."""
        if len(full_target.shape) >= 2:
            path_indices = [i % full_target.size(-1) for i in path]
            return full_target[..., path_indices]
        else:
            return full_target
    
    def compute_pi_cpi_regularizer(
        self,
        pi_cpi_scores: Dict[str, float],
        parameter_changes: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the PI-CPI regularization term.
        
        L_reg = Σⱼ PI_CPI_j (θⱼ - θ_old_j)²
        
        Args:
            pi_cpi_scores: Path-integrated CPI scores for each parameter
            parameter_changes: Dictionary of parameter changes (θⱼ - θ_old_j)
            
        Returns:
            regularizer: PI-CPI regularization loss
        """
        total_reg = torch.tensor(0.0, device=self.device)
        
        for param_name, pi_cpi_score in pi_cpi_scores.items():
            if param_name in parameter_changes:
                param_change = parameter_changes[param_name]
                
                # PI-CPI weighted L2 regularization
                weighted_change = pi_cpi_score * (param_change ** 2)
                total_reg += weighted_change.sum()
        
        return total_reg
    
    def get_top_influential_parameters(
        self,
        pi_cpi_scores: Dict[str, float],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most influential parameters based on PI-CPI scores.
        """
        sorted_params = sorted(
            pi_cpi_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_params[:k]
    
    def analyze_path_contributions(
        self,
        causal_graph: nx.DiGraph,
        input_batch: torch.Tensor
    ) -> Dict[Tuple[int, ...], float]:
        """
        Analyze contribution of different causal paths to overall influence.
        """
        important_paths = self._importance_sample_paths(causal_graph)
        
        path_contributions = {}
        
        for path, path_weight in important_paths:
            # Compute total influence along this path
            path_influences = self._compute_path_influence(
                path, input_batch, None, nn.MSELoss()
            )
            
            total_influence = sum(path_influences.values()) * path_weight
            path_contributions[tuple(path)] = total_influence
        
        return path_contributions
    
    def clear_cache(self):
        """Clear cached computations."""
        self.path_influences_cache.clear()
        self.parameter_influences_cache.clear()
        
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory usage statistics for caching."""
        return {
            'path_cache_entries': len(self.path_influences_cache),
            'parameter_cache_entries': len(self.parameter_influences_cache),
            'total_parameters_tracked': len(self.param_names)
        }
    
    def forward(
        self,
        causal_graph: nx.DiGraph,
        input_batch: torch.Tensor,
        target_batch: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Forward pass computing both PI-CPI scores and regularization loss.
        
        Returns:
            pi_cpi_scores: Parameter importance scores
            regularizer: Regularization term (zero if no old parameters provided)
        """
        pi_cpi_scores = self.compute_path_integrated_cpi(
            causal_graph, input_batch, target_batch, loss_fn
        )
        
        # Return zero regularizer since we don't have old parameters here
        regularizer = torch.tensor(0.0, device=self.device)
        
        return pi_cpi_scores, regularizer