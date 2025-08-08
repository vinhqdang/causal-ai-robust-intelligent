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
        top_k_paths: int = 10,
        influence_threshold: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.model = model
        self.max_path_length = max_path_length
        self.integration_steps = integration_steps
        self.top_k_paths = top_k_paths
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
        
        # Get all causal paths up to max length
        all_paths = self._enumerate_causal_paths(causal_graph)
        
        # Select top-k most important paths based on structural importance
        important_paths = self._select_important_paths(all_paths, causal_graph)
        
        # Compute path-integrated influences
        pi_cpi_scores = defaultdict(float)
        
        for path in important_paths:
            path_influences = self._compute_path_influence(
                path, input_batch, target_batch, loss_fn
            )
            
            # Aggregate influences across parameters for this path
            for param_name, influence in path_influences.items():
                pi_cpi_scores[param_name] += influence
                
        return dict(pi_cpi_scores)
    
    def _enumerate_causal_paths(self, graph: nx.DiGraph) -> List[List[int]]:
        """
        Enumerate all causal paths in the graph up to max_path_length.
        """
        all_paths = []
        nodes = list(graph.nodes())
        
        # Find all simple paths between all pairs of nodes
        for source in nodes:
            for target in nodes:
                if source != target:
                    try:
                        # Get all simple paths from source to target
                        simple_paths = list(nx.all_simple_paths(
                            graph, source, target, cutoff=self.max_path_length
                        ))
                        all_paths.extend(simple_paths)
                    except nx.NetworkXNoPath:
                        continue
        
        return all_paths
    
    def _select_important_paths(
        self, 
        all_paths: List[List[int]], 
        graph: nx.DiGraph
    ) -> List[List[int]]:
        """
        Select the most structurally important paths based on graph properties.
        """
        if len(all_paths) <= self.top_k_paths:
            return all_paths
            
        # Score paths by structural importance
        path_scores = []
        
        for path in all_paths:
            score = 0.0
            
            # Length penalty (shorter paths are more direct)
            score += 1.0 / len(path)
            
            # Node centrality bonus
            for node in path:
                score += graph.degree(node) / (2 * len(path))
            
            # Connectivity bonus for bridging nodes
            for node in path[1:-1]:  # Intermediate nodes
                predecessors = len(list(graph.predecessors(node)))
                successors = len(list(graph.successors(node)))
                score += (predecessors * successors) / (len(path) ** 2)
                
            path_scores.append((score, path))
        
        # Sort by score and select top-k
        path_scores.sort(reverse=True, key=lambda x: x[0])
        return [path for _, path in path_scores[:self.top_k_paths]]
    
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
        all_paths = self._enumerate_causal_paths(causal_graph)
        important_paths = self._select_important_paths(all_paths, causal_graph)
        
        path_contributions = {}
        
        for path in important_paths:
            # Compute total influence along this path
            path_influences = self._compute_path_influence(
                path, input_batch, None, nn.MSELoss()
            )
            
            total_influence = sum(path_influences.values())
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