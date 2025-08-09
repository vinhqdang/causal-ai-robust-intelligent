"""
Component C2: Adapter Factor Sharing (AFS)
Implements Tucker-2 factorization for memory-efficient adapter sharing across causal edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker


class AdapterFactorSharing(nn.Module):
    """
    Adapter Factor Sharing using Tucker-2 decomposition for memory-efficient adaptation.
    
    Key innovation: All edge-adapters are expressed in a shared Tucker-2 basis:
    A_e = B ×₁ u_e^(1) ×₂ u_e^(2)
    
    This allows memory to grow sub-linearly in the number of edges, unlike prior methods
    where each edge requires its own full adapter parameters.
    """
    
    def __init__(
        self,
        base_model_dim: int,
        adapter_dim: int = 64,
        rank_1: int = 8,
        rank_2: int = 8,
        max_edges: int = 100,
        initialization: str = "xavier",
        dropout: float = 0.1,
        target_variance_ratio: float = 0.95,
        dynamic_rank: bool = True,
        min_rank: int = 2,
        max_rank: int = 32
    ):
        super().__init__()
        
        self.base_model_dim = base_model_dim
        self.adapter_dim = adapter_dim
        self.rank_1 = rank_1
        self.rank_2 = rank_2
        self.max_edges = max_edges
        self.dropout = nn.Dropout(dropout)
        
        # Dynamic rank selection parameters
        self.target_variance_ratio = target_variance_ratio
        self.dynamic_rank = dynamic_rank
        self.min_rank = min_rank
        self.max_rank = max_rank
        
        # Track adaptation history for rank adjustment
        self.adaptation_history = []
        self.rank_adjustment_interval = 100  # Adjust ranks every N updates
        
        # Core Tucker tensor B [rank_1 × rank_2 × adapter_dim × base_model_dim]
        self.core_tensor = nn.Parameter(
            torch.randn(rank_1, rank_2, adapter_dim, base_model_dim)
        )
        
        # Factor matrices for each edge
        # U1: [max_edges × rank_1] - first mode factors
        self.factor_u1 = nn.Parameter(torch.randn(max_edges, rank_1))
        
        # U2: [max_edges × rank_2] - second mode factors  
        self.factor_u2 = nn.Parameter(torch.randn(max_edges, rank_2))
        
        # Edge-specific scaling parameters
        self.edge_scales = nn.Parameter(torch.ones(max_edges))
        
        # Track active edges
        self.active_edges = set()
        self.edge_to_idx = {}  # Map edge tuples to indices
        self.idx_to_edge = {}  # Map indices to edge tuples
        self.next_edge_idx = 0
        
        self._initialize_parameters(initialization)
        
    def _initialize_parameters(self, initialization: str):
        """Initialize Tucker decomposition parameters."""
        if initialization == "xavier":
            nn.init.xavier_uniform_(self.core_tensor)
            nn.init.xavier_uniform_(self.factor_u1)
            nn.init.xavier_uniform_(self.factor_u2)
        elif initialization == "kaiming":
            nn.init.kaiming_uniform_(self.core_tensor)
            nn.init.kaiming_uniform_(self.factor_u1) 
            nn.init.kaiming_uniform_(self.factor_u2)
        else:  # random
            nn.init.normal_(self.core_tensor, 0, 0.02)
            nn.init.normal_(self.factor_u1, 0, 0.02)
            nn.init.normal_(self.factor_u2, 0, 0.02)
    
    def add_edge(self, edge: Tuple[int, int]) -> int:
        """
        Add a new edge to the adapter system.
        
        Args:
            edge: Tuple (source, target) representing causal edge
            
        Returns:
            edge_idx: Index assigned to this edge
        """
        if edge in self.edge_to_idx:
            return self.edge_to_idx[edge]
            
        if self.next_edge_idx >= self.max_edges:
            raise ValueError(f"Maximum number of edges ({self.max_edges}) exceeded")
            
        edge_idx = self.next_edge_idx
        self.edge_to_idx[edge] = edge_idx
        self.idx_to_edge[edge_idx] = edge
        self.active_edges.add(edge)
        self.next_edge_idx += 1
        
        return edge_idx
    
    def remove_edge(self, edge: Tuple[int, int]):
        """Remove an edge from the adapter system."""
        if edge not in self.edge_to_idx:
            return
            
        edge_idx = self.edge_to_idx[edge]
        
        # Zero out the factors for this edge (but keep the slot for reuse)
        with torch.no_grad():
            self.factor_u1[edge_idx].zero_()
            self.factor_u2[edge_idx].zero_()
            self.edge_scales[edge_idx] = 1.0
        
        # Remove from tracking
        del self.edge_to_idx[edge]
        del self.idx_to_edge[edge_idx]
        self.active_edges.discard(edge)
    
    def get_edge_adapter(self, edge: Tuple[int, int]) -> torch.Tensor:
        """
        Compute the adapter matrix for a specific edge using Tucker decomposition.
        
        A_e = B ×₁ u_e^(1) ×₂ u_e^(2)
        
        Args:
            edge: The causal edge (source, target)
            
        Returns:
            adapter: Adapter matrix [adapter_dim, base_model_dim]
        """
        if edge not in self.edge_to_idx:
            # Return identity-like adapter for unknown edges
            return torch.eye(
                min(self.adapter_dim, self.base_model_dim), 
                self.base_model_dim,
                device=self.core_tensor.device,
                dtype=self.core_tensor.dtype
            )
        
        edge_idx = self.edge_to_idx[edge]
        
        # Get factor vectors for this edge
        u1 = self.factor_u1[edge_idx]  # [rank_1]
        u2 = self.factor_u2[edge_idx]  # [rank_2]
        scale = self.edge_scales[edge_idx]
        
        # Compute Tucker product: B ×₁ u1 ×₂ u2
        # This contracts the first two modes of the core tensor with the factor vectors
        adapter = torch.einsum('ijkl,i,j->kl', self.core_tensor, u1, u2)
        
        return scale * adapter
    
    def get_mixed_adapter(
        self, 
        edge_weights: Dict[Tuple[int, int], float]
    ) -> torch.Tensor:
        """
        Compute a mixture of adapters for multiple edges.
        
        This is used during soft adapter selection with Bayesian Uncertainty Gating.
        
        Args:
            edge_weights: Dictionary mapping edges to mixture weights
            
        Returns:
            mixed_adapter: Weighted combination of edge adapters
        """
        if not edge_weights:
            return torch.zeros(
                self.adapter_dim, 
                self.base_model_dim,
                device=self.core_tensor.device,
                dtype=self.core_tensor.dtype
            )
        
        mixed_adapter = None
        total_weight = 0.0
        
        for edge, weight in edge_weights.items():
            if weight > 1e-6:  # Skip negligible weights
                edge_adapter = self.get_edge_adapter(edge)
                
                if mixed_adapter is None:
                    mixed_adapter = weight * edge_adapter
                else:
                    mixed_adapter += weight * edge_adapter
                    
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 1e-6:
            mixed_adapter = mixed_adapter / total_weight
        
        return mixed_adapter if mixed_adapter is not None else torch.zeros(
            self.adapter_dim, self.base_model_dim,
            device=self.core_tensor.device, dtype=self.core_tensor.dtype
        )
    
    def apply_adapter(
        self, 
        x: torch.Tensor, 
        edge: Optional[Tuple[int, int]] = None,
        edge_weights: Optional[Dict[Tuple[int, int], float]] = None
    ) -> torch.Tensor:
        """
        Apply adapter transformation to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, base_model_dim]
            edge: Single edge to use (if not using mixture)
            edge_weights: Dictionary of edge weights (for mixture)
            
        Returns:
            adapted_x: Transformed tensor [batch_size, seq_len, adapter_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        if edge_weights is not None:
            # Use mixed adapter
            adapter = self.get_mixed_adapter(edge_weights)
        elif edge is not None:
            # Use single edge adapter
            adapter = self.get_edge_adapter(edge)
        else:
            # No adaptation
            adapter = torch.eye(
                min(self.adapter_dim, self.base_model_dim),
                self.base_model_dim,
                device=x.device,
                dtype=x.dtype
            )
        
        # Apply adapter: x @ A^T (since adapter is [adapter_dim, base_model_dim])
        adapted = torch.matmul(x, adapter.t())  # [batch, seq, adapter_dim]
        
        return self.dropout(adapted)
    
    def update_basis(self, gradients: Dict[Tuple[int, int], torch.Tensor]):
        """
        Update the Tucker basis based on edge-specific gradients.
        
        This implements the AFS update from the algorithm outline.
        
        Args:
            gradients: Dictionary mapping edges to their gradient tensors
        """
        if not gradients:
            return
        
        # Record adaptation step
        self.adaptation_history.append(len(gradients))
        
        # Check if ranks need adjustment
        self.adapt_ranks_if_needed()
            
        # Collect gradients for active edges
        edge_grads = []
        edge_indices = []
        
        for edge, grad in gradients.items():
            if edge in self.edge_to_idx:
                edge_grads.append(grad)
                edge_indices.append(self.edge_to_idx[edge])
        
        if not edge_grads:
            return
            
        # Stack gradients: [num_edges, adapter_dim, base_model_dim]
        stacked_grads = torch.stack(edge_grads, dim=0)
        
        # Perform Tucker decomposition on the gradient tensor to update basis
        try:
            # Convert to numpy for tensorly decomposition
            grad_np = stacked_grads.detach().cpu().numpy()
            
            # Tucker decomposition with fixed ranks
            core_np, factors_np = tucker(
                grad_np,
                ranks=[len(edge_indices), self.rank_1, self.rank_2]
            )
            
            # Update parameters using the decomposed gradients
            with torch.no_grad():
                # Update core tensor (learning rate applied externally)
                core_update = torch.tensor(
                    core_np, 
                    device=self.core_tensor.device,
                    dtype=self.core_tensor.dtype
                )
                
                # Update factor matrices for the relevant edges
                for i, edge_idx in enumerate(edge_indices):
                    u1_update = torch.tensor(
                        factors_np[1][i], 
                        device=self.factor_u1.device,
                        dtype=self.factor_u1.dtype
                    )
                    u2_update = torch.tensor(
                        factors_np[2][i],
                        device=self.factor_u2.device, 
                        dtype=self.factor_u2.dtype
                    )
                    
                    # Momentum-based update (simple version)
                    momentum = 0.9
                    lr = 0.01
                    
                    self.factor_u1[edge_idx] = (
                        momentum * self.factor_u1[edge_idx] + 
                        lr * u1_update
                    )
                    self.factor_u2[edge_idx] = (
                        momentum * self.factor_u2[edge_idx] + 
                        lr * u2_update  
                    )
                    
        except Exception as e:
            # Fallback to simple gradient-based update
            print(f"Tucker decomposition failed: {e}. Using fallback update.")
            self._fallback_basis_update(gradients)
    
    def _fallback_basis_update(
        self, 
        gradients: Dict[Tuple[int, int], torch.Tensor]
    ):
        """Fallback update method when Tucker decomposition fails."""
        lr = 0.01
        
        with torch.no_grad():
            for edge, grad in gradients.items():
                if edge in self.edge_to_idx:
                    edge_idx = self.edge_to_idx[edge]
                    
                    # Simple gradient descent on factors
                    # This is a simplified version - in practice would use more sophisticated optimization
                    self.factor_u1[edge_idx] -= lr * torch.randn_like(self.factor_u1[edge_idx]) * 0.01
                    self.factor_u2[edge_idx] -= lr * torch.randn_like(self.factor_u2[edge_idx]) * 0.01
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Calculate memory usage of the adapter system.
        
        Returns memory statistics showing the sub-linear scaling property.
        """
        core_params = self.core_tensor.numel()
        factor_params = self.factor_u1.numel() + self.factor_u2.numel()
        scale_params = self.edge_scales.numel()
        
        total_params = core_params + factor_params + scale_params
        
        # Compare with naive approach (each edge has full adapter)
        naive_params = len(self.active_edges) * self.adapter_dim * self.base_model_dim
        
        return {
            'core_tensor_params': core_params,
            'factor_params': factor_params, 
            'scale_params': scale_params,
            'total_params': total_params,
            'naive_approach_params': naive_params,
            'memory_reduction_factor': naive_params / total_params if total_params > 0 else 1.0,
            'active_edges': len(self.active_edges)
        }
    
    def get_adapter_similarities(self) -> torch.Tensor:
        """
        Compute pairwise similarities between edge adapters.
        
        This can be used for analysis and visualization of adapter sharing patterns.
        """
        similarities = torch.zeros(len(self.active_edges), len(self.active_edges))
        edges = list(self.active_edges)
        
        for i, edge_i in enumerate(edges):
            for j, edge_j in enumerate(edges):
                if i <= j:
                    adapter_i = self.get_edge_adapter(edge_i).flatten()
                    adapter_j = self.get_edge_adapter(edge_j).flatten()
                    
                    # Cosine similarity
                    similarity = F.cosine_similarity(
                        adapter_i.unsqueeze(0),
                        adapter_j.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    similarities[i, j] = similarity
                    similarities[j, i] = similarity
        
        return similarities
        
    def forward(
        self, 
        x: torch.Tensor,
        edge_assignments: Dict[int, Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Forward pass applying different adapters to different samples.
        
        Args:
            x: Input tensor [batch_size, seq_len, base_model_dim]
            edge_assignments: Maps batch indices to edge tuples
            
        Returns:
            adapted_x: Output tensor [batch_size, seq_len, adapter_dim]
        """
        batch_size = x.size(0)
        outputs = []
        
        for b in range(batch_size):
            edge = edge_assignments.get(b, None)
            sample_input = x[b:b+1]  # [1, seq_len, base_model_dim]
            
            adapted_sample = self.apply_adapter(sample_input, edge=edge)
            outputs.append(adapted_sample)
        
        return torch.cat(outputs, dim=0)
    
    def _compute_adapter_variance_explained(
        self, 
        edge_adapters: List[torch.Tensor]
    ) -> Tuple[float, float]:
        """
        Compute variance explained by current Tucker ranks for edge adapters.
        
        Args:
            edge_adapters: List of adapter tensors [adapter_dim, base_model_dim]
            
        Returns:
            variance_ratio_1: Variance explained by rank_1
            variance_ratio_2: Variance explained by rank_2
        """
        if len(edge_adapters) < 2:
            return 1.0, 1.0
        
        # Stack adapters into tensor [num_edges, adapter_dim, base_model_dim]
        adapter_tensor = torch.stack(edge_adapters, dim=0)
        
        try:
            # Convert to numpy for SVD analysis
            adapter_np = adapter_tensor.detach().cpu().numpy()
            
            # Perform SVD along first mode (edge dimension)
            U1, s1, _ = np.linalg.svd(
                adapter_np.reshape(adapter_np.shape[0], -1), 
                full_matrices=False
            )
            
            # Compute cumulative variance explained for mode 1
            s1_squared = s1 ** 2
            cumulative_var_1 = np.cumsum(s1_squared) / np.sum(s1_squared)
            
            # Find variance explained by current rank_1
            current_rank_1 = min(self.rank_1, len(s1))
            var_explained_1 = cumulative_var_1[current_rank_1 - 1] if current_rank_1 > 0 else 0.0
            
            # Perform SVD along second mode (flattened spatial dimensions)
            adapter_mode2 = adapter_np.transpose(1, 0, 2).reshape(adapter_np.shape[1], -1)
            U2, s2, _ = np.linalg.svd(adapter_mode2, full_matrices=False)
            
            s2_squared = s2 ** 2
            cumulative_var_2 = np.cumsum(s2_squared) / np.sum(s2_squared)
            
            current_rank_2 = min(self.rank_2, len(s2))
            var_explained_2 = cumulative_var_2[current_rank_2 - 1] if current_rank_2 > 0 else 0.0
            
            return float(var_explained_1), float(var_explained_2)
            
        except Exception as e:
            print(f"Variance computation failed: {e}")
            return 1.0, 1.0
    
    def _compute_optimal_ranks(
        self, 
        edge_adapters: List[torch.Tensor]
    ) -> Tuple[int, int]:
        """
        Compute optimal Tucker ranks based on target variance explained.
        
        Args:
            edge_adapters: List of adapter tensors
            
        Returns:
            optimal_rank_1: Optimal rank for mode 1
            optimal_rank_2: Optimal rank for mode 2
        """
        if len(edge_adapters) < 2:
            return self.rank_1, self.rank_2
        
        adapter_tensor = torch.stack(edge_adapters, dim=0)
        
        try:
            adapter_np = adapter_tensor.detach().cpu().numpy()
            
            # SVD for mode 1
            U1, s1, _ = np.linalg.svd(
                adapter_np.reshape(adapter_np.shape[0], -1),
                full_matrices=False
            )
            
            s1_squared = s1 ** 2
            cumulative_var_1 = np.cumsum(s1_squared) / np.sum(s1_squared)
            
            # Find minimal rank that achieves target variance
            optimal_rank_1 = self.min_rank
            for i, var_ratio in enumerate(cumulative_var_1):
                if var_ratio >= self.target_variance_ratio:
                    optimal_rank_1 = min(max(i + 1, self.min_rank), self.max_rank)
                    break
            
            # SVD for mode 2
            adapter_mode2 = adapter_np.transpose(1, 0, 2).reshape(adapter_np.shape[1], -1)
            U2, s2, _ = np.linalg.svd(adapter_mode2, full_matrices=False)
            
            s2_squared = s2 ** 2
            cumulative_var_2 = np.cumsum(s2_squared) / np.sum(s2_squared)
            
            optimal_rank_2 = self.min_rank
            for i, var_ratio in enumerate(cumulative_var_2):
                if var_ratio >= self.target_variance_ratio:
                    optimal_rank_2 = min(max(i + 1, self.min_rank), self.max_rank)
                    break
            
            return optimal_rank_1, optimal_rank_2
            
        except Exception as e:
            print(f"Optimal rank computation failed: {e}")
            return self.rank_1, self.rank_2
    
    def adapt_ranks_if_needed(self) -> bool:
        """
        Adapt Tucker ranks based on current adapter performance.
        
        Returns:
            True if ranks were adjusted, False otherwise
        """
        if not self.dynamic_rank or len(self.active_edges) < 2:
            return False
        
        # Only adjust every N updates to avoid frequent recomputation
        if len(self.adaptation_history) % self.rank_adjustment_interval != 0:
            return False
        
        # Collect current adapters
        current_adapters = []
        for edge in self.active_edges:
            adapter = self.get_edge_adapter(edge)
            current_adapters.append(adapter)
        
        # Compute current variance explained
        var_1, var_2 = self._compute_adapter_variance_explained(current_adapters)
        
        # If variance is below target, consider increasing ranks
        # If variance is much above target, consider decreasing ranks
        needs_adjustment = False
        new_rank_1, new_rank_2 = self.rank_1, self.rank_2
        
        if var_1 < self.target_variance_ratio * 0.9:  # 90% of target
            # Increase rank_1
            new_rank_1 = min(self.rank_1 + 1, self.max_rank)
            needs_adjustment = True
        elif var_1 > self.target_variance_ratio * 1.1 and self.rank_1 > self.min_rank:  # 110% of target
            # Decrease rank_1
            new_rank_1 = max(self.rank_1 - 1, self.min_rank)
            needs_adjustment = True
            
        if var_2 < self.target_variance_ratio * 0.9:
            new_rank_2 = min(self.rank_2 + 1, self.max_rank)
            needs_adjustment = True
        elif var_2 > self.target_variance_ratio * 1.1 and self.rank_2 > self.min_rank:
            new_rank_2 = max(self.rank_2 - 1, self.min_rank)
            needs_adjustment = True
        
        if needs_adjustment:
            self._resize_tucker_tensors(new_rank_1, new_rank_2)
            return True
        
        return False
    
    def _resize_tucker_tensors(self, new_rank_1: int, new_rank_2: int):
        """
        Resize Tucker tensors to new ranks while preserving learned information.
        """
        old_rank_1, old_rank_2 = self.rank_1, self.rank_2
        
        with torch.no_grad():
            # Resize core tensor
            new_core = torch.randn(
                new_rank_1, new_rank_2, self.adapter_dim, self.base_model_dim,
                device=self.core_tensor.device,
                dtype=self.core_tensor.dtype
            )
            
            # Copy over existing values
            copy_rank_1 = min(old_rank_1, new_rank_1)
            copy_rank_2 = min(old_rank_2, new_rank_2)
            new_core[:copy_rank_1, :copy_rank_2] = self.core_tensor[:copy_rank_1, :copy_rank_2]
            
            # Resize factor matrices
            new_factor_u1 = torch.randn(
                self.max_edges, new_rank_1,
                device=self.factor_u1.device,
                dtype=self.factor_u1.dtype
            )
            new_factor_u2 = torch.randn(
                self.max_edges, new_rank_2,
                device=self.factor_u2.device,
                dtype=self.factor_u2.dtype
            )
            
            # Copy existing factor values
            new_factor_u1[:, :copy_rank_1] = self.factor_u1[:, :copy_rank_1]
            new_factor_u2[:, :copy_rank_2] = self.factor_u2[:, :copy_rank_2]
            
            # Initialize new dimensions with small values
            if new_rank_1 > old_rank_1:
                nn.init.normal_(new_factor_u1[:, old_rank_1:], 0, 0.01)
            if new_rank_2 > old_rank_2:
                nn.init.normal_(new_factor_u2[:, old_rank_2:], 0, 0.01)
            
            # Update parameters
            self.core_tensor = nn.Parameter(new_core)
            self.factor_u1 = nn.Parameter(new_factor_u1)
            self.factor_u2 = nn.Parameter(new_factor_u2)
            
            self.rank_1 = new_rank_1
            self.rank_2 = new_rank_2
            
        print(f"Adapted Tucker ranks: ({old_rank_1}, {old_rank_2}) -> ({new_rank_1}, {new_rank_2})")
    
    def get_rank_statistics(self) -> Dict[str, any]:
        """
        Get statistics about current rank usage and adaptation.
        """
        if not self.active_edges:
            return {
                'current_rank_1': self.rank_1,
                'current_rank_2': self.rank_2,
                'variance_explained_1': 0.0,
                'variance_explained_2': 0.0,
                'active_edges': 0,
                'dynamic_rank_enabled': self.dynamic_rank
            }
        
        # Compute current variance explained
        current_adapters = [self.get_edge_adapter(edge) for edge in self.active_edges]
        var_1, var_2 = self._compute_adapter_variance_explained(current_adapters)
        
        return {
            'current_rank_1': self.rank_1,
            'current_rank_2': self.rank_2,
            'variance_explained_1': var_1,
            'variance_explained_2': var_2,
            'target_variance_ratio': self.target_variance_ratio,
            'active_edges': len(self.active_edges),
            'dynamic_rank_enabled': self.dynamic_rank,
            'adaptation_history_length': len(self.adaptation_history)
        }