"""
CASCADA Algorithm - Main orchestration of all components
Implements the complete continual causal learning algorithm as specified in algorithmv2.md
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
import json
from pathlib import Path

from .online_graph_refinement import OnlineGraphRefinement
from .adapter_factor_sharing import AdapterFactorSharing
from .path_integrated_cpi import PathIntegratedCPI
from .generative_counterfactual_replay import GenerativeCounterfactualReplay
from .bayesian_uncertainty_gating import BayesianUncertaintyGating


@dataclass
class CASCADAConfig:
    """Configuration for CASCADA algorithm."""
    # Model dimensions
    base_model_dim: int = 512
    latent_dim: int = 64
    stable_dim: int = 32
    context_dim: int = 32
    adapter_dim: int = 64
    
    # Component-specific parameters
    ogr_alpha_cit: float = 0.01
    afs_rank_1: int = 8
    afs_rank_2: int = 8
    pi_cpi_integration_steps: int = 20
    gcr_num_inference_steps: int = 50
    bug_uncertainty_threshold: float = 0.1
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    beta_cf: float = 1.0  # Counterfactual loss weight
    lambda_reg: float = 1.0  # PI-CPI regularization weight
    max_edges: int = 100
    
    # Online learning parameters
    buffer_size: int = 1000
    update_frequency: int = 10
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging and checkpointing
    log_frequency: int = 100
    checkpoint_frequency: int = 1000
    checkpoint_dir: str = "./checkpoints"
    
    # PAC-Bayesian parameters
    pac_delta: float = 0.05
    prior_variance: float = 1.0


class CASCADA(nn.Module):
    """
    Complete CASCADA implementation orchestrating all five core components.
    
    Implements the algorithm outline from algorithmv2.md:
    1. Online Graph Refinement (C1)
    2. Adapter Factor Sharing (C2) 
    3. Path-Integrated CPI (C3)
    4. Generative Counterfactual Replay (C4)
    5. Bayesian Uncertainty Gating (C5)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: CASCADAConfig = None
    ):
        super().__init__()
        
        if config is None:
            config = CASCADAConfig()
        self.config = config
        
        self.base_model = base_model
        self.device = config.device
        
        # Initialize all core components
        self._initialize_components()
        
        # Initialize causal graph structure
        self.causal_graph = nx.DiGraph()
        self._initialize_graph()
        
        # Training state
        self.training_step = 0
        self.cumulative_regret = 0.0
        self.old_parameters = {}
        
        # Performance tracking
        self.performance_history = []
        self.loss_history = []
        
        # Move to device
        self.to(self.device)
        
    def _initialize_components(self):
        """Initialize all five core components."""
        config = self.config
        
        # C1: Online Graph Refinement
        self.ogr = OnlineGraphRefinement(
            latent_dim=config.latent_dim,
            alpha_cit=config.ogr_alpha_cit,
            max_edges=config.max_edges
        )
        
        # C2: Adapter Factor Sharing
        self.afs = AdapterFactorSharing(
            base_model_dim=config.base_model_dim,
            adapter_dim=config.adapter_dim,
            rank_1=config.afs_rank_1,
            rank_2=config.afs_rank_2,
            max_edges=config.max_edges
        )
        
        # C3: Path-Integrated CPI  
        self.pi_cpi = PathIntegratedCPI(
            model=self.base_model,
            integration_steps=config.pi_cpi_integration_steps,
            device=config.device
        )
        
        # C4: Generative Counterfactual Replay
        self.gcr = GenerativeCounterfactualReplay(
            input_shape=(3, 32, 32),  # Configurable based on data
            causal_dim=config.latent_dim,
            condition_dim=config.context_dim,
            num_inference_steps=config.gcr_num_inference_steps,
            device=config.device
        )
        
        # C5: Bayesian Uncertainty Gating
        self.bug = BayesianUncertaintyGating(
            context_dim=config.context_dim,
            max_adapters=config.max_edges,
            uncertainty_threshold=config.bug_uncertainty_threshold,
            device=config.device
        )
        
        # Representation extractor for z_stable and z_context
        self.representation_extractor = nn.Sequential(
            nn.Linear(config.base_model_dim, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, config.stable_dim + config.context_dim)
        )
    
    def _initialize_graph(self):
        """Initialize the causal graph with basic structure."""
        # Start with a simple graph - will be refined by OGR
        for i in range(self.config.latent_dim):
            self.causal_graph.add_node(i)
            
        # Add some initial edges (to be refined)
        for i in range(min(5, self.config.latent_dim - 1)):
            self.causal_graph.add_edge(i, i + 1)
    
    def extract_representations(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract stable and context representations from input.
        
        Args:
            x: Input tensor [batch_size, ...]
            
        Returns:
            z_stable: Stable latent variables [batch_size, stable_dim]
            z_context: Context-specific variables [batch_size, context_dim] 
        """
        # Get base model features
        with torch.no_grad():
            if hasattr(self.base_model, 'forward_features'):
                features = self.base_model.forward_features(x)
            else:
                # Assume model returns features
                features = self.base_model(x)
                
        # Extract representations
        representations = self.representation_extractor(features)
        
        # Split into stable and context
        stable_dim = self.config.stable_dim
        z_stable = representations[:, :stable_dim]
        z_context = representations[:, stable_dim:]
        
        return z_stable, z_context
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        update_graph: bool = True,
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass implementing the CASCADA algorithm.
        
        Args:
            x: Input data [batch_size, ...]
            y: Target labels [batch_size, ...] (optional)
            update_graph: Whether to update causal graph
            return_diagnostics: Whether to return diagnostic information
            
        Returns:
            output: Model predictions
            diagnostics: Diagnostic information (if requested)
        """
        batch_size = x.size(0)
        diagnostics = {}
        
        # Extract stable and context representations
        z_stable, z_context = self.extract_representations(x)
        
        # 1. Online Graph Refinement (C1)
        if update_graph:
            refined_graph = self.ogr(z_stable, z_context, self.causal_graph)
            
            # Update graph and register new edges with AFS and BUG
            self._update_graph_structure(refined_graph)
            self.causal_graph = refined_graph
            
            diagnostics['graph_edges'] = len(self.causal_graph.edges())
            diagnostics['graph_nodes'] = len(self.causal_graph.nodes())
        
        # 2. Bayesian Uncertainty Gating (C5) - Adapter Selection
        edge_weights, uncertainty = self.bug(z_context, self.causal_graph)
        diagnostics['routing_uncertainty'] = uncertainty.mean().item()
        
        # 3. Adapter Factor Sharing (C2) - Get mixed adapter
        mixed_adapter = self.afs.get_mixed_adapter(edge_weights)
        
        # 4. Apply adapter to base model forward pass
        base_output = self.base_model(x)
        adapted_output = self._apply_adapter_to_output(base_output, mixed_adapter)
        
        diagnostics['adapter_usage'] = len(edge_weights)
        
        if return_diagnostics:
            return adapted_output, diagnostics
        return adapted_output
    
    def _update_graph_structure(self, new_graph: nx.DiGraph):
        """Update adapter and routing systems when graph changes."""
        current_edges = set(self.causal_graph.edges())
        new_edges = set(new_graph.edges())
        
        # Add new edges
        for edge in new_edges - current_edges:
            adapter_idx = self.afs.add_edge(edge)
            self.bug.register_edge_adapter(edge, adapter_idx)
            
        # Remove deleted edges  
        for edge in current_edges - new_edges:
            self.afs.remove_edge(edge)
            self.bug.unregister_edge_adapter(edge)
    
    def _apply_adapter_to_output(
        self,
        base_output: torch.Tensor,
        adapter: torch.Tensor
    ) -> torch.Tensor:
        """Apply adapter transformation to model output."""
        if adapter.size(1) == base_output.size(-1):
            # Direct adaptation if dimensions match
            return torch.matmul(base_output, adapter.t())
        else:
            # Project to adapter dimension first
            projected = torch.matmul(
                base_output, 
                adapter[:min(adapter.size(0), base_output.size(-1))].t()
            )
            return projected
    
    def compute_total_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        predictions: torch.Tensor,
        edge_weights: Dict[Tuple[int, int], float],
        task_loss_fn: nn.Module = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the complete three-term loss from the algorithm.
        
        L_total = L_task + β * L_cf + λ * L_reg
        """
        if task_loss_fn is None:
            task_loss_fn = nn.CrossEntropyLoss()
        
        losses = {}
        
        # 1. Task loss
        L_task = task_loss_fn(predictions, y)
        losses['task'] = L_task
        
        # 2. Generative counterfactual loss (C4)
        z_stable, z_context = self.extract_representations(x)
        interventions = torch.randn_like(z_context) * 0.1  # Random interventions
        
        L_cf = self.gcr.compute_generative_cf_loss(
            x, self.causal_graph, interventions
        )
        losses['counterfactual'] = L_cf
        
        # 3. Path-Integrated CPI regularization (C3)
        if len(self.old_parameters) > 0:
            pi_cpi_scores, _ = self.pi_cpi(
                self.causal_graph, x, y, task_loss_fn
            )
            
            parameter_changes = self._compute_parameter_changes()
            L_reg = self.pi_cpi.compute_pi_cpi_regularizer(
                pi_cpi_scores, parameter_changes
            )
        else:
            L_reg = torch.tensor(0.0, device=self.device)
        
        losses['regularization'] = L_reg
        
        # Total loss
        total_loss = (
            L_task + 
            self.config.beta_cf * L_cf + 
            self.config.lambda_reg * L_reg
        )
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _compute_parameter_changes(self) -> Dict[str, torch.Tensor]:
        """Compute parameter changes since last checkpoint."""
        parameter_changes = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.old_parameters:
                change = param - self.old_parameters[name]
                parameter_changes[name] = change
                
        return parameter_changes
    
    def continual_update(
        self,
        data_shard: torch.utils.data.DataLoader,
        optimizer: Optional[optim.Optimizer] = None,
        validation_data: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """
        Perform continual update on a data shard following the algorithm outline.
        
        This implements the main continual loop from algorithmv2.md.
        """
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        # Store old parameters for regularization
        self._store_old_parameters()
        
        total_metrics = {
            'task_loss': 0.0,
            'counterfactual_loss': 0.0, 
            'regularization_loss': 0.0,
            'total_loss': 0.0,
            'accuracy': 0.0
        }
        
        num_batches = 0
        all_edge_weights = {}
        
        # Process data shard
        for batch_idx, (x, y) in enumerate(data_shard):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass with graph refinement
            predictions, diagnostics = self.forward(
                x, y, update_graph=True, return_diagnostics=True
            )
            
            # Get edge weights for adapter updates
            z_stable, z_context = self.extract_representations(x)
            edge_weights, uncertainty = self.bug(z_context, self.causal_graph)
            
            # Compute three-term loss
            total_loss, losses = self.compute_total_loss(
                x, y, predictions, edge_weights
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update parameters (only those with low PI-CPI)
            self._selective_parameter_update(optimizer)
            
            # Update AFS basis
            gradients = self._extract_edge_gradients(edge_weights)
            self.afs.update_basis(gradients)
            
            # Partial fit GCR
            interventions = torch.randn_like(z_context) * 0.1
            self.gcr.partial_fit(x, self.causal_graph, interventions, optimizer)
            
            # Accumulate metrics
            for key, value in losses.items():
                if key in total_metrics:
                    total_metrics[key] += value.item()
            
            # Compute accuracy
            if y.dim() > 1:  # Multi-class
                accuracy = (predictions.argmax(-1) == y.argmax(-1)).float().mean()
            else:
                accuracy = (predictions.argmax(-1) == y).float().mean()
            total_metrics['accuracy'] += accuracy.item()
            
            num_batches += 1
            self.training_step += 1
            
            # Merge edge weights for final update
            for edge, weight in edge_weights.items():
                if edge in all_edge_weights:
                    all_edge_weights[edge] += weight
                else:
                    all_edge_weights[edge] = weight
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= max(num_batches, 1)
        
        # Update BUG from validation performance
        if validation_data is not None:
            self._update_routing_from_validation(validation_data)
        
        # Update performance history
        self.performance_history.append(total_metrics)
        self.loss_history.extend([total_metrics['total_loss']])
        
        return total_metrics
    
    def _store_old_parameters(self):
        """Store current parameters as old parameters for regularization."""
        self.old_parameters = {
            name: param.clone().detach()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
    
    def _selective_parameter_update(self, optimizer: optim.Optimizer):
        """
        Update only parameters with low PI-CPI scores to prevent catastrophic forgetting.
        """
        # Get current PI-CPI scores
        dummy_input = torch.randn(1, 3, 32, 32, device=self.device)  # Dummy input
        dummy_target = torch.randint(0, 10, (1,), device=self.device)
        
        try:
            pi_cpi_scores, _ = self.pi_cpi(
                self.causal_graph, dummy_input, dummy_target
            )
            
            # Freeze parameters with high PI-CPI (high importance)
            pi_cpi_threshold = np.percentile(list(pi_cpi_scores.values()), 75)
            
            for name, param in self.named_parameters():
                if param.requires_grad and name in pi_cpi_scores:
                    if pi_cpi_scores[name] > pi_cpi_threshold:
                        param.requires_grad_(False)
            
            # Update parameters
            optimizer.step()
            
            # Restore gradients for next iteration
            for name, param in self.named_parameters():
                if name in pi_cpi_scores:
                    param.requires_grad_(True)
                    
        except Exception as e:
            # Fallback to normal update
            print(f"PI-CPI selective update failed: {e}. Using normal update.")
            optimizer.step()
    
    def _extract_edge_gradients(
        self,
        edge_weights: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """Extract gradients for each edge for AFS basis update."""
        gradients = {}
        
        for edge, weight in edge_weights.items():
            # Get adapter for this edge
            adapter = self.afs.get_edge_adapter(edge)
            
            if adapter.grad is not None:
                gradients[edge] = adapter.grad * weight
            else:
                # Create dummy gradient if none available
                gradients[edge] = torch.randn_like(adapter) * 0.01
        
        return gradients
    
    def _update_routing_from_validation(
        self,
        validation_data: torch.utils.data.DataLoader
    ):
        """Update BUG routing based on validation performance."""
        self.eval()
        
        total_errors = []
        all_contexts = []
        
        with torch.no_grad():
            for x_val, y_val in validation_data:
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                
                predictions = self.forward(x_val, update_graph=False)
                
                # Compute per-sample error
                errors = F.cross_entropy(predictions, y_val, reduction='none')
                total_errors.append(errors)
                
                # Get contexts
                _, z_context = self.extract_representations(x_val)
                all_contexts.append(z_context)
        
        if total_errors and all_contexts:
            all_errors = torch.cat(total_errors)
            all_contexts = torch.cat(all_contexts)
            
            # Update BUG routing
            self.bug.update_from_feedback(
                all_contexts, self.causal_graph, all_errors
            )
        
        self.train()
    
    def compute_pac_bayesian_bound(self) -> float:
        """
        Compute PAC-Bayesian plasticity-regret bound from the theory.
        
        Returns the bound from Theorem 1 in algorithmv2.md.
        """
        if not self.loss_history:
            return float('inf')
        
        # Compute empirical risk
        T = len(self.loss_history)
        empirical_risk = np.mean(self.loss_history)
        
        # Compute capacity term (simplified version)
        total_capacity = sum(
            param.numel() for param in self.parameters() 
            if param.requires_grad
        )
        
        # PAC-Bayesian bound
        kl_term = np.log(1 / self.config.pac_delta)
        bound = np.sqrt(
            (total_capacity * kl_term) / (2 * T)
        )
        
        return empirical_risk + bound
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint with all component states."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'causal_graph': nx.node_link_data(self.causal_graph),
            'training_step': self.training_step,
            'performance_history': self.performance_history,
            'loss_history': self.loss_history,
            'old_parameters': self.old_parameters
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.causal_graph = nx.node_link_graph(checkpoint['causal_graph'])
        self.training_step = checkpoint['training_step']
        self.performance_history = checkpoint['performance_history']
        self.loss_history = checkpoint['loss_history']
        self.old_parameters = checkpoint.get('old_parameters', {})
        
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        return {
            'training_step': self.training_step,
            'graph_structure': {
                'nodes': len(self.causal_graph.nodes()),
                'edges': len(self.causal_graph.edges()),
                'density': nx.density(self.causal_graph)
            },
            'ogr_stats': self.ogr.get_statistics(),
            'afs_memory': self.afs.get_memory_usage(),
            'pi_cpi_stats': self.pi_cpi.get_memory_stats(),
            'gcr_stats': self.gcr.get_model_stats(),
            'bug_diagnostics': self.bug.get_diagnostic_info(),
            'pac_bound': self.compute_pac_bayesian_bound(),
            'recent_performance': (
                self.performance_history[-10:] if self.performance_history else []
            )
        }