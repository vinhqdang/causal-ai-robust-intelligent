"""
Component C5: Bayesian Uncertainty Gating (BUG)
Implements Dirichlet posterior routing for adapter selection with epistemic uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import networkx as nx
from scipy.special import digamma, loggamma
from torch.distributions import Dirichlet, Categorical, Beta, Normal


class StickBreakingVAERouter(nn.Module):
    """
    Stick-breaking VAE router for continuous relaxation of Dirichlet routing.
    
    Uses stick-breaking construction with reparameterizable gradients for
    end-to-end differentiable adapter selection. This provides better
    gradient flow compared to discrete Dirichlet sampling.
    """
    
    def __init__(
        self,
        context_dim: int,
        max_adapters: int = 50,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        beta_concentration: float = 1.0,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.max_adapters = max_adapters
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta_concentration = beta_concentration
        self.temperature = temperature
        self.device = device
        
        # VAE encoder: context -> latent variables
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Latent space parameters (mean and log-variance)
        self.latent_mean = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Stick-breaking decoder: latent -> stick-breaking parameters
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_adapters - 1)  # K-1 stick-breaking variables
        )
        
        # Graph structure embedding
        self.graph_encoder = nn.Sequential(
            nn.Linear(max_adapters * max_adapters, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Adapter availability mask
        self.register_buffer(
            'adapter_mask', 
            torch.zeros(max_adapters, dtype=torch.bool)
        )
        
        # Track active adapters
        self.active_adapters = set()
        self.adapter_to_edge = {}
        self.edge_to_adapter = {}
        
        # KL loss tracking
        self.kl_losses = []
        
    def register_adapter(self, edge: Tuple[int, int], adapter_idx: int):
        """Register a new adapter for a causal edge."""
        if adapter_idx >= self.max_adapters:
            raise ValueError(f"Adapter index {adapter_idx} exceeds maximum {self.max_adapters}")
            
        self.active_adapters.add(adapter_idx)
        self.adapter_to_edge[adapter_idx] = edge
        self.edge_to_adapter[edge] = adapter_idx
        self.adapter_mask[adapter_idx] = True
    
    def unregister_adapter(self, edge: Tuple[int, int]):
        """Unregister an adapter for a causal edge."""
        if edge in self.edge_to_adapter:
            adapter_idx = self.edge_to_adapter[edge]
            self.active_adapters.discard(adapter_idx)
            del self.adapter_to_edge[adapter_idx]
            del self.edge_to_adapter[edge]
            self.adapter_mask[adapter_idx] = False
    
    def encode(
        self, 
        z_context: torch.Tensor, 
        causal_graph: Optional[nx.DiGraph] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode context to latent variables.
        
        Args:
            z_context: Context variables [batch_size, context_dim]
            causal_graph: Optional causal graph for structural conditioning
            
        Returns:
            mu: Latent means [batch_size, latent_dim]
            logvar: Latent log-variances [batch_size, latent_dim]
        """
        batch_size = z_context.size(0)
        
        # Encode context
        h = self.encoder(z_context)
        
        # Get latent parameters
        mu = self.latent_mean(h)
        logvar = self.latent_logvar(h)
        
        # Add graph structure conditioning if available
        if causal_graph is not None:
            graph_embedding = self._encode_graph_structure(causal_graph, batch_size)
            mu = mu + 0.1 * graph_embedding  # Small additive conditioning
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling latent variables.
        
        Args:
            mu: Latent means [batch_size, latent_dim]
            logvar: Latent log-variances [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent variables [batch_size, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode_to_stick_breaking(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variables to stick-breaking parameters.
        
        Args:
            z: Latent variables [batch_size, latent_dim]
            
        Returns:
            v_params: Stick-breaking parameters [batch_size, max_adapters-1]
        """
        # Decode to stick-breaking logits
        v_logits = self.decoder(z)  # [batch_size, max_adapters-1]
        
        # Apply sigmoid to get Beta parameters
        v_params = torch.sigmoid(v_logits)
        
        return v_params
    
    def stick_breaking_to_probabilities(self, v_params: torch.Tensor) -> torch.Tensor:
        """
        Convert stick-breaking parameters to probability simplex.
        
        Implements: π_k = v_k * ∏_{j<k} (1 - v_j)
        
        Args:
            v_params: Stick-breaking parameters [batch_size, max_adapters-1]
            
        Returns:
            probabilities: Probability simplex [batch_size, max_adapters]
        """
        batch_size = v_params.size(0)
        
        # Initialize probabilities
        probabilities = []
        
        # Cumulative product of (1 - v_j)
        remaining_stick = torch.ones(batch_size, device=self.device)
        
        for k in range(self.max_adapters - 1):
            v_k = v_params[:, k]
            
            # π_k = v_k * remaining_stick
            prob_k = v_k * remaining_stick
            probabilities.append(prob_k)
            
            # Update remaining stick: remaining *= (1 - v_k)
            remaining_stick = remaining_stick * (1.0 - v_k)
        
        # Last probability gets all remaining stick
        probabilities.append(remaining_stick)
        
        # Stack and apply adapter mask
        probabilities = torch.stack(probabilities, dim=1)  # [batch_size, max_adapters]
        probabilities = probabilities * self.adapter_mask.float()
        
        # Renormalize to ensure valid probabilities
        prob_sum = probabilities.sum(dim=1, keepdim=True)
        probabilities = probabilities / (prob_sum + 1e-8)
        
        return probabilities
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss for VAE regularization.
        
        KL[q(z|x) || p(z)] where p(z) = N(0, I)
        
        Args:
            mu: Latent means [batch_size, latent_dim]
            logvar: Latent log-variances [batch_size, latent_dim]
            
        Returns:
            kl_loss: KL divergence loss (scalar)
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()
        
        self.kl_losses.append(kl_loss.item())
        return kl_loss
    
    def forward(
        self,
        z_context: torch.Tensor,
        causal_graph: nx.DiGraph,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through stick-breaking VAE router.
        
        Args:
            z_context: Context variables [batch_size, context_dim]
            causal_graph: Current causal graph structure
            return_components: Whether to return intermediate components
            
        Returns:
            probabilities: Adapter probabilities [batch_size, max_adapters]
            If return_components=True, also returns (mu, logvar, kl_loss)
        """
        # Encode to latent space
        mu, logvar = self.encode(z_context, causal_graph)
        
        # Sample latent variables
        z = self.reparameterize(mu, logvar)
        
        # Decode to stick-breaking parameters
        v_params = self.decode_to_stick_breaking(z)
        
        # Convert to probabilities
        probabilities = self.stick_breaking_to_probabilities(v_params)
        
        # Apply temperature scaling
        probabilities = F.softmax(
            torch.log(probabilities + 1e-8) / self.temperature,
            dim=-1
        )
        
        if return_components:
            kl_loss = self.compute_kl_loss(mu, logvar)
            return probabilities, mu, logvar, kl_loss
        
        return probabilities
    
    def _encode_graph_structure(self, causal_graph: nx.DiGraph, batch_size: int) -> torch.Tensor:
        """
        Encode causal graph structure for conditioning.
        
        Args:
            causal_graph: Causal graph
            batch_size: Batch size
            
        Returns:
            graph_embedding: Graph structure embedding [batch_size, latent_dim]
        """
        # Create adjacency matrix representation
        adj_matrix = torch.zeros(self.max_adapters, self.max_adapters, device=self.device)
        
        # Fill adjacency based on active adapters and their edges
        for adapter_idx in self.active_adapters:
            if adapter_idx in self.adapter_to_edge:
                edge = self.adapter_to_edge[adapter_idx]
                source, target = edge
                
                # Map nodes to adapter indices if possible
                source_adapter = self.edge_to_adapter.get((source, source), None)
                target_adapter = self.edge_to_adapter.get((target, target), None)
                
                if source_adapter is not None and target_adapter is not None:
                    if source_adapter < self.max_adapters and target_adapter < self.max_adapters:
                        adj_matrix[source_adapter, target_adapter] = 1.0
        
        # Flatten and encode
        adj_flat = adj_matrix.view(-1)
        graph_embedding = self.graph_encoder(adj_flat)
        
        # Expand for batch
        graph_embedding = graph_embedding.unsqueeze(0).expand(batch_size, -1)
        
        return graph_embedding
    
    def get_soft_adapter_mixture(
        self,
        probabilities: torch.Tensor,
        threshold: float = 0.01
    ) -> Dict[Tuple[int, int], float]:
        """
        Convert probabilities to soft adapter mixture.
        
        Args:
            probabilities: Adapter probabilities [batch_size, max_adapters]
            threshold: Minimum probability threshold
            
        Returns:
            edge_weights: Dictionary mapping edges to mixture weights
        """
        # Average probabilities across batch
        avg_probs = probabilities.mean(dim=0)
        
        edge_weights = {}
        
        for adapter_idx in self.active_adapters:
            prob = avg_probs[adapter_idx].item()
            
            if prob > threshold and adapter_idx in self.adapter_to_edge:
                edge = self.adapter_to_edge[adapter_idx]
                edge_weights[edge] = prob
        
        # Normalize weights
        total_weight = sum(edge_weights.values())
        if total_weight > 0:
            edge_weights = {
                edge: weight / total_weight
                for edge, weight in edge_weights.items()
            }
        
        return edge_weights
    
    def compute_reconstruction_loss(
        self,
        z_context: torch.Tensor,
        target_probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for VAE training.
        
        Args:
            z_context: Input context [batch_size, context_dim]
            target_probabilities: Target probability distribution [batch_size, max_adapters]
            
        Returns:
            recon_loss: Reconstruction loss (scalar)
        """
        # Forward pass
        pred_probabilities = self.forward(z_context, None)
        
        # KL divergence between predicted and target distributions
        recon_loss = F.kl_div(
            torch.log(pred_probabilities + 1e-8),
            target_probabilities,
            reduction='batchmean'
        )
        
        return recon_loss
    
    def get_vae_statistics(self) -> Dict[str, float]:
        """Get VAE training statistics."""
        return {
            'avg_kl_loss': np.mean(self.kl_losses[-100:]) if self.kl_losses else 0.0,
            'latent_dim': self.latent_dim,
            'active_adapters': len(self.active_adapters),
            'temperature': self.temperature,
            'total_kl_updates': len(self.kl_losses)
        }


class DirichletRouter(nn.Module):
    """
    Dirichlet-based router that maintains uncertainty over adapter selections.
    
    Treats adapter routing as a Dirichlet posterior, allowing CASCADA to sample
    or marginalize adapters according to epistemic uncertainty.
    """
    
    def __init__(
        self,
        context_dim: int,
        max_adapters: int = 50,
        hidden_dim: int = 128,
        concentration_init: float = 1.0,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.max_adapters = max_adapters
        self.hidden_dim = hidden_dim
        self.concentration_init = concentration_init
        self.temperature = temperature
        self.device = device
        
        # Context encoder for routing decisions
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_adapters)
        )
        
        # Learnable concentration parameters (log-space for positivity)
        self.log_concentrations = nn.Parameter(
            torch.log(torch.ones(max_adapters) * concentration_init)
        )
        
        # Adapter availability mask
        self.register_buffer(
            'adapter_mask', 
            torch.zeros(max_adapters, dtype=torch.bool)
        )
        
        # Routing history for adaptation
        self.routing_history = []
        self.validation_errors = []
        
        # Track active adapters
        self.active_adapters = set()
        self.adapter_to_edge = {}  # Map adapter indices to edge tuples
        self.edge_to_adapter = {}  # Map edge tuples to adapter indices
    
    def register_adapter(self, edge: Tuple[int, int], adapter_idx: int):
        """Register a new adapter for a causal edge."""
        if adapter_idx >= self.max_adapters:
            raise ValueError(f"Adapter index {adapter_idx} exceeds maximum {self.max_adapters}")
            
        self.active_adapters.add(adapter_idx)
        self.adapter_to_edge[adapter_idx] = edge
        self.edge_to_adapter[edge] = adapter_idx
        self.adapter_mask[adapter_idx] = True
    
    def unregister_adapter(self, edge: Tuple[int, int]):
        """Unregister an adapter for a causal edge."""
        if edge in self.edge_to_adapter:
            adapter_idx = self.edge_to_adapter[edge]
            self.active_adapters.discard(adapter_idx)
            del self.adapter_to_edge[adapter_idx]
            del self.edge_to_adapter[edge]
            self.adapter_mask[adapter_idx] = False
    
    def forward(
        self,
        z_context: torch.Tensor,
        causal_graph: nx.DiGraph,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute Dirichlet posterior over adapters.
        
        Args:
            z_context: Context variables [batch_size, context_dim]
            causal_graph: Current causal graph structure  
            return_uncertainty: Whether to return uncertainty measures
            
        Returns:
            probabilities: Adapter selection probabilities [batch_size, max_adapters]
            uncertainties: Epistemic uncertainty measures (if requested)
        """
        batch_size = z_context.size(0)
        
        # Encode context to get routing logits
        routing_logits = self.context_encoder(z_context)  # [batch_size, max_adapters]
        
        # Get concentration parameters (ensure positivity)
        concentrations = torch.exp(self.log_concentrations)  # [max_adapters]
        
        # Modulate concentrations based on context and graph structure
        modulated_concentrations = self._modulate_concentrations(
            concentrations, routing_logits, causal_graph
        )  # [batch_size, max_adapters]
        
        # Apply adapter mask to zero out inactive adapters
        masked_concentrations = modulated_concentrations * self.adapter_mask.float()
        
        # Ensure minimum concentration for numerical stability
        masked_concentrations = torch.clamp(masked_concentrations, min=1e-6)
        
        # Create Dirichlet distribution
        dirichlet = Dirichlet(masked_concentrations)
        
        # Sample or use mean
        if self.training:
            # Sample from posterior during training
            probabilities = dirichlet.rsample()
        else:
            # Use mean during inference
            probabilities = dirichlet.mean
        
        # Apply temperature scaling for sharpness control
        probabilities = F.softmax(
            torch.log(probabilities + 1e-8) / self.temperature, 
            dim=-1
        )
        
        if return_uncertainty:
            uncertainty = self._compute_uncertainty(masked_concentrations)
            return probabilities, uncertainty
        
        return probabilities
    
    def _modulate_concentrations(
        self,
        base_concentrations: torch.Tensor,
        routing_logits: torch.Tensor,
        causal_graph: nx.DiGraph
    ) -> torch.Tensor:
        """
        Modulate concentration parameters based on context and graph structure.
        """
        batch_size = routing_logits.size(0)
        
        # Expand base concentrations for batch
        concentrations = base_concentrations.unsqueeze(0).expand(
            batch_size, -1
        )  # [batch_size, max_adapters]
        
        # Context modulation: routing logits influence concentrations
        context_modulation = torch.sigmoid(routing_logits)
        modulated = concentrations * (1.0 + context_modulation)
        
        # Graph structure modulation based on edge importance
        graph_modulation = self._compute_graph_modulation(causal_graph, batch_size)
        modulated = modulated * graph_modulation
        
        return modulated
    
    def _compute_graph_modulation(
        self, 
        causal_graph: nx.DiGraph, 
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute graph structure-based modulation of concentrations.
        """
        modulation = torch.ones(
            batch_size, self.max_adapters, 
            device=self.device
        )
        
        # Modulate based on edge centrality and importance
        for adapter_idx in self.active_adapters:
            if adapter_idx in self.adapter_to_edge:
                edge = self.adapter_to_edge[adapter_idx]
                source, target = edge
                
                # Check if nodes exist in graph
                if source in causal_graph.nodes() and target in causal_graph.nodes():
                    # Edge importance based on node degrees
                    source_degree = causal_graph.degree(source)
                    target_degree = causal_graph.degree(target)
                    
                    # Higher degree nodes get higher concentration
                    importance = 1.0 + 0.1 * (source_degree + target_degree)
                    modulation[:, adapter_idx] *= importance
        
        return modulation
    
    def _compute_uncertainty(
        self, 
        concentrations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute epistemic uncertainty measures from Dirichlet parameters.
        
        Returns multiple uncertainty measures:
        - Entropy: H[p] = -Σ p_i log p_i  
        - Mutual information: I[y,θ] ≈ H[E[p]] - E[H[p]]
        - Concentration magnitude: ||α||
        """
        batch_size = concentrations.size(0)
        
        # Dirichlet mean (expected probabilities)
        alpha_0 = concentrations.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        mean_probs = concentrations / alpha_0  # [batch_size, max_adapters]
        
        # Entropy of expected probabilities
        entropy_of_expected = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        
        # Expected entropy (analytical formula for Dirichlet)
        digamma_alpha_0 = torch.tensor(
            [digamma(a.item()) for a in alpha_0.squeeze()], 
            device=self.device
        )
        digamma_concentrations = torch.tensor(
            [[digamma(c.item()) for c in conc] for conc in concentrations],
            device=self.device
        )
        
        expected_entropy = (
            digamma_alpha_0.unsqueeze(1) - digamma_concentrations
        ).sum(dim=-1) * (concentrations / alpha_0).sum(dim=-1)
        
        # Mutual information (epistemic uncertainty)
        mutual_info = entropy_of_expected - expected_entropy
        
        # Concentration magnitude (confidence measure)
        concentration_magnitude = alpha_0.squeeze()
        
        # Combine into uncertainty tensor
        uncertainty = torch.stack([
            entropy_of_expected,
            mutual_info,
            concentration_magnitude
        ], dim=-1)  # [batch_size, 3]
        
        return uncertainty
    
    def sample_adapters(
        self,
        probabilities: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample adapter indices from probability distribution.
        
        Args:
            probabilities: Adapter probabilities [batch_size, max_adapters]
            num_samples: Number of samples per batch item
            
        Returns:
            samples: Sampled adapter indices [batch_size, num_samples]
        """
        batch_size = probabilities.size(0)
        
        # Create categorical distribution
        categorical = Categorical(probabilities)
        
        # Sample adapter indices
        samples = categorical.sample((num_samples,)).T  # [batch_size, num_samples]
        
        return samples
    
    def get_soft_adapter_mixture(
        self,
        probabilities: torch.Tensor,
        threshold: float = 0.01
    ) -> Dict[Tuple[int, int], float]:
        """
        Convert probabilities to soft adapter mixture for differentiable routing.
        
        Args:
            probabilities: Adapter probabilities [batch_size, max_adapters]
            threshold: Minimum probability threshold
            
        Returns:
            edge_weights: Dictionary mapping edges to mixture weights
        """
        # Average probabilities across batch (assumes similar context)
        avg_probs = probabilities.mean(dim=0)
        
        edge_weights = {}
        
        for adapter_idx in self.active_adapters:
            prob = avg_probs[adapter_idx].item()
            
            if prob > threshold and adapter_idx in self.adapter_to_edge:
                edge = self.adapter_to_edge[adapter_idx]
                edge_weights[edge] = prob
        
        # Normalize weights
        total_weight = sum(edge_weights.values())
        if total_weight > 0:
            edge_weights = {
                edge: weight / total_weight 
                for edge, weight in edge_weights.items()
            }
        
        return edge_weights
    
    def update_from_validation(
        self,
        probabilities: torch.Tensor,
        validation_errors: torch.Tensor
    ):
        """
        Update router based on validation performance.
        
        This implements the validation-based update from the algorithm outline.
        """
        batch_size = probabilities.size(0)
        
        # Store for history
        self.routing_history.append(probabilities.detach().cpu())
        self.validation_errors.append(validation_errors.detach().cpu())
        
        # Compute performance-based concentration updates
        # Lower error -> higher concentration for used adapters
        performance_weight = 1.0 / (validation_errors + 1e-8)  # [batch_size]
        
        # Update concentrations based on performance
        with torch.no_grad():
            for adapter_idx in self.active_adapters:
                adapter_probs = probabilities[:, adapter_idx]  # [batch_size]
                
                # Weighted average performance for this adapter
                weighted_performance = (adapter_probs * performance_weight).sum()
                usage_count = adapter_probs.sum() + 1e-8
                
                avg_performance = weighted_performance / usage_count
                
                # Update log concentration (gradient-free)
                learning_rate = 0.01
                current_log_conc = self.log_concentrations[adapter_idx]
                
                # Increase concentration for good performance, decrease for bad
                update = learning_rate * (avg_performance - 1.0)
                self.log_concentrations[adapter_idx] = current_log_conc + update
        
        # Keep history bounded
        max_history = 1000
        if len(self.routing_history) > max_history:
            self.routing_history = self.routing_history[-max_history:]
            self.validation_errors = self.validation_errors[-max_history:]


class BayesianUncertaintyGating(nn.Module):
    """
    Complete Bayesian Uncertainty Gating system combining routing and uncertainty estimation.
    """
    
    def __init__(
        self,
        context_dim: int,
        max_adapters: int = 50,
        uncertainty_threshold: float = 0.1,
        exploration_rate: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        router_type: str = "dirichlet",  # "dirichlet" or "stick_breaking_vae"
        vae_latent_dim: int = 32
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.max_adapters = max_adapters
        self.uncertainty_threshold = uncertainty_threshold
        self.exploration_rate = exploration_rate
        self.device = device
        self.router_type = router_type
        
        # Initialize appropriate router
        if router_type == "stick_breaking_vae":
            self.router = StickBreakingVAERouter(
                context_dim=context_dim,
                max_adapters=max_adapters,
                latent_dim=vae_latent_dim,
                device=device
            )
        elif router_type == "dirichlet":
            self.router = DirichletRouter(
                context_dim=context_dim,
                max_adapters=max_adapters,
                device=device
            )
        else:
            raise ValueError(f"Unknown router type: {router_type}. Choose 'dirichlet' or 'stick_breaking_vae'")
        
        # Uncertainty-based exploration strategy
        self.exploration_strategy = "thompson"  # "thompson", "ucb", "epsilon_greedy"
        
        # Track routing performance
        self.routing_stats = {
            'total_decisions': 0,
            'high_uncertainty_decisions': 0,
            'exploration_decisions': 0,
            'avg_uncertainty': 0.0
        }
    
    def forward(
        self,
        z_context: torch.Tensor,
        causal_graph: nx.DiGraph,
        exploration: bool = True
    ) -> Tuple[Dict[Tuple[int, int], float], torch.Tensor]:
        """
        Complete forward pass with uncertainty-aware routing.
        
        Args:
            z_context: Context variables [batch_size, context_dim] 
            causal_graph: Current causal graph structure
            exploration: Whether to apply exploration strategy
            
        Returns:
            edge_weights: Soft adapter mixture weights
            uncertainty: Uncertainty measures [batch_size, 3]
        """
        # Get probabilities and uncertainty from router
        if self.router_type == "stick_breaking_vae":
            probabilities, mu, logvar, kl_loss = self.router(
                z_context, causal_graph, return_components=True
            )
            # Convert VAE outputs to uncertainty measures (simplified)
            uncertainty = torch.stack([
                -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1),  # Entropy
                kl_loss.expand(z_context.size(0)),  # KL as mutual info proxy
                logvar.mean(dim=1)  # Latent variance as confidence measure
            ], dim=1)
        else:
            # Dirichlet router
            probabilities, uncertainty = self.router(
                z_context, causal_graph, return_uncertainty=True
            )
        
        # Apply exploration if enabled
        if exploration and self.training:
            probabilities = self._apply_exploration_strategy(
                probabilities, uncertainty
            )
        
        # Convert to edge weights for adapter mixing
        edge_weights = self.router.get_soft_adapter_mixture(probabilities)
        
        # Update statistics
        self._update_routing_stats(uncertainty)
        
        return edge_weights, uncertainty
    
    def _apply_exploration_strategy(
        self,
        probabilities: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply uncertainty-guided exploration strategy.
        """
        batch_size = probabilities.size(0)
        
        if self.exploration_strategy == "thompson":
            # Thompson sampling: sample from posterior
            concentrations = probabilities * 10.0  # Scale up for sampling
            dirichlet = Dirichlet(concentrations + 1e-6)
            return dirichlet.rsample()
            
        elif self.exploration_strategy == "ucb":
            # Upper Confidence Bound based on uncertainty
            mutual_info = uncertainty[:, 1]  # Use mutual information
            ucb_bonus = torch.sqrt(mutual_info).unsqueeze(1)
            
            # Add uncertainty bonus to probabilities
            return F.softmax(
                torch.log(probabilities + 1e-8) + ucb_bonus, 
                dim=-1
            )
            
        elif self.exploration_strategy == "epsilon_greedy":
            # Epsilon-greedy with uncertainty-adaptive epsilon
            mutual_info = uncertainty[:, 1]
            adaptive_epsilon = self.exploration_rate * torch.sigmoid(mutual_info)
            
            # Mix with uniform distribution based on adaptive epsilon
            uniform = torch.ones_like(probabilities) / probabilities.size(-1)
            
            explore_mask = torch.rand(batch_size, 1, device=self.device) < adaptive_epsilon.unsqueeze(1)
            
            return torch.where(explore_mask, uniform, probabilities)
        
        return probabilities
    
    def _update_routing_stats(self, uncertainty: torch.Tensor):
        """Update routing statistics for monitoring."""
        batch_size = uncertainty.size(0)
        
        # Count high uncertainty decisions
        mutual_info = uncertainty[:, 1]
        high_uncertainty = (mutual_info > self.uncertainty_threshold).sum().item()
        
        # Update running statistics
        self.routing_stats['total_decisions'] += batch_size
        self.routing_stats['high_uncertainty_decisions'] += high_uncertainty
        
        # Running average of uncertainty
        alpha = 0.01
        current_avg = mutual_info.mean().item()
        self.routing_stats['avg_uncertainty'] = (
            alpha * current_avg + 
            (1 - alpha) * self.routing_stats['avg_uncertainty']
        )
    
    def register_edge_adapter(self, edge: Tuple[int, int], adapter_idx: int):
        """Register adapter for causal edge."""
        self.router.register_adapter(edge, adapter_idx)
    
    def unregister_edge_adapter(self, edge: Tuple[int, int]):
        """Unregister adapter for causal edge."""
        self.router.unregister_adapter(edge)
    
    def update_from_feedback(
        self,
        z_context: torch.Tensor,
        causal_graph: nx.DiGraph,
        validation_errors: torch.Tensor
    ):
        """
        Update routing based on validation feedback.
        
        This implements the BUG update from the algorithm outline.
        """
        # Get current routing probabilities
        probabilities = self.router(z_context, causal_graph)
        
        # Update router based on validation performance
        self.router.update_from_validation(probabilities, validation_errors)
    
    def get_routing_confidence(
        self,
        z_context: torch.Tensor,
        causal_graph: nx.DiGraph
    ) -> torch.Tensor:
        """
        Get confidence measure for current routing decision.
        
        Returns:
            confidence: Confidence scores [batch_size] (higher = more confident)
        """
        _, uncertainty = self.router(z_context, causal_graph, return_uncertainty=True)
        
        # Use inverse of mutual information as confidence
        mutual_info = uncertainty[:, 1]
        confidence = 1.0 / (1.0 + mutual_info)
        
        return confidence
    
    def get_diagnostic_info(self) -> Dict:
        """Get diagnostic information about routing behavior."""
        total = self.routing_stats['total_decisions']
        
        base_info = {
            'total_routing_decisions': total,
            'high_uncertainty_rate': (
                self.routing_stats['high_uncertainty_decisions'] / max(total, 1)
            ),
            'average_uncertainty': self.routing_stats['avg_uncertainty'],
            'active_adapters': len(self.router.active_adapters),
            'exploration_strategy': self.exploration_strategy,
            'uncertainty_threshold': self.uncertainty_threshold,
            'router_type': self.router_type
        }
        
        # Add router-specific diagnostics
        if self.router_type == "stick_breaking_vae":
            vae_stats = self.router.get_vae_statistics()
            base_info.update({
                'vae_avg_kl_loss': vae_stats['avg_kl_loss'],
                'vae_latent_dim': vae_stats['latent_dim'],
                'vae_temperature': vae_stats['temperature'],
                'vae_kl_updates': vae_stats['total_kl_updates']
            })
        
        return base_info
    
    def compute_vae_loss(
        self,
        z_context: torch.Tensor,
        causal_graph: nx.DiGraph,
        beta: float = 1.0
    ) -> Optional[torch.Tensor]:
        """
        Compute VAE loss for Stick-Breaking router (if applicable).
        
        Args:
            z_context: Context variables [batch_size, context_dim]
            causal_graph: Current causal graph
            beta: Beta-VAE regularization weight
            
        Returns:
            vae_loss: Combined reconstruction + KL loss (or None if not VAE router)
        """
        if self.router_type != "stick_breaking_vae":
            return None
        
        # Forward pass with components
        probabilities, mu, logvar, kl_loss = self.router(
            z_context, causal_graph, return_components=True
        )
        
        # For stick-breaking VAE, reconstruction loss is implicit in the probabilistic formulation
        # We focus on the KL regularization
        vae_loss = beta * kl_loss
        
        return vae_loss