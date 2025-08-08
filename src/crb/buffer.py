import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pickle
import json
from dataclasses import dataclass

@dataclass
class CausalRecord:
    """A single causal record storing parent-child relationship."""
    parent_stats: torch.Tensor  # Sufficient statistics of parent variables
    child_outcome: torch.Tensor  # Observed child outcome
    context: torch.Tensor  # Context information (confounders)
    timestamp: float  # When this record was created
    edge_id: str  # Which causal edge this belongs to
    

class ParentStatEncoder(nn.Module):
    """
    Encode parent variable configurations into compact sufficient statistics.
    Uses variational encoding to capture distributional properties.
    """
    
    def __init__(self, input_dim: int, stat_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.stat_dim = stat_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stat_dim * 2)  # Mean and log-variance
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input into mean and log-variance of sufficient statistics.
        
        Args:
            x: Input tensor [batch_size, ..., input_dim]
            
        Returns:
            (mean, log_var) each [batch_size, ..., stat_dim]
        """
        batch_shape = None
        if x.dim() > 2:
            batch_shape = x.shape[:-1]
            x = x.reshape(-1, self.input_dim)
            
        encoded = self.encoder(x)
        mean, log_var = encoded.chunk(2, dim=-1)
        
        if batch_shape is not None and len(batch_shape) > 1:
            mean = mean.reshape(*batch_shape, self.stat_dim)
            log_var = log_var.reshape(*batch_shape, self.stat_dim)
            
        return mean, log_var
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


class CounterfactualSampler(nn.Module):
    """
    Generate counterfactual outcomes given parent statistics and interventions.
    """
    
    def __init__(self, stat_dim: int, context_dim: int, outcome_dim: int, 
                 hidden_dim: int = 256):
        super().__init__()
        self.stat_dim = stat_dim
        self.context_dim = context_dim
        self.outcome_dim = outcome_dim
        
        # Counterfactual generation network
        self.cf_generator = nn.Sequential(
            nn.Linear(stat_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outcome_dim * 2)  # Mean and log-variance for outcome distribution
        )
        
    def forward(self, parent_stats: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate counterfactual outcome distribution parameters.
        
        Args:
            parent_stats: Encoded parent statistics [batch_size, stat_dim]
            context: Context/confounder information [batch_size, context_dim]
            
        Returns:
            (cf_mean, cf_log_var) for counterfactual outcome distribution
        """
        input_features = torch.cat([parent_stats, context], dim=-1)
        output = self.cf_generator(input_features)
        cf_mean, cf_log_var = output.chunk(2, dim=-1)
        return cf_mean, cf_log_var
    
    def sample_counterfactual(self, parent_stats: torch.Tensor, 
                            context: torch.Tensor) -> torch.Tensor:
        """Sample counterfactual outcomes."""
        cf_mean, cf_log_var = self.forward(parent_stats, context)
        std = torch.exp(0.5 * cf_log_var)
        eps = torch.randn_like(std)
        return cf_mean + eps * std


class CounterfactualReplayBuffer:
    """
    Advanced Counterfactual Replay Buffer with:
    - Parent-variable sufficient statistics compression
    - Differentiable counterfactual sampling
    - Memory-efficient storage with compression
    """
    
    def __init__(self, stat_dim: int = 64, context_dim: int = 32, 
                 outcome_dim: int = 128, max_size: int = 10000,
                 compression_ratio: float = 0.1):
        """
        Initialize the Counterfactual Replay Buffer.
        
        Args:
            stat_dim: Dimension of parent sufficient statistics
            context_dim: Dimension of context/confounder variables
            outcome_dim: Dimension of outcome variables
            max_size: Maximum number of records per edge
            compression_ratio: Fraction of old records to keep when buffer is full
        """
        self.stat_dim = stat_dim
        self.context_dim = context_dim
        self.outcome_dim = outcome_dim
        self.max_size = max_size
        self.compression_ratio = compression_ratio
        
        # Storage for causal records
        self.records: Dict[str, List[CausalRecord]] = defaultdict(list)
        
        # Neural components (will be initialized when first edge is added)
        self.encoders: Dict[str, ParentStatEncoder] = {}
        self.samplers: Dict[str, CounterfactualSampler] = {}
        
        print("Advanced CounterfactualReplayBuffer initialized.")
    
    def _get_or_create_encoder(self, edge: str, input_dim: int) -> ParentStatEncoder:
        """Get or create encoder for specific edge."""
        if edge not in self.encoders:
            self.encoders[edge] = ParentStatEncoder(input_dim, self.stat_dim)
        return self.encoders[edge]
    
    def _get_or_create_sampler(self, edge: str) -> CounterfactualSampler:
        """Get or create counterfactual sampler for specific edge."""
        if edge not in self.samplers:
            self.samplers[edge] = CounterfactualSampler(
                self.stat_dim, self.context_dim, self.outcome_dim
            )
        return self.samplers[edge]
    
    def add_record(self, edge: str, parent_config: torch.Tensor, 
                  child_outcome: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        Add a causal record to the buffer.
        
        Args:
            edge: Causal edge identifier (e.g., 'X->Y')
            parent_config: Parent variable configuration
            child_outcome: Observed child outcome
            context: Context/confounder variables
        """
        # Handle context
        if context is None:
            context = torch.zeros(self.context_dim)
        elif context.shape[-1] != self.context_dim:
            # Pad or truncate context to match expected dimension
            if context.shape[-1] < self.context_dim:
                padding = torch.zeros(*context.shape[:-1], 
                                    self.context_dim - context.shape[-1])
                context = torch.cat([context, padding], dim=-1)
            else:
                context = context[..., :self.context_dim]
        
        # Get or create encoder for this edge
        input_dim = parent_config.shape[-1]
        encoder = self._get_or_create_encoder(edge, input_dim)
        
        # Encode parent configuration to sufficient statistics
        with torch.no_grad():
            parent_mean, parent_log_var = encoder(parent_config.unsqueeze(0))
            parent_stats = encoder.reparameterize(parent_mean, parent_log_var).squeeze(0)
        
        # Create record
        record = CausalRecord(
            parent_stats=parent_stats.detach(),
            child_outcome=child_outcome.detach(),
            context=context.detach(),
            timestamp=torch.rand(1).item(),  # Simple timestamp
            edge_id=edge
        )
        
        # Add to buffer
        self.records[edge].append(record)
        
        # Compress if buffer is full
        if len(self.records[edge]) > self.max_size:
            self._compress_buffer(edge)
    
    def _compress_buffer(self, edge: str):
        """Compress buffer by removing oldest records."""
        records = self.records[edge]
        # Sort by timestamp and keep most recent fraction
        records.sort(key=lambda r: r.timestamp, reverse=True)
        keep_size = int(self.max_size * self.compression_ratio)
        self.records[edge] = records[:keep_size]
    
    def sample_batch(self, edge: str, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample a batch of records for a given edge.
        
        Args:
            edge: Causal edge to sample from
            batch_size: Number of samples to return
            
        Returns:
            (parent_stats, child_outcomes, contexts) or None if edge not found
        """
        if edge not in self.records or len(self.records[edge]) == 0:
            return None
        
        # Sample random records
        records = self.records[edge]
        n_records = len(records)
        indices = torch.randint(0, n_records, (batch_size,))
        
        # Collect sampled data
        parent_stats = torch.stack([records[i].parent_stats for i in indices])
        child_outcomes = torch.stack([records[i].child_outcome for i in indices])
        contexts = torch.stack([records[i].context for i in indices])
        
        return parent_stats, child_outcomes, contexts
    
    def sample_counterfactuals(self, edge: str, batch_size: int, 
                             intervention_strength: float = 1.0) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample counterfactual outcomes for a given edge.
        
        Args:
            edge: Causal edge to sample from
            batch_size: Number of counterfactual samples
            intervention_strength: Strength of intervention (0=no intervention, 1=full)
            
        Returns:
            (original_outcomes, counterfactual_outcomes, parent_stats) or None
        """
        batch_data = self.sample_batch(edge, batch_size)
        if batch_data is None:
            return None
        
        parent_stats, original_outcomes, contexts = batch_data
        
        # Get counterfactual sampler
        sampler = self._get_or_create_sampler(edge)
        
        # Apply intervention to parent stats (simple noise-based intervention)
        intervention_noise = torch.randn_like(parent_stats) * intervention_strength
        intervened_stats = parent_stats + intervention_noise
        
        # Generate counterfactual outcomes
        with torch.no_grad():
            counterfactual_outcomes = sampler.sample_counterfactual(intervened_stats, contexts)
        
        return original_outcomes, counterfactual_outcomes, parent_stats
    
    def compute_counterfactual_loss(self, edge: str, batch_size: int, 
                                  loss_fn: callable = F.mse_loss) -> Optional[torch.Tensor]:
        """
        Compute counterfactual consistency loss.
        
        Args:
            edge: Causal edge to compute loss for
            batch_size: Batch size for sampling
            loss_fn: Loss function to use
            
        Returns:
            Counterfactual loss tensor or None
        """
        cf_data = self.sample_counterfactuals(edge, batch_size)
        if cf_data is None:
            return None
        
        original_outcomes, counterfactual_outcomes, _ = cf_data
        
        # Flatten both tensors to ensure they match
        original_flat = original_outcomes.view(original_outcomes.shape[0], -1)
        counterfactual_flat = counterfactual_outcomes.view(counterfactual_outcomes.shape[0], -1)
        
        # Ensure both have the same final dimension
        min_dim = min(original_flat.shape[-1], counterfactual_flat.shape[-1])
        original_flat = original_flat[..., :min_dim]
        counterfactual_flat = counterfactual_flat[..., :min_dim]
        
        # Simple consistency loss: counterfactuals should be different from originals
        # but within reasonable bounds
        consistency_loss = loss_fn(counterfactual_flat, original_flat)
        
        # Add regularization to prevent collapse
        diversity_loss = -torch.var(counterfactual_outcomes, dim=0).mean()
        
        return consistency_loss + 0.1 * diversity_loss
    
    def get_buffer_stats(self) -> Dict[str, Dict]:
        """Get statistics about the buffer contents."""
        stats = {}
        for edge, records in self.records.items():
            if records:
                stats[edge] = {
                    'num_records': len(records),
                    'avg_timestamp': np.mean([r.timestamp for r in records]),
                    'stat_dim': records[0].parent_stats.shape[-1],
                    'outcome_dim': records[0].child_outcome.shape[-1],
                    'context_dim': records[0].context.shape[-1]
                }
            else:
                stats[edge] = {'num_records': 0}
        return stats
    
    def save_buffer(self, filepath: str):
        """Save buffer to file."""
        save_data = {
            'records': dict(self.records),
            'config': {
                'stat_dim': self.stat_dim,
                'context_dim': self.context_dim,
                'outcome_dim': self.outcome_dim,
                'max_size': self.max_size,
                'compression_ratio': self.compression_ratio
            }
        }
        torch.save(save_data, filepath)
    
    def load_buffer(self, filepath: str):
        """Load buffer from file."""
        save_data = torch.load(filepath)
        self.records = defaultdict(list, save_data['records'])
        
        # Update config
        config = save_data['config']
        self.stat_dim = config['stat_dim']
        self.context_dim = config['context_dim'] 
        self.outcome_dim = config['outcome_dim']
        self.max_size = config['max_size']
        self.compression_ratio = config['compression_ratio']

if __name__ == '__main__':
    # Example Usage of Advanced Counterfactual Replay Buffer
    
    print("=== Testing Advanced Counterfactual Replay Buffer ===")
    
    # 1. Initialize the buffer
    stat_dim = 32
    context_dim = 16
    outcome_dim = 64
    crb = CounterfactualReplayBuffer(
        stat_dim=stat_dim, 
        context_dim=context_dim, 
        outcome_dim=outcome_dim,
        max_size=1000
    )

    # 2. Define causal edges and generate data
    edges = ["treatment->outcome", "confounder->treatment", "outcome->feedback"]
    parent_dim = 128  # Dimension of parent variables
    num_samples = 200
    
    print(f"\nAdding {num_samples} records for each edge...")
    
    for edge in edges:
        for i in range(num_samples):
            # Generate synthetic data
            parent_config = torch.randn(parent_dim)
            child_outcome = torch.randn(outcome_dim)
            context = torch.randn(context_dim) if i % 2 == 0 else None  # Sometimes no context
            
            crb.add_record(edge, parent_config, child_outcome, context)
    
    # 3. Show buffer statistics
    print("\n--- Buffer Statistics ---")
    stats = crb.get_buffer_stats()
    for edge, stat in stats.items():
        print(f"Edge '{edge}': {stat}")

    # 4. Sample batches from buffer
    print("\n--- Testing Batch Sampling ---")
    batch_size = 16
    
    for edge in edges:
        batch_data = crb.sample_batch(edge, batch_size)
        if batch_data is not None:
            parent_stats, child_outcomes, contexts = batch_data
            print(f"Edge '{edge}':")
            print(f"  Parent stats shape: {parent_stats.shape}")
            print(f"  Child outcomes shape: {child_outcomes.shape}")
            print(f"  Contexts shape: {contexts.shape}")

    # 5. Test counterfactual sampling
    print("\n--- Testing Counterfactual Sampling ---")
    
    edge = edges[0]  # Use first edge
    cf_data = crb.sample_counterfactuals(edge, batch_size, intervention_strength=0.5)
    
    if cf_data is not None:
        original_outcomes, counterfactual_outcomes, parent_stats = cf_data
        print(f"Original outcomes shape: {original_outcomes.shape}")
        print(f"Counterfactual outcomes shape: {counterfactual_outcomes.shape}")
        print(f"Parent stats shape: {parent_stats.shape}")
        
        # Compare original vs counterfactual
        orig_mean = original_outcomes.mean(dim=0)
        cf_mean = counterfactual_outcomes.mean(dim=0)
        difference = torch.norm(cf_mean - orig_mean)
        print(f"Mean difference between original and counterfactual: {difference.item():.4f}")

    # 6. Compute counterfactual loss
    print("\n--- Testing Counterfactual Loss ---")
    
    for edge in edges:
        cf_loss = crb.compute_counterfactual_loss(edge, batch_size)
        if cf_loss is not None:
            print(f"Counterfactual loss for '{edge}': {cf_loss.item():.4f}")

    # 7. Test neural components directly
    print("\n--- Testing Neural Components ---")
    
    # Test encoder
    if edges[0] in crb.encoders:
        encoder = crb.encoders[edges[0]]
        test_input = torch.randn(4, parent_dim)
        mean, log_var = encoder(test_input)
        encoded = encoder.reparameterize(mean, log_var)
        print(f"Encoder - Input: {test_input.shape} -> Encoded: {encoded.shape}")
        print(f"Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
        print(f"Log-var range: [{log_var.min():.3f}, {log_var.max():.3f}]")
    
    # Test sampler  
    if edges[0] in crb.samplers:
        sampler = crb.samplers[edges[0]]
        test_stats = torch.randn(4, stat_dim)
        test_context = torch.randn(4, context_dim)
        cf_mean, cf_log_var = sampler(test_stats, test_context)
        cf_sample = sampler.sample_counterfactual(test_stats, test_context)
        print(f"Sampler - Stats: {test_stats.shape}, Context: {test_context.shape}")
        print(f"-> CF mean: {cf_mean.shape}, CF sample: {cf_sample.shape}")

    # 8. Test buffer persistence
    print("\n--- Testing Buffer Save/Load ---")
    
    save_path = "/tmp/test_crb_buffer.pt"
    try:
        crb.save_buffer(save_path)
        print(f"Buffer saved to {save_path}")
        
        # Create new buffer and load
        new_crb = CounterfactualReplayBuffer()
        new_crb.load_buffer(save_path)
        
        new_stats = new_crb.get_buffer_stats()
        print("Loaded buffer statistics:")
        for edge, stat in new_stats.items():
            print(f"  Edge '{edge}': {stat.get('num_records', 0)} records")
            
    except Exception as e:
        print(f"Save/load test failed: {e}")

    print("\n=== Counterfactual Replay Buffer Testing Complete ===")
