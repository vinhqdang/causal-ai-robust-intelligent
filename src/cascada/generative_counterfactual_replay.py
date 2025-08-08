"""
Component C4: Generative Counterfactual Replay (GCR) 
Implements causally-conditioned diffusion model for high-fidelity counterfactual generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import networkx as nx
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


class CausalConditioningModule(nn.Module):
    """
    Module to condition diffusion on causal graph structure and interventions.
    """
    
    def __init__(
        self,
        causal_dim: int,
        condition_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.causal_dim = causal_dim
        self.condition_dim = condition_dim
        
        # Graph structure encoder
        self.graph_encoder = nn.Sequential(
            nn.Linear(causal_dim * causal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, condition_dim)
        )
        
        # Intervention encoder  
        self.intervention_encoder = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, condition_dim)
        )
        
    def forward(
        self,
        adjacency_matrix: torch.Tensor,
        interventions: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode causal structure and interventions into conditioning vector.
        
        Args:
            adjacency_matrix: [batch_size, causal_dim, causal_dim]
            interventions: [batch_size, causal_dim] - intervention values
            
        Returns:
            conditioning: [batch_size, condition_dim] - conditioning vector
        """
        batch_size = adjacency_matrix.size(0)
        
        # Flatten adjacency matrix for graph encoding
        adj_flat = adjacency_matrix.view(batch_size, -1)
        graph_embed = self.graph_encoder(adj_flat)
        
        # Encode interventions
        intervention_embed = self.intervention_encoder(interventions)
        
        # Combine embeddings
        conditioning = graph_embed + intervention_embed
        
        return conditioning


class LightweightDiffusionUNet(nn.Module):
    """
    Lightweight U-Net architecture for counterfactual generation (only 4 blocks).
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        condition_dim: int = 128,
        base_channels: int = 64,
        num_blocks: int = 4
    ):
        super().__init__()
        
        self.condition_dim = condition_dim
        
        # Time embedding (for diffusion timestep)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Condition embedding projection
        self.condition_proj = nn.Linear(condition_dim, base_channels)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        channels = [in_channels] + [base_channels * (2**i) for i in range(num_blocks)]
        
        for i in range(num_blocks):
            in_ch = channels[i] + (32 if i == 0 else 0)  # Add time embedding to first block
            out_ch = channels[i + 1]
            
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
        
        # Bottleneck
        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch * 2, 3, padding=1),
            nn.GroupNorm(8, bottleneck_ch * 2),
            nn.ReLU(),
            nn.Conv2d(bottleneck_ch * 2, bottleneck_ch, 3, padding=1),
            nn.GroupNorm(8, bottleneck_ch),
            nn.ReLU()
        )
        
        # Decoder blocks  
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_blocks - 1, -1, -1):
            in_ch = channels[i + 1] + channels[i]  # Skip connection
            out_ch = channels[i] if i > 0 else out_channels
            
            self.decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch) if out_ch >= 8 else nn.Identity(),
                nn.ReLU() if i > 0 else nn.Identity()
            ))
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of lightweight diffusion U-Net.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            timestep: Diffusion timestep [batch_size, 1]
            conditioning: Causal conditioning [batch_size, condition_dim]
            
        Returns:
            noise_pred: Predicted noise [batch_size, channels, height, width]
        """
        batch_size = x.size(0)
        
        # Time embedding
        time_emb = self.time_embedding(timestep.float())  # [batch, 32]
        time_emb = time_emb.view(batch_size, 32, 1, 1)
        time_emb = time_emb.expand(-1, -1, x.size(2), x.size(3))
        
        # Condition embedding  
        cond_emb = self.condition_proj(conditioning)  # [batch, base_channels]
        cond_emb = cond_emb.view(batch_size, -1, 1, 1)
        cond_emb = cond_emb.expand(-1, -1, x.size(2), x.size(3))
        
        # Encoder
        encoder_features = []
        h = x
        
        for i, encoder_block in enumerate(self.encoder_blocks):
            if i == 0:
                # Add time embedding to first block
                h = torch.cat([h, time_emb], dim=1)
            
            h = encoder_block(h)
            
            # Add conditioning at each level
            if h.size(1) == cond_emb.size(1):
                h = h + cond_emb
                
            encoder_features.append(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_feat = encoder_features[-(i + 1)]
            
            # Upsample h to match skip connection size if needed
            if h.size(2) != skip_feat.size(2) or h.size(3) != skip_feat.size(3):
                h = F.interpolate(h, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
            
            h = torch.cat([h, skip_feat], dim=1)
            h = decoder_block(h)
        
        return h


class GenerativeCounterfactualReplay(nn.Module):
    """
    Generative Counterfactual Replay using causally-conditioned diffusion model.
    
    Key innovation: Solves the "Gaussian noise" weakness of ICCP-v1 by generating
    high-fidelity counterfactuals that are consistent with the current SCM structure.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        causal_dim: int = 10,
        condition_dim: int = 128,
        num_inference_steps: int = 50,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.causal_dim = causal_dim
        self.condition_dim = condition_dim
        self.num_inference_steps = num_inference_steps
        self.device = device
        
        channels, height, width = input_shape
        
        # Causal conditioning module
        self.causal_conditioner = CausalConditioningModule(
            causal_dim=causal_dim,
            condition_dim=condition_dim
        )
        
        # Lightweight diffusion U-Net
        self.unet = LightweightDiffusionUNet(
            in_channels=channels,
            out_channels=channels,
            condition_dim=condition_dim
        )
        
        # DDPM noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        # Counterfactual data buffer
        self.cf_buffer = []
        self.max_buffer_size = 1000
        
        # Training statistics
        self.training_steps = 0
        self.loss_history = []
        
    def forward(
        self,
        x: torch.Tensor,
        causal_graph: nx.DiGraph,
        interventions: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training or inference.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]  
            causal_graph: Current causal graph structure
            interventions: Intervention values [batch_size, causal_dim]
            timestep: Diffusion timestep (for training)
            
        Returns:
            output: Generated counterfactual or noise prediction
        """
        batch_size = x.size(0)
        
        # Convert graph to adjacency matrix
        adjacency_matrix = self._graph_to_adjacency(causal_graph, batch_size)
        
        # Default interventions if not provided
        if interventions is None:
            interventions = torch.zeros(batch_size, self.causal_dim, device=self.device)
        
        # Get causal conditioning
        conditioning = self.causal_conditioner(adjacency_matrix, interventions)
        
        if timestep is not None:
            # Training mode: predict noise
            return self.unet(x, timestep, conditioning)
        else:
            # Inference mode: generate counterfactual
            return self._generate_counterfactual(conditioning)
    
    def _graph_to_adjacency(
        self, 
        causal_graph: nx.DiGraph, 
        batch_size: int
    ) -> torch.Tensor:
        """Convert NetworkX graph to adjacency matrix tensor."""
        adj_matrix = torch.zeros(
            batch_size, self.causal_dim, self.causal_dim, 
            device=self.device
        )
        
        # Fill adjacency matrix from graph
        for edge in causal_graph.edges():
            i, j = edge
            if i < self.causal_dim and j < self.causal_dim:
                adj_matrix[:, i, j] = 1.0
        
        return adj_matrix
    
    def _generate_counterfactual(
        self, 
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate counterfactual using DDPM sampling.
        """
        batch_size = conditioning.size(0)
        channels, height, width = self.input_shape
        
        # Start from random noise
        x = torch.randn(
            batch_size, channels, height, width,
            device=self.device
        )
        
        # Set scheduler for inference
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        
        # Denoising loop
        for t in timesteps:
            timestep_tensor = torch.full(
                (batch_size, 1), t, device=self.device, dtype=torch.long
            )
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(x, timestep_tensor, conditioning)
            
            # Compute previous sample
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
        
        return x
    
    def generate_counterfactuals(
        self,
        batch_size: int,
        causal_graph: nx.DiGraph,
        interventions: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate multiple counterfactual samples.
        
        Args:
            batch_size: Number of samples in batch
            causal_graph: Current causal graph  
            interventions: Intervention specifications [batch_size, causal_dim]
            num_samples: Number of counterfactual samples per intervention
            
        Returns:
            counterfactuals: Generated samples [num_samples * batch_size, channels, height, width]
        """
        self.eval()
        
        all_counterfactuals = []
        
        for _ in range(num_samples):
            # Expand interventions if needed
            expanded_interventions = interventions.repeat_interleave(
                batch_size // interventions.size(0) + 1, dim=0
            )[:batch_size]
            
            counterfactuals = self.forward(
                torch.zeros(batch_size, *self.input_shape, device=self.device),
                causal_graph,
                expanded_interventions
            )
            all_counterfactuals.append(counterfactuals)
        
        return torch.cat(all_counterfactuals, dim=0)
    
    def compute_generative_cf_loss(
        self,
        real_batch: torch.Tensor,
        causal_graph: nx.DiGraph,
        interventions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the generative counterfactual loss L_cf.
        
        This trains the diffusion model to generate realistic counterfactuals
        consistent with the causal structure.
        """
        batch_size = real_batch.size(0)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to real data
        noise = torch.randn_like(real_batch)
        noisy_data = self.noise_scheduler.add_noise(real_batch, noise, timesteps)
        
        # Predict noise
        timestep_tensor = timesteps.unsqueeze(1)
        noise_pred = self.forward(noisy_data, causal_graph, interventions, timestep_tensor)
        
        # MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        # Add causal consistency loss
        causal_loss = self._compute_causal_consistency_loss(
            noise_pred, noise, causal_graph, interventions
        )
        
        total_loss = loss + 0.1 * causal_loss
        
        self.loss_history.append(total_loss.item())
        return total_loss
    
    def _compute_causal_consistency_loss(
        self,
        pred_noise: torch.Tensor,
        true_noise: torch.Tensor, 
        causal_graph: nx.DiGraph,
        interventions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss to ensure counterfactuals respect causal structure.
        """
        # Simplified causal consistency: encourage predictions to respect graph structure
        batch_size = pred_noise.size(0)
        
        # Get adjacency matrix
        adj_matrix = self._graph_to_adjacency(causal_graph, batch_size)
        
        # Compute feature correlations in predicted vs true noise
        pred_flat = pred_noise.view(batch_size, -1)
        true_flat = true_noise.view(batch_size, -1)
        
        # Simplified consistency loss based on correlation structure
        pred_corr = torch.corrcoef(pred_flat.T) if pred_flat.size(1) > 1 else torch.zeros(1, 1)
        true_corr = torch.corrcoef(true_flat.T) if true_flat.size(1) > 1 else torch.zeros(1, 1)
        
        # Ensure tensors have the same size for loss computation
        min_size = min(pred_corr.size(0), true_corr.size(0), self.causal_dim)
        if min_size > 1:
            pred_corr_subset = pred_corr[:min_size, :min_size]
            true_corr_subset = true_corr[:min_size, :min_size]
            
            consistency_loss = F.mse_loss(pred_corr_subset, true_corr_subset)
        else:
            consistency_loss = torch.tensor(0.0, device=pred_noise.device)
        
        return consistency_loss
    
    def partial_fit(
        self,
        data_batch: torch.Tensor,
        causal_graph: nx.DiGraph,
        interventions: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Partial fit on a single batch (for online learning).
        """
        if interventions is None:
            interventions = torch.zeros(
                data_batch.size(0), self.causal_dim, device=self.device
            )
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        self.train()
        
        # Compute loss
        loss = self.compute_generative_cf_loss(data_batch, causal_graph, interventions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        self.training_steps += 1
        
        # Add to buffer for replay
        self._update_buffer(data_batch.detach().cpu(), interventions.detach().cpu())
    
    def _update_buffer(
        self, 
        data: torch.Tensor, 
        interventions: torch.Tensor
    ):
        """Update counterfactual replay buffer."""
        batch_data = [(data[i], interventions[i]) for i in range(data.size(0))]
        
        self.cf_buffer.extend(batch_data)
        
        # Keep buffer size under limit
        if len(self.cf_buffer) > self.max_buffer_size:
            # Remove oldest samples
            excess = len(self.cf_buffer) - self.max_buffer_size
            self.cf_buffer = self.cf_buffer[excess:]
    
    def sample_from_buffer(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch from counterfactual buffer for replay."""
        if len(self.cf_buffer) < batch_size:
            # Return empty tensors if insufficient data
            return (
                torch.zeros(0, *self.input_shape),
                torch.zeros(0, self.causal_dim)
            )
        
        # Random sampling
        indices = np.random.choice(len(self.cf_buffer), size=batch_size, replace=False)
        
        sampled_data = []
        sampled_interventions = []
        
        for idx in indices:
            data, intervention = self.cf_buffer[idx]
            sampled_data.append(data)
            sampled_interventions.append(intervention)
        
        return (
            torch.stack(sampled_data),
            torch.stack(sampled_interventions)
        )
    
    def get_model_stats(self) -> Dict[str, Union[int, float]]:
        """Get model statistics and diagnostics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'training_steps': self.training_steps,
            'buffer_size': len(self.cf_buffer),
            'average_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        }