"""
Main script demonstrating CASCADA v2 algorithm usage
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
import logging

from cascada.cascada_algorithm import CASCADA, CASCADAConfig


def create_synthetic_continual_dataset(
    num_shards: int = 5,
    samples_per_shard: int = 1000,
    input_dim: Tuple[int, int, int] = (3, 32, 32),
    num_classes: int = 10,
    drift_severity: float = 0.5
) -> List[DataLoader]:
    """
    Create synthetic continual learning dataset with concept drift.
    
    Args:
        num_shards: Number of data shards (tasks)
        samples_per_shard: Samples per shard
        input_dim: Input dimensions (C, H, W)
        num_classes: Number of output classes
        drift_severity: Amount of concept drift between shards
        
    Returns:
        List of DataLoaders for each shard
    """
    shards = []
    
    base_mean = torch.zeros(input_dim)
    base_std = torch.ones(input_dim)
    
    for shard_idx in range(num_shards):
        # Introduce concept drift
        drift_factor = shard_idx * drift_severity
        
        # Generate data with drift
        X_shard = []
        y_shard = []
        
        for _ in range(samples_per_shard):
            # Sample class
            y = np.random.randint(0, num_classes)
            
            # Generate input with class-dependent and drift-dependent patterns
            x = torch.randn(input_dim) * base_std + base_mean
            
            # Add class-specific patterns
            if y < num_classes // 2:
                x += 0.5 * torch.ones_like(x)
            else:
                x -= 0.5 * torch.ones_like(x)
            
            # Add drift
            drift_noise = torch.randn_like(x) * drift_factor * 0.1
            x += drift_noise
            
            X_shard.append(x)
            y_shard.append(y)
        
        X_shard = torch.stack(X_shard)
        y_shard = torch.tensor(y_shard, dtype=torch.long)
        
        # Create DataLoader
        dataset = TensorDataset(X_shard, y_shard)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        shards.append(dataloader)
    
    return shards


def create_base_model(input_dim: Tuple[int, int, int], num_classes: int) -> nn.Module:
    """Create a base CNN model."""
    c, h, w = input_dim
    
    model = nn.Sequential(
        # Convolutional layers
        nn.Conv2d(c, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        
        # Flatten and fully connected
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes)
    )
    
    return model


def evaluate_model(model: CASCADA, dataloader: DataLoader) -> Dict[str, float]:
    """Evaluate model performance on a dataset."""
    model.eval()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(model.device), y.to(model.device)
            
            # Forward pass (without graph updates during evaluation)
            predictions = model.forward(x, update_graph=False)
            
            # Compute loss
            loss = criterion(predictions, y)
            total_loss += loss.item()
            
            # Compute accuracy
            predicted_classes = predictions.argmax(dim=1)
            correct_predictions += (predicted_classes == y).sum().item()
            total_samples += y.size(0)
    
    model.train()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct_predictions / total_samples,
        'total_samples': total_samples
    }


def plot_training_progress(
    performance_history: List[Dict[str, float]],
    save_path: str = None
):
    """Plot training progress."""
    if not performance_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    shards = list(range(len(performance_history)))
    
    # Task Loss
    task_losses = [metrics['task_loss'] for metrics in performance_history]
    axes[0, 0].plot(shards, task_losses, 'b-', marker='o')
    axes[0, 0].set_title('Task Loss')
    axes[0, 0].set_xlabel('Shard')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Total Loss
    total_losses = [metrics['total_loss'] for metrics in performance_history]
    axes[0, 1].plot(shards, total_losses, 'r-', marker='s')
    axes[0, 1].set_title('Total Loss (Task + CF + Reg)')
    axes[0, 1].set_xlabel('Shard')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Accuracy
    accuracies = [metrics['accuracy'] for metrics in performance_history]
    axes[1, 0].plot(shards, accuracies, 'g-', marker='^')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].set_xlabel('Shard')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True)
    
    # Counterfactual Loss
    cf_losses = [metrics['counterfactual_loss'] for metrics in performance_history]
    axes[1, 1].plot(shards, cf_losses, 'm-', marker='d')
    axes[1, 1].set_title('Counterfactual Loss')
    axes[1, 1].set_xlabel('Shard')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function demonstrating CASCADA v2 usage."""
    parser = argparse.ArgumentParser(description='CASCADA v2 Demonstration')
    parser.add_argument('--num_shards', type=int, default=5,
                        help='Number of continual learning shards')
    parser.add_argument('--samples_per_shard', type=int, default=500,
                        help='Samples per shard')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create synthetic dataset
    logger.info("Creating synthetic continual dataset...")
    input_dim = (3, 32, 32)
    num_classes = 10
    
    train_shards = create_synthetic_continual_dataset(
        num_shards=args.num_shards,
        samples_per_shard=args.samples_per_shard,
        input_dim=input_dim,
        num_classes=num_classes,
        drift_severity=0.3
    )
    
    # Create validation data (static)
    val_dataloader = create_synthetic_continual_dataset(
        num_shards=1,
        samples_per_shard=200,
        input_dim=input_dim,
        num_classes=num_classes,
        drift_severity=0.0
    )[0]
    
    # Create base model
    logger.info("Creating base model...")
    base_model = create_base_model(input_dim, num_classes)
    
    # Configure CASCADA
    config = CASCADAConfig(
        base_model_dim=256,
        latent_dim=16,
        stable_dim=8,
        context_dim=8,
        adapter_dim=64,
        max_edges=20,
        learning_rate=1e-3,
        beta_cf=0.5,
        lambda_reg=1.0,
        device=device
    )
    
    # Create CASCADA system
    logger.info("Initializing CASCADA system...")
    cascada = CASCADA(base_model, config)
    
    # Training loop
    logger.info("Starting continual training...")
    
    for shard_idx, train_dataloader in enumerate(train_shards):
        logger.info(f"\n=== Training on Shard {shard_idx + 1}/{len(train_shards)} ===")
        
        # Continual update on this shard
        metrics = cascada.continual_update(
            train_dataloader,
            validation_data=val_dataloader
        )
        
        logger.info(f"Shard {shard_idx + 1} Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Evaluate on validation data
        val_metrics = evaluate_model(cascada, val_dataloader)
        logger.info(f"Validation Metrics:")
        for key, value in val_metrics.items():
            logger.info(f"  val_{key}: {value:.4f}")
        
        # Get system diagnostics
        diagnostics = cascada.get_system_diagnostics()
        logger.info(f"System Diagnostics:")
        logger.info(f"  Graph edges: {diagnostics['graph_structure']['edges']}")
        logger.info(f"  Active adapters: {diagnostics['bug_diagnostics']['active_adapters']}")
        logger.info(f"  PAC-Bayesian bound: {diagnostics['pac_bound']:.4f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"cascada_shard_{shard_idx + 1}.pt"
        cascada.save_checkpoint(str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Plot training progress
    logger.info("Plotting training progress...")
    plot_path = output_dir / "training_progress.png"
    plot_training_progress(cascada.performance_history, str(plot_path))
    
    # Final evaluation on all shards
    logger.info("\n=== Final Evaluation ===")
    final_results = {}
    
    for shard_idx, shard_dataloader in enumerate(train_shards):
        metrics = evaluate_model(cascada, shard_dataloader)
        final_results[f'shard_{shard_idx + 1}'] = metrics
        
        logger.info(f"Final Shard {shard_idx + 1} Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
    
    # Save final results
    results_path = output_dir / "final_results.json"
    import json
    with open(results_path, 'w') as f:
        # Convert tensors to floats for JSON serialization
        serializable_results = {
            k: {sk: float(sv) if isinstance(sv, (torch.Tensor, np.ndarray)) else sv 
                for sk, sv in v.items()}
            for k, v in final_results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Final results saved: {results_path}")
    logger.info("CASCADA v2 demonstration completed!")


if __name__ == "__main__":
    main()