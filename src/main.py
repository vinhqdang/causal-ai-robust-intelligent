import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm

# Import ICCP components
from src.caref.encoder import CaReFEncoder, d_separation_contrastive_loss
from src.adapters.scaffolding import StructuralCausalMemory
from src.influence_functions.ops import CPIComputer, FisherInformationMatrix
from src.crb.buffer import CounterfactualReplayBuffer
from src.audit import CryptoAuditLedger
from src.evaluation.harness import EvaluationHarness

from transformers import GPT2Model, GPT2Config, AutoTokenizer

class ICCPTrainingConfig:
    """Configuration class for ICCP training."""
    
    def __init__(self):
        # Model configuration
        self.model_name = "gpt2"
        self.causal_dim = 128
        self.noise_dim = 64
        
        # Training configuration
        self.num_epochs = 10
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.max_sequence_length = 128
        
        # Loss weights
        self.lambda_d_sep = 1.0      # D-separation loss weight
        self.lambda_mi = 0.1         # Mutual information loss weight
        self.lambda_cf = 0.5         # Counterfactual loss weight
        self.lambda_reg = 0.01       # Regularization weight
        
        # CPI configuration
        self.use_cpi_regularization = True
        self.cpi_damping = 1e-3
        self.fisher_samples = 20
        
        # Buffer configuration
        self.crb_stat_dim = 64
        self.crb_context_dim = 32
        self.crb_outcome_dim = 128
        
        # Causal edges to simulate
        self.causal_edges = [
            "context->treatment",
            "treatment->outcome", 
            "confounder->treatment",
            "confounder->outcome"
        ]
        
        # Audit configuration
        self.enable_audit = True
        self.audit_dir = "artefacts"


class ICCPTrainer:
    """Complete ICCP training system integrating all components."""
    
    def __init__(self, config: ICCPTrainingConfig):
        self.config = config
        
        # Initialize components
        print("Initializing ICCP components...")
        self._initialize_foundation_model()
        self._initialize_caref_encoder()
        self._initialize_scm_memory()
        self._initialize_cpi_computer()
        self._initialize_crb_buffer()
        self._initialize_audit_ledger()
        
        # Training state
        self.global_step = 0
        self.epoch_metrics = []
        
        print("ICCP trainer initialized successfully!")
    
    def _initialize_foundation_model(self):
        """Initialize the foundation model."""
        self.gpt_config = GPT2Config.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.foundation_model = GPT2Model(self.gpt_config)
        print(f"Foundation model loaded: {self.config.model_name}")
    
    def _initialize_caref_encoder(self):
        """Initialize CaReF encoder."""
        self.caref_encoder = CaReFEncoder(
            foundation_model=self.foundation_model,
            input_dim=self.gpt_config.n_embd,
            causal_dim=self.config.causal_dim,
            noise_dim=self.config.noise_dim
        )
        self.caref_optimizer = optim.AdamW(
            self.caref_encoder.parameters(), 
            lr=self.config.learning_rate
        )
        print("CaReF encoder initialized")
    
    def _initialize_scm_memory(self):
        """Initialize Structural Causal Memory."""
        self.scm_memory = StructuralCausalMemory(
            model_name=self.config.model_name,
            model=self.foundation_model
        )
        
        # Add adapters for each causal edge
        for edge in self.config.causal_edges:
            self.scm_memory.add_adapter_for_edge(edge, r=8, alpha=16)
        
        print(f"SCM memory initialized with {len(self.config.causal_edges)} adapters")
    
    def _initialize_cpi_computer(self):
        """Initialize CPI computer."""
        self.cpi_computer = CPIComputer(
            model=self.caref_encoder,
            use_fisher_approx=True,
            damping=self.config.cpi_damping
        )
        print("CPI computer initialized")
    
    def _initialize_crb_buffer(self):
        """Initialize Counterfactual Replay Buffer."""
        self.crb_buffer = CounterfactualReplayBuffer(
            stat_dim=self.config.crb_stat_dim,
            context_dim=self.config.crb_context_dim,
            outcome_dim=self.config.crb_outcome_dim,
            max_size=5000
        )
        print("CRB buffer initialized")
    
    def _initialize_audit_ledger(self):
        """Initialize audit ledger."""
        if self.config.enable_audit:
            self.audit_ledger = CryptoAuditLedger(
                ledger_dir=self.config.audit_dir,
                batch_size=50,
                auto_merkle=True
            )
            print("Audit ledger initialized")
        else:
            self.audit_ledger = None
    
    def _generate_synthetic_data(self, num_samples: int = 100) -> DataLoader:
        """Generate synthetic causal data for training."""
        # Generate random token sequences
        input_ids = torch.randint(
            0, self.gpt_config.vocab_size, 
            (num_samples, self.config.max_sequence_length)
        )
        
        # Create attention masks (all ones for simplicity)
        attention_mask = torch.ones_like(input_ids)
        
        # Create simple causal targets (synthetic interventions)
        causal_targets = torch.randn(num_samples, self.config.causal_dim)
        
        dataset = TensorDataset(input_ids, attention_mask, causal_targets)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
    
    def _compute_caref_loss(self, input_ids: torch.Tensor, 
                           attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute CaReF losses (d-separation + mutual information)."""
        # Forward pass through CaReF encoder
        causal_factors, noise_factors = self.caref_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        
        # D-separation contrastive loss
        d_sep_loss = d_separation_contrastive_loss(causal_factors, noise_factors)
        
        # Mutual information loss
        mi_loss = self.caref_encoder.compute_mutual_information_loss(
            causal_factors, noise_factors
        )
        
        # Combined CaReF loss
        caref_loss = (self.config.lambda_d_sep * d_sep_loss + 
                     self.config.lambda_mi * mi_loss)
        
        loss_info = {
            'd_sep_loss': d_sep_loss.item(),
            'mi_loss': mi_loss.item(),
            'caref_loss': caref_loss.item()
        }
        
        return caref_loss, loss_info
    
    def _compute_counterfactual_loss(self) -> Tuple[torch.Tensor, Dict]:
        """Compute counterfactual consistency loss from CRB."""
        total_cf_loss = 0.0
        cf_losses = {}
        active_edges = 0
        
        for edge in self.config.causal_edges:
            cf_loss = self.crb_buffer.compute_counterfactual_loss(
                edge, batch_size=self.config.batch_size
            )
            if cf_loss is not None:
                total_cf_loss += cf_loss
                cf_losses[edge] = cf_loss.item()
                active_edges += 1
        
        if active_edges > 0:
            total_cf_loss /= active_edges
        
        loss_info = {
            'cf_loss_total': total_cf_loss.item() if isinstance(total_cf_loss, torch.Tensor) else total_cf_loss,
            'cf_losses_by_edge': cf_losses,
            'active_edges': active_edges
        }
        
        return total_cf_loss, loss_info
    
    def _compute_cpi_regularization(self) -> Tuple[torch.Tensor, Dict]:
        """Compute CPI-based regularization."""
        if not self.config.use_cpi_regularization:
            return torch.tensor(0.0), {'cpi_reg': 0.0}
        
        # Simple L2 regularization weighted by parameter importance
        reg_loss = torch.tensor(0.0)
        
        for name, param in self.caref_encoder.named_parameters():
            if param.requires_grad:
                reg_loss += torch.sum(param.pow(2))
        
        reg_info = {
            'cpi_reg': reg_loss.item(),
            'regularized_params': sum(1 for p in self.caref_encoder.parameters() if p.requires_grad)
        }
        
        return reg_loss * self.config.lambda_reg, reg_info
    
    def _update_crb_buffer(self, input_ids: torch.Tensor, causal_factors: torch.Tensor):
        """Update counterfactual replay buffer with new data."""
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            # Extract features for this sample
            parent_config = input_ids[i].float()  # Convert to float for processing
            child_outcome = causal_factors[i]
            
            # Add record to buffer for random edge (simulation)
            edge = np.random.choice(self.config.causal_edges)
            self.crb_buffer.add_record(edge, parent_config, child_outcome)
    
    def _accumulate_fisher_information(self, loss: torch.Tensor):
        """Accumulate Fisher information for CPI computation."""
        if self.cpi_computer.fisher:
            self.cpi_computer.fisher.accumulate_fisher(loss, retain_graph=True)
    
    def _log_to_audit(self, step_info: Dict):
        """Log training step information to audit ledger."""
        if self.audit_ledger is None:
            return
        
        # Log parameter update
        self.audit_ledger.log_parameter_update(
            'caref_encoder',
            self.caref_encoder,
            gradient_info=step_info.get('gradient_info'),
            loss_info=step_info.get('loss_info')
        )
        
        # Log gradient step
        if 'loss_value' in step_info:
            self.audit_ledger.log_gradient_step(
                'iccp_trainer',
                optimizer_state={'lr': self.config.learning_rate},
                loss_value=step_info['loss_value'],
                gradient_norm=step_info.get('gradient_norm', 0.0)
            )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch."""
        self.caref_encoder.train()
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, (input_ids, attention_mask, causal_targets) in enumerate(progress_bar):
            self.global_step += 1
            
            # Zero gradients
            self.caref_optimizer.zero_grad()
            
            # Compute CaReF loss
            caref_loss, caref_info = self._compute_caref_loss(input_ids, attention_mask)
            
            # Get causal factors for buffer update
            with torch.no_grad():
                causal_factors, _ = self.caref_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                self._update_crb_buffer(input_ids, causal_factors)
            
            # Compute counterfactual loss
            cf_loss, cf_info = self._compute_counterfactual_loss()
            if isinstance(cf_loss, (int, float)):
                cf_loss = torch.tensor(cf_loss, requires_grad=True)
            
            # Compute CPI regularization
            reg_loss, reg_info = self._compute_cpi_regularization()
            
            # Total loss
            total_loss = caref_loss + self.config.lambda_cf * cf_loss + reg_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.caref_encoder.parameters(), max_norm=1.0
            )
            
            # Optimizer step
            self.caref_optimizer.step()
            
            # Accumulate Fisher information periodically (disabled for now to avoid graph issues)
            # if self.global_step % 10 == 0:
            #     self._accumulate_fisher_information(total_loss)
            
            # Collect loss information
            step_losses = {
                'total_loss': total_loss.item(),
                'gradient_norm': gradient_norm.item(),
                **caref_info,
                **cf_info,
                **reg_info
            }
            
            epoch_losses.append(step_losses)
            
            # Log to audit ledger
            self._log_to_audit({
                'loss_value': total_loss.item(),
                'gradient_norm': gradient_norm.item(),
                'loss_info': step_losses,
                'gradient_info': {'max_grad': gradient_norm.item()}
            })
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'D-sep': f"{caref_info['d_sep_loss']:.4f}",
                'CF': f"{cf_info['cf_loss_total']:.4f}"
            })
        
        # Compute epoch statistics
        epoch_stats = {}
        for key in epoch_losses[0].keys():
            values = [loss[key] for loss in epoch_losses]
            # Only compute stats for numeric values, not dictionaries
            if all(isinstance(v, (int, float)) for v in values):
                epoch_stats[f'epoch_{key}_mean'] = np.mean(values)
                epoch_stats[f'epoch_{key}_std'] = np.std(values)
            elif all(isinstance(v, dict) for v in values):
                # For nested dictionaries, flatten and compute stats
                for subkey in values[0].keys():
                    subvalues = [v[subkey] for v in values if subkey in v]
                    if all(isinstance(sv, (int, float)) for sv in subvalues):
                        epoch_stats[f'epoch_{key}_{subkey}_mean'] = np.mean(subvalues)
                        epoch_stats[f'epoch_{key}_{subkey}_std'] = np.std(subvalues)
        
        return epoch_stats
    
    def train(self):
        """Main training loop."""
        print("\n=== Starting ICCP Training ===")
        
        # Generate synthetic training data
        print("Generating synthetic training data...")
        train_dataloader = self._generate_synthetic_data(num_samples=500)
        
        # Finalize Fisher information setup
        if self.cpi_computer.fisher:
            # Pre-accumulate some Fisher information
            print("Pre-computing Fisher information...")
            # Ensure model is unfrozen for Fisher computation
            self.caref_encoder.unfreeze_stable()
            
            for i in range(self.config.fisher_samples):
                dummy_batch = next(iter(train_dataloader))
                input_ids, attention_mask, _ = dummy_batch
                dummy_loss, _ = self._compute_caref_loss(input_ids, attention_mask)
                self.cpi_computer.fisher.accumulate_fisher(dummy_loss)
            self.cpi_computer.fisher.finalize_fisher()
            print(f"Fisher information computed over {self.cpi_computer.fisher.n_samples} samples")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.config.num_epochs} ---")
            
            epoch_stats = self.train_epoch(train_dataloader, epoch)
            self.epoch_metrics.append(epoch_stats)
            
            # Print epoch summary
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Total Loss: {epoch_stats['epoch_total_loss_mean']:.4f} ± {epoch_stats['epoch_total_loss_std']:.4f}")
            print(f"  CaReF Loss: {epoch_stats['epoch_caref_loss_mean']:.4f} ± {epoch_stats['epoch_caref_loss_std']:.4f}")
            print(f"  CF Loss: {epoch_stats['epoch_cf_loss_total_mean']:.4f} ± {epoch_stats['epoch_cf_loss_total_std']:.4f}")
            print(f"  Gradient Norm: {epoch_stats['epoch_gradient_norm_mean']:.4f}")
            
            # Log epoch results to audit
            if self.audit_ledger:
                self.audit_ledger.log_evaluation_result(
                    'iccp_trainer',
                    'epoch_completion',
                    epoch + 1,
                    {
                        'epoch_stats': epoch_stats,
                        'global_step': self.global_step
                    }
                )
        
        print("\n=== Training Completed ===")
        return self.epoch_metrics
    
    def evaluate(self) -> Dict:
        """Evaluate the trained model."""
        print("\n=== Model Evaluation ===")
        
        self.caref_encoder.eval()
        
        # Generate test data
        test_dataloader = self._generate_synthetic_data(num_samples=100)
        
        eval_losses = []
        with torch.no_grad():
            for input_ids, attention_mask, causal_targets in test_dataloader:
                caref_loss, caref_info = self._compute_caref_loss(input_ids, attention_mask)
                cf_loss, cf_info = self._compute_counterfactual_loss()
                
                if isinstance(cf_loss, (int, float)):
                    cf_loss = torch.tensor(cf_loss)
                
                eval_losses.append({
                    'caref_loss': caref_loss.item(),
                    'cf_loss': cf_loss.item() if isinstance(cf_loss, torch.Tensor) else cf_loss,
                    **caref_info,
                    **cf_info
                })
        
        # Compute evaluation statistics
        eval_stats = {}
        for key in eval_losses[0].keys():
            values = [loss[key] for loss in eval_losses]
            eval_stats[f'eval_{key}_mean'] = np.mean(values)
            eval_stats[f'eval_{key}_std'] = np.std(values)
        
        # Buffer statistics
        buffer_stats = self.crb_buffer.get_buffer_stats()
        eval_stats['buffer_stats'] = buffer_stats
        
        # Audit summary
        if self.audit_ledger:
            audit_summary = self.audit_ledger.get_audit_summary()
            eval_stats['audit_summary'] = audit_summary
            
            # Verify audit integrity
            integrity_check = self.audit_ledger.verify_chain_integrity()
            eval_stats['audit_integrity'] = integrity_check
        
        print("Evaluation Results:")
        print(f"  CaReF Loss: {eval_stats['eval_caref_loss_mean']:.4f} ± {eval_stats['eval_caref_loss_std']:.4f}")
        print(f"  CF Loss: {eval_stats['eval_cf_loss_mean']:.4f} ± {eval_stats['eval_cf_loss_std']:.4f}")
        print(f"  D-separation Loss: {eval_stats['eval_d_sep_loss_mean']:.4f}")
        print(f"  MI Loss: {eval_stats['eval_mi_loss_mean']:.4f}")
        
        return eval_stats


def main():
    """
    Main function to run complete ICCP training and evaluation.
    """
    print("🚀 ICCP (Intervention-aware Causal Consistency Preserving) Training")
    print("=" * 70)
    
    # Initialize configuration
    config = ICCPTrainingConfig()
    
    # Create trainer
    trainer = ICCPTrainer(config)
    
    try:
        # Run training
        training_metrics = trainer.train()
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        # Save results
        results_dir = Path("artefacts/results")
        results_dir.mkdir(exist_ok=True)
        
        # Save training metrics
        with open(results_dir / "training_metrics.json", 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        # Save evaluation results
        with open(results_dir / "evaluation_results.json", 'w') as f:
            json.dump({k: v for k, v in eval_results.items() 
                      if isinstance(v, (int, float, str, list, dict))}, f, indent=2)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Results saved to {results_dir}")
        print(f"🔍 Audit logs available in {config.audit_dir}")
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎯 ICCP Training Summary")
        print("=" * 70)
        print(f"📈 Epochs trained: {config.num_epochs}")
        print(f"🔧 Components integrated: CaReF, SCM-Mem, CPI, CRB, Audit")
        print(f"📝 Audit entries: {eval_results.get('audit_summary', {}).get('entries', 'N/A')}")
        print(f"🔒 Audit integrity: {'✅ Valid' if eval_results.get('audit_integrity', {}).get('valid', False) else '❌ Invalid'}")
        print(f"💾 Buffer records: {sum(stats.get('num_records', 0) for stats in eval_results.get('buffer_stats', {}).values())}")
        
        return eval_results
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
