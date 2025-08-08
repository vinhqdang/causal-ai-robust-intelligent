import torch
import torch.nn as nn
import sys
import os
import pytest
import numpy as np
from unittest.mock import patch
import tempfile
import json

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import ICCP components
from src.caref.encoder import CaReFEncoder, BifurcatedMLPHead, MINEEstimator, d_separation_contrastive_loss
from src.adapters.scaffolding import StructuralCausalMemory, CosineGatingNetwork, AdapterMetadata
from src.influence_functions.ops import CPIComputer, FisherInformationMatrix, ParameterHasher
from src.crb.buffer import CounterfactualReplayBuffer, ParentStatEncoder, CounterfactualSampler
from src.audit import CryptoAuditLedger, MerkleTree, ParameterHasher as AuditParameterHasher
from src.evaluation.metrics import (
    causal_consistency_score, fate_score, causal_disentanglement_score,
    comprehensive_causal_evaluation
)
from src.main import ICCPTrainer, ICCPTrainingConfig

from transformers import GPT2Model, GPT2Config


class TestCaReFEncoder:
    """Test suite for CaReF encoder components."""
    
    @pytest.fixture
    def gpt_model(self):
        """Create a small GPT model for testing."""
        config = GPT2Config(
            vocab_size=1000, n_embd=128, n_layer=2, n_head=4,
            max_length=64, max_position_embeddings=64
        )
        return GPT2Model(config)
    
    @pytest.fixture
    def caref_encoder(self, gpt_model):
        """Create CaReF encoder for testing."""
        return CaReFEncoder(
            foundation_model=gpt_model,
            input_dim=128,
            causal_dim=32,
            noise_dim=24
        )
    
    def test_bifurcated_mlp_head(self):
        """Test BifurcatedMLPHead functionality."""
        head = BifurcatedMLPHead(input_dim=128, causal_dim=32, noise_dim=24)
        x = torch.randn(4, 10, 128)  # batch_size, seq_len, input_dim
        
        causal_factors, noise_factors = head(x)
        
        assert causal_factors.shape == (4, 10, 32)
        assert noise_factors.shape == (4, 10, 24)
        assert not torch.isnan(causal_factors).any()
        assert not torch.isnan(noise_factors).any()
    
    def test_mine_estimator(self):
        """Test MINE estimator functionality."""
        mine = MINEEstimator(causal_dim=32, noise_dim=32)
        
        causal_factors = torch.randn(16, 32)
        noise_factors = torch.randn(16, 32)
        
        mi_estimate = mine(causal_factors, noise_factors)
        
        assert isinstance(mi_estimate, torch.Tensor)
        assert mi_estimate.shape == ()
        assert not torch.isnan(mi_estimate)
    
    def test_caref_encoder_forward(self, caref_encoder):
        """Test CaReF encoder forward pass."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        
        causal_factors, noise_factors = caref_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        
        assert causal_factors.shape == (2, 10, 32)
        assert noise_factors.shape == (2, 10, 32)  # After projection
    
    def test_freeze_functionality(self, caref_encoder):
        """Test freezing and unfreezing functionality."""
        # Initially unfrozen
        assert caref_encoder.foundation_model.wte.weight.requires_grad
        
        # Freeze
        caref_encoder.freeze_stable()
        assert not caref_encoder.foundation_model.wte.weight.requires_grad
        
        # Unfreeze
        caref_encoder.unfreeze_stable()
        assert caref_encoder.foundation_model.wte.weight.requires_grad
    
    def test_mutual_information_loss(self, caref_encoder):
        """Test MI loss computation."""
        input_ids = torch.randint(0, 1000, (4, 8))
        causal_factors, noise_factors = caref_encoder(input_ids)
        
        mi_loss = caref_encoder.compute_mutual_information_loss(causal_factors, noise_factors)
        
        assert isinstance(mi_loss, torch.Tensor)
        assert not torch.isnan(mi_loss)
    
    def test_d_separation_loss(self):
        """Test d-separation contrastive loss."""
        causal_factors = torch.randn(8, 16, 32)
        noise_factors = torch.randn(8, 16, 32)
        
        loss = d_separation_contrastive_loss(causal_factors, noise_factors)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestSCMMemory:
    """Test suite for Structural Causal Memory."""
    
    def test_cosine_gating_network(self):
        """Test cosine gating network."""
        gating = CosineGatingNetwork(key_dim=64)
        
        # Register some keys
        gating.register_adapter_key("edge1")
        gating.register_adapter_key("edge2")
        
        query = torch.randn(64)
        weights = gating(query, temperature=1.0)
        
        assert len(weights) == 2
        assert "edge1" in weights
        assert "edge2" in weights
        assert all(0 <= w <= 1 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-5  # Should sum to 1
    
    def test_adapter_metadata(self):
        """Test adapter metadata tracking."""
        metadata = AdapterMetadata()
        
        # Add some entries
        metadata.add_entry("edge1", "create", {"param": "value"})
        metadata.add_entry("edge2", "update", {"param": "value2"})
        
        history = metadata.get_history()
        
        assert len(history) == 2
        assert history[0]['edge_name'] == "edge1"
        assert history[1]['edge_name'] == "edge2"
        assert all('hash' in entry for entry in history)
    
    @patch('src.adapters.scaffolding.AutoModelForCausalLM')
    @patch('src.adapters.scaffolding.AutoTokenizer')
    def test_scm_memory_initialization(self, mock_tokenizer, mock_model):
        """Test SCM memory initialization."""
        mock_model_instance = mock_model.from_pretrained.return_value
        mock_tokenizer_instance = mock_tokenizer.from_pretrained.return_value
        mock_tokenizer_instance.pad_token = "[PAD]"
        
        scm = StructuralCausalMemory("gpt2")
        
        assert scm.base_model is not None
        assert len(scm.adapters) == 0
        assert len(scm.adapter_configs) == 0


class TestCPIComputer:
    """Test suite for Causal Parameter Importance."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def test_parameter_hasher(self):
        """Test parameter hashing functionality."""
        tensor = torch.randn(5, 5)
        hash1 = ParameterHasher.hash_tensor(tensor)
        hash2 = ParameterHasher.hash_tensor(tensor)
        
        assert hash1 == hash2  # Same tensor should produce same hash
        assert len(hash1) == 64  # SHA256 hex string length
    
    def test_fisher_information_matrix(self, simple_model):
        """Test Fisher information matrix computation."""
        fisher = FisherInformationMatrix(simple_model)
        
        # Simulate some losses
        for _ in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x).sum()
            fisher.accumulate_fisher(y)
        
        fisher.finalize_fisher()
        
        assert fisher.n_samples == 5
        assert len(fisher.fisher_dict) > 0
    
    def test_cpi_computer_initialization(self, simple_model):
        """Test CPI computer initialization."""
        cpi = CPIComputer(simple_model, use_fisher_approx=True)
        
        assert cpi.model == simple_model
        assert cpi.use_fisher_approx
        assert cpi.fisher is not None
    
    def test_parameter_importance_computation(self, simple_model):
        """Test parameter importance computation (simplified)."""
        cpi = CPIComputer(simple_model, use_fisher_approx=True)
        
        # Create dummy loss functions
        def train_loss():
            x = torch.randn(4, 10)
            return simple_model(x).sum()
        
        def test_loss():
            x = torch.randn(4, 10)
            return simple_model(x).sum()
        
        # Accumulate some Fisher info
        for _ in range(3):
            loss = train_loss()
            cpi.fisher.accumulate_fisher(loss)
        cpi.fisher.finalize_fisher()
        
        importance_scores = cpi.compute_parameter_importance(train_loss, test_loss)
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) > 0
        assert all(isinstance(v, float) for v in importance_scores.values())


class TestCRBBuffer:
    """Test suite for Counterfactual Replay Buffer."""
    
    def test_parent_stat_encoder(self):
        """Test parent statistics encoder."""
        encoder = ParentStatEncoder(input_dim=64, stat_dim=32)
        
        x = torch.randn(8, 64)
        mean, log_var = encoder(x)
        encoded = encoder.reparameterize(mean, log_var)
        
        assert mean.shape == (8, 32)
        assert log_var.shape == (8, 32)
        assert encoded.shape == (8, 32)
    
    def test_counterfactual_sampler(self):
        """Test counterfactual sampler."""
        sampler = CounterfactualSampler(stat_dim=32, context_dim=16, outcome_dim=64)
        
        parent_stats = torch.randn(4, 32)
        context = torch.randn(4, 16)
        
        cf_mean, cf_log_var = sampler(parent_stats, context)
        cf_sample = sampler.sample_counterfactual(parent_stats, context)
        
        assert cf_mean.shape == (4, 64)
        assert cf_log_var.shape == (4, 64)
        assert cf_sample.shape == (4, 64)
    
    def test_crb_buffer_operations(self):
        """Test CRB buffer record operations."""
        crb = CounterfactualReplayBuffer(stat_dim=32, context_dim=16, outcome_dim=64)
        
        # Add some records
        for i in range(10):
            parent_config = torch.randn(128)
            child_outcome = torch.randn(64)
            context = torch.randn(16)
            
            crb.add_record(f"edge_{i%3}", parent_config, child_outcome, context)
        
        # Test sampling
        batch_data = crb.sample_batch("edge_0", batch_size=4)
        assert batch_data is not None
        
        parent_stats, child_outcomes, contexts = batch_data
        assert parent_stats.shape == (4, 32)  # stat_dim
        assert child_outcomes.shape == (4, 64)  # outcome_dim
        assert contexts.shape == (4, 16)  # context_dim
    
    def test_counterfactual_sampling(self):
        """Test counterfactual sampling from buffer."""
        crb = CounterfactualReplayBuffer(stat_dim=16, context_dim=8, outcome_dim=32)
        
        # Add records
        for i in range(20):
            crb.add_record("test_edge", torch.randn(64), torch.randn(32), torch.randn(8))
        
        cf_data = crb.sample_counterfactuals("test_edge", batch_size=5)
        assert cf_data is not None
        
        original, counterfactual, parent_stats = cf_data
        assert original.shape == (5, 32)
        assert counterfactual.shape == (5, 32)
        assert parent_stats.shape == (5, 16)


class TestAuditLedger:
    """Test suite for audit ledger."""
    
    def test_merkle_tree(self):
        """Test Merkle tree implementation."""
        hashes = ["hash1", "hash2", "hash3", "hash4"]
        root = MerkleTree.build_tree(hashes)
        
        assert len(root) == 64  # SHA256 hex string
        assert root != "hash1"  # Should be different from input
        
        # Test with single hash
        single_root = MerkleTree.build_tree(["single"])
        assert single_root == "single"
        
        # Test with empty list
        empty_root = MerkleTree.build_tree([])
        assert len(empty_root) == 64
    
    def test_parameter_hasher(self):
        """Test audit parameter hasher."""
        model = nn.Linear(5, 3)
        state_dict = model.state_dict()
        
        hashes = AuditParameterHasher.hash_state_dict(state_dict)
        
        assert len(hashes) == len(state_dict)
        assert all(len(h) == 64 for h in hashes.values())  # SHA256 hex length
    
    def test_audit_ledger_basic_operations(self):
        """Test basic audit ledger operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = CryptoAuditLedger(ledger_dir=temp_dir, batch_size=3, auto_merkle=True)
            
            # Test logging different types of entries
            model = nn.Linear(10, 5)
            
            ledger.log_parameter_update("test_component", model)
            ledger.log_gradient_step("test_component", {"lr": 0.001}, 0.5, 0.1)
            ledger.log_evaluation_result("test_component", "accuracy", 0.85, {"test": True})
            
            # Check summary
            summary = ledger.get_audit_summary()
            assert summary['entries'] == 3
            assert "test_component" in summary['components']
            
            # Check integrity
            integrity = ledger.verify_chain_integrity()
            assert integrity['valid']
            assert integrity['entries'] == 3


class TestEvaluationMetrics:
    """Test suite for evaluation metrics."""
    
    def test_causal_consistency_score(self):
        """Test causal consistency score computation."""
        old_effects = torch.randn(10, 20)
        new_effects = old_effects + torch.randn(10, 20) * 0.1
        
        ccs = causal_consistency_score(old_effects, new_effects)
        
        assert isinstance(ccs, (float, torch.Tensor))
        assert 0 <= abs(float(ccs)) <= 1
    
    def test_fate_score(self):
        """Test FATE score computation."""
        base_effects = torch.randn(10, 15)
        old_effects = base_effects + torch.randn(10, 15) * 0.2
        new_effects = old_effects + torch.randn(10, 15) * 0.1
        
        fate = fate_score(old_effects, new_effects, base_effects)
        
        assert isinstance(fate, (float, torch.Tensor))
        assert float(fate) >= 0
    
    def test_causal_disentanglement_score(self):
        """Test causal disentanglement score."""
        causal_factors = torch.randn(50, 20)
        noise_factors = torch.randn(50, 15)
        
        score = causal_disentanglement_score(causal_factors, noise_factors)
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive causal evaluation."""
        causal_factors = torch.randn(20, 16)
        noise_factors = torch.randn(20, 12)
        original_outcomes = torch.randn(20, 16)
        counterfactual_outcomes = original_outcomes + torch.randn(20, 16) * 0.2
        importance_scores = {"param1": 0.5, "param2": 0.8}
        losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.38]
        
        metrics = comprehensive_causal_evaluation(
            causal_factors, noise_factors, original_outcomes, 
            counterfactual_outcomes, importance_scores, losses
        )
        
        assert isinstance(metrics, dict)
        assert "composite_score" in metrics
        assert "causal_consistency" in metrics
        assert "disentanglement" in metrics
        assert "efficiency" in metrics


class TestICCPTrainer:
    """Test suite for ICCP trainer integration."""
    
    def test_training_config(self):
        """Test ICCP training configuration."""
        config = ICCPTrainingConfig()
        
        assert config.model_name == "gpt2"
        assert config.causal_dim > 0
        assert config.num_epochs > 0
        assert len(config.causal_edges) > 0
    
    @patch('src.main.AutoTokenizer')
    @patch('src.main.GPT2Model')
    @patch('src.main.GPT2Config')
    def test_trainer_initialization(self, mock_config, mock_model, mock_tokenizer):
        """Test trainer initialization (mocked)."""
        # Mock the GPT2 components
        mock_config_instance = mock_config.from_pretrained.return_value
        mock_config_instance.n_embd = 128
        mock_config_instance.vocab_size = 1000
        
        mock_model_instance = mock_model.return_value
        mock_tokenizer_instance = mock_tokenizer.from_pretrained.return_value
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "[EOS]"
        
        config = ICCPTrainingConfig()
        config.num_epochs = 1  # Reduce for testing
        
        trainer = ICCPTrainer(config)
        
        assert trainer.config == config
        assert trainer.caref_encoder is not None
        assert trainer.scm_memory is not None
        assert trainer.cpi_computer is not None
        assert trainer.crb_buffer is not None


def test_smoke():
    """
    Comprehensive smoke test that runs a minimal ICCP training.
    """
    with patch('src.main.GPT2Config') as mock_config_cls, \
         patch('src.main.GPT2Model') as mock_model_cls, \
         patch('src.main.AutoTokenizer') as mock_tokenizer_cls:
        
        # Mock configuration
        mock_config = mock_config_cls.from_pretrained.return_value
        mock_config.n_embd = 64
        mock_config.vocab_size = 1000
        mock_config.max_length = 32
        mock_config.max_position_embeddings = 32
        mock_config.n_layer = 1
        mock_config.n_head = 2
        
        # Mock model
        mock_model = nn.Module()
        mock_model.wte = nn.Embedding(1000, 64)
        mock_model_cls.return_value = mock_model
        
        # Mock tokenizer
        mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        
        try:
            # Create minimal config for testing
            config = ICCPTrainingConfig()
            config.num_epochs = 2
            config.batch_size = 4
            config.causal_dim = 32
            config.noise_dim = 24
            config.enable_audit = False  # Disable audit for simplicity
            
            trainer = ICCPTrainer(config)
            
            # Test data generation
            dataloader = trainer._generate_synthetic_data(num_samples=20)
            assert len(dataloader) > 0
            
            # Test single step components
            input_ids = torch.randint(0, 1000, (4, 32))
            attention_mask = torch.ones_like(input_ids)
            
            caref_loss, caref_info = trainer._compute_caref_loss(input_ids, attention_mask)
            assert isinstance(caref_loss, torch.Tensor)
            assert isinstance(caref_info, dict)
            
            cf_loss, cf_info = trainer._compute_counterfactual_loss()
            assert isinstance(cf_info, dict)
            
            reg_loss, reg_info = trainer._compute_cpi_regularization()
            assert isinstance(reg_loss, torch.Tensor)
            assert isinstance(reg_info, dict)
            
            print("✅ Comprehensive smoke test passed!")
            
        except Exception as e:
            pytest.fail(f"Smoke test failed with exception: {e}")


if __name__ == "__main__":
    # Run specific test if called directly
    test_smoke()
