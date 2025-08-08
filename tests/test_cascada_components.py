"""
Comprehensive test suite for all CASCADA components
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cascada.online_graph_refinement import OnlineGraphRefinement
from cascada.adapter_factor_sharing import AdapterFactorSharing
from cascada.path_integrated_cpi import PathIntegratedCPI
from cascada.generative_counterfactual_replay import GenerativeCounterfactualReplay
from cascada.bayesian_uncertainty_gating import BayesianUncertaintyGating
from cascada.cascada_algorithm import CASCADA, CASCADAConfig


class TestOnlineGraphRefinement:
    """Test cases for Online Graph Refinement (OGR) component."""
    
    @pytest.fixture
    def ogr(self):
        return OnlineGraphRefinement(
            latent_dim=8,
            kernel_bandwidth=1.0,
            alpha_cit=0.05,
            min_samples=10
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        batch_size = 16
        stable_dim = 4
        context_dim = 4
        
        z_stable = torch.randn(batch_size, stable_dim)
        z_context = torch.randn(batch_size, context_dim)
        
        return z_stable, z_context
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample causal graph."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        return G
    
    def test_initialization(self, ogr):
        """Test OGR initialization."""
        assert ogr.latent_dim == 8
        assert ogr.alpha_cit == 0.05
        assert ogr.min_samples == 10
        assert isinstance(ogr.adjacency_matrix, torch.Tensor)
        assert ogr.adjacency_matrix.shape == (8, 8)
    
    def test_forward_insufficient_samples(self, ogr, sample_data, sample_graph):
        """Test forward pass with insufficient samples."""
        z_stable, z_context = sample_data
        
        # Clear buffer to simulate insufficient samples
        ogr.latent_buffer = []
        
        refined_graph = ogr.forward(z_stable, z_context, sample_graph)
        
        # Should return unchanged graph
        assert set(refined_graph.edges()) == set(sample_graph.edges())
    
    def test_forward_sufficient_samples(self, ogr, sample_data, sample_graph):
        """Test forward pass with sufficient samples."""
        z_stable, z_context = sample_data
        
        # Add multiple batches to exceed min_samples
        for _ in range(3):
            refined_graph = ogr.forward(z_stable, z_context, sample_graph)
        
        # Should have processed refinement (may or may not change graph)
        assert isinstance(refined_graph, nx.DiGraph)
        assert len(ogr.latent_buffer) > 0
    
    def test_conditional_independence_test(self, ogr):
        """Test conditional independence testing."""
        # Create correlated data
        n_samples = 50
        samples = np.random.randn(n_samples, 4)
        
        # Make variables 0 and 1 correlated
        samples[:, 1] = samples[:, 0] + 0.1 * np.random.randn(n_samples)
        
        # Test marginal independence (should be False for correlated variables)
        result = ogr._conditional_independence_test(samples, 0, 1, [])
        assert isinstance(result, bool)
    
    def test_edge_strength_computation(self, ogr):
        """Test edge strength computation."""
        samples = np.random.randn(30, 4)
        strengths = ogr.get_edge_strengths(samples)
        
        assert isinstance(strengths, dict)
        assert len(strengths) == 12  # 4 * 3 pairs (excluding self-loops)
        
        for edge, strength in strengths.items():
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert isinstance(strength, float)


class TestAdapterFactorSharing:
    """Test cases for Adapter Factor Sharing (AFS) component."""
    
    @pytest.fixture
    def afs(self):
        return AdapterFactorSharing(
            base_model_dim=64,
            adapter_dim=32,
            rank_1=4,
            rank_2=4,
            max_edges=10
        )
    
    def test_initialization(self, afs):
        """Test AFS initialization."""
        assert afs.base_model_dim == 64
        assert afs.adapter_dim == 32
        assert afs.rank_1 == 4
        assert afs.rank_2 == 4
        assert afs.core_tensor.shape == (4, 4, 32, 64)
        assert len(afs.active_edges) == 0
    
    def test_edge_management(self, afs):
        """Test adding and removing edges."""
        edge = (0, 1)
        
        # Add edge
        edge_idx = afs.add_edge(edge)
        assert edge in afs.active_edges
        assert afs.edge_to_idx[edge] == edge_idx
        assert afs.idx_to_edge[edge_idx] == edge
        
        # Remove edge
        afs.remove_edge(edge)
        assert edge not in afs.active_edges
        assert edge not in afs.edge_to_idx
    
    def test_adapter_computation(self, afs):
        """Test adapter matrix computation."""
        edge = (0, 1)
        afs.add_edge(edge)
        
        adapter = afs.get_edge_adapter(edge)
        
        assert isinstance(adapter, torch.Tensor)
        assert adapter.shape == (32, 64)  # (adapter_dim, base_model_dim)
    
    def test_mixed_adapter(self, afs):
        """Test mixed adapter computation."""
        edges = [(0, 1), (1, 2), (2, 3)]
        edge_weights = {}
        
        for edge in edges:
            afs.add_edge(edge)
            edge_weights[edge] = 1.0 / len(edges)
        
        mixed_adapter = afs.get_mixed_adapter(edge_weights)
        
        assert isinstance(mixed_adapter, torch.Tensor)
        assert mixed_adapter.shape == (32, 64)
    
    def test_apply_adapter(self, afs):
        """Test adapter application to input tensor."""
        edge = (0, 1)
        afs.add_edge(edge)
        
        batch_size, seq_len = 8, 16
        x = torch.randn(batch_size, seq_len, 64)
        
        adapted_x = afs.apply_adapter(x, edge=edge)
        
        assert adapted_x.shape == (batch_size, seq_len, 32)
    
    def test_memory_usage_calculation(self, afs):
        """Test memory usage calculation."""
        # Add some edges
        for i in range(3):
            afs.add_edge((i, i + 1))
        
        memory_stats = afs.get_memory_usage()
        
        assert 'total_params' in memory_stats
        assert 'naive_approach_params' in memory_stats
        assert 'memory_reduction_factor' in memory_stats
        assert memory_stats['active_edges'] == 3


class TestPathIntegratedCPI:
    """Test cases for Path-Integrated CPI component."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        return model
    
    @pytest.fixture
    def pi_cpi(self, mock_model):
        return PathIntegratedCPI(
            model=mock_model,
            max_path_length=3,
            integration_steps=5,
            top_k_paths=3
        )
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample causal graph."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
        return G
    
    def test_initialization(self, pi_cpi):
        """Test PI-CPI initialization."""
        assert pi_cpi.max_path_length == 3
        assert pi_cpi.integration_steps == 5
        assert pi_cpi.top_k_paths == 3
        assert len(pi_cpi.param_names) > 0
    
    def test_path_enumeration(self, pi_cpi, sample_graph):
        """Test causal path enumeration."""
        paths = pi_cpi._enumerate_causal_paths(sample_graph)
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        
        # Check that all paths are valid
        for path in paths:
            assert isinstance(path, list)
            assert len(path) <= pi_cpi.max_path_length + 1
    
    def test_path_selection(self, pi_cpi, sample_graph):
        """Test important path selection."""
        all_paths = pi_cpi._enumerate_causal_paths(sample_graph)
        important_paths = pi_cpi._select_important_paths(all_paths, sample_graph)
        
        assert len(important_paths) <= pi_cpi.top_k_paths
        assert len(important_paths) <= len(all_paths)
    
    def test_pi_cpi_computation(self, pi_cpi, mock_model, sample_graph):
        """Test PI-CPI score computation."""
        batch_size = 4
        input_batch = torch.randn(batch_size, 32)
        target_batch = torch.randint(0, 10, (batch_size,))
        
        pi_cpi_scores = pi_cpi.compute_path_integrated_cpi(
            sample_graph, input_batch, target_batch
        )
        
        assert isinstance(pi_cpi_scores, dict)
        assert len(pi_cpi_scores) > 0
        
        for param_name, score in pi_cpi_scores.items():
            assert isinstance(param_name, str)
            assert isinstance(score, float)
            assert score >= 0.0
    
    def test_regularizer_computation(self, pi_cpi):
        """Test PI-CPI regularizer computation."""
        pi_cpi_scores = {'param1': 0.5, 'param2': 1.0}
        parameter_changes = {
            'param1': torch.tensor(0.1),
            'param2': torch.tensor(0.2)
        }
        
        regularizer = pi_cpi.compute_pi_cpi_regularizer(
            pi_cpi_scores, parameter_changes
        )
        
        assert isinstance(regularizer, torch.Tensor)
        assert regularizer.numel() == 1


class TestGenerativeCounterfactualReplay:
    """Test cases for Generative Counterfactual Replay component."""
    
    @pytest.fixture
    def gcr(self):
        return GenerativeCounterfactualReplay(
            input_shape=(3, 32, 32),
            causal_dim=8,
            condition_dim=16,
            num_inference_steps=10  # Reduced for testing
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        batch_size = 4
        real_batch = torch.randn(batch_size, 3, 32, 32)
        interventions = torch.randn(batch_size, 8)
        return real_batch, interventions
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample causal graph."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        return G
    
    def test_initialization(self, gcr):
        """Test GCR initialization."""
        assert gcr.input_shape == (3, 32, 32)
        assert gcr.causal_dim == 8
        assert gcr.condition_dim == 16
        assert hasattr(gcr, 'unet')
        assert hasattr(gcr, 'noise_scheduler')
    
    def test_graph_to_adjacency(self, gcr, sample_graph):
        """Test conversion of graph to adjacency matrix."""
        batch_size = 2
        adj_matrix = gcr._graph_to_adjacency(sample_graph, batch_size)
        
        assert adj_matrix.shape == (batch_size, 8, 8)
        assert adj_matrix.dtype == torch.float32
        
        # Check that edges are represented
        assert adj_matrix[0, 0, 1] == 1.0  # Edge (0, 1) exists
        assert adj_matrix[0, 1, 2] == 1.0  # Edge (1, 2) exists
    
    def test_counterfactual_generation(self, gcr, sample_graph):
        """Test counterfactual generation."""
        batch_size = 2
        interventions = torch.randn(batch_size, 8)
        
        counterfactuals = gcr.generate_counterfactuals(
            batch_size, sample_graph, interventions, num_samples=1
        )
        
        assert counterfactuals.shape == (batch_size, 3, 32, 32)
        assert counterfactuals.dtype == torch.float32
    
    def test_loss_computation(self, gcr, sample_data, sample_graph):
        """Test generative loss computation."""
        real_batch, interventions = sample_data
        
        loss = gcr.compute_generative_cf_loss(
            real_batch, sample_graph, interventions
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.requires_grad
    
    def test_buffer_operations(self, gcr, sample_data):
        """Test counterfactual buffer operations."""
        real_batch, interventions = sample_data
        
        # Update buffer
        gcr._update_buffer(real_batch, interventions)
        assert len(gcr.cf_buffer) == real_batch.size(0)
        
        # Sample from buffer
        sampled_data, sampled_interventions = gcr.sample_from_buffer(2)
        assert sampled_data.shape[0] <= 2
        assert sampled_interventions.shape[0] <= 2


class TestBayesianUncertaintyGating:
    """Test cases for Bayesian Uncertainty Gating component."""
    
    @pytest.fixture
    def bug(self):
        return BayesianUncertaintyGating(
            context_dim=16,
            max_adapters=10,
            uncertainty_threshold=0.1
        )
    
    @pytest.fixture
    def sample_context(self):
        """Generate sample context data."""
        batch_size = 4
        context_dim = 16
        return torch.randn(batch_size, context_dim)
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample causal graph."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        return G
    
    def test_initialization(self, bug):
        """Test BUG initialization."""
        assert bug.context_dim == 16
        assert bug.max_adapters == 10
        assert bug.uncertainty_threshold == 0.1
        assert hasattr(bug, 'router')
    
    def test_adapter_registration(self, bug):
        """Test adapter registration and unregistration."""
        edge = (0, 1)
        adapter_idx = 3
        
        # Register adapter
        bug.register_edge_adapter(edge, adapter_idx)
        assert adapter_idx in bug.router.active_adapters
        assert bug.router.edge_to_adapter[edge] == adapter_idx
        
        # Unregister adapter
        bug.unregister_edge_adapter(edge)
        assert adapter_idx not in bug.router.active_adapters
        assert edge not in bug.router.edge_to_adapter
    
    def test_forward_pass(self, bug, sample_context, sample_graph):
        """Test forward pass with uncertainty computation."""
        # Register some adapters first
        for i, edge in enumerate([(0, 1), (1, 2)]):
            bug.register_edge_adapter(edge, i)
        
        edge_weights, uncertainty = bug(sample_context, sample_graph)
        
        assert isinstance(edge_weights, dict)
        assert isinstance(uncertainty, torch.Tensor)
        assert uncertainty.shape == (sample_context.size(0), 3)  # 3 uncertainty measures
    
    def test_routing_confidence(self, bug, sample_context, sample_graph):
        """Test routing confidence computation."""
        # Register some adapters
        for i, edge in enumerate([(0, 1), (1, 2)]):
            bug.register_edge_adapter(edge, i)
        
        confidence = bug.get_routing_confidence(sample_context, sample_graph)
        
        assert isinstance(confidence, torch.Tensor)
        assert confidence.shape == (sample_context.size(0),)
        assert torch.all(confidence >= 0.0) and torch.all(confidence <= 1.0)
    
    def test_feedback_update(self, bug, sample_context, sample_graph):
        """Test update from validation feedback."""
        # Register some adapters
        for i, edge in enumerate([(0, 1), (1, 2)]):
            bug.register_edge_adapter(edge, i)
        
        validation_errors = torch.randn(sample_context.size(0))
        
        # Should not raise an error
        bug.update_from_feedback(sample_context, sample_graph, validation_errors)
    
    def test_diagnostic_info(self, bug):
        """Test diagnostic information retrieval."""
        diagnostics = bug.get_diagnostic_info()
        
        assert isinstance(diagnostics, dict)
        assert 'total_routing_decisions' in diagnostics
        assert 'active_adapters' in diagnostics
        assert 'uncertainty_threshold' in diagnostics


class TestCASCADAIntegration:
    """Integration tests for the complete CASCADA system."""
    
    @pytest.fixture
    def base_model(self):
        """Create a simple base model for testing."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    @pytest.fixture
    def cascada(self, base_model):
        """Create CASCADA system for testing."""
        config = CASCADAConfig(
            base_model_dim=256,
            latent_dim=8,
            stable_dim=4,
            context_dim=4,
            max_edges=5,
            device='cpu'  # Use CPU for testing
        )
        return CASCADA(base_model, config)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        return x, y
    
    def test_initialization(self, cascada):
        """Test CASCADA system initialization."""
        assert hasattr(cascada, 'ogr')
        assert hasattr(cascada, 'afs')
        assert hasattr(cascada, 'pi_cpi')
        assert hasattr(cascada, 'gcr')
        assert hasattr(cascada, 'bug')
        assert isinstance(cascada.causal_graph, nx.DiGraph)
    
    def test_representation_extraction(self, cascada, sample_data):
        """Test representation extraction."""
        x, _ = sample_data
        
        z_stable, z_context = cascada.extract_representations(x)
        
        assert z_stable.shape == (4, 4)  # (batch_size, stable_dim)
        assert z_context.shape == (4, 4)  # (batch_size, context_dim)
    
    def test_forward_pass(self, cascada, sample_data):
        """Test forward pass."""
        x, y = sample_data
        
        output = cascada.forward(x, y)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == x.shape[0]  # Same batch size
    
    def test_forward_with_diagnostics(self, cascada, sample_data):
        """Test forward pass with diagnostics."""
        x, y = sample_data
        
        output, diagnostics = cascada.forward(
            x, y, return_diagnostics=True
        )
        
        assert isinstance(output, torch.Tensor)
        assert isinstance(diagnostics, dict)
        assert 'graph_edges' in diagnostics
        assert 'routing_uncertainty' in diagnostics
    
    def test_loss_computation(self, cascada, sample_data):
        """Test total loss computation."""
        x, y = sample_data
        
        predictions = cascada.forward(x, y)
        z_stable, z_context = cascada.extract_representations(x)
        edge_weights, _ = cascada.bug(z_context, cascada.causal_graph)
        
        total_loss, losses = cascada.compute_total_loss(
            x, y, predictions, edge_weights
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert isinstance(losses, dict)
        assert 'task' in losses
        assert 'counterfactual' in losses
        assert 'regularization' in losses
    
    def test_checkpoint_operations(self, cascada, tmp_path):
        """Test saving and loading checkpoints."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        # Save checkpoint
        cascada.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # Modify some state
        original_step = cascada.training_step
        cascada.training_step = 999
        
        # Load checkpoint
        cascada.load_checkpoint(str(checkpoint_path))
        assert cascada.training_step == original_step
    
    def test_system_diagnostics(self, cascada):
        """Test system diagnostics."""
        diagnostics = cascada.get_system_diagnostics()
        
        assert isinstance(diagnostics, dict)
        assert 'training_step' in diagnostics
        assert 'graph_structure' in diagnostics
        assert 'ogr_stats' in diagnostics
        assert 'afs_memory' in diagnostics
        assert 'pac_bound' in diagnostics
    
    def test_pac_bayesian_bound(self, cascada):
        """Test PAC-Bayesian bound computation."""
        # Add some loss history
        cascada.loss_history = [1.0, 0.9, 0.8, 0.7, 0.6]
        
        bound = cascada.compute_pac_bayesian_bound()
        
        assert isinstance(bound, float)
        assert bound >= 0.0


@pytest.mark.integration
class TestCASCADATraining:
    """Integration tests for CASCADA training process."""
    
    @pytest.fixture
    def setup_training(self):
        """Set up training environment."""
        # Simple base model
        base_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 64),  # Smaller for faster testing
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        # CASCADA config
        config = CASCADAConfig(
            base_model_dim=64,
            latent_dim=4,
            stable_dim=2,
            context_dim=2,
            max_edges=3,
            device='cpu'
        )
        
        # CASCADA system
        cascada = CASCADA(base_model, config)
        
        # Sample data
        batch_size = 8
        x = torch.randn(batch_size, 3, 8, 8)
        y = torch.randint(0, 10, (batch_size,))
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        return cascada, dataloader
    
    def test_continual_update(self, setup_training):
        """Test continual update on data shard."""
        cascada, dataloader = setup_training
        
        # Perform continual update
        metrics = cascada.continual_update(dataloader)
        
        assert isinstance(metrics, dict)
        assert 'task_loss' in metrics
        assert 'total_loss' in metrics
        assert 'accuracy' in metrics
        
        # Check that training step was incremented
        assert cascada.training_step > 0
        
        # Check that performance history was updated
        assert len(cascada.performance_history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])