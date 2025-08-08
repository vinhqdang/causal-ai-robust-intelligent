# CASCADA v2 Developer Guide

## 📖 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Deep Dive](#component-deep-dive)
3. [Algorithm Implementation](#algorithm-implementation)
4. [Extending CASCADA](#extending-cascada)
5. [Performance Optimization](#performance-optimization)
6. [Debugging & Troubleshooting](#debugging--troubleshooting)
7. [API Reference](#api-reference)

## Architecture Overview

### System Design Philosophy

CASCADA v2 follows a modular, component-based architecture where each of the five core components (C1-C5) can be developed, tested, and optimized independently while maintaining clean interfaces for integration.

```
┌─────────────────────────────────────────────────────────┐
│                    CASCADA System                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │     OGR     │  │     AFS     │  │   PI-CPI    │     │
│  │     (C1)    │  │     (C2)    │  │     (C3)    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────┐  ┌─────────────┐                      │
│  │     GCR     │  │     BUG     │                      │
│  │     (C4)    │  │     (C5)    │                      │
│  └─────────────┘  └─────────────┘                      │
├─────────────────────────────────────────────────────────┤
│           Base Model Integration Layer                   │
├─────────────────────────────────────────────────────────┤
│                 PyTorch Backend                         │
└─────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Modularity**: Each component is self-contained with minimal dependencies
2. **Configurability**: All hyperparameters exposed via `CASCADAConfig`
3. **Extensibility**: Clean interfaces for custom implementations
4. **Efficiency**: Memory and compute optimized for production use
5. **Debuggability**: Comprehensive diagnostics and logging

## Component Deep Dive

### C1: Online Graph Refinement (OGR)

**File**: `src/cascada/online_graph_refinement.py`

#### Purpose
Dynamically learns and updates the causal graph structure using kernelised conditional independence (CI) tests on latent representations.

#### Key Classes

```python
class OnlineGraphRefinement(nn.Module):
    def __init__(self, latent_dim: int, kernel_bandwidth: float, 
                 alpha_cit: float, min_samples: int, max_edges: int)
    
    def forward(self, z_stable: torch.Tensor, z_context: torch.Tensor, 
                current_graph: nx.DiGraph) -> nx.DiGraph
```

#### Algorithm Details

1. **Conditional Independence Testing**: Uses HSIC (Hilbert-Schmidt Independence Criterion) with RBF kernels
2. **Graph Updates**: Adds edges when variables are dependent, removes when independent
3. **Statistical Significance**: Uses chi-square test with configurable α level

#### Implementation Notes

- Maintains a rolling buffer of latent representations for statistical tests
- Uses regularized kernel matrix inversion for numerical stability
- Implements both marginal and conditional independence tests

#### Extending OGR

```python
class CustomOGR(OnlineGraphRefinement):
    def _conditional_independence_test(self, samples, i, j, conditioning_set):
        # Custom CI test implementation
        return custom_test_result
    
    def _custom_kernel(self, X):
        # Custom kernel function
        return kernel_matrix
```

### C2: Adapter Factor Sharing (AFS)

**File**: `src/cascada/adapter_factor_sharing.py`

#### Purpose
Provides memory-efficient adaptation using Tucker-2 decomposition to share parameters across causal edges.

#### Key Innovation
Instead of storing full adapter matrices for each edge, AFS factorizes all adapters as:
```
A_e = B ×₁ u_e^(1) ×₂ u_e^(2)
```

Where `B` is a shared core tensor and `u_e^(1)`, `u_e^(2)` are edge-specific factor vectors.

#### Memory Complexity
- **Naive approach**: O(E × d_adapter × d_model)
- **AFS approach**: O(r₁ × r₂ × d_adapter × d_model + E × (r₁ + r₂))
- **Reduction**: Up to 70% for E=100, r₁=r₂=8

#### Key Methods

```python
def get_edge_adapter(self, edge: Tuple[int, int]) -> torch.Tensor:
    """Compute adapter matrix using Tucker decomposition"""
    
def update_basis(self, gradients: Dict[Tuple[int, int], torch.Tensor]):
    """Update shared Tucker basis from edge gradients"""
```

#### Extending AFS

```python
class CustomAFS(AdapterFactorSharing):
    def _custom_factorization(self, adapter_grads):
        # Alternative factorization method (e.g., CP decomposition)
        return core, factors
```

### C3: Path-Integrated CPI (PI-CPI)

**File**: `src/cascada/path_integrated_cpi.py`

#### Purpose
Computes parameter importance along entire causal paths rather than individual edges, preventing "over-freeze" of important parameters.

#### Mathematical Foundation
```
PI-CPI_j = Σ_{paths p} ∫₀¹ ∂_{θⱼ} τₚ(θ₀ + α Δθⱼ) dα
```

Where:
- `τₚ` is the path function from source to target
- `θⱼ` is parameter j
- Integration captures full influence along the path

#### Implementation Strategy

1. **Path Enumeration**: Find all causal paths up to max length
2. **Path Selection**: Score paths by structural importance
3. **Numerical Integration**: Trapezoidal rule over α ∈ [0,1]
4. **Gradient Computation**: Use PyTorch autograd for ∂τₙ/∂θⱼ

#### Key Methods

```python
def compute_path_integrated_cpi(self, causal_graph: nx.DiGraph, 
                               input_batch: torch.Tensor) -> Dict[str, float]:
    """Main CPI computation method"""

def _compute_path_influence_at_alpha(self, path: List[int], alpha: float, 
                                   ...) -> Dict[str, float]:
    """Influence computation at specific interpolation point"""
```

#### Extending PI-CPI

```python
class AdaptivePI_CPI(PathIntegratedCPI):
    def _adaptive_integration_steps(self, path_complexity):
        # Adjust integration steps based on path properties
        return adaptive_steps
```

### C4: Generative Counterfactual Replay (GCR)

**File**: `src/cascada/generative_counterfactual_replay.py`

#### Purpose
Generates high-fidelity counterfactuals using a lightweight diffusion model conditioned on causal graph structure and interventions.

#### Architecture

```python
class GenerativeCounterfactualReplay(nn.Module):
    def __init__(self, input_shape, causal_dim, condition_dim, num_inference_steps)
    
    # Core components:
    self.causal_conditioner: CausalConditioningModule  # Graph + intervention encoder
    self.unet: LightweightDiffusionUNet                # 4-block U-Net
    self.noise_scheduler: DDPMScheduler                # DDPM sampling
```

#### Diffusion Process

1. **Forward Process**: Add noise to real data over T steps
2. **Conditioning**: Encode graph adjacency + interventions  
3. **Denoising**: Lightweight U-Net predicts noise
4. **Sampling**: DDPM scheduler generates counterfactuals

#### Causal Conditioning

The model conditions on:
- **Graph Structure**: Adjacency matrix → graph embedding
- **Interventions**: Intervention values → intervention embedding
- **Combined**: Additive combination of embeddings

#### Key Methods

```python
def generate_counterfactuals(self, batch_size: int, causal_graph: nx.DiGraph,
                           interventions: torch.Tensor) -> torch.Tensor:
    """Generate counterfactual samples"""

def compute_generative_cf_loss(self, real_batch: torch.Tensor, 
                              causal_graph: nx.DiGraph) -> torch.Tensor:
    """Training loss for diffusion model"""
```

#### Extending GCR

```python
class AdvancedGCR(GenerativeCounterfactualReplay):
    def __init__(self, ...):
        super().__init__(...)
        # Use more sophisticated conditioning
        self.attention_conditioner = GraphAttentionConditioner()
        
    def _advanced_causal_conditioning(self, graph, interventions):
        # Custom conditioning strategy
        return advanced_conditioning
```

### C5: Bayesian Uncertainty Gating (BUG)

**File**: `src/cascada/bayesian_uncertainty_gating.py`

#### Purpose
Routes adapter selection through Dirichlet posterior, enabling uncertainty-aware decision making and exploration.

#### Bayesian Framework

1. **Prior**: Dirichlet(α) over adapter selection probabilities
2. **Posterior**: Updated based on validation performance  
3. **Sampling**: Thompson sampling or UCB for exploration
4. **Uncertainty**: Mutual information as epistemic uncertainty measure

#### Key Components

```python
class DirichletRouter(nn.Module):
    def __init__(self, context_dim, max_adapters, concentration_init)
    
    def forward(self, z_context, causal_graph, return_uncertainty=False):
        """Compute Dirichlet posterior over adapters"""

class BayesianUncertaintyGating(nn.Module):
    def __init__(self, context_dim, max_adapters, uncertainty_threshold)
    
    def forward(self, z_context, causal_graph, exploration=True):
        """Complete uncertainty-aware routing"""
```

#### Uncertainty Measures

1. **Entropy of Expected**: H[E[p]]
2. **Expected Entropy**: E[H[p]]  
3. **Mutual Information**: H[E[p]] - E[H[p]] (epistemic uncertainty)

#### Extending BUG

```python
class HierarchicalBUG(BayesianUncertaintyGating):
    def __init__(self, ...):
        super().__init__(...)
        self.hierarchical_router = HierarchicalDirichletProcess()
    
    def _hierarchical_routing(self, context, graph):
        # Multi-level routing with HDP
        return hierarchical_weights
```

## Algorithm Implementation

### Main Training Loop

The core algorithm follows the pseudocode from `algorithmv2.md`:

```python
def continual_update(self, data_shard: DataLoader) -> Dict[str, float]:
    # Store old parameters for PI-CPI regularization
    self._store_old_parameters()
    
    for batch_idx, (x, y) in enumerate(data_shard):
        # 1. Extract representations
        z_stable, z_context = self.extract_representations(x)
        
        # 2. Online Graph Refinement (C1)
        refined_graph = self.ogr(z_stable, z_context, self.causal_graph)
        self._update_graph_structure(refined_graph)
        
        # 3. Bayesian Uncertainty Gating (C5)
        edge_weights, uncertainty = self.bug(z_context, self.causal_graph)
        
        # 4. Adapter Factor Sharing (C2)
        mixed_adapter = self.afs.get_mixed_adapter(edge_weights)
        
        # 5. Forward pass with adaptation
        predictions = self._adapted_forward(x, mixed_adapter)
        
        # 6. Three-term loss computation
        L_task = task_loss_fn(predictions, y)
        L_cf = self.gcr.compute_generative_cf_loss(x, self.causal_graph, interventions)
        L_reg = self._compute_pi_cpi_regularizer()  # C3
        
        total_loss = L_task + β * L_cf + λ * L_reg
        
        # 7. Selective parameter update based on PI-CPI
        self._selective_parameter_update(optimizer, total_loss)
        
        # 8. Update component-specific parameters
        self.afs.update_basis(self._extract_edge_gradients(edge_weights))
        self.gcr.partial_fit(x, self.causal_graph, interventions)
        self.bug.update_from_feedback(z_context, self.causal_graph, validation_errors)
    
    return metrics
```

### Component Integration

#### Graph Structure Updates

```python
def _update_graph_structure(self, new_graph: nx.DiGraph):
    """Synchronize all components when graph changes"""
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
```

#### Memory Management

```python
def _optimize_memory_usage(self):
    """Dynamic memory optimization"""
    # AFS basis compression
    if self.afs.get_memory_usage()['memory_reduction_factor'] < 1.5:
        self.afs._compress_basis()
    
    # GCR buffer pruning
    if len(self.gcr.cf_buffer) > self.config.buffer_size:
        self.gcr._prune_buffer()
    
    # PI-CPI cache cleanup
    self.pi_cpi.clear_cache()
```

## Extending CASCADA

### Custom Components

#### 1. Custom Graph Refinement

```python
class CustomGraphRefinement(OnlineGraphRefinement):
    def __init__(self, latent_dim, custom_param):
        super().__init__(latent_dim)
        self.custom_param = custom_param
    
    def _custom_structure_learning(self, data):
        # Implement custom graph learning algorithm
        # e.g., PC algorithm, GES, or neural approaches
        pass
```

#### 2. Custom Adapter Architecture

```python
class HyperNetworkAFS(AdapterFactorSharing):
    def __init__(self, base_model_dim, hypernetwork_dim):
        super().__init__(base_model_dim)
        self.hypernetwork = nn.Sequential(
            nn.Linear(hypernetwork_dim, 128),
            nn.ReLU(),
            nn.Linear(128, base_model_dim * adapter_dim)
        )
    
    def get_edge_adapter(self, edge):
        # Generate adapter using hypernetwork
        edge_embedding = self._encode_edge(edge)
        adapter_params = self.hypernetwork(edge_embedding)
        return adapter_params.view(self.adapter_dim, self.base_model_dim)
```

#### 3. Multi-Modal Extensions

```python
class MultiModalCASCADA(CASCADA):
    def __init__(self, vision_model, language_model, config):
        super().__init__(vision_model, config)
        self.language_model = language_model
        self.cross_modal_fusion = CrossModalFusion(config)
    
    def forward(self, vision_input, language_input):
        # Process both modalities
        vision_repr = self.extract_representations(vision_input)
        language_repr = self.extract_representations(language_input)
        
        # Cross-modal fusion
        fused_repr = self.cross_modal_fusion(vision_repr, language_repr)
        
        # Continue with CASCADA processing
        return self._cascada_forward(fused_repr)
```

### Configuration Extensions

```python
@dataclass
class CustomCASCADAConfig(CASCADAConfig):
    # Custom graph learning parameters
    structure_learning_algorithm: str = "pc"
    pc_alpha: float = 0.01
    
    # Custom adaptation parameters
    hypernetwork_dim: int = 64
    cross_modal_fusion: bool = False
    
    # Custom training parameters
    curriculum_learning: bool = True
    adaptive_regularization: bool = True
```

## Performance Optimization

### Memory Optimization

#### 1. Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class MemoryOptimizedCASCADA(CASCADA):
    def forward(self, x):
        # Use gradient checkpointing for memory-intensive components
        z_stable, z_context = checkpoint(self.extract_representations, x)
        
        # Checkpoint PI-CPI computation
        pi_cpi_scores = checkpoint(
            lambda: self.pi_cpi.compute_path_integrated_cpi(
                self.causal_graph, x
            )
        )
        
        return self._continue_forward(z_stable, z_context, pi_cpi_scores)
```

#### 2. Dynamic Precision

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionCASCADA(CASCADA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()
    
    def continual_update(self, data_shard):
        for x, y in data_shard:
            with autocast():
                output = self.forward(x)
                loss = self.compute_total_loss(x, y, output)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
```

### Compute Optimization

#### 1. Batch Processing

```python
class BatchOptimizedCASCADA(CASCADA):
    def _batch_process_graph_updates(self, batch_representations):
        """Process multiple graph updates in parallel"""
        # Batch conditional independence tests
        batch_ci_results = self.ogr.batch_ci_test(batch_representations)
        
        # Aggregate graph updates
        consensus_graph = self._consensus_graph_update(batch_ci_results)
        return consensus_graph
```

#### 2. Sparse Operations

```python
class SparseCASCADA(CASCADA):
    def __init__(self, *args, sparsity_threshold=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity_threshold = sparsity_threshold
    
    def _sparse_adapter_routing(self, edge_weights):
        """Use sparse routing for efficiency"""
        # Filter out low-weight edges
        sparse_weights = {
            edge: weight for edge, weight in edge_weights.items()
            if weight > self.sparsity_threshold
        }
        return sparse_weights
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedCASCADA(CASCADA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Wrap components in DDP
        self.base_model = DDP(self.base_model)
        self.gcr = DDP(self.gcr)
    
    def distributed_graph_consensus(self):
        """Achieve consensus on graph structure across nodes"""
        local_graph_edges = list(self.causal_graph.edges())
        
        # Gather all local graphs
        gathered_graphs = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_graphs, local_graph_edges)
        
        # Compute consensus graph
        consensus_graph = self._merge_graphs(gathered_graphs)
        return consensus_graph
```

## Debugging & Troubleshooting

### Common Issues

#### 1. Memory Issues

**Symptoms**: CUDA out of memory, slow training
**Solutions**:
```python
# Reduce batch size and adapter dimensions
config.batch_size = 16  # Instead of 32
config.afs_rank_1 = 4   # Instead of 8
config.afs_rank_2 = 4   # Instead of 8

# Enable gradient checkpointing
config.use_gradient_checkpointing = True

# Clear caches periodically
if step % 100 == 0:
    torch.cuda.empty_cache()
    cascada.pi_cpi.clear_cache()
```

#### 2. Graph Learning Issues

**Symptoms**: Graph remains static, no edge updates
**Solutions**:
```python
# Check CI test sensitivity
config.ogr_alpha_cit = 0.05  # Increase from 0.01

# Ensure sufficient samples
config.ogr_min_samples = 30  # Reduce from 50

# Debug CI tests
diagnostics = cascada.ogr.get_statistics()
print(f"Buffer size: {diagnostics['buffer_size']}")
print(f"Current edges: {diagnostics['current_edges']}")
```

#### 3. Adapter Routing Issues

**Symptoms**: All weight concentrated on few adapters
**Solutions**:
```python
# Increase exploration
config.bug_exploration_rate = 0.2  # Increase from 0.1

# Check concentration parameters
router_diag = cascada.bug.get_diagnostic_info()
print(f"High uncertainty rate: {router_diag['high_uncertainty_rate']}")

# Manual router inspection
concentrations = torch.exp(cascada.bug.router.log_concentrations)
print(f"Concentration range: {concentrations.min():.3f} - {concentrations.max():.3f}")
```

### Debugging Tools

#### 1. Component Diagnostics

```python
def debug_cascada_state(cascada):
    """Comprehensive debugging information"""
    diagnostics = cascada.get_system_diagnostics()
    
    print("=== CASCADA Debug Report ===")
    print(f"Training Step: {diagnostics['training_step']}")
    print(f"Graph Density: {diagnostics['graph_structure']['density']:.3f}")
    print(f"Active Adapters: {diagnostics['afs_memory']['active_edges']}")
    print(f"Memory Reduction: {diagnostics['afs_memory']['memory_reduction_factor']:.2f}x")
    print(f"PI-CPI Cache Size: {diagnostics['pi_cpi_stats']['path_cache_entries']}")
    print(f"GCR Buffer Size: {diagnostics['gcr_stats']['buffer_size']}")
    print(f"PAC Bound: {diagnostics['pac_bound']:.4f}")
    
    return diagnostics
```

#### 2. Performance Profiling

```python
import cProfile
import pstats

def profile_cascada_forward(cascada, sample_batch):
    """Profile CASCADA forward pass"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    output = cascada.forward(sample_batch)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return output
```

#### 3. Visualization Tools

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_causal_graph(cascada):
    """Visualize current causal graph structure"""
    G = cascada.causal_graph
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, arrowsize=20)
    plt.title(f"Causal Graph (Edges: {len(G.edges())})")
    plt.show()

def plot_adapter_usage(cascada):
    """Plot adapter usage statistics"""
    afs_stats = cascada.afs.get_memory_usage()
    
    plt.figure(figsize=(8, 6))
    plt.bar(['AFS Total', 'Naive Equivalent'], 
            [afs_stats['total_params'], afs_stats['naive_approach_params']])
    plt.title('Memory Usage Comparison')
    plt.ylabel('Number of Parameters')
    plt.show()
```

## API Reference

### Core Classes

#### `CASCADA`
Main algorithm class orchestrating all components.

```python
class CASCADA(nn.Module):
    def __init__(self, base_model: nn.Module, config: CASCADAConfig = None)
    def forward(self, x: torch.Tensor, y: torch.Tensor = None, 
                update_graph: bool = True, return_diagnostics: bool = False)
    def continual_update(self, data_shard: DataLoader, 
                        optimizer: optim.Optimizer = None, 
                        validation_data: DataLoader = None) -> Dict[str, float]
    def save_checkpoint(self, path: str)
    def load_checkpoint(self, path: str)
    def get_system_diagnostics(self) -> Dict[str, Any]
```

#### `CASCADAConfig`
Configuration dataclass for all hyperparameters.

```python
@dataclass
class CASCADAConfig:
    # Model dimensions
    base_model_dim: int = 512
    latent_dim: int = 64
    stable_dim: int = 32
    context_dim: int = 32
    adapter_dim: int = 64
    
    # Component parameters
    ogr_alpha_cit: float = 0.01
    afs_rank_1: int = 8
    afs_rank_2: int = 8
    pi_cpi_integration_steps: int = 20
    gcr_num_inference_steps: int = 50
    bug_uncertainty_threshold: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    beta_cf: float = 1.0
    lambda_reg: float = 1.0
    max_edges: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

### Component APIs

#### OnlineGraphRefinement

```python
class OnlineGraphRefinement(nn.Module):
    def forward(self, z_stable: torch.Tensor, z_context: torch.Tensor, 
                current_graph: nx.DiGraph) -> nx.DiGraph
    def get_edge_strengths(self, samples: np.ndarray) -> Dict[Tuple[int, int], float]
    def reset_buffer(self)
    def get_statistics(self) -> Dict
```

#### AdapterFactorSharing

```python
class AdapterFactorSharing(nn.Module):
    def add_edge(self, edge: Tuple[int, int]) -> int
    def remove_edge(self, edge: Tuple[int, int])
    def get_edge_adapter(self, edge: Tuple[int, int]) -> torch.Tensor
    def get_mixed_adapter(self, edge_weights: Dict[Tuple[int, int], float]) -> torch.Tensor
    def apply_adapter(self, x: torch.Tensor, edge: Tuple[int, int] = None, 
                     edge_weights: Dict[Tuple[int, int], float] = None) -> torch.Tensor
    def update_basis(self, gradients: Dict[Tuple[int, int], torch.Tensor])
    def get_memory_usage(self) -> Dict[str, int]
```

#### PathIntegratedCPI

```python
class PathIntegratedCPI(nn.Module):
    def compute_path_integrated_cpi(self, causal_graph: nx.DiGraph, 
                                   input_batch: torch.Tensor, 
                                   target_batch: torch.Tensor = None, 
                                   loss_fn: Callable = None) -> Dict[str, float]
    def compute_pi_cpi_regularizer(self, pi_cpi_scores: Dict[str, float], 
                                  parameter_changes: Dict[str, torch.Tensor]) -> torch.Tensor
    def get_top_influential_parameters(self, pi_cpi_scores: Dict[str, float], 
                                      k: int = 10) -> List[Tuple[str, float]]
    def clear_cache(self)
    def get_memory_stats(self) -> Dict[str, int]
```

#### GenerativeCounterfactualReplay

```python
class GenerativeCounterfactualReplay(nn.Module):
    def generate_counterfactuals(self, batch_size: int, causal_graph: nx.DiGraph, 
                               interventions: torch.Tensor, 
                               num_samples: int = 1) -> torch.Tensor
    def compute_generative_cf_loss(self, real_batch: torch.Tensor, 
                                  causal_graph: nx.DiGraph, 
                                  interventions: torch.Tensor) -> torch.Tensor
    def partial_fit(self, data_batch: torch.Tensor, causal_graph: nx.DiGraph, 
                   interventions: torch.Tensor = None, 
                   optimizer: optim.Optimizer = None)
    def sample_from_buffer(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]
    def get_model_stats(self) -> Dict[str, Union[int, float]]
```

#### BayesianUncertaintyGating

```python
class BayesianUncertaintyGating(nn.Module):
    def forward(self, z_context: torch.Tensor, causal_graph: nx.DiGraph, 
                exploration: bool = True) -> Tuple[Dict[Tuple[int, int], float], torch.Tensor]
    def register_edge_adapter(self, edge: Tuple[int, int], adapter_idx: int)
    def unregister_edge_adapter(self, edge: Tuple[int, int])
    def update_from_feedback(self, z_context: torch.Tensor, causal_graph: nx.DiGraph, 
                           validation_errors: torch.Tensor)
    def get_routing_confidence(self, z_context: torch.Tensor, 
                             causal_graph: nx.DiGraph) -> torch.Tensor
    def get_diagnostic_info(self) -> Dict
```

### Utility Functions

```python
# Visualization utilities
def visualize_causal_graph(graph: nx.DiGraph, save_path: str = None)
def plot_training_progress(performance_history: List[Dict], save_path: str = None)
def plot_component_diagnostics(cascada: CASCADA)

# Performance utilities
def profile_component_performance(cascada: CASCADA, sample_batch: torch.Tensor)
def benchmark_memory_usage(cascada: CASCADA)
def analyze_graph_evolution(cascada: CASCADA, num_steps: int)

# Debugging utilities
def debug_component_states(cascada: CASCADA)
def validate_component_integration(cascada: CASCADA)
def check_numerical_stability(cascada: CASCADA)
```

---

This developer guide provides the foundational knowledge needed to understand, extend, and optimize CASCADA v2. For specific implementation details, refer to the source code and inline documentation.