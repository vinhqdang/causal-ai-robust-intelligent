import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

def causal_consistency_score(old_effects: torch.Tensor, new_effects: torch.Tensor) -> float:
    """
    Computes the Causal Consistency Score (CCS).
    CCS is defined as the cosine similarity between the treatment effect vectors
    before and after the update.

    Args:
        old_effects (torch.Tensor): The treatment effects before the update.
        new_effects (torch.Tensor): The treatment effects after the update.

    Returns:
        float: The Causal Consistency Score.
    """
    old_effects_flat = old_effects.flatten()
    new_effects_flat = new_effects.flatten()
    
    if torch.norm(old_effects_flat) == 0 or torch.norm(new_effects_flat) == 0:
        return 0.0
        
    return torch.dot(old_effects_flat, new_effects_flat) / \
           (torch.norm(old_effects_flat) * torch.norm(new_effects_flat))

def fate_score(old_effects: torch.Tensor, new_effects: torch.Tensor, 
               base_effects: torch.Tensor) -> float:
    """
    Computes the Forgetting-Adjusted Treatment-Effect Error (FATE).
    FATE measures how much the model has forgotten the original treatment effects.

    Args:
        old_effects (torch.Tensor): The treatment effects before the update.
        new_effects (torch.Tensor): The treatment effects after the update.
        base_effects (torch.Tensor): The initial treatment effects before any updates.

    Returns:
        float: The FATE score.
    """
    return torch.norm(new_effects - old_effects) / (torch.norm(base_effects) + 1e-8)


def causal_disentanglement_score(causal_factors: torch.Tensor, 
                                noise_factors: torch.Tensor) -> float:
    """
    Measure how well causal and noise factors are disentangled.
    Uses mutual information estimation.
    
    Args:
        causal_factors: Causal representation [batch_size, causal_dim]
        noise_factors: Noise representation [batch_size, noise_dim]
        
    Returns:
        Disentanglement score (lower is better, 0 = perfect disentanglement)
    """
    # Convert to numpy and discretize for MI estimation
    causal_np = causal_factors.detach().cpu().numpy()
    noise_np = noise_factors.detach().cpu().numpy()
    
    # Compute pairwise mutual information
    mi_scores = []
    
    for i in range(min(causal_np.shape[1], 10)):  # Limit to first 10 dims for efficiency
        for j in range(min(noise_np.shape[1], 10)):
            # Discretize continuous variables
            c_discrete = np.digitize(causal_np[:, i], np.linspace(causal_np[:, i].min(), 
                                                                 causal_np[:, i].max(), 20))
            n_discrete = np.digitize(noise_np[:, j], np.linspace(noise_np[:, j].min(), 
                                                                noise_np[:, j].max(), 20))
            
            mi = mutual_info_score(c_discrete, n_discrete)
            mi_scores.append(mi)
    
    return np.mean(mi_scores) if mi_scores else 0.0


def intervention_consistency_score(pre_intervention: torch.Tensor,
                                  post_intervention: torch.Tensor,
                                  intervention_strength: float = 1.0) -> float:
    """
    Measure consistency of causal effects under interventions.
    
    Args:
        pre_intervention: Model outputs before intervention
        post_intervention: Model outputs after intervention  
        intervention_strength: Expected strength of intervention effect
        
    Returns:
        Consistency score (higher is better)
    """
    effect_size = torch.norm(post_intervention - pre_intervention)
    expected_effect = intervention_strength
    
    # Score based on how close the effect is to expected
    consistency = 1.0 / (1.0 + abs(effect_size.item() - expected_effect))
    return consistency


def structural_stability_score(causal_graph_old: Dict[str, List[str]],
                              causal_graph_new: Dict[str, List[str]]) -> float:
    """
    Measure stability of causal structure after updates.
    
    Args:
        causal_graph_old: Previous causal graph structure
        causal_graph_new: Current causal graph structure
        
    Returns:
        Stability score (1.0 = identical structure, 0.0 = completely different)
    """
    if not causal_graph_old and not causal_graph_new:
        return 1.0
    
    all_nodes = set(causal_graph_old.keys()) | set(causal_graph_new.keys())
    if not all_nodes:
        return 1.0
    
    edge_overlap = 0
    total_edges = 0
    
    for node in all_nodes:
        old_children = set(causal_graph_old.get(node, []))
        new_children = set(causal_graph_new.get(node, []))
        
        edge_overlap += len(old_children & new_children)
        total_edges += len(old_children | new_children)
    
    return edge_overlap / max(total_edges, 1)


def counterfactual_validity_score(original_outcomes: torch.Tensor,
                                 counterfactual_outcomes: torch.Tensor,
                                 min_effect_size: float = 0.1) -> float:
    """
    Evaluate quality of counterfactual generation.
    
    Args:
        original_outcomes: Original model outputs
        counterfactual_outcomes: Counterfactual model outputs
        min_effect_size: Minimum expected effect size
        
    Returns:
        Validity score measuring counterfactual quality
    """
    # Check if counterfactuals are sufficiently different
    effect_size = torch.norm(counterfactual_outcomes - original_outcomes, dim=-1).mean()
    diversity_score = torch.var(counterfactual_outcomes, dim=0).mean()
    
    # Penalize if effects are too small (counterfactuals too similar to originals)
    effect_penalty = max(0, min_effect_size - effect_size.item())
    
    # Reward diversity in counterfactual outcomes  
    validity = diversity_score.item() - effect_penalty
    
    return max(0.0, validity)


def parameter_importance_coherence(importance_scores: Dict[str, float],
                                  gradient_norms: Dict[str, float]) -> float:
    """
    Check coherence between parameter importance and gradient magnitudes.
    
    Args:
        importance_scores: CPI importance scores for parameters
        gradient_norms: Gradient norms for same parameters
        
    Returns:
        Coherence score (higher = more coherent)
    """
    common_params = set(importance_scores.keys()) & set(gradient_norms.keys())
    
    if len(common_params) < 2:
        return 0.0
    
    importance_vals = [importance_scores[p] for p in common_params]
    gradient_vals = [gradient_norms[p] for p in common_params]
    
    # Compute correlation
    importance_vals = np.array(importance_vals)
    gradient_vals = np.array(gradient_vals)
    
    if np.std(importance_vals) == 0 or np.std(gradient_vals) == 0:
        return 0.0
    
    correlation = np.corrcoef(importance_vals, gradient_vals)[0, 1]
    return max(0.0, correlation)  # Return only positive correlations


def adaptation_efficiency_score(losses_over_time: List[float],
                               convergence_threshold: float = 0.01) -> Dict[str, float]:
    """
    Measure how efficiently the model adapts to new data.
    
    Args:
        losses_over_time: Training losses over time
        convergence_threshold: Threshold for considering convergence
        
    Returns:
        Dictionary with efficiency metrics
    """
    if len(losses_over_time) < 2:
        return {'efficiency': 0.0, 'convergence_step': -1, 'final_loss': losses_over_time[0] if losses_over_time else 0.0}
    
    losses = np.array(losses_over_time)
    
    # Find convergence point
    convergence_step = -1
    for i in range(1, len(losses)):
        if abs(losses[i] - losses[i-1]) < convergence_threshold:
            convergence_step = i
            break
    
    # Compute efficiency as inverse of convergence time (normalized)
    if convergence_step > 0:
        efficiency = 1.0 / (convergence_step / len(losses))
    else:
        # Penalize if no convergence
        efficiency = 0.1
    
    # Also consider final loss reduction
    loss_reduction = (losses[0] - losses[-1]) / max(losses[0], 1e-8)
    efficiency *= max(0.1, loss_reduction)
    
    return {
        'efficiency': efficiency,
        'convergence_step': convergence_step,
        'final_loss': losses[-1],
        'loss_reduction': loss_reduction
    }


def comprehensive_causal_evaluation(causal_factors: torch.Tensor,
                                   noise_factors: torch.Tensor,
                                   original_outcomes: torch.Tensor,
                                   counterfactual_outcomes: torch.Tensor,
                                   importance_scores: Dict[str, float],
                                   losses_over_time: List[float]) -> Dict[str, float]:
    """
    Comprehensive evaluation combining multiple causal metrics.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    metrics = {}
    
    # Core causal metrics
    metrics['causal_consistency'] = causal_consistency_score(
        original_outcomes[:-1], original_outcomes[1:]
    ).item() if len(original_outcomes) > 1 else 0.0
    
    metrics['disentanglement'] = causal_disentanglement_score(
        causal_factors, noise_factors
    )
    
    metrics['counterfactual_validity'] = counterfactual_validity_score(
        original_outcomes, counterfactual_outcomes
    )
    
    # Adaptation metrics
    adaptation_metrics = adaptation_efficiency_score(losses_over_time)
    metrics.update(adaptation_metrics)
    
    # Composite score
    weights = {
        'causal_consistency': 0.3,
        'disentanglement': -0.2,  # Lower is better
        'counterfactual_validity': 0.2,
        'efficiency': 0.3
    }
    
    composite_score = sum(weights.get(k, 0) * v for k, v in metrics.items() 
                         if k in weights)
    metrics['composite_score'] = composite_score
    
    return metrics

if __name__ == '__main__':
    print("=== Testing Enhanced Causal Evaluation Metrics ===")
    
    # Generate synthetic data for testing
    batch_size = 50
    causal_dim = 32
    noise_dim = 24
    
    # 1. Test basic metrics
    print("\n--- Testing Basic Metrics ---")
    old_effects = torch.randn(batch_size, causal_dim)
    new_effects = old_effects + torch.randn(batch_size, causal_dim) * 0.1
    base_effects = torch.randn(batch_size, causal_dim)
    
    ccs = causal_consistency_score(old_effects, new_effects)
    fate = fate_score(old_effects, new_effects, base_effects)
    
    print(f"Causal Consistency Score: {ccs:.4f}")
    print(f"FATE Score: {fate:.4f}")
    
    # 2. Test disentanglement score
    print("\n--- Testing Disentanglement Score ---")
    causal_factors = torch.randn(batch_size, causal_dim)
    noise_factors = torch.randn(batch_size, noise_dim)
    
    disentanglement = causal_disentanglement_score(causal_factors, noise_factors)
    print(f"Disentanglement Score: {disentanglement:.4f}")
    
    # 3. Test intervention consistency
    print("\n--- Testing Intervention Consistency ---")
    pre_intervention = torch.randn(batch_size, 10)
    post_intervention = pre_intervention + torch.randn(batch_size, 10) * 0.5
    
    intervention_score = intervention_consistency_score(pre_intervention, post_intervention, 0.5)
    print(f"Intervention Consistency: {intervention_score:.4f}")
    
    # 4. Test structural stability
    print("\n--- Testing Structural Stability ---")
    old_graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"]}
    new_graph = {"A": ["B", "C"], "B": ["D"], "C": []}  # Lost C->D edge
    
    stability = structural_stability_score(old_graph, new_graph)
    print(f"Structural Stability: {stability:.4f}")
    
    # 5. Test counterfactual validity
    print("\n--- Testing Counterfactual Validity ---")
    original_outcomes = torch.randn(batch_size, causal_dim)
    counterfactual_outcomes = original_outcomes + torch.randn(batch_size, causal_dim) * 0.3
    
    cf_validity = counterfactual_validity_score(original_outcomes, counterfactual_outcomes)
    print(f"Counterfactual Validity: {cf_validity:.4f}")
    
    # 6. Test parameter importance coherence
    print("\n--- Testing Parameter Importance Coherence ---")
    importance_scores = {"param1": 0.8, "param2": 0.3, "param3": 0.6}
    gradient_norms = {"param1": 0.7, "param2": 0.2, "param3": 0.5}
    
    coherence = parameter_importance_coherence(importance_scores, gradient_norms)
    print(f"Parameter Importance Coherence: {coherence:.4f}")
    
    # 7. Test adaptation efficiency
    print("\n--- Testing Adaptation Efficiency ---")
    losses_over_time = [1.0, 0.8, 0.6, 0.45, 0.42, 0.41, 0.40, 0.40, 0.39]
    
    efficiency_metrics = adaptation_efficiency_score(losses_over_time, convergence_threshold=0.02)
    print(f"Adaptation Efficiency: {efficiency_metrics['efficiency']:.4f}")
    print(f"Convergence Step: {efficiency_metrics['convergence_step']}")
    print(f"Loss Reduction: {efficiency_metrics['loss_reduction']:.4f}")
    
    # 8. Test comprehensive evaluation
    print("\n--- Testing Comprehensive Evaluation ---")
    comprehensive_metrics = comprehensive_causal_evaluation(
        causal_factors,
        noise_factors,
        original_outcomes,
        counterfactual_outcomes,
        importance_scores,
        losses_over_time
    )
    
    print("Comprehensive Metrics:")
    for metric_name, value in comprehensive_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print("\n=== All tests completed successfully! ===")

