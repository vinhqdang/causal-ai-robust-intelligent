import torch

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
    return torch.norm(new_effects - old_effects) / torch.norm(base_effects)

if __name__ == '__main__':
    # Example Usage
    
    old_effects = torch.randn(10)
    new_effects = old_effects + torch.randn(10) * 0.1 # Simulate a small change
    base_effects = torch.randn(10)
    
    ccs = causal_consistency_score(old_effects, new_effects)
    fate = fate_score(old_effects, new_effects, base_effects)
    
    print(f"Causal Consistency Score: {ccs:.4f}")
    print(f"FATE Score: {fate:.4f}")

