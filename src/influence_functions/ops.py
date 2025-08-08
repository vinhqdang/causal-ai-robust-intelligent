import torch
import torch.nn as nn
from torch.autograd import grad

def compute_treatment_effect(model: nn.Module, C_a: torch.Tensor, C_b: torch.Tensor):
    """
    A placeholder function to compute the average treatment effect (ATE).
    This is a simplified version and would be more complex in a real scenario.
    ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
    Here, we'll just use the model's prediction as a proxy for the outcome.
    """
    # In a real implementation, this would involve a more rigorous
    # estimation of the treatment effect.
    # For now, we'll just return the difference in model output.
    output_a = model(C_a)
    if hasattr(output_a, 'last_hidden_state'):
        output_a = output_a.last_hidden_state
    elif isinstance(output_a, tuple):
        output_a = output_a[0]
        
    output_b = model(C_b)
    if hasattr(output_b, 'last_hidden_state'):
        output_b = output_b.last_hidden_state
    elif isinstance(output_b, tuple):
        output_b = output_b[0]
        
    return output_a.mean() - output_b.mean()

def causal_parameter_importance(model: nn.Module, param: torch.Tensor, 
                                C_a: torch.Tensor, C_b: torch.Tensor):
    """
    Computes the Causal Parameter Importance (CPI) for a given parameter.
    CPI is defined as the squared norm of the gradient of the treatment effect
    with respect to the parameter.
    
    Args:
        model (nn.Module): The model being evaluated.
        param (torch.Tensor): The parameter for which to compute CPI.
        C_a (torch.Tensor): Input tensor representing the treatment group.
        C_b (torch.Tensor): Input tensor representing the control group.
        
    Returns:
        float: The CPI value for the parameter.
    """
    if not param.requires_grad:
        print("Parameter does not require gradients.")
        return 0.0

    # 1. Compute the treatment effect
    treatment_effect = compute_treatment_effect(model, C_a, C_b)
    
    # 2. Compute the gradient of the treatment effect w.r.t. the parameter
    grads = grad(treatment_effect, param, retain_graph=True, allow_unused=True)
    
    if grads[0] is None:
        return 0.0
        
    # 3. Compute the squared norm of the gradient
    cpi = torch.norm(grads[0])**2
    
    return cpi.item()

if __name__ == '__main__':
    # Example Usage

    # 1. Define a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 1)
        
        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    model = SimpleModel()

    # 2. Create dummy data for treatment and control groups
    batch_size = 64
    treatment_data = torch.randn(batch_size, 10)
    control_data = torch.randn(batch_size, 10)

    # 3. Compute CPI for a specific parameter
    target_param = model.linear1.weight
    
    cpi_value = causal_parameter_importance(model, target_param, treatment_data, control_data)
    
    print(f"CPI for 'linear1.weight': {cpi_value}")

    # 4. Compute CPI for another parameter
    target_param_2 = model.linear2.weight
    cpi_value_2 = causal_parameter_importance(model, target_param_2, treatment_data, control_data)
    print(f"CPI for 'linear2.weight': {cpi_value_2}")

    # 5. Example of how you might use CPI to freeze parameters
    importance_threshold = 0.1
    
    if cpi_value > importance_threshold:
        print("'linear1.weight' is causally important and should be frozen or regularized.")
        target_param.requires_grad = False # Example of freezing
    
    print(f"requires_grad for 'linear1.weight' after check: {target_param.requires_grad}")
