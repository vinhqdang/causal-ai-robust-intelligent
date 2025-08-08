import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
from collections import defaultdict

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
                                C_a: torch.Tensor, C_b: torch.Tensor,
                                test_loss_fn=None):
    """
    Computes the Causal Parameter Importance (CPI) for a given parameter.
    This is a simplified implementation of the influence function, where we
    approximate the Hessian-vector product with a gradient calculation.
    
    Args:
        model (nn.Module): The model being evaluated.
        param (torch.Tensor): The parameter for which to compute CPI.
        C_a (torch.Tensor): Input tensor representing the treatment group.
        C_b (torch.Tensor): Input tensor representing the control group.
        test_loss_fn (callable, optional): A function to compute the test loss. 
                                           If None, a dummy loss is used.
        
    Returns:
        float: The CPI value for the parameter.
    """
    if not param.requires_grad:
        print("Parameter does not require gradients.")
        return 0.0

    # 1. Compute the gradient of the treatment effect w.r.t. the parameter
    treatment_effect = compute_treatment_effect(model, C_a, C_b)
    s_test = grad(treatment_effect, param, retain_graph=True, allow_unused=True)[0]
    
    if s_test is None:
        return 0.0

    # 2. Compute the gradient of the test loss w.r.t. the parameter
    if test_loss_fn is None:
        # Dummy test loss
        test_loss = compute_treatment_effect(model, C_a, C_b)
    else:
        test_loss = test_loss_fn()
        
    test_loss_grads = grad(test_loss, param, retain_graph=True, allow_unused=True)[0]

    if test_loss_grads is None:
        return 0.0

    # 3. Compute the influence score (dot product of the two gradients)
    influence = -torch.dot(s_test.flatten(), test_loss_grads.flatten())
    
    return influence.item()


class FisherInformationMatrix:
    """
    Compute and manage Fisher Information Matrix for approximating Hessians.
    """
    
    def __init__(self, model: nn.Module, damping: float = 1e-3):
        self.model = model
        self.damping = damping
        self.fisher_dict = {}
        self.n_samples = 0
        
    def accumulate_fisher(self, loss: torch.Tensor, retain_graph: bool = True):
        """
        Accumulate Fisher information from a single loss.
        
        Args:
            loss: Scalar loss tensor
            retain_graph: Whether to retain computation graph
        """
        # Compute gradients
        grads = grad(loss, self.model.parameters(), 
                    retain_graph=retain_graph, create_graph=False)
        
        # Accumulate Fisher information (outer product of gradients)
        for (name, param), g in zip(self.model.named_parameters(), grads):
            if g is not None:
                if name not in self.fisher_dict:
                    self.fisher_dict[name] = torch.zeros_like(param)
                # Use diagonal approximation: F_ii = E[g_i^2]
                self.fisher_dict[name] += g.pow(2)
        
        self.n_samples += 1
    
    def finalize_fisher(self):
        """Finalize Fisher matrix by averaging and adding damping."""
        for name in self.fisher_dict:
            self.fisher_dict[name] /= self.n_samples
            self.fisher_dict[name] += self.damping
    
    def fisher_vector_product(self, vector_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher-vector product F * v.
        
        Args:
            vector_dict: Dictionary mapping parameter names to vectors
            
        Returns:
            Dictionary of Fisher-vector products
        """
        result = {}
        for name, vec in vector_dict.items():
            if name in self.fisher_dict:
                result[name] = self.fisher_dict[name] * vec
            else:
                result[name] = torch.zeros_like(vec)
        return result


class HessianVectorProduct:
    """
    Efficient computation of Hessian-Vector Products using automatic differentiation.
    """
    
    def __init__(self, model: nn.Module, loss_fn: Callable):
        self.model = model
        self.loss_fn = loss_fn
        
    def hvp(self, vector_dict: Dict[str, torch.Tensor], 
            *loss_args, **loss_kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute Hessian-Vector Product H * v.
        
        Args:
            vector_dict: Dictionary mapping parameter names to vectors
            *loss_args, **loss_kwargs: Arguments for loss function
            
        Returns:
            Dictionary of HVP results
        """
        # Compute loss and gradients
        loss = self.loss_fn(*loss_args, **loss_kwargs)
        grads = grad(loss, self.model.parameters(), create_graph=True)
        
        # Compute dot product with vector
        dot_product = 0
        for (name, param), g in zip(self.model.named_parameters(), grads):
            if name in vector_dict and g is not None:
                dot_product += torch.sum(g * vector_dict[name])
        
        # Compute second-order gradients
        hvp_grads = grad(dot_product, self.model.parameters(), retain_graph=False)
        
        # Package results
        result = {}
        for (name, param), hvp_grad in zip(self.model.named_parameters(), hvp_grads):
            if hvp_grad is not None:
                result[name] = hvp_grad
            else:
                result[name] = torch.zeros_like(param)
                
        return result


class CPIComputer:
    """
    Causal Parameter Importance computer with proper influence function implementation.
    """
    
    def __init__(self, model: nn.Module, use_fisher_approx: bool = True, 
                 damping: float = 1e-3, recursion_depth: int = 50):
        self.model = model
        self.use_fisher_approx = use_fisher_approx
        self.damping = damping
        self.recursion_depth = recursion_depth
        self.fisher = FisherInformationMatrix(model, damping) if use_fisher_approx else None
        
    def compute_s_test(self, test_loss_fn: Callable, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute s_test = H^{-1} * grad_test_loss using iterative methods.
        
        Args:
            test_loss_fn: Function that computes test loss
            *args, **kwargs: Arguments for test loss function
            
        Returns:
            Dictionary of s_test vectors
        """
        # Compute test loss gradient
        test_loss = test_loss_fn(*args, **kwargs)
        test_grads = grad(test_loss, self.model.parameters(), retain_graph=False)
        
        # Package gradients into dictionary
        grad_dict = {}
        for (name, param), g in zip(self.model.named_parameters(), test_grads):
            if g is not None:
                grad_dict[name] = g.detach()
            else:
                grad_dict[name] = torch.zeros_like(param)
        
        if self.use_fisher_approx:
            # Use Fisher approximation: s_test ≈ F^{-1} * grad_test
            return self._fisher_inverse_hvp(grad_dict)
        else:
            # Use exact Hessian (expensive)
            return self._hessian_inverse_hvp(grad_dict, test_loss_fn, *args, **kwargs)
    
    def _fisher_inverse_hvp(self, grad_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Approximate H^{-1} * v using Fisher matrix."""
        if self.fisher.n_samples == 0:
            # Fisher not computed yet, return scaled gradient
            return {name: g / self.damping for name, g in grad_dict.items()}
        
        result = {}
        for name, g in grad_dict.items():
            if name in self.fisher.fisher_dict:
                result[name] = g / self.fisher.fisher_dict[name]
            else:
                result[name] = g / self.damping
        return result
    
    def _hessian_inverse_hvp(self, grad_dict: Dict[str, torch.Tensor], 
                           loss_fn: Callable, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute H^{-1} * v using conjugate gradient method."""
        hvp_computer = HessianVectorProduct(self.model, loss_fn)
        
        # Initialize
        x = {name: torch.zeros_like(g) for name, g in grad_dict.items()}
        r = grad_dict.copy()
        p = grad_dict.copy()
        rsold = sum(torch.sum(r[name] * r[name]) for name in r.keys())
        
        # Conjugate gradient iterations
        for i in range(self.recursion_depth):
            Ap = hvp_computer.hvp(p, *args, **kwargs)
            # Add damping
            for name in Ap.keys():
                Ap[name] += self.damping * p[name]
            
            pAp = sum(torch.sum(p[name] * Ap[name]) for name in p.keys())
            alpha = rsold / (pAp + 1e-10)
            
            # Update x and residual
            for name in x.keys():
                x[name] = x[name] + alpha * p[name]
                r[name] = r[name] - alpha * Ap[name]
            
            rsnew = sum(torch.sum(r[name] * r[name]) for name in r.keys())
            
            if rsnew < 1e-10:
                break
                
            beta = rsnew / rsold
            for name in p.keys():
                p[name] = r[name] + beta * p[name]
                
            rsold = rsnew
        
        return x
    
    def compute_influence_score(self, s_test: Dict[str, torch.Tensor], 
                               train_loss_fn: Callable, *args, **kwargs) -> float:
        """
        Compute influence score: s_test^T * grad_train_loss.
        
        Args:
            s_test: s_test vectors from compute_s_test
            train_loss_fn: Function that computes training loss for specific example
            
        Returns:
            Scalar influence score
        """
        train_loss = train_loss_fn(*args, **kwargs)
        train_grads = grad(train_loss, self.model.parameters(), retain_graph=False)
        
        influence = 0.0
        for (name, param), train_grad in zip(self.model.named_parameters(), train_grads):
            if train_grad is not None and name in s_test:
                influence += torch.sum(s_test[name] * train_grad).item()
        
        return -influence  # Negative sign for influence
    
    def compute_parameter_importance(self, train_loss_fn: Callable, 
                                   test_loss_fn: Callable,
                                   train_args: tuple = (), train_kwargs: dict = {},
                                   test_args: tuple = (), test_kwargs: dict = {},
                                   ) -> Dict[str, float]:
        """
        Compute parameter importance scores for all parameters.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        # Compute s_test
        s_test = self.compute_s_test(test_loss_fn, *test_args, **test_kwargs)
        
        # Compute influence score
        total_influence = self.compute_influence_score(s_test, train_loss_fn, 
                                                     *train_args, **train_kwargs)
        
        # Compute parameter-wise importance (magnitude of s_test components)
        importance_scores = {}
        for name, s_vec in s_test.items():
            importance_scores[name] = torch.norm(s_vec).item()
        
        return importance_scores
    
    def get_lambda_weights(self, importance_scores: Dict[str, float], 
                         base_lambda: float = 1.0, 
                         scaling: str = 'linear') -> Dict[str, float]:
        """
        Convert importance scores to regularization weights λ_j.
        
        Args:
            importance_scores: Parameter importance scores
            base_lambda: Base regularization strength
            scaling: Scaling method ('linear', 'exp', 'rank')
            
        Returns:
            Dictionary of lambda weights for each parameter
        """
        if not importance_scores:
            return {}
        
        scores = np.array(list(importance_scores.values()))
        names = list(importance_scores.keys())
        
        if scaling == 'linear':
            # Linear scaling
            max_score = scores.max()
            if max_score > 0:
                normalized = scores / max_score
            else:
                normalized = np.ones_like(scores)
        elif scaling == 'exp':
            # Exponential scaling
            normalized = np.exp(scores - scores.max())
        elif scaling == 'rank':
            # Rank-based scaling
            ranks = np.argsort(np.argsort(scores))
            normalized = ranks / len(ranks)
        else:
            normalized = np.ones_like(scores)
        
        lambda_weights = {}
        for name, norm_score in zip(names, normalized):
            lambda_weights[name] = base_lambda * norm_score
            
        return lambda_weights

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

    # 2. Create dummy data
    batch_size = 32
    train_data = torch.randn(batch_size, 10)
    train_targets = torch.randn(batch_size, 1)
    test_data = torch.randn(batch_size, 10)
    test_targets = torch.randn(batch_size, 1)

    # 3. Define loss functions
    def train_loss_fn():
        outputs = model(train_data)
        return F.mse_loss(outputs, train_targets)
    
    def test_loss_fn():
        outputs = model(test_data)
        return F.mse_loss(outputs, test_targets)

    # 4. Create CPI computer
    print("=== Testing Advanced CPI Implementation ===")
    cpi_computer = CPIComputer(model, use_fisher_approx=True, damping=1e-3)

    # 5. Accumulate Fisher information
    print("\nAccumulating Fisher information...")
    for _ in range(10):  # Multiple samples for Fisher
        loss = train_loss_fn()
        cpi_computer.fisher.accumulate_fisher(loss)
    cpi_computer.fisher.finalize_fisher()
    print(f"Fisher computed over {cpi_computer.fisher.n_samples} samples")

    # 6. Compute parameter importance
    print("\nComputing parameter importance...")
    importance_scores = cpi_computer.compute_parameter_importance(
        train_loss_fn, test_loss_fn
    )
    
    print("\nParameter Importance Scores:")
    for name, score in importance_scores.items():
        print(f"  {name}: {score:.6f}")

    # 7. Get lambda weights for regularization
    lambda_weights = cpi_computer.get_lambda_weights(importance_scores, base_lambda=1.0)
    
    print("\nRegularization Weights (λ):")
    for name, weight in lambda_weights.items():
        print(f"  {name}: {weight:.6f}")

    # 8. Test legacy CPI function for comparison
    print("\n=== Testing Legacy CPI Implementation ===")
    treatment_data = torch.randn(batch_size, 10)
    control_data = torch.randn(batch_size, 10)

    target_param = model.linear1.weight
    cpi_value = causal_parameter_importance(model, target_param, treatment_data, control_data)
    print(f"Legacy CPI for 'linear1.weight': {cpi_value:.6f}")

    # 9. Demonstrate parameter freezing based on importance
    print("\n=== Parameter Freezing Example ===")
    importance_threshold = 0.5
    
    for name, score in importance_scores.items():
        if score > importance_threshold:
            print(f"Parameter '{name}' (importance: {score:.4f}) is highly important - should be regularized strongly")
        else:
            print(f"Parameter '{name}' (importance: {score:.4f}) has low importance - can use lighter regularization")
