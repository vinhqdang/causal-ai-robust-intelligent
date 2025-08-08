import torch
from collections import defaultdict

class CounterfactualReplayBuffer:
    """
    A buffer to store summaries of parent-child relationships from a causal graph
    and generate counterfactuals for computing the causal consistency loss.
    """
    def __init__(self):
        """
        Initializes the CounterfactualReplayBuffer.
        The buffer stores data in a dictionary where keys are causal edges.
        """
        self.buffer = defaultdict(list)
        print("CounterfactualReplayBuffer initialized.")

    def add(self, edge: str, parent_config: torch.Tensor, outcome: torch.Tensor):
        """
        Adds a new parent-outcome summary to the buffer.

        Args:
            edge (str): The causal edge name (e.g., 'X->Y').
            parent_config (torch.Tensor): A tensor representing the parent variable's state.
            outcome (torch.Tensor): A tensor representing the outcome.
        """
        # For simplicity, we store the tensors directly.
        # In a real implementation, these would be low-rank summaries.
        self.buffer[edge].append((parent_config.clone(), outcome.clone()))

    def sample(self, edge: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a batch of parent-outcome pairs for a given edge.

        Args:
            edge (str): The causal edge to sample from.
            batch_size (int): The number of samples to return.

        Returns:
            A tuple of tensors (parent_configs, outcomes).
        """
        if not self.buffer[edge]:
            return None, None
            
        indices = torch.randint(0, len(self.buffer[edge]), (batch_size,))
        
        parent_configs = torch.stack([self.buffer[edge][i][0] for i in indices])
        outcomes = torch.stack([self.buffer[edge][i][1] for i in indices])
        
        return parent_configs, outcomes

    def generate_counterfactual(self, model, edge: str, parent_config: torch.Tensor, new_val: float):
        """
        Generates a counterfactual outcome for a given parent configuration.
        This is a simplified placeholder.
        """
        # In a real implementation, this would involve a more sophisticated
        # counterfactual generation process.
        counterfactual_parent = parent_config.clone()
        counterfactual_parent[0] = new_val # do(C_a = new_val)
        
        with torch.no_grad():
            counterfactual_outcome = model(counterfactual_parent)
            
        return counterfactual_outcome

if __name__ == '__main__':
    # Example Usage
    
    # 1. Initialize the buffer
    crb = CounterfactualReplayBuffer()

    # 2. Define a causal edge and some data
    edge_name = "treatment->outcome"
    num_samples = 100
    parent_dim = 10
    
    for _ in range(num_samples):
        parent = torch.randn(1, parent_dim)
        outcome = torch.randn(1, 1) # Assuming a single-dimensional outcome
        crb.add(edge_name, parent, outcome)

    print(f"Added {len(crb.buffer[edge_name])} samples for edge '{edge_name}'.")

    # 3. Sample from the buffer
    batch_size = 16
    parent_batch, outcome_batch = crb.sample(edge_name, batch_size)
    
    if parent_batch is not None:
        print(f"\nSampled parent batch shape: {parent_batch.shape}")
        print(f"Sampled outcome batch shape: {outcome_batch.shape}")

    # 4. Generate a counterfactual
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(parent_dim, 1)
        def forward(self, x):
            return self.linear(x)
            
    dummy_model = SimpleModel()
    
    # Take one of the sampled parents
    example_parent = parent_batch[0]
    
    counterfactual_outcome = crb.generate_counterfactual(
        model=dummy_model,
        edge=edge_name,
        parent_config=example_parent,
        new_val=5.0
    )
    
    original_outcome = dummy_model(example_parent)

    print(f"\nOriginal parent config: {example_parent}")
    print(f"Original outcome: {original_outcome.item()}")
    print(f"Counterfactual outcome (when first element is 5.0): {counterfactual_outcome.item()}")
