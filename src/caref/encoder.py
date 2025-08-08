import torch
import torch.nn as nn

class BifurcatedMLPHead(nn.Module):
    """
    A bifurcated MLP head that takes features from a foundation model
    and projects them into causal and noise/contextual factors.
    """
    def __init__(self, input_dim: int, causal_dim: int, noise_dim: int, hidden_dim: int = 512):
        """
        Initializes the BifurcatedMLPHead.

        Args:
            input_dim (int): The dimension of the input features (e.g., from the FM).
            causal_dim (int): The dimension of the causal factor space.
            noise_dim (int): The dimension of the noise/contextual factor space.
            hidden_dim (int): The dimension of the hidden layers in the MLPs.
        """
        super().__init__()
        self.causal_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, causal_dim)
        )
        self.noise_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, noise_dim)
        )
        print("BifurcatedMLPHead initialized.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the BifurcatedMLPHead.

        Args:
            x (torch.Tensor): The input tensor from the foundation model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the causal factors
                                               and the noise/contextual factors.
        """
        causal_factors = self.causal_projector(x)
        noise_factors = self.noise_projector(x)
        return causal_factors, noise_factors

class CaReFEncoder(nn.Module):
    """
    Wraps a foundation model and attaches the BifurcatedMLPHead to its output.
    This model is intended to be frozen after initial pre-training.
    """
    def __init__(self, foundation_model: nn.Module, input_dim: int, causal_dim: int, noise_dim: int):
        """
        Initializes the CaReFEncoder.

        Args:
            foundation_model (nn.Module): The pre-trained foundation model.
            input_dim (int): The output dimension of the foundation model.
            causal_dim (int): The dimension of the causal factor space.
            noise_dim (int): The dimension of the noise/contextual factor space.
        """
        super().__init__()
        self.foundation_model = foundation_model
        self.bifurcated_head = BifurcatedMLPHead(input_dim, causal_dim, noise_dim)
        print("CaReFEncoder initialized.")

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CaReFEncoder.

        Args:
            *args: Positional arguments to be passed to the foundation model.
            **kwargs: Keyword arguments to be passed to the foundation model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the causal factors
                                               and the noise/contextual factors.
        """
        # We assume the foundation model returns a single tensor of features.
        # If it returns a tuple, you might need to adjust this.
        features = self.foundation_model(*args, **kwargs)
        
        # If the output is a transformer output object, get the last hidden state
        if hasattr(features, 'last_hidden_state'):
            features = features.last_hidden_state
        # If the output is a tuple (e.g., from Hugging Face models), take the last hidden state.
        elif isinstance(features, tuple):
            features = features[0]

        return self.bifurcated_head(features)

    def __getattr__(self, name):
        """
        Delegates attribute access to the foundation model.
        This is useful for methods and attributes that are not explicitly
        wrapped by CaReFEncoder, but are required by libraries like `adapters`.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.foundation_model, name)

def d_separation_contrastive_loss(C: torch.Tensor, N: torch.Tensor, margin: float = 1.0):
    """
    A simple d-separation contrastive loss.
    This is a placeholder for a more sophisticated implementation.
    The goal is to make C and N independent. A simple way is to
    minimize the cosine similarity between them.
    """
    # Normalize the tensors to prevent the loss from exploding
    c_norm = C / C.norm(dim=1, keepdim=True)
    n_norm = N / N.norm(dim=1, keepdim=True)
    
    # Cosine similarity
    similarity = torch.einsum("...i,...i->...", c_norm, n_norm)
    
    # We want to push similarity to 0.
    # A simple L1 loss on the similarity.
    loss = torch.mean(torch.abs(similarity))
    return loss

if __name__ == '__main__':
    from transformers import GPT2Model, GPT2Config
    # Example Usage
    
    # 1. Use a real foundation model
    config = GPT2Config.from_pretrained("gpt2")
    dummy_fm = GPT2Model(config)

    # 2. Instantiate the CaReFEncoder
    causal_dim = 128
    noise_dim = 64
    
    caref_encoder = CaReFEncoder(
        foundation_model=dummy_fm,
        input_dim=config.n_embd,
        causal_dim=causal_dim,
        noise_dim=noise_dim
    )

    # 3. Create some dummy data (token IDs)
    input_data = torch.randint(0, config.vocab_size, (1, 10))

    # 4. Forward pass
    causal_factors, noise_factors = caref_encoder(input_data)

    # 5. Calculate the d-separation loss
    loss = d_separation_contrastive_loss(causal_factors, noise_factors)

    print(f"Input shape: {input_data.shape}")
    print(f"Causal factors shape: {causal_factors.shape}")
    print(f"Noise factors shape: {noise_factors.shape}")
    print(f"D-separation loss: {loss.item()}")

    # Example of backpropagation
    loss.backward()
    print("Backward pass successful.")
    for name, param in caref_encoder.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name} has shape: {param.grad.shape}")

