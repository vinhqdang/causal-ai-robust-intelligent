import adapters
from transformers import AutoTokenizer, AutoModelForCausalLM
from adapters import AdapterConfig

class StructuralCausalMemory:
    """
    A class to manage adapters in a foundation model, where each adapter
    corresponds to a causal edge in a Structural Causal Model (SCM).
    """
    def __init__(self, model_name: str, model=None):
        """
        Initializes the StructuralCausalMemory.

        Args:
            model_name (str): The name of the pre-trained model from Hugging Face.
            model (optional): An existing model instance. If None, a new model is loaded.
        """
        if model:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Initialize adapters for the model
        adapters.init(self.model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.adapters = {}
        print(f"StructuralCausalMemory initialized with model: {model_name}")

    def add_adapter_for_edge(self, edge_name: str, config: str = 'lora'):
        """
        Adds a new adapter for a specific causal edge.

        Args:
            edge_name (str): A unique name for the causal edge (e.g., 'X->Y').
            config (str): The configuration for the adapter (e.g., 'lora').
        """
        if edge_name in self.model.adapters_config.adapters:
            print(f"Adapter '{edge_name}' already exists.")
            return

        # Configure the adapter
        adapter_config = AdapterConfig.load(config, reduction_factor=16)
        
        # Add the adapter to the model
        self.model.add_adapter(edge_name, config=adapter_config)
        self.model.train_adapter(edge_name)
        self.adapters[edge_name] = adapter_config
        print(f"Added and activated adapter for edge: '{edge_name}'")

    def set_active_adapter(self, edge_name: str):
        """
        Sets the active adapter for the model.

        Args:
            edge_name (str): The name of the edge adapter to activate.
        """
        if edge_name not in self.adapters:
            print(f"Adapter for edge '{edge_name}' not found.")
            return
        
        self.model.set_active_adapters(edge_name)
        print(f"Active adapter set to: '{edge_name}'")

    def forward(self, text: str):
        """
        A simple forward pass to demonstrate adapter usage.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    # Example Usage
    
    # 1. Initialize the StructuralCausalMemory with a small model for demonstration
    model_name = "gpt2"
    scm_memory = StructuralCausalMemory(model_name)

    # 2. Define some causal edges
    causal_edges = ["treatment->outcome", "confounder->treatment"]

    # 3. Add adapters for each edge
    for edge in causal_edges:
        scm_memory.add_adapter_for_edge(edge)

    # 4. List the added adapters
    print("\nAvailable adapters:")
    print(scm_memory.model.adapters_config.to_dict())

    # 5. Set an active adapter and perform a forward pass
    active_edge = "treatment->outcome"
    scm_memory.set_active_adapter(active_edge)
    
    # 6. Example forward pass
    prompt = "Once upon a time"
    generated_text = scm_memory.forward(prompt)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated text with '{active_edge}' adapter: '{generated_text}'")

    # 7. Switch to another adapter
    active_edge_2 = "confounder->treatment"
    scm_memory.set_active_adapter(active_edge_2)
    generated_text_2 = scm_memory.forward(prompt)
    print(f"Generated text with '{active_edge_2}' adapter: '{generated_text_2}'")
