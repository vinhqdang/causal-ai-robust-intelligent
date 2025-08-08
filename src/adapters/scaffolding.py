import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib
import json
from datetime import datetime

class CosineGatingNetwork(nn.Module):
    """
    Cosine-key gating mechanism for selecting appropriate adapters.
    """
    def __init__(self, key_dim: int = 128):
        super().__init__()
        self.key_dim = key_dim
        self.keys = nn.ParameterDict()
        
    def register_adapter_key(self, edge_name: str, key: Optional[torch.Tensor] = None):
        """Register a key for a specific adapter."""
        if key is None:
            key = torch.randn(self.key_dim)
        self.keys[edge_name] = nn.Parameter(key / key.norm())
        
    def forward(self, query: torch.Tensor, temperature: float = 1.0) -> Dict[str, float]:
        """
        Compute cosine similarities and return attention weights.
        
        Args:
            query: Query vector [batch_size, key_dim] or [key_dim]
            temperature: Temperature for softmax
            
        Returns:
            Dict mapping adapter names to attention weights
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = query / query.norm(dim=-1, keepdim=True)
        
        similarities = {}
        for edge_name, key in self.keys.items():
            sim = F.cosine_similarity(query, key.unsqueeze(0), dim=-1)
            similarities[edge_name] = sim
        
        if not similarities:
            return {}
            
        # Convert to attention weights
        sim_tensor = torch.stack(list(similarities.values()))
        weights = F.softmax(sim_tensor / temperature, dim=0)
        
        return {edge: weight.item() for edge, weight in zip(similarities.keys(), weights)}


class AdapterMetadata:
    """Metadata for tracking adapter changes in append-only fashion."""
    
    def __init__(self):
        self.history = []
        
    def add_entry(self, edge_name: str, action: str, parameters: Dict, timestamp: Optional[str] = None):
        """Add an entry to the append-only log."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        entry = {
            'timestamp': timestamp,
            'edge_name': edge_name,
            'action': action,
            'parameters': parameters,
            'hash': self._compute_hash(edge_name, action, parameters, timestamp)
        }
        self.history.append(entry)
        
    def _compute_hash(self, edge_name: str, action: str, parameters: Dict, timestamp: str) -> str:
        """Compute SHA-256 hash of the entry."""
        content = f"{timestamp}:{edge_name}:{action}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
        
    def get_history(self) -> List[Dict]:
        """Get the full history (read-only)."""
        return self.history.copy()


class StructuralCausalMemory:
    """
    Advanced adapter management system for Structural Causal Models with:
    - PEFT LoRA integration
    - Cosine-key gating
    - Append-only metadata storage
    """
    
    def __init__(self, model_name: str, model=None, key_dim: int = 128):
        """
        Initialize the SCM memory system.
        
        Args:
            model_name: HuggingFace model name
            model: Optional existing model instance
            key_dim: Dimension for cosine gating keys
        """
        self.model_name = model_name  # Store model name for reference
        
        if model:
            self.base_model = model
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Adapter management
        self.adapters: Dict[str, PeftModel] = {}
        self.adapter_configs: Dict[str, LoraConfig] = {}
        
        # Gating network
        self.gating_network = CosineGatingNetwork(key_dim)
        
        # Metadata tracking
        self.metadata = AdapterMetadata()
        
        # Current active adapters
        self.active_adapters: Dict[str, float] = {}
        
        print(f"StructuralCausalMemory initialized with model: {model_name}")

    def add_adapter_for_edge(self, edge_name: str, 
                           r: int = 16, 
                           alpha: int = 32,
                           target_modules: Optional[List[str]] = None,
                           key: Optional[torch.Tensor] = None):
        """
        Add a LoRA adapter for a specific causal edge.
        
        Args:
            edge_name: Unique identifier for the causal edge
            r: LoRA rank
            alpha: LoRA alpha parameter
            target_modules: Target modules for LoRA (None for auto-detection)
            key: Optional custom key for gating (None for random)
        """
        if edge_name in self.adapters:
            print(f"Adapter '{edge_name}' already exists.")
            return
            
        # Create LoRA configuration with model-specific target modules
        if target_modules is None:
            # GPT-2 specific target modules
            if "gpt2" in self.model_name.lower():
                target_modules = ["c_attn", "c_proj", "c_fc"]  # GPT-2 attention and MLP modules
            else:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Standard transformer modules
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Create PEFT model
        peft_model = get_peft_model(self.base_model, lora_config)
        
        # Store adapter
        self.adapters[edge_name] = peft_model
        self.adapter_configs[edge_name] = lora_config
        
        # Register gating key
        self.gating_network.register_adapter_key(edge_name, key)
        
        # Log metadata
        self.metadata.add_entry(
            edge_name=edge_name,
            action="create_adapter",
            parameters={
                "r": r,
                "alpha": alpha,
                "target_modules": target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        )
        
        print(f"Added LoRA adapter for edge: '{edge_name}' (r={r}, alpha={alpha})")

    def route(self, query: torch.Tensor, temperature: float = 1.0, top_k: int = 1) -> Dict[str, float]:
        """
        Route query through gating network to select adapters.
        
        Args:
            query: Query vector for adapter selection
            temperature: Gating temperature
            top_k: Number of top adapters to activate
            
        Returns:
            Dictionary mapping adapter names to weights
        """
        if not self.adapters:
            return {}
            
        # Get similarity scores
        similarities = self.gating_network(query, temperature)
        
        # Select top-k adapters
        sorted_adapters = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        selected_adapters = dict(sorted_adapters[:top_k])
        
        # Update active adapters
        self.active_adapters = selected_adapters
        
        return selected_adapters

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                query: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass through the model with adapter routing.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            query: Query for adapter selection (if None, use input embedding mean)
            **kwargs: Additional arguments for the model
            
        Returns:
            Model outputs
        """
        # If no query provided, use mean of input embeddings
        if query is None and self.adapters:
            with torch.no_grad():
                embeddings = self.base_model.get_input_embeddings()(input_ids)
                query = embeddings.mean(dim=1).mean(dim=0)  # [hidden_dim]
        
        # Route to select adapters
        if query is not None and self.adapters:
            selected_adapters = self.route(query)
            
            if selected_adapters:
                # For simplicity, use the top adapter
                top_adapter = max(selected_adapters.items(), key=lambda x: x[1])
                adapter_name = top_adapter[0]
                
                # Use the selected adapter
                adapter_model = self.adapters[adapter_name]
                return adapter_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Fallback to base model
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate_text(self, prompt: str, max_length: int = 50, **kwargs) -> str:
        """Generate text using the routed model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.forward(**inputs, **kwargs)
            
        # Generate continuation
        generated = self.base_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_length=max_length,
            do_sample=True,
            temperature=0.8,
            **kwargs
        )
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def get_adapter_parameters(self, edge_name: str) -> Dict[str, torch.Tensor]:
        """Get parameters of a specific adapter."""
        if edge_name not in self.adapters:
            return {}
            
        adapter_params = {}
        for name, param in self.adapters[edge_name].named_parameters():
            if 'lora' in name:
                adapter_params[name] = param.data.clone()
        return adapter_params

    def get_metadata_history(self) -> List[Dict]:
        """Get the append-only metadata history."""
        return self.metadata.get_history()

if __name__ == '__main__':
    # Example Usage
    
    # 1. Initialize the StructuralCausalMemory
    model_name = "gpt2"
    scm_memory = StructuralCausalMemory(model_name)

    # 2. Define causal edges
    causal_edges = ["treatment->outcome", "confounder->treatment", "outcome->feedback"]

    # 3. Add LoRA adapters for each edge
    for edge in causal_edges:
        scm_memory.add_adapter_for_edge(edge, r=8, alpha=16)

    # 4. Test routing with different queries
    print("\n--- Testing Adapter Routing ---")
    query_vector = torch.randn(128)  # Random query
    selected_adapters = scm_memory.route(query_vector, temperature=1.0, top_k=2)
    print(f"Selected adapters: {selected_adapters}")

    # 5. Generate text with adapter routing
    prompt = "The treatment had a significant effect"
    generated_text = scm_memory.generate_text(prompt, max_length=30)
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated text: '{generated_text}'")

    # 6. Show adapter parameters
    if causal_edges:
        edge_name = causal_edges[0]
        params = scm_memory.get_adapter_parameters(edge_name)
        print(f"\nAdapter '{edge_name}' has {len(params)} LoRA parameters")

    # 7. Show metadata history
    print("\n--- Adapter Metadata History ---")
    history = scm_memory.get_metadata_history()
    for entry in history:
        print(f"[{entry['timestamp']}] {entry['action']} for {entry['edge_name']}")
        print(f"  Hash: {entry['hash'][:16]}...")

    # 8. Test gating network directly
    print("\n--- Testing Gating Network ---")
    if scm_memory.gating_network.keys:
        test_query = torch.randn(128)
        similarities = scm_memory.gating_network(test_query, temperature=0.5)
        print(f"Gating similarities: {similarities}")
