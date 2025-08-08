import json
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from pathlib import Path

@dataclass
class AuditEntry:
    """Single audit entry with complete metadata."""
    entry_id: str
    timestamp: str
    action_type: str  # 'parameter_update', 'model_checkpoint', 'gradient_step', etc.
    component: str  # e.g., 'caref_encoder', 'adapter_X->Y', 'cpi_computer'
    metadata: Dict[str, Any]
    content_hash: str  # SHA-256 of the actual content
    previous_hash: str  # Hash of previous entry for chaining
    merkle_root: str  # Root of Merkle tree for this batch
    signature: Optional[str] = None  # For cryptographic signing


class ParameterHasher:
    """Efficient hashing of PyTorch model parameters."""
    
    @staticmethod
    def hash_tensor(tensor: torch.Tensor, precision: int = 8) -> str:
        """Hash a tensor with controlled precision to avoid floating-point issues."""
        # Convert to numpy and round to specified precision
        rounded = torch.round(tensor * (10 ** precision)) / (10 ** precision)
        tensor_bytes = rounded.detach().cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()
    
    @staticmethod
    def hash_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """Hash all tensors in a state dict."""
        hashes = {}
        for name, tensor in state_dict.items():
            hashes[name] = ParameterHasher.hash_tensor(tensor)
        return hashes
    
    @staticmethod
    def hash_parameter_delta(old_state: Dict[str, torch.Tensor], 
                           new_state: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """Hash the differences between parameter states."""
        delta_hashes = {}
        for name in new_state.keys():
            if name in old_state:
                delta = new_state[name] - old_state[name]
                delta_hashes[name] = ParameterHasher.hash_tensor(delta)
            else:
                delta_hashes[name] = ParameterHasher.hash_tensor(new_state[name])
        return delta_hashes


class MerkleTree:
    """Simple Merkle tree implementation for batch integrity."""
    
    @staticmethod
    def hash_pair(left: str, right: str) -> str:
        """Hash a pair of strings."""
        combined = (left + right).encode('utf-8')
        return hashlib.sha256(combined).hexdigest()
    
    @staticmethod
    def build_tree(hashes: List[str]) -> str:
        """Build Merkle tree and return root hash."""
        if not hashes:
            return hashlib.sha256(b'').hexdigest()
        
        if len(hashes) == 1:
            return hashes[0]
        
        # Make even number of hashes
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])  # Duplicate last hash
        
        next_level = []
        for i in range(0, len(hashes), 2):
            next_level.append(MerkleTree.hash_pair(hashes[i], hashes[i + 1]))
        
        return MerkleTree.build_tree(next_level)


class CryptoAuditLedger:
    """
    Advanced audit ledger with:
    - SHA-256 hash chaining
    - Merkle tree batching
    - Parameter delta tracking
    - Cryptographic integrity
    """
    
    def __init__(self, ledger_dir: str = "artefacts", 
                 batch_size: int = 100,
                 auto_merkle: bool = True):
        """
        Initialize the cryptographic audit ledger.
        
        Args:
            ledger_dir: Directory to store ledger files
            batch_size: Number of entries before creating Merkle root
            auto_merkle: Whether to automatically create Merkle roots
        """
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(exist_ok=True)
        
        self.ledger_file = self.ledger_dir / "ledger.jsonl"
        self.hash_chain_file = self.ledger_dir / "hash_chain.json"
        self.merkle_file = self.ledger_dir / "merkle_roots.json"
        
        self.batch_size = batch_size
        self.auto_merkle = auto_merkle
        
        # Load existing state
        self.entry_count = 0
        self.last_hash = self._load_last_hash()
        self.current_batch: List[str] = []
        
        # Parameter tracking
        self.component_states: Dict[str, Dict[str, torch.Tensor]] = {}
        
        print(f"CryptoAuditLedger initialized in {ledger_dir}")
    
    def _load_last_hash(self) -> str:
        """Load the last hash from the chain for continuity."""
        if self.hash_chain_file.exists():
            try:
                with open(self.hash_chain_file, 'r') as f:
                    chain_data = json.load(f)
                    self.entry_count = chain_data.get('entry_count', 0)
                    return chain_data.get('last_hash', '')
            except:
                pass
        return ''  # Genesis hash
    
    def _save_hash_chain_state(self):
        """Save current hash chain state."""
        chain_data = {
            'last_hash': self.last_hash,
            'entry_count': self.entry_count,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        with open(self.hash_chain_file, 'w') as f:
            json.dump(chain_data, f, indent=2)
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"audit_{timestamp}_{self.entry_count:06d}"
    
    def _compute_content_hash(self, action_type: str, component: str, 
                            metadata: Dict[str, Any]) -> str:
        """Compute hash of entry content."""
        content = {
            'action_type': action_type,
            'component': component,
            'metadata': metadata
        }
        content_str = json.dumps(content, sort_keys=True).encode('utf-8')
        return hashlib.sha256(content_str).hexdigest()
    
    def log_parameter_update(self, component: str, model: nn.Module, 
                           gradient_info: Optional[Dict] = None,
                           loss_info: Optional[Dict] = None):
        """
        Log a parameter update with full state tracking.
        
        Args:
            component: Name of the component being updated
            model: PyTorch model with parameters
            gradient_info: Optional gradient information
            loss_info: Optional loss information
        """
        current_state = model.state_dict()
        
        # Compute parameter hashes
        param_hashes = ParameterHasher.hash_state_dict(current_state)
        
        # Compute delta if we have previous state
        delta_hashes = {}
        if component in self.component_states:
            delta_hashes = ParameterHasher.hash_parameter_delta(
                self.component_states[component], current_state
            )
        
        # Store current state
        self.component_states[component] = {
            name: param.detach().clone() for name, param in current_state.items()
        }
        
        # Build metadata
        metadata = {
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'parameter_hashes': param_hashes,
            'delta_hashes': delta_hashes,
            'model_architecture': str(type(model).__name__)
        }
        
        if gradient_info:
            metadata['gradient_info'] = gradient_info
        if loss_info:
            metadata['loss_info'] = loss_info
        
        self._add_entry('parameter_update', component, metadata)
    
    def log_gradient_step(self, component: str, optimizer_state: Dict,
                         loss_value: float, gradient_norm: float):
        """Log gradient step information."""
        metadata = {
            'loss_value': loss_value,
            'gradient_norm': gradient_norm,
            'optimizer_state': optimizer_state,
            'step_type': 'gradient_descent'
        }
        self._add_entry('gradient_step', component, metadata)
    
    def log_model_checkpoint(self, component: str, checkpoint_path: str,
                           model_metrics: Optional[Dict] = None):
        """Log model checkpoint creation."""
        metadata = {
            'checkpoint_path': checkpoint_path,
            'file_size_bytes': os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0
        }
        
        if model_metrics:
            metadata['metrics'] = model_metrics
        
        self._add_entry('model_checkpoint', component, metadata)
    
    def log_causal_intervention(self, edge: str, intervention_type: str,
                              intervention_params: Dict):
        """Log causal interventions."""
        metadata = {
            'edge': edge,
            'intervention_type': intervention_type,
            'intervention_params': intervention_params
        }
        self._add_entry('causal_intervention', f"edge_{edge}", metadata)
    
    def log_evaluation_result(self, component: str, metric_name: str,
                            metric_value: float, evaluation_context: Dict):
        """Log evaluation results."""
        metadata = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'evaluation_context': evaluation_context
        }
        self._add_entry('evaluation_result', component, metadata)
    
    def _add_entry(self, action_type: str, component: str, metadata: Dict[str, Any]):
        """Add entry to the ledger with full integrity checks."""
        entry_id = self._generate_entry_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Compute content hash
        content_hash = self._compute_content_hash(action_type, component, metadata)
        
        # Create entry
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            action_type=action_type,
            component=component,
            metadata=metadata,
            content_hash=content_hash,
            previous_hash=self.last_hash,
            merkle_root=""  # Will be filled when batch is complete
        )
        
        # Update chain
        entry_hash = self._hash_entry(entry)
        self.current_batch.append(entry_hash)
        self.last_hash = entry_hash
        self.entry_count += 1
        
        # Write to ledger
        with open(self.ledger_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
        
        # Check if batch is complete
        if self.auto_merkle and len(self.current_batch) >= self.batch_size:
            self._finalize_batch()
        
        # Save chain state
        self._save_hash_chain_state()
    
    def _hash_entry(self, entry: AuditEntry) -> str:
        """Compute hash of an audit entry."""
        entry_dict = asdict(entry)
        entry_str = json.dumps(entry_dict, sort_keys=True).encode('utf-8')
        return hashlib.sha256(entry_str).hexdigest()
    
    def _finalize_batch(self):
        """Finalize current batch with Merkle root."""
        if not self.current_batch:
            return
        
        merkle_root = MerkleTree.build_tree(self.current_batch)
        
        # Update all entries in current batch with Merkle root
        self._update_merkle_roots(merkle_root)
        
        # Save Merkle root
        self._save_merkle_root(merkle_root)
        
        # Clear batch
        self.current_batch = []
    
    def _update_merkle_roots(self, merkle_root: str):
        """Update Merkle roots for current batch entries."""
        # This is a simplified implementation
        # In practice, you might want to keep the ledger immutable
        # and store Merkle roots separately
        pass
    
    def _save_merkle_root(self, merkle_root: str):
        """Save Merkle root to file."""
        merkle_data = {
            'merkle_root': merkle_root,
            'batch_end_entry': self.entry_count,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'batch_size': len(self.current_batch)
        }
        
        # Append to Merkle roots file
        with open(self.merkle_file, 'a') as f:
            f.write(json.dumps(merkle_data) + '\n')
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire chain."""
        if not self.ledger_file.exists():
            return {'valid': True, 'entries': 0, 'message': 'Empty ledger'}
        
        previous_hash = ''
        entry_count = 0
        integrity_issues = []
        
        with open(self.ledger_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry_dict = json.loads(line.strip())
                    
                    # Verify hash chain
                    if entry_dict['previous_hash'] != previous_hash:
                        integrity_issues.append(f"Line {line_num}: Hash chain broken")
                    
                    # Verify content hash
                    expected_content_hash = self._compute_content_hash(
                        entry_dict['action_type'], 
                        entry_dict['component'], 
                        entry_dict['metadata']
                    )
                    if entry_dict['content_hash'] != expected_content_hash:
                        integrity_issues.append(f"Line {line_num}: Content hash mismatch")
                    
                    # Update for next iteration
                    entry = AuditEntry(**entry_dict)
                    previous_hash = self._hash_entry(entry)
                    entry_count += 1
                    
                except Exception as e:
                    integrity_issues.append(f"Line {line_num}: Parse error - {e}")
        
        return {
            'valid': len(integrity_issues) == 0,
            'entries': entry_count,
            'issues': integrity_issues,
            'last_hash': previous_hash
        }
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit ledger."""
        if not self.ledger_file.exists():
            return {'entries': 0, 'components': [], 'actions': []}
        
        components = set()
        actions = set()
        entry_count = 0
        
        with open(self.ledger_file, 'r') as f:
            for line in f:
                try:
                    entry_dict = json.loads(line.strip())
                    components.add(entry_dict['component'])
                    actions.add(entry_dict['action_type'])
                    entry_count += 1
                except:
                    continue
        
        return {
            'entries': entry_count,
            'components': sorted(list(components)),
            'actions': sorted(list(actions)),
            'current_batch_size': len(self.current_batch)
        }

# Legacy AuditLedger for backward compatibility
class AuditLedger(CryptoAuditLedger):
    """Legacy interface for backward compatibility."""
    
    def __init__(self, log_file: str = "artefacts/ledger.jsonl"):
        # Extract directory from log file path
        ledger_dir = str(Path(log_file).parent)
        super().__init__(ledger_dir=ledger_dir, batch_size=10, auto_merkle=True)
        
    def log_update(self, edge_name: str, params: dict):
        """Log update in legacy format."""
        metadata = {
            'edge_name': edge_name,
            'parameters': params,
            'param_hash': self.hash_params(params)
        }
        self._add_entry('parameter_update', edge_name, metadata)
        
    def hash_params(self, params: dict) -> str:
        """Legacy parameter hashing."""
        param_string = json.dumps(params, sort_keys=True).encode("utf-8")
        return hashlib.sha256(param_string).hexdigest()


if __name__ == '__main__':
    print("=== Testing Advanced Crypto Audit Ledger ===")
    
    # Initialize ledger
    ledger = CryptoAuditLedger(ledger_dir="artefacts", batch_size=5)
    
    # Test with simple models
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model1 = SimpleModel()
    model2 = SimpleModel()
    
    # 1. Log parameter updates
    print("\n--- Logging Parameter Updates ---")
    ledger.log_parameter_update(
        'caref_encoder', 
        model1, 
        gradient_info={'grad_norm': 0.1, 'num_params': 11},
        loss_info={'loss': 0.5, 'loss_type': 'mse'}
    )
    
    # Simulate parameter change
    with torch.no_grad():
        model1.linear.weight += 0.01 * torch.randn_like(model1.linear.weight)
    
    ledger.log_parameter_update(
        'caref_encoder',
        model1,
        gradient_info={'grad_norm': 0.08, 'num_params': 11},
        loss_info={'loss': 0.45, 'loss_type': 'mse'}
    )
    
    # 2. Log gradient steps
    print("Logging gradient step...")
    ledger.log_gradient_step(
        'adapter_X->Y',
        optimizer_state={'lr': 0.001, 'momentum': 0.9},
        loss_value=0.3,
        gradient_norm=0.12
    )
    
    # 3. Log causal interventions
    print("Logging causal intervention...")
    ledger.log_causal_intervention(
        'treatment->outcome',
        'do_intervention',
        {'intervention_value': 1.0, 'target_var': 'treatment'}
    )
    
    # 4. Log evaluation results
    print("Logging evaluation result...")
    ledger.log_evaluation_result(
        'cpi_computer',
        'causal_consistency_score',
        0.85,
        {'dataset': 'synthetic', 'n_samples': 1000}
    )
    
    # Add more entries to trigger batch completion
    for i in range(3):
        ledger.log_gradient_step(
            f'component_{i}',
            {'lr': 0.001},
            loss_value=0.1 * i,
            gradient_norm=0.05
        )
    
    # 5. Get audit summary
    print("\n--- Audit Summary ---")
    summary = ledger.get_audit_summary()
    print(f"Total entries: {summary['entries']}")
    print(f"Components tracked: {summary['components']}")
    print(f"Action types: {summary['actions']}")
    print(f"Current batch size: {summary['current_batch_size']}")
    
    # 6. Verify chain integrity
    print("\n--- Chain Integrity Verification ---")
    integrity_result = ledger.verify_chain_integrity()
    print(f"Chain valid: {integrity_result['valid']}")
    print(f"Entries verified: {integrity_result['entries']}")
    if not integrity_result['valid']:
        print(f"Issues found: {integrity_result['issues']}")
    else:
        print("No integrity issues found!")
    
    # 7. Test parameter hashing
    print("\n--- Testing Parameter Hashing ---")
    state_dict = model1.state_dict()
    param_hashes = ParameterHasher.hash_state_dict(state_dict)
    print("Parameter hashes:")
    for name, hash_val in param_hashes.items():
        print(f"  {name}: {hash_val[:16]}...")
    
    # 8. Test Merkle tree
    print("\n--- Testing Merkle Tree ---")
    test_hashes = ['hash1', 'hash2', 'hash3', 'hash4', 'hash5']
    merkle_root = MerkleTree.build_tree(test_hashes)
    print(f"Merkle root for {len(test_hashes)} hashes: {merkle_root[:16]}...")
    
    # 9. Test legacy interface
    print("\n--- Testing Legacy Interface ---")
    legacy_ledger = AuditLedger()
    
    params_to_log = {
        "learning_rate": 0.001,
        "lambda_cpi": 0.01
    }
    
    legacy_ledger.log_update("edge_treatment->outcome", params_to_log)
    print("Legacy update logged successfully")
    
    # Final summary
    final_summary = ledger.get_audit_summary()
    print(f"\n=== Final Summary ===")
    print(f"Total audit entries: {final_summary['entries']}")
    print("Crypto audit ledger testing complete!")
