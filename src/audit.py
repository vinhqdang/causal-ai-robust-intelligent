import json
import hashlib
from datetime import datetime

class AuditLedger:
    """
    A simple audit ledger to log model updates.
    """
    def __init__(self, log_file: str = "artefacts/ledger.jsonl"):
        """
        Initializes the AuditLedger.

        Args:
            log_file (str): The path to the log file.
        """
        self.log_file = log_file

    def log_update(self, edge_name: str, params: dict):
        """
        Logs a model update to the ledger.

        Args:
            edge_name (str): The name of the causal edge that was updated.
            params (dict): A dictionary of parameters to log.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "edge_name": edge_name,
            "parameters": params,
            "param_hash": self.hash_params(params)
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def hash_params(self, params: dict) -> str:
        """
        Computes a SHA256 hash of the model parameters.

        Args:
            params (dict): A dictionary of parameters to hash.

        Returns:
            str: The SHA256 hash of the parameters.
        """
        param_string = json.dumps(params, sort_keys=True).encode("utf-8")
        return hashlib.sha256(param_string).hexdigest()

if __name__ == '__main__':
    ledger = AuditLedger()
    
    # Example usage
    params_to_log = {
        "learning_rate": 0.001,
        "lambda_cpi": 0.01
    }
    
    ledger.log_update("edge_0", params_to_log)
    print(f"Logged update to {ledger.log_file}")
