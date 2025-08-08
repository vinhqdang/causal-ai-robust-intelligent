"""
CASCADA - Causal Adaptation via Structure-aware Continual Arbitrary Drift Accounting
Version 2.0 - ICLR 2026 Research Implementation
"""

from .online_graph_refinement import OnlineGraphRefinement
from .adapter_factor_sharing import AdapterFactorSharing
from .path_integrated_cpi import PathIntegratedCPI
from .generative_counterfactual_replay import GenerativeCounterfactualReplay
from .bayesian_uncertainty_gating import BayesianUncertaintyGating
from .cascada_algorithm import CASCADA

__version__ = "2.0.0"
__author__ = "CASCADA Research Team"

__all__ = [
    "OnlineGraphRefinement",
    "AdapterFactorSharing", 
    "PathIntegratedCPI",
    "GenerativeCounterfactualReplay",
    "BayesianUncertaintyGating",
    "CASCADA"
]