# Causal AI: Integrating Causality and Machine Learning for Robust Intelligent Systems

This repository contains the implementation of the **Intervention-Aware Continual Causal Pre-training (ICCP)** algorithm, a novel framework for continually updating foundation models without eroding their learned causal mechanisms.

## Overview

The ICCP algorithm is designed to address the challenge of "model rot" in large language models, where models become less effective over time as the data they were trained on becomes outdated. ICCP treats each model update as a causal intervention, safeguarding the discovered causal mechanisms throughout the model's lifetime.

The core components of the ICCP algorithm are:

- **CaReF (Causal Representation Factorization):** A module for extracting causal features from the input data.
- **Structural Causal Memory:** A system for managing adapters that correspond to causal edges in a Structural Causal Model (SCM).
- **Causal Parameter Importance (CPI):** A method for identifying the parameters that are most important for preserving the model's causal reasoning abilities.
- **Counterfactual Replay Buffer (CRB):** A buffer for storing and replaying counterfactual examples to ensure causal consistency.
- **Audit Ledger:** A system for logging all model updates to ensure transparency and auditability.

## Getting Started

### Prerequisites

- Python 3.10+
- Conda

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/vinhqdang/causal-ai-robust-intelligent.git
    cd causal-ai-robust-intelligent
    ```

2.  Create and activate the conda environment:
    ```bash
    conda create -n causal-ai python=3.10
    conda activate causal-ai
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulation

To run the ICCP simulation, execute the following command:

```bash
python3 -m src.main
```

This will run a simple simulation with a GPT-2 model and print the training progress to the console. The audit ledger will be saved to `artefacts/ledger.jsonl`.

### Running the Tests

To run the smoke test, execute the following command:

```bash
pytest
```
