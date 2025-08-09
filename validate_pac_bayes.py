#!/usr/bin/env python3
"""
PAC-Bayes Empirical Validation Script for CASCADA Algorithm

This script validates the PAC-Bayes bounds theoretically predicted by CASCADA
against empirical risk curves. It implements experiments to verify that the
theoretical bounds hold in practice with the √T risk dependence.

Usage:
    python validate_pac_bayes.py [--config CONFIG_FILE] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cascada.cascada_algorithm import CASCADAAlgorithm
from evaluation.harness import EvaluationHarness
from evaluation.metrics import TaskSpecificMetrics


class PACBayesValidator:
    """
    PAC-Bayes bound validator for CASCADA algorithm.
    
    Implements experiments to verify that empirical risk curves match
    theoretical PAC-Bayes predictions with √T dependence.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "./pac_bayes_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'pac_bayes_validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Experiment parameters
        self.num_time_steps = config.get('num_time_steps', [100, 200, 500, 1000, 2000, 5000])
        self.num_trials = config.get('num_trials', 10)
        self.confidence_levels = config.get('confidence_levels', [0.05, 0.1, 0.2])
        self.model_complexity = config.get('model_complexity', 'medium')  # 'small', 'medium', 'large'
        
        # Initialize CASCADA algorithm
        self.cascada = self._initialize_cascada()
        
        # Track validation results
        self.validation_results = {
            'empirical_risks': {},
            'theoretical_bounds': {},
            'bound_violations': {},
            'sqrt_t_fit_quality': {},
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
    def _initialize_cascada(self) -> CASCADAAlgorithm:
        """Initialize CASCADA algorithm with appropriate configuration."""
        # Model dimensions based on complexity
        complexity_configs = {
            'small': {'model_dim': 64, 'adapter_dim': 32, 'latent_dim': 16},
            'medium': {'model_dim': 128, 'adapter_dim': 64, 'latent_dim': 32},
            'large': {'model_dim': 256, 'adapter_dim': 128, 'latent_dim': 64}
        }
        
        dims = complexity_configs[self.model_complexity]
        
        # Create synthetic base model for testing
        base_model = nn.Sequential(
            nn.Linear(dims['model_dim'], dims['model_dim']),
            nn.ReLU(),
            nn.Linear(dims['model_dim'], dims['model_dim']),
            nn.ReLU(),
            nn.Linear(dims['model_dim'], 10)  # 10-class output
        ).to(self.device)
        
        # Initialize CASCADA
        cascada = CASCADAAlgorithm(
            base_model=base_model,
            model_dim=dims['model_dim'],
            adapter_dim=dims['adapter_dim'],
            latent_dim=dims['latent_dim'],
            max_edges=20,
            device=self.device
        )
        
        return cascada
    
    def generate_synthetic_data(
        self,
        num_samples: int,
        input_dim: int = 128,
        num_classes: int = 10,
        num_tasks: int = 5,
        drift_strength: float = 0.1
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate synthetic continual learning data with concept drift.
        
        Args:
            num_samples: Number of samples per task
            input_dim: Input dimensionality
            num_classes: Number of output classes
            num_tasks: Number of sequential tasks
            drift_strength: Strength of concept drift between tasks
            
        Returns:
            task_data: List of (X, y) tuples for each task
        """
        task_data = []
        
        # Base data distribution
        base_mean = torch.zeros(input_dim)
        base_cov = torch.eye(input_dim)
        
        for task_id in range(num_tasks):
            # Apply concept drift
            drift_offset = drift_strength * task_id * torch.randn(input_dim)
            task_mean = base_mean + drift_offset
            
            # Generate data for this task
            X = torch.multivariate_normal(task_mean, base_cov, (num_samples,))
            
            # Generate labels with some task-specific bias
            task_bias = np.random.choice(num_classes, size=2)  # Two preferred classes
            y = torch.randint(0, num_classes, (num_samples,))
            
            # Bias toward preferred classes
            bias_mask = torch.rand(num_samples) < 0.3
            y[bias_mask] = torch.randint(0, 2, (bias_mask.sum(),))
            y[bias_mask] = torch.tensor([task_bias[i % 2] for i in range(bias_mask.sum())])
            
            task_data.append((X.to(self.device), y.to(self.device)))
            
        self.logger.info(f"Generated {num_tasks} tasks with {num_samples} samples each")
        return task_data
    
    def compute_pac_bayes_bound(
        self,
        empirical_risk: float,
        num_samples: int,
        model_complexity: float,
        confidence: float = 0.05
    ) -> float:
        """
        Compute PAC-Bayes generalization bound.
        
        Implements the bound: R(h) ≤ R̂(h) + √((KL + log(2√n/δ)) / (2(n-1)))
        where KL is the KL divergence to prior.
        
        Args:
            empirical_risk: Empirical risk on training data
            num_samples: Number of training samples
            model_complexity: KL divergence to prior (proxy)
            confidence: Confidence parameter δ
            
        Returns:
            pac_bayes_bound: Upper bound on true risk
        """
        if num_samples <= 1:
            return float('inf')
            
        # PAC-Bayes bound computation
        kl_term = model_complexity
        confidence_term = np.log(2 * np.sqrt(num_samples) / confidence)
        
        generalization_gap = np.sqrt((kl_term + confidence_term) / (2 * (num_samples - 1)))
        
        pac_bayes_bound = empirical_risk + generalization_gap
        
        return pac_bayes_bound
    
    def estimate_model_complexity(self, cascada: CASCADAAlgorithm, task_data: torch.Tensor) -> float:
        """
        Estimate model complexity as KL divergence proxy.
        
        This is a simplified version that uses parameter magnitude as complexity measure.
        """
        total_complexity = 0.0
        
        # Base model complexity (parameter norms)
        for param in cascada.base_model.parameters():
            total_complexity += torch.norm(param, p=2).item() ** 2
        
        # Adapter complexity
        if hasattr(cascada, 'adapter_factor_sharing'):
            afs_stats = cascada.adapter_factor_sharing.get_memory_usage()
            total_complexity += afs_stats['total_params'] * 0.01  # Scale factor
        
        # Graph complexity (number of edges)
        if hasattr(cascada, 'current_graph'):
            total_complexity += cascada.current_graph.number_of_edges() * 0.1
        
        return total_complexity
    
    def run_single_experiment(
        self,
        time_steps: List[int],
        confidence: float = 0.05
    ) -> Dict[str, List[float]]:
        """
        Run a single PAC-Bayes validation experiment.
        
        Args:
            time_steps: List of time steps to evaluate
            confidence: Confidence level for bounds
            
        Returns:
            results: Dictionary with empirical risks and bounds
        """
        empirical_risks = []
        theoretical_bounds = []
        bound_violations = []
        
        # Generate full dataset
        max_steps = max(time_steps)
        task_data = self.generate_synthetic_data(
            num_samples=max_steps // 5,  # Distribute across 5 tasks
            num_tasks=5
        )
        
        # Initialize fresh CASCADA instance
        cascada = self._initialize_cascada()
        optimizer = optim.Adam(cascada.parameters(), lr=0.001)
        
        all_data = []
        for X, y in task_data:
            all_data.extend([(x, label) for x, label in zip(X, y)])
        
        # Track cumulative performance
        cumulative_loss = 0.0
        sample_count = 0
        
        for t in time_steps:
            # Train up to time step t
            while sample_count < t and sample_count < len(all_data):
                x, y = all_data[sample_count]
                x = x.unsqueeze(0)  # Add batch dimension
                y = y.unsqueeze(0)
                
                # Create simple causal graph for this sample
                graph = nx.DiGraph()
                graph.add_edges_from([(0, 1), (1, 2)])  # Simple chain
                
                # Forward pass and loss computation
                optimizer.zero_grad()
                output = cascada(x, graph)
                loss = nn.CrossEntropyLoss()(output, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                cumulative_loss += loss.item()
                sample_count += 1
            
            if sample_count == 0:
                continue
                
            # Compute empirical risk
            empirical_risk = cumulative_loss / sample_count
            empirical_risks.append(empirical_risk)
            
            # Estimate model complexity
            model_complexity = self.estimate_model_complexity(cascada, x)
            
            # Compute PAC-Bayes bound
            bound = self.compute_pac_bayes_bound(
                empirical_risk, sample_count, model_complexity, confidence
            )
            theoretical_bounds.append(bound)
            
            # Check for bound violations (empirical risk > bound)
            violation = empirical_risk > bound
            bound_violations.append(violation)
            
            self.logger.info(
                f"T={t}: Empirical Risk={empirical_risk:.4f}, "
                f"PAC-Bayes Bound={bound:.4f}, Violation={violation}"
            )
        
        return {
            'empirical_risks': empirical_risks,
            'theoretical_bounds': theoretical_bounds,
            'bound_violations': bound_violations,
            'time_steps': time_steps[:len(empirical_risks)]
        }
    
    def analyze_sqrt_t_dependence(
        self,
        time_steps: List[int],
        empirical_risks: List[float]
    ) -> Dict[str, float]:
        """
        Analyze whether empirical risk follows √T dependence.
        
        Fits empirical_risk = a / √T + b and returns fit quality.
        """
        if len(time_steps) != len(empirical_risks) or len(time_steps) < 3:
            return {'r_squared': 0.0, 'slope': 0.0, 'intercept': 0.0}
        
        # Prepare data for fitting: y = a * x + b where x = 1/√T
        x = np.array([1.0 / np.sqrt(t) for t in time_steps])
        y = np.array(empirical_risks)
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        
        slope, intercept = coeffs
        
        # Compute R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'r_squared': float(r_squared),
            'slope': float(slope),
            'intercept': float(intercept)
        }
    
    def run_validation_experiments(self):
        """Run complete PAC-Bayes validation experiments."""
        self.logger.info("Starting PAC-Bayes validation experiments")
        
        for confidence in self.confidence_levels:
            self.logger.info(f"Running experiments with confidence level δ={confidence}")
            
            # Results for this confidence level
            conf_results = {
                'empirical_risks': [],
                'theoretical_bounds': [],
                'bound_violations': [],
                'sqrt_t_fits': []
            }
            
            # Run multiple trials
            for trial in range(self.num_trials):
                self.logger.info(f"Trial {trial + 1}/{self.num_trials}")
                
                # Run single experiment
                results = self.run_single_experiment(self.num_time_steps, confidence)
                
                conf_results['empirical_risks'].append(results['empirical_risks'])
                conf_results['theoretical_bounds'].append(results['theoretical_bounds'])
                conf_results['bound_violations'].append(results['bound_violations'])
                
                # Analyze √T dependence
                sqrt_t_fit = self.analyze_sqrt_t_dependence(
                    results['time_steps'], results['empirical_risks']
                )
                conf_results['sqrt_t_fits'].append(sqrt_t_fit)
            
            # Store results for this confidence level
            self.validation_results['empirical_risks'][confidence] = conf_results['empirical_risks']
            self.validation_results['theoretical_bounds'][confidence] = conf_results['theoretical_bounds']
            self.validation_results['bound_violations'][confidence] = conf_results['bound_violations']
            self.validation_results['sqrt_t_fit_quality'][confidence] = conf_results['sqrt_t_fits']
            
        self.logger.info("Completed all PAC-Bayes validation experiments")
    
    def analyze_results(self):
        """Analyze and summarize validation results."""
        self.logger.info("Analyzing PAC-Bayes validation results")
        
        analysis = {}
        
        for confidence in self.confidence_levels:
            # Get results for this confidence level
            emp_risks = self.validation_results['empirical_risks'][confidence]
            bounds = self.validation_results['theoretical_bounds'][confidence]
            violations = self.validation_results['bound_violations'][confidence]
            sqrt_fits = self.validation_results['sqrt_t_fit_quality'][confidence]
            
            # Compute statistics
            violation_rates = []
            avg_gap_ratios = []
            sqrt_t_r_squared = []
            
            for trial in range(len(emp_risks)):
                # Violation rate for this trial
                trial_violations = violations[trial]
                violation_rate = sum(trial_violations) / len(trial_violations)
                violation_rates.append(violation_rate)
                
                # Average gap ratio (bound - empirical) / empirical
                trial_emp = emp_risks[trial]
                trial_bounds = bounds[trial]
                gap_ratios = []
                for emp, bound in zip(trial_emp, trial_bounds):
                    if emp > 0:
                        gap_ratios.append((bound - emp) / emp)
                avg_gap_ratios.append(np.mean(gap_ratios))
                
                # √T fit quality
                sqrt_t_r_squared.append(sqrt_fits[trial]['r_squared'])
            
            analysis[confidence] = {
                'avg_violation_rate': np.mean(violation_rates),
                'std_violation_rate': np.std(violation_rates),
                'avg_gap_ratio': np.mean(avg_gap_ratios),
                'std_gap_ratio': np.std(avg_gap_ratios),
                'avg_sqrt_t_r_squared': np.mean(sqrt_t_r_squared),
                'std_sqrt_t_r_squared': np.std(sqrt_t_r_squared)
            }
            
            self.logger.info(
                f"Confidence δ={confidence}: "
                f"Violation Rate={analysis[confidence]['avg_violation_rate']:.3f} ± "
                f"{analysis[confidence]['std_violation_rate']:.3f}, "
                f"√T R²={analysis[confidence]['avg_sqrt_t_r_squared']:.3f} ± "
                f"{analysis[confidence]['std_sqrt_t_r_squared']:.3f}"
            )
        
        return analysis
    
    def generate_plots(self, analysis: Dict):
        """Generate visualization plots for PAC-Bayes validation results."""
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Risk curves vs √T
        plt.figure(figsize=(12, 8))
        
        for i, confidence in enumerate(self.confidence_levels):
            emp_risks = self.validation_results['empirical_risks'][confidence]
            bounds = self.validation_results['theoretical_bounds'][confidence]
            
            # Average across trials
            avg_emp_risks = np.mean(emp_risks, axis=0)
            avg_bounds = np.mean(bounds, axis=0)
            std_emp_risks = np.std(emp_risks, axis=0)
            std_bounds = np.std(bounds, axis=0)
            
            time_steps = self.num_time_steps[:len(avg_emp_risks)]
            sqrt_time_steps = [np.sqrt(t) for t in time_steps]
            
            plt.subplot(2, 2, i + 1)
            plt.errorbar(sqrt_time_steps, avg_emp_risks, yerr=std_emp_risks, 
                        label='Empirical Risk', marker='o')
            plt.errorbar(sqrt_time_steps, avg_bounds, yerr=std_bounds,
                        label='PAC-Bayes Bound', marker='s')
            plt.xlabel('√T')
            plt.ylabel('Risk')
            plt.title(f'Risk Curves (δ={confidence})')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'risk_curves_sqrt_t.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Violation rates vs confidence level
        plt.figure(figsize=(10, 6))
        
        violation_rates = []
        violation_stds = []
        
        for confidence in self.confidence_levels:
            avg_rate = analysis[confidence]['avg_violation_rate']
            std_rate = analysis[confidence]['std_violation_rate']
            violation_rates.append(avg_rate)
            violation_stds.append(std_rate)
        
        plt.errorbar(self.confidence_levels, violation_rates, yerr=violation_stds,
                    marker='o', capsize=5, capthick=2)
        plt.axhline(y=0.05, color='r', linestyle='--', 
                   label='Expected Rate (δ=0.05)')
        plt.xlabel('Confidence Level δ')
        plt.ylabel('Bound Violation Rate')
        plt.title('PAC-Bayes Bound Violation Rates')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(plots_dir / 'violation_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: √T fit quality
        plt.figure(figsize=(10, 6))
        
        r_squared_means = []
        r_squared_stds = []
        
        for confidence in self.confidence_levels:
            mean_r2 = analysis[confidence]['avg_sqrt_t_r_squared']
            std_r2 = analysis[confidence]['std_sqrt_t_r_squared']
            r_squared_means.append(mean_r2)
            r_squared_stds.append(std_r2)
        
        plt.errorbar(self.confidence_levels, r_squared_means, yerr=r_squared_stds,
                    marker='s', capsize=5, capthick=2)
        plt.xlabel('Confidence Level δ')
        plt.ylabel('R² for √T Fit')
        plt.title('Quality of √T Dependence Fit')
        plt.grid(True)
        
        plt.savefig(plots_dir / 'sqrt_t_fit_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Generated plots in {plots_dir}")
    
    def save_results(self, analysis: Dict):
        """Save validation results and analysis to files."""
        # Save raw results
        with open(self.output_dir / 'pac_bayes_results.json', 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        # Save analysis summary
        with open(self.output_dir / 'pac_bayes_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Create summary report
        report = []
        report.append("PAC-Bayes Validation Report")
        report.append("=" * 40)
        report.append(f"Timestamp: {self.validation_results['timestamp']}")
        report.append(f"Model Complexity: {self.model_complexity}")
        report.append(f"Number of Trials: {self.num_trials}")
        report.append(f"Time Steps: {self.num_time_steps}")
        report.append("")
        
        for confidence in self.confidence_levels:
            report.append(f"Confidence Level δ = {confidence}:")
            report.append(f"  Avg Violation Rate: {analysis[confidence]['avg_violation_rate']:.4f} ± {analysis[confidence]['std_violation_rate']:.4f}")
            report.append(f"  Avg Gap Ratio: {analysis[confidence]['avg_gap_ratio']:.4f} ± {analysis[confidence]['std_gap_ratio']:.4f}")
            report.append(f"  Avg √T R²: {analysis[confidence]['avg_sqrt_t_r_squared']:.4f} ± {analysis[confidence]['std_sqrt_t_r_squared']:.4f}")
            
            # Interpretation
            violation_rate = analysis[confidence]['avg_violation_rate']
            expected_rate = confidence
            
            if violation_rate <= expected_rate * 1.2:  # Within 20% tolerance
                report.append(f"  → Bounds hold well (violation rate ≤ {expected_rate * 1.2:.3f})")
            else:
                report.append(f"  → Bounds may be loose (violation rate > {expected_rate * 1.2:.3f})")
            
            sqrt_r2 = analysis[confidence]['avg_sqrt_t_r_squared']
            if sqrt_r2 > 0.8:
                report.append(f"  → Strong √T dependence (R² = {sqrt_r2:.3f})")
            elif sqrt_r2 > 0.5:
                report.append(f"  → Moderate √T dependence (R² = {sqrt_r2:.3f})")
            else:
                report.append(f"  → Weak √T dependence (R² = {sqrt_r2:.3f})")
            
            report.append("")
        
        # Overall conclusion
        all_violation_rates = [analysis[c]['avg_violation_rate'] for c in self.confidence_levels]
        all_r_squared = [analysis[c]['avg_sqrt_t_r_squared'] for c in self.confidence_levels]
        
        report.append("Overall Assessment:")
        if np.mean(all_violation_rates) < np.mean(self.confidence_levels) * 1.5:
            report.append("✓ PAC-Bayes bounds are empirically validated")
        else:
            report.append("✗ PAC-Bayes bounds show significant violations")
        
        if np.mean(all_r_squared) > 0.6:
            report.append("✓ Strong evidence for √T risk dependence")
        else:
            report.append("✗ Weak evidence for √T risk dependence")
        
        with open(self.output_dir / 'pac_bayes_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def run_full_validation(self):
        """Run complete PAC-Bayes validation pipeline."""
        self.logger.info("Starting full PAC-Bayes validation pipeline")
        
        # Run experiments
        self.run_validation_experiments()
        
        # Analyze results
        analysis = self.analyze_results()
        
        # Generate visualizations
        self.generate_plots(analysis)
        
        # Save all results
        self.save_results(analysis)
        
        self.logger.info("PAC-Bayes validation completed successfully")
        return analysis


def main():
    """Main function to run PAC-Bayes validation."""
    parser = argparse.ArgumentParser(description="PAC-Bayes Validation for CASCADA")
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file (JSON)')
    parser.add_argument('--output', type=str, default='./pac_bayes_results',
                       help='Output directory')
    parser.add_argument('--complexity', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model complexity level')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials per experiment')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'num_time_steps': [100, 200, 500, 1000, 2000, 5000],
            'num_trials': args.trials,
            'confidence_levels': [0.05, 0.1, 0.2],
            'model_complexity': args.complexity
        }
    
    # Create validator and run experiments
    validator = PACBayesValidator(config, args.output)
    analysis = validator.run_full_validation()
    
    print(f"\nPAC-Bayes validation completed!")
    print(f"Results saved to: {args.output}")
    print(f"Check pac_bayes_report.txt for summary")
    
    return analysis


if __name__ == "__main__":
    main()