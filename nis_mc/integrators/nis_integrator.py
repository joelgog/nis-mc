"""
Neural Importance Sampling Integrator NumPy Implementation.

Implements the normalizing flow-based integration algorithm using NumPy and scipy.optimize.
"""

from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
import scipy.optimize

from nis_mc.integrators.base import BaseIntegrator, IntegrationResult
from nis_mc.flows.affine_coupling import AffineCouplingLayer1D


class NISIntegratorNumPy(BaseIntegrator):
    """
    Neural Importance Sampling integrator using NumPy (no PyTorch).

    Trains a 1D sequence of affine coupling layers to learn the target distribution
    for simple teaching and demonstration purposes.
    """

    def __init__(self, n_train_samples: int = 1000):
        """
        Initialize the NISIntegrator.
        
        Args:
            n_train_samples: Fixed number of samples drawn to estimate the loss during training.
        """
        # Initialize a single minimal affine coupling layer
        self.flow = AffineCouplingLayer1D(s=1.0, t=0.0)
        
        self.n_train_samples = n_train_samples
        self.training_history: List[Tuple[int, float]] = []

    def integrate(self, f: Callable, bounds: list[tuple[float, float]], n_eval: int) -> IntegrationResult:
        """
        Compute the Monte Carlo integral of f over the given bounds using NIS.

        Args:
            f: The integrand. Must accept a numpy array of shape (n_points, n_dim)
               and return a 1D array of shape (n_points,).
            bounds: Integration bounds as a list of (min, max) tuples for each dimension.
            n_eval: Total number of function evaluations (training + integration).

        Returns:
            IntegrationResult
        """
        if len(bounds) != 1:
            raise ValueError("NISIntegratorNumPy only supports 1D integration for this teaching example.")
            
        a, b = bounds[0]
        
        # ==========================================
        # 1. Training loop: minimize reverse KL
        # ==========================================
        self.training_history = []
        step_counter = [0]
        
        # Fix the random noise batch for stable numerical gradients in SciPy.
        # Clip strictly inside (0,1) — required because the new architecture uses logit(u).
        u_base = np.clip(np.random.uniform(0, 1, self.n_train_samples), 1e-6, 1 - 1e-6)
        
        
        # Defensive Importance Sampling mixture: 90% Flow, 10% Uniform.
        # This guarantees bounded variance because the Breit-Wigner has heavy 
        # polynomial tails, while our simple logit-sigmoid flow has thin 
        # exponential tails. A 10% uniform floor prevents q(x) -> 0.
        alpha = 0.90
        
        def objective(params: np.ndarray) -> float:
            """
            Computes Reverse KL with defensive mixture:
            E_{x~u}[ -log(f(x) / (alpha*q(x) + (1-alpha)*U)) ]
            """
            self.flow.s = params[0]
            self.flow.t = params[1]
            
            # Map U(0,1) through flow to get z in (0,1)
            z_flow = self.flow.forward(u_base)
            x_phys = a + z_flow * (b - a)
            
            # Density q(x)
            log_q_z = self.flow.log_prob(z_flow)
            jacobian_bound = (b - a)
            q_flow_x = np.exp(log_q_z) / jacobian_bound
            
            # Mix with uniform density 1/(b-a)
            q_mix_x = alpha * q_flow_x + (1.0 - alpha) / jacobian_bound
            
            # Evaluate integrand f(x)
            f_vals = f(x_phys.reshape(-1, 1))
            if hasattr(f_vals, 'squeeze'):
                f_vals = f_vals.squeeze()
            f_vals = np.maximum(f_vals, 1e-30) # Prevent log(0)
            
            loss = np.mean(-np.log(f_vals / q_mix_x))
            
            self.training_history.append((step_counter[0], float(loss)))
            step_counter[0] += 1
            
            return float(loss)

        # Optimize flow parameters using L-BFGS-B (numerical gradients)
        initial_params = np.array([self.flow.s, self.flow.t])
        
        res = scipy.optimize.minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=[(1e-3, 5.0), (-5.0, 5.0)]  # small s concentrates near sigmoid(t)
        )
        
        # Set to optimal parameters
        self.flow.s = res.x[0]
        self.flow.t = res.x[1]
        
        # ==========================================
        # 2. Integration: Importance Sampling
        # ==========================================
        n_int_samples = n_eval - self.n_train_samples
        if n_int_samples <= 0:
            n_int_samples = n_eval # Fallback if n_eval budget was too small 

        n_flow_samples = int(alpha * n_int_samples)
        n_uni_samples = n_int_samples - n_flow_samples

        # Draw samples from the optimal flow
        u_int = np.random.uniform(0, 1, n_flow_samples)
        z_flow = self.flow.forward(u_int)
        x_flow = a + z_flow * (b - a)
        
        # Draw samples from the uniform background
        x_uni = np.random.uniform(a, b, n_uni_samples)
        
        # Combine
        x_phys = np.concatenate([x_flow, x_uni])
        z_all = (x_phys - a) / (b - a)
        
        # Compute exact final defensive density
        log_q_z_all = self.flow.log_prob(z_all)
        q_flow_x = np.exp(log_q_z_all) / (b - a)
        q_mix_x = alpha * q_flow_x + (1.0 - alpha) / (b - a)
        
        # Compute weights: w = f(x) / q_mix(x)
        f_vals = f(x_phys.reshape(-1, 1))
        if hasattr(f_vals, 'squeeze'):
            f_vals = f_vals.squeeze()
        weights = f_vals / q_mix_x
        
        # Compute mean and standard error
        integral_mean = float(np.mean(weights))
        integral_var = float(np.var(weights, ddof=1)) if n_int_samples > 1 else 0.0
        integral_stderr = float(np.sqrt(integral_var / n_int_samples))

        return IntegrationResult(
            mean=integral_mean,
            std=integral_stderr,
            chi2=0.0,
            n_evals=n_eval,
            metadata={
                "training_history": self.training_history,
                "optimized_s": self.flow.s,
                "optimized_t": self.flow.t,
            }
        )

    def benchmark(self, f: Callable, bounds: list[tuple[float, float]], n_eval_list: list[int]) -> pd.DataFrame:
        """
        Run the integrator across a sequence of evaluation budgets for benchmarking.
        """
        results = []
        for n in n_eval_list:
            res = self.integrate(f, bounds, n)
            results.append({
                "n_evals": n,
                "mean": res.mean,
                "std": res.std
            })
        return pd.DataFrame(results)
