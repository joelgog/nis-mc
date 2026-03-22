"""
Neural Importance Sampling Integrator PyTorch Implementation.

Implements the normalizing flow-based integration algorithm using PyTorch
"""

from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from nis_mc.integrators.base import BaseIntegrator, IntegrationResult
from nis_mc.flows.affine_coupling import AffineCouplingLayer1D


class NISIntegrator(BaseIntegrator):
    """
    Neural Importance Sampling integrator using PyTorch.

    Trains a 1D sequence of affine coupling layers to learn the target distribution
    for simple teaching and demonstration purposes.
    """

    def __init__(self, n_train_samples: int = 1000, epochs: int = 100, lr: float = 0.1):
        """
        Initialize the NISIntegrator.

        Args:
            n_train_samples: Fixed number of samples drawn to estimate the loss during training.
            epochs: Number of training iterations.
            lr: Learning rate for the Adam optimizer.
        """
        self.flow = AffineCouplingLayer1D(s=1.0, t=0.0)

        self.n_train_samples = n_train_samples
        self.epochs = epochs
        self.lr = lr
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
            raise ValueError("NISIntegrator only supports 1D integration for this teaching example.")

        a, b = bounds[0]

        # ==========================================
        # 1. Training loop: minimize reverse KL
        # ==========================================
        self.training_history = []

        # Defensive Importance Sampling mixture: 90% Flow, 10% Uniform.
        alpha = 0.90

        optimizer = optim.Adam(self.flow.parameters(), lr=self.lr)

        # Train
        self.flow.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Draw base samples
            u_base = torch.rand(self.n_train_samples).clamp(1e-6, 1 - 1e-6)

            # Map U(0,1) through flow to get z in (0,1)
            z_flow = self.flow.forward(u_base)
            x_phys = a + z_flow * (b - a)

            # Density q(x)
            log_q_z = self.flow.log_prob(z_flow)
            jacobian_bound = (b - a)
            q_flow_x = torch.exp(log_q_z) / jacobian_bound

            # Mix with uniform density 1/(b-a)
            q_mix_x = alpha * q_flow_x + (1.0 - alpha) / jacobian_bound

            # Evaluate integrand f(x)
            # Try evaluating with PyTorch for autograd, fallback to no-grad if f is strictly numpy
            try:
                f_vals = f(x_phys.view(-1, 1))
                if hasattr(f_vals, 'squeeze'):
                    f_vals = f_vals.squeeze()
                if not isinstance(f_vals, torch.Tensor):
                    f_vals = torch.tensor(f_vals, dtype=torch.float32)
                f_vals = torch.clamp(f_vals, min=1e-30)
            except Exception:
                x_phys_np = x_phys.detach().numpy()
                f_vals_np = f(x_phys_np.reshape(-1, 1))
                if hasattr(f_vals_np, 'squeeze'):
                    f_vals_np = f_vals_np.squeeze()
                f_vals_np = np.maximum(f_vals_np, 1e-30)
                f_vals = torch.tensor(f_vals_np, dtype=torch.float32)

            # Loss = KL divergence = E_q[-log(f(x) / q(x))]
            # Wait, standard formulation with reparameterization is evaluating mean( -log(f(x_u) / q(x_u)) )
            # because we sample u, x_u = g(u). But then the distribution depends on parameters.
            # Using reparameterization trick, x_u depends on theta, and q(x_u) depends on theta.
            loss = torch.mean(-torch.log(f_vals / q_mix_x))

            loss.backward()
            optimizer.step()

            self.training_history.append((epoch, loss.item()))

        self.flow.eval()

        # ==========================================
        # 2. Integration: Importance Sampling
        # ==========================================
        n_int_samples = n_eval - self.n_train_samples * self.epochs
        if n_int_samples <= 0:
            n_int_samples = n_eval # Fallback if n_eval budget was too small

        n_flow_samples = int(alpha * n_int_samples)
        n_uni_samples = n_int_samples - n_flow_samples

        with torch.no_grad():
            # Draw samples from the optimal flow
            u_int = torch.rand(n_flow_samples).clamp(1e-6, 1 - 1e-6)
            z_flow = self.flow.forward(u_int)
            x_flow = a + z_flow * (b - a)
            x_flow_np = x_flow.numpy()

            # Draw samples from the uniform background
            x_uni_np = np.random.uniform(a, b, n_uni_samples)

            # Combine
            x_phys_np = np.concatenate([x_flow_np, x_uni_np])
            x_phys = torch.tensor(x_phys_np, dtype=torch.float32)
            z_all = (x_phys - a) / (b - a)

            # Compute exact final defensive density
            log_q_z_all = self.flow.log_prob(z_all)
            q_flow_x = torch.exp(log_q_z_all) / (b - a)
            q_mix_x = alpha * q_flow_x + (1.0 - alpha) / (b - a)
            q_mix_x_np = q_mix_x.numpy()

            # Compute weights: w = f(x) / q_mix(x)
            f_vals_res = f(x_phys_np.reshape(-1, 1))
            if hasattr(f_vals_res, 'squeeze'):
                f_vals_res = f_vals_res.squeeze()
            weights = f_vals_res / q_mix_x_np

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
                "optimized_s": self.flow.s.item(),
                "optimized_t": self.flow.t.item(),
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
