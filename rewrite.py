content1 = '''"""
Affine Coupling module.

Implements a single, minimal affine coupling layer using PyTorch for teaching.

Architecture: y = sigmoid(s · logit(u) + t) where u ~ U(0,1).

Key property: by applying logit FIRST (mapping U(0,1) → Logistic(0,1))
before the affine shift-and-scale, the output y covers the FULL interval
(0,1) for any finite (s, t). This is essential for unbiased importance
sampling over the entire integration domain.

Behaviour:
  s=1, t=0  →  y = u  (identity, uniform proposal)
  s→0, t=0  →  y concentrates around 0.5
  s→0, t=L  →  y concentrates around sigmoid(L)
"""

import torch
import torch.nn as nn

def _logit(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-7, 1.0 - 1e-7)
    return torch.log(x / (1.0 - x))

def _sigmoid(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)

class AffineCouplingLayer1D(nn.Module):
    """
    A minimal 1D Affine Coupling Layer for Normalizing Flows.

    Maps u ~ U(0,1) to y ~ q(y) via:
        z = logit(u)              # U(0,1) -> Logistic(0,1)
        w = s * z + t             # shift & scale
        y = sigmoid(w)            # -> (0,1), full support

    Attributes:
        s (nn.Parameter): Scale parameter. s=1 gives uniform; smaller s concentrates.
        t (nn.Parameter): Translation. sigmoid(t) is the mode of the distribution.
    """

    def __init__(self, s: float = 1.0, t: float = 0.0):
        super().__init__()
        self.s = nn.Parameter(torch.tensor([float(s)]))
        self.t = nn.Parameter(torch.tensor([float(t)]))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward map: u ~ U(0,1)  →  y in (0,1) with full support.

            y = sigmoid(s · logit(u) + t)

        Special cases:
            s=1, t=0  →  y = sigmoid(logit(u)) = u  (identity)
            s<1, t=0  →  distribution concentrates around 0.5
        """
        z = _logit(u)            # U(0,1) → Logistic(0,1)
        w = self.s * z + self.t  # affine transform
        y = _sigmoid(w)          # → (0,1)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse map: y  →  u

            u = sigmoid((logit(y) - t) / s)
        """
        w = _logit(y)
        z = (w - self.t) / self.s
        u = _sigmoid(z)
        return u

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        """
        Log density log q(y) under the logit-sigmoid flow.

        Derivation (change-of-variables from u ~ U(0,1)):
            y = sigmoid(s · logit(u) + t)
            dy/du = s · y(1-y) / (u(1-u))
            log q(y) = -log|dy/du|
                     = log(u) + log(1-u) - log|s| - log(y) - log(1-y)
            where u = sigmoid((logit(y) - t) / s)  [the inverse]

        Properties:
            - Integrates to 1 over (0,1) for any (s,t)  ✓
            - s=1,t=0:  log q(y) = 0  (uniform)  ✓
            - Small s:  density peaks sharply at sigmoid(t)  ✓
        """
        y = torch.clamp(y, 1e-7, 1.0 - 1e-7)
        u = self.inverse(y)
        u = torch.clamp(u, 1e-7, 1.0 - 1e-7)

        log_q_y = (
            torch.log(u) + torch.log(1.0 - u)
            - torch.log(torch.abs(self.s))
            - torch.log(y) - torch.log(1.0 - y)
        )
        return log_q_y

    def diagram_ascii(self):
        """Prints an ASCII diagram of the transformation."""
        s_val = self.s.item()
        t_val = self.t.item()
        print(f"""
        AffineCouplingLayer1D (logit-sigmoid):
        ======================================

           u ~ U(0,1)     z=logit(u)    w=s·z+t    y=sigmoid(w)
          [Base Space]  → Logistic(0,1) →  affine  →  [Target Space]

        Current Parameters:
          s = {s_val:.4f}   (s<1 concentrates, s=1 is uniform)
          t = {t_val:.4f}   (mode at sigmoid(t) = {_sigmoid(torch.tensor(t_val)):.4f})

        Jacobian |dy/du| = s · y(1-y) / (u(1-u))
        """)
'''

with open('nis_mc/flows/affine_coupling.py', 'w', encoding='utf-8') as f:
    f.write(content1)

content2 = '''"""
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
            x_phys_np = x_phys.detach().numpy()
            f_vals_np = f(x_phys_np.reshape(-1, 1))
            if hasattr(f_vals_np, 'squeeze'):
                f_vals_np = f_vals_np.squeeze()
            f_vals_np = np.maximum(f_vals_np, 1e-30) # Prevent log(0)

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
'''

with open('nis_mc/integrators/nis_integrator.py', 'w', encoding='utf-8') as f:
    f.write(content2)
