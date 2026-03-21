"""
Neural Importance Sampling Integrator module.

Implements the normalizing flow-based integration algorithm designed to
perfectly trace the geometry of the matrix element and reduce variance.
"""

from typing import Callable
import pandas as pd
from nis_mc.integrators.base import BaseIntegrator, IntegrationResult


class NISIntegrator(BaseIntegrator):
    """
    Neural Importance Sampling integrator.

    Trains a normalizing flow to learn the joint geometry of the integrand,
    enabling it to capture diagonal correlations that separable grid methods miss.
    """

    def integrate(self, f: Callable, bounds: list[tuple[float, float]], n_eval: int) -> IntegrationResult:
        """
        Compute the Monte Carlo integral of f over the given bounds using NIS.

        Args:
            f: The integrand. Must accept a numpy array of shape (n_points, n_dim)
               and return a 1D array of shape (n_points,).
            bounds: Integration bounds as a list of (min, max) tuples for each dimension.
            n_eval: Total number of function evaluations to use (including flow training).

        Returns:
            IntegrationResult containing the estimated mean, std, chi2/dof,
            number of evaluations, and any specific metadata (like loss curve).

        Raises:
            RuntimeError: If the normalizing flow fails to train properly.
        """
        pass

    def benchmark(self, f: Callable, bounds: list[tuple[float, float]], n_eval_list: list[int]) -> pd.DataFrame:
        """
        Run the NIS integrator across a sequence of evaluation budgets.

        Args:
            f: The integrand to evaluate.
            bounds: Integration bounds as a list of (min, max) tuples.
            n_eval_list: A list of integers specifying the evaluation budgets.

        Returns:
            A pandas DataFrame summarizing the performance at each budget.

        Raises:
            ValueError: If any element in n_eval_list is not positive.
        """
        pass
