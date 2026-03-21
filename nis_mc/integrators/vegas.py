"""
VEGAS Integrator module.

Provides a benchmark standard using the VEGAS adaptive grid algorithm.
"""

from typing import Callable
import pandas as pd
from nis_mc.integrators.base import BaseIntegrator, IntegrationResult


class VegasIntegrator(BaseIntegrator):
    """
    VEGAS integration algorithm wrapper.

    Implements the standard VEGAS adaptive importance sampling method
    for benchmarking against Neural Importance Sampling.
    """

    def integrate(self, f: Callable, bounds: list[tuple[float, float]], n_eval: int) -> IntegrationResult:
        """
        Compute the Monte Carlo integral of f over the given bounds using VEGAS.

        Uses VEGAS adaptive importance sampling. Performs n_warmup adaptation steps
        (10% of n_eval) before the main integration.

        Args:
            f: The integrand. Must accept a numpy array of shape (n_points, n_dim)
               and return a 1D array of shape (n_points,).
            bounds: Integration bounds as a list of (min, max) tuples for each dimension.
            n_eval: Total number of function evaluations (including warmup).

        Returns:
            IntegrationResult with mean, std (1-sigma), chi2/dof, and n_evals used.

        Raises:
            ValueError: If bounds are not all finite.
            ConvergenceWarning: If chi2/dof > 2.0 (poor VEGAS adaptation).
        """
        pass

    def benchmark(self, f: Callable, bounds: list[tuple[float, float]], n_eval_list: list[int]) -> pd.DataFrame:
        """
        Run the VEGAS integrator across a sequence of evaluation budgets.

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
