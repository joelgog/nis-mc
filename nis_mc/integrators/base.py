"""
Base Integrator Module.

This module defines the abstract base class and required data structures
for all integration algorithms to establish a uniform benchmarking interface.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class IntegrationResult:
    """
    Data container for the results of a Monte Carlo integration pass.

    Attributes:
        mean: The estimated mean of the integral.
        std: The estimated 1-sigma standard deviation (error in the mean).
        chi2: The chi-squared per degree of freedom for the iterations.
        n_evals: The total number of integrand evaluations performed.
        metadata: Any additional algorithm-specific output or state.
    """
    mean: float
    std: float
    chi2: float
    n_evals: int
    metadata: dict[str, Any]


class BaseIntegrator(ABC):
    """
    Abstract base class for all integrators in nis-mc.

    Every integrator must implement the `integrate` and `benchmark`
    methods to ensure a uniform interface for comparing performance.
    """

    @abstractmethod
    def integrate(self, f: Callable, bounds: list[tuple[float, float]], n_eval: int) -> IntegrationResult:
        """
        Compute the Monte Carlo integral of f over the given bounds.

        Args:
            f: The integrand. Must accept a numpy array of shape (n_points, n_dim)
               and return a 1D array of shape (n_points,).
            bounds: Integration bounds as a list of (min, max) tuples for each dimension.
            n_eval: Total number of function evaluations to use.

        Returns:
            IntegrationResult containing the estimated mean, std, chi2/dof,
            number of evaluations, and any specific metadata.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        pass

    @abstractmethod
    def benchmark(self, f: Callable, bounds: list[tuple[float, float]], n_eval_list: list[int]) -> pd.DataFrame:
        """
        Run the integrator across a sequence of evaluation budgets for benchmarking.

        Args:
            f: The integrand to evaluate.
            bounds: Integration bounds as a list of (min, max) tuples.
            n_eval_list: A list of integers specifying the evaluation budgets to benchmark.

        Returns:
            A pandas DataFrame summarizing the performance at each budget.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        pass
