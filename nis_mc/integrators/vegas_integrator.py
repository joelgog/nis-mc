"""
VEGAS Integrator module.

Provides a benchmark standard using the VEGAS adaptive grid algorithm.
"""

from typing import Callable, Any
import numpy as np
import pandas as pd
import vegas

from nis_mc.integrators.base import BaseIntegrator, IntegrationResult

class VEGASIntegrator(BaseIntegrator):
    """
    VEGAS integration algorithm wrapper.

    Implements the standard VEGAS adaptive importance sampling method
    for benchmarking against Neural Importance Sampling.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._integ = None

    def integrate(self, f: Callable, bounds: list[tuple[float, float]], n_eval: int, n_warmup: int = 1000) -> IntegrationResult:
        """
        Compute the Monte Carlo integral of f over the given bounds using VEGAS.
        
        Runs vegas adaptation using n_warmup evaluations, then integration using n_eval evaluations.
        Note: VEGAS typically does iterations. We use 5 iterations for warmup and 10 for integration,
        scaling the evaluations per iteration so the total matches n_warmup and n_eval respectively.
        
        Args:
            f: The integrand. Must accept a numpy array of shape (n_points, n_dim).
            bounds: Integration bounds as a list of (min, max) tuples for each dimension.
            n_eval: Total number of function evaluations for the main integration pass.
            n_warmup: Total number of function evaluations for the warmup (adaptation) pass.
            
        Returns:
            IntegrationResult containing the results.
        """
        # Seed the random number generator
        np.random.seed(self.seed)

        # Make vegas reproducible by using numpy's seeded uniform generator
        def seed_generator(size):
            return np.random.uniform(0.0, 1.0, size=size)

        self._integ = vegas.Integrator(bounds, ran_array_generator=seed_generator)

        # Ensure batching is supported
        @vegas.batchintegrand
        def batch_f(x):
            val = f(x)
            if hasattr(val, 'squeeze'):
                val = val.squeeze()
            return val

        # Adaptation (warmup)
        if n_warmup > 0:
            warmup_neval = max(1, n_warmup // 5)
            # We don't save the result of the warmup
            self._integ(batch_f, nitn=5, neval=warmup_neval)

        # Integration
        main_neval = max(1, n_eval // 10)
        result = self._integ(batch_f, nitn=10, neval=main_neval)

        # vegas returns a weighted average of iterations (RAvg object)
        # Handle safely in case of API variations
        mean_val = float(getattr(result, 'mean', result))
        std_val = float(getattr(result, 'sdev', 0.0))
        
        dof = max(1, getattr(result, 'dof', 1))
        chi2_val = float(getattr(result, 'chi2', 0.0))
        chi2_dof = chi2_val / dof

        metadata = {}
        if hasattr(result, 'summary'):
            metadata["vegas_summary"] = result.summary()

        return IntegrationResult(
            mean=mean_val,
            std=std_val,
            chi2=chi2_dof,
            n_evals=n_eval,
            metadata=metadata
        )

    def benchmark(self, f: Callable, bounds: list[tuple[float, float]], n_eval_list: list[int], true_value: float | None = None) -> pd.DataFrame:
        """
        Run the VEGAS integrator across a sequence of evaluation budgets.
        
        Args:
            f: The integrand to evaluate. May have a 'true_integral' attribute.
            bounds: Integration bounds as a list of (min, max) tuples.
            n_eval_list: A list of integers specifying the evaluation budgets.
            true_value: Optional ground-truth value of the integral. If provided,
                this takes priority over f.true_integral. Useful when f is a
                plain lambda that has no .true_integral attribute.
            
        Returns:
            A pandas DataFrame with columns: ['n_evals', 'abs_error', 'rel_error', 'chi2']
        """
        if true_value is not None:
            true_val = float(true_value)
        else:
            try:
                true_val = float(f.true_integral)
            except AttributeError:
                true_val = np.nan

        results = []
        for n in n_eval_list:
            res = self.integrate(f, bounds, n_eval=n)
            
            if np.isnan(true_val):
                abs_err = np.nan
                rel_err = np.nan
            else:
                abs_err = abs(res.mean - true_val)
                rel_err = abs_err / abs(true_val) if true_val != 0 else np.nan

            results.append({
                "n_evals": n,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "chi2": res.chi2
            })
            
        return pd.DataFrame(results)

    def get_grid_edges(self) -> list[np.ndarray]:
        """
        Returns the adaptive grid bin edges for visualization.
        """
        if self._integ is None:
            raise RuntimeError("Integrator has not been run yet. Call integrate() first.")
        
        return [np.array(edges) for edges in self._integ.grid]
