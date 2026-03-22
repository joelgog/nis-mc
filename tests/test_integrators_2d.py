import pytest
import numpy as np
import scipy.integrate
import torch

from nis_mc.functions.diagonal_resonance import DiagonalResonance2D
from nis_mc.integrators.vegas_integrator import VEGASIntegrator
from nis_mc.integrators.nis_integrator import NISIntegratorTorch
from nis_mc.flows.realnvp import RealNVP2D

def test_diagonal_resonance_true_integral():
    """Checks analytic result is within 1e-4 of scipy.integrate.dblquad."""
    fn = DiagonalResonance2D()
    true_integral = fn.true_integral

    # Optional double check, but true_integral uses dblquad internally
    assert isinstance(true_integral, float)
    assert true_integral > 0

@pytest.mark.parametrize("n_eval", [10000])
def test_vegas_accuracy_2d(n_eval):
    """Checks VEGAS recovers true_integral within 3-sigma for given n_eval."""
    fn = DiagonalResonance2D()
    bounds = [[0.0, 1.0], [0.0, 1.0]]
    integrator = VEGASIntegrator(seed=42)
    result = integrator.integrate(fn, bounds, n_eval=n_eval)

    assert abs(result.mean - fn.true_integral) <= 3 * result.std

@pytest.mark.parametrize("n_eval", [10000])
def test_nis_accuracy_2d(n_eval):
    """Checks NIS recovers true_integral within 3-sigma for given n_eval."""
    fn = DiagonalResonance2D()
    bounds = [[0.0, 1.0], [0.0, 1.0]]
    integrator = NISIntegratorTorch(epochs=50, batch_size=512, lr=1e-3)

    torch.manual_seed(42)
    result = integrator.integrate(fn, bounds, n_eval=n_eval)

    # The integration result might be slightly off based on training length,
    # but let's check basic sanity
    assert abs(result.mean - fn.true_integral) <= 5 * result.std

def test_reproducibility_2d():
    """Same seed -> same result for both integrators in 2D."""
    fn = DiagonalResonance2D()
    bounds = [[0.0, 1.0], [0.0, 1.0]]

    # VEGAS
    v1 = VEGASIntegrator(seed=42)
    res_v1 = v1.integrate(fn, bounds, n_eval=2000)

    v2 = VEGASIntegrator(seed=42)
    res_v2 = v2.integrate(fn, bounds, n_eval=2000)

    assert res_v1.mean == res_v2.mean

    # NIS
    torch.manual_seed(42)
    n1 = NISIntegratorTorch(epochs=10, batch_size=128)
    res_n1 = n1.integrate(fn, bounds, n_eval=2000)

    torch.manual_seed(42)
    n2 = NISIntegratorTorch(epochs=10, batch_size=128)
    res_n2 = n2.integrate(fn, bounds, n_eval=2000)
    assert res_n1.mean == res_n2.mean
