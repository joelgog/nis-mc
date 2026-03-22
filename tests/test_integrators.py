import pytest
import numpy as np
import scipy.integrate

from nis_mc.functions.breit_wigner import BreitWigner1D
from nis_mc.integrators.vegas_integrator import VEGASIntegrator
from nis_mc.integrators.nis_integrator import NISIntegrator


def test_breit_wigner_true_integral():
    """Checks analytic result is within 1e-10 of scipy.integrate.quad."""
    bw = BreitWigner1D()
    true_integral = bw.true_integral
    quad_res, quad_err = scipy.integrate.quad(bw, bw.integration_bounds[0], bw.integration_bounds[1])
    assert np.isclose(true_integral, quad_res, atol=1e-10)


@pytest.mark.parametrize("n_eval", [10000])
def test_vegas_accuracy(n_eval):
    """Checks VEGAS recovers true_integral within 3-sigma for given n_eval."""
    bw = BreitWigner1D()
    integrator = VEGASIntegrator(seed=42)
    result = integrator.integrate(bw, [bw.integration_bounds], n_eval=n_eval)
    
    assert abs(result.mean - bw.true_integral) <= 3 * result.std


@pytest.mark.parametrize("n_eval", [10000])
def test_nis_accuracy(n_eval):
    """Checks NIS recovers true_integral within 3-sigma for given n_eval."""
    bw = BreitWigner1D()
    integrator = NISIntegrator(n_train_samples=1000)
    
    np.random.seed(42)
    import torch; torch.manual_seed(42)
    result = integrator.integrate(bw, [bw.integration_bounds], n_eval=n_eval)
    
    assert abs(result.mean - bw.true_integral) <= 3 * result.std


def test_benchmark_dataframe_columns():
    """Checks benchmark() returns correct columns."""
    bw = BreitWigner1D()
    
    v_integrator = VEGASIntegrator(seed=42)
    df_v = v_integrator.benchmark(bw, [bw.integration_bounds], [1000])
    assert list(df_v.columns) == ['n_evals', 'abs_error', 'rel_error', 'chi2']
    
    n_integrator = NISIntegrator(n_train_samples=100)
    df_n = n_integrator.benchmark(bw, [bw.integration_bounds], [200])
    assert list(df_n.columns) == ['n_evals', 'mean', 'std']


def test_reproducibility():
    """Same seed -> same result for both integrators."""
    bw = BreitWigner1D()
    bounds = [bw.integration_bounds]
    
    # VEGAS
    v1 = VEGASIntegrator(seed=42)
    res_v1 = v1.integrate(bw, bounds, n_eval=2000)
    
    v2 = VEGASIntegrator(seed=42)
    res_v2 = v2.integrate(bw, bounds, n_eval=2000)
    
    assert res_v1.mean == res_v2.mean
    
    # NIS
    n1 = NISIntegrator(n_train_samples=100)
    np.random.seed(42)
    import torch; torch.manual_seed(42)
    res_n1 = n1.integrate(bw, bounds, n_eval=2000)
    
    n2 = NISIntegrator(n_train_samples=100)
    np.random.seed(42)
    import torch; torch.manual_seed(42)
    res_n2 = n2.integrate(bw, bounds, n_eval=2000)
    
    assert res_n1.mean == res_n2.mean
