import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import os

from nis_mc.functions.breit_wigner import BreitWigner1D
from nis_mc.functions.diagonal_resonance import DiagonalResonance2D
from nis_mc.integrators.vegas_integrator import VEGASIntegrator
from nis_mc.integrators.nis_integrator import NISIntegrator, NISIntegratorTorch
from nis_mc.flows.realnvp import RealNVP2D
from nis_mc.visualization.comparison_plot import plot_level1_comparison
from nis_mc.visualization.vegas_failure_plot import plot_vegas_failure

def run_level1(seed):
    print("Running Level 1 Benchmark...")
    np.random.seed(seed)
    torch.manual_seed(seed)

    bw = BreitWigner1D()
    bounds = [[bw.bounds[0], bw.bounds[1]]]

    n_evals = [1000, 2000, 5000, 10000, 20000]

    vegas = VEGASIntegrator(seed=seed)
    df_vegas = vegas.benchmark(bw, bounds, n_evals)

    nis = NISIntegrator(n_train_samples=2000)
    df_nis = nis.benchmark(bw, bounds, n_evals)

    os.makedirs('results', exist_ok=True)
    plot_level1_comparison(bw, df_vegas, df_nis, 'results/level1_comparison.png')
    print("Level 1 saved to results/level1_comparison.png")

def run_level2(seed):
    print("Running Level 2 Benchmark...")
    np.random.seed(seed)
    torch.manual_seed(seed)

    fn = DiagonalResonance2D()
    bounds = [[0.0, 1.0], [0.0, 1.0]]

    n_evals = [1000, 5000, 10000, 50000, 100000]

    print("Running VEGAS...")
    vegas = VEGASIntegrator(seed=seed)
    df_vegas = vegas.benchmark(fn, bounds, n_evals)

    print("Running NIS...")

    res_list = []
    nis = None
    for ne in n_evals:
        nis = NISIntegratorTorch(epochs=200, batch_size=2048, lr=1e-3)
        res = nis.integrate(fn, bounds, n_eval=ne)

        abs_err = abs(res.mean - fn.true_integral)
        rel_err = abs_err / fn.true_integral
        res_list.append({
            'n_evals': ne,
            'mean': res.mean,
            'std': res.std,
            'abs_error': abs_err,
            'rel_error': rel_err
        })
    df_nis = pd.DataFrame(res_list)

    vegas_rel_err = df_vegas['rel_error'].iloc[-1]
    nis_rel_err = df_nis['rel_error'].iloc[-1]

    os.makedirs('results', exist_ok=True)
    fig = plot_vegas_failure(fn, vegas, nis, vegas_rel_err, nis_rel_err, df_vegas, df_nis)
    fig.savefig('results/level2_vegas_failure.png', bbox_inches='tight', dpi=300)
    print("Level 2 saved to results/level2_vegas_failure.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NIS vs VEGAS benchmarks")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2], help="Benchmark level (1 or 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.level == 1:
        run_level1(args.seed)
    elif args.level == 2:
        run_level2(args.seed)
