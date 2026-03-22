import torch
import numpy as np
torch.manual_seed(0)
np.random.seed(0)
from nis_mc.integrators.nis_integrator import NISIntegrator
from nis_mc.functions.breit_wigner import BreitWigner1D
bw = BreitWigner1D()
integrator = NISIntegrator(n_train_samples=8000)
bounds = [bw.integration_bounds]
res = integrator.integrate(lambda x: bw(x[:, 0]), bounds, 1000_000)
print('Optimized s:', integrator.flow.s.item(), 't:', integrator.flow.t.item())
