import numpy as np
from nis_mc.integrators.nis_integrator import NISIntegrator
from nis_mc.functions.breit_wigner import BreitWigner1D
bw = BreitWigner1D()
integrator = NISIntegrator(n_train_samples=800)
bounds = [bw.integration_bounds]
res = integrator.integrate(lambda x: bw(x[:, 0]), bounds, 100000)
print('Optimized s:', integrator.flow.s.item())
print('Optimized t:', integrator.flow.t.item())
print('History len:', len(integrator.training_history))
print('First loss:', integrator.training_history[0][1])
print('Last loss:', integrator.training_history[-1][1])
