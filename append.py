with open('nis_mc/integrators/nis_integrator.py', 'a') as f: f.write('from nis_mc.flows.realnvp import RealNVP2D\n\nclass NISIntegratorTorch(BaseIntegrator):\n    pass\n')
