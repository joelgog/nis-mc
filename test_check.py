import warnings
warnings.filterwarnings('ignore')
from nis_mc.functions.diagonal_resonance import DiagonalResonance2D
import numpy as np
func = DiagonalResonance2D()
print('Integral:', func.true_integral)
res = func(np.array([[0.5, 0.5]]))
print('val:', res)
