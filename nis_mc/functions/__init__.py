"""
Physical integrands module.

Contains modules for computing physical integrands.
Strict rule: No ML (machine learning code) is allowed here.
"""

from .breit_wigner import BreitWigner1D
from .diagonal_resonance import DiagonalResonance2D

__all__ = [
    "BreitWigner1D",
    "DiagonalResonance2D"
]
