"""
Physical integrands module.

Contains modules for computing physical integrands.
Strict rule: No ML (machine learning code) is allowed here.
"""

from .breit_wigner import BreitWigner1D

__all__ = [
    "BreitWigner1D"
]
