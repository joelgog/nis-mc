"""
Breit-Wigner (Lorentzian) peak implementation.
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

class BreitWigner1D:
    """
    1D Breit-Wigner (Lorentzian) unnormalized resonance.

    This function mimics a physical resonance such as a Z or Higgs boson. It is 
    characterized by a continuous distribution with a very sharp peak. Because of 
    the narrow width (Γ), naive Monte Carlo integration is highly inefficient
    and will waste >95% of its random samples simply missing the resonance peak.
    """
    def __init__(
        self, 
        center: float = 0.5, 
        width: float = 0.02, 
        integration_bounds: Tuple[float, float] = (0.0, 1.0)
    ):
        self.center = center
        self.width = width
        self.integration_bounds = integration_bounds

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        """Returns the unnormalized Lorentzian peak density."""
        a = self.width / 2.0
        return 1.0 / ((x - self.center)**2 + a**2)

    @property
    def true_integral(self) -> float:
        """
        Analytically computed ground truth of the integral over the bounds.
        
        Uses the exact formula:
        integral of 1/(x^2 + a^2) from b to c = (1/a)[arctan(c/a) - arctan(b/a)]
        """
        a = self.width / 2.0
        # shift variables to match formula integration 1/(x^2 + a^2)
        b = self.integration_bounds[0] - self.center
        c = self.integration_bounds[1] - self.center
        return (1.0 / a) * (np.arctan(c / a) - np.arctan(b / a))

    def plot(self) -> plt.Figure:
        """
        Returns a matplotlib Figure showing the function over the integration bounds.
        The region under the curve is shaded. Includes an annotation showing the width Γ 
        and the center, returning an aesthetically pleasing academic style plot with a subtle grid.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Clean academic style styling: no top/right spines, subtle dashed grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Determine domain bounds
        x_min, x_max = self.integration_bounds
        x = np.linspace(x_min, x_max, 1000)
        y = self(x)
        
        # Plot curve and shade under
        ax.plot(x, y, color='black', linewidth=1.5, label='Theory Curve')
        ax.fill_between(x, y, color='steelblue', alpha=0.2)
        
        # Highlight center
        ax.axvline(self.center, color='firebrick', linestyle=':', alpha=0.5)
        
        # Annotation text for width and center
        info_str = f"Center ($M$): {self.center}\nWidth ($\\Gamma$): {self.width}"
        ax.text(
            0.95, 0.95, info_str, 
            transform=ax.transAxes, 
            fontsize=11,
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
        )
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('1D Breit-Wigner Resonance', fontsize=14, pad=10)
        
        fig.tight_layout()
        return fig
