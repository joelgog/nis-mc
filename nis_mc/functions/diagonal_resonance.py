"""
Diagonal 2D Resonance function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

from nis_mc.functions.breit_wigner import BreitWigner1D


class DiagonalResonance2D:
    """
    This function mimics collinear photon radiation in e+e- → γγ,
    where energy-momentum conservation forces correlations between the two photon momenta.
    """

    def __init__(self, bounds=((0.0, 1.0), (0.0, 1.0))):
        self.bounds = bounds

        # A sharp ridge along the anti-diagonal x1 + x2 = 1 (simulating collinear radiation)
        self.bw_anti_diag = BreitWigner1D(center=0.0, width=0.03)

        # Crossed by a softer ridge along x1 = x2
        self.bw_diag = BreitWigner1D(center=0.0, width=0.1)

        # Compute ground truth (cached)
        self._true_integral = self._compute_true_integral()

    def _compute_true_integral(self) -> float:
        """Computes the ground truth integral using scipy.dblquad."""
        def integrand(x2, x1):
            # dblquad passes scalar values
            return self(np.array([[x1, x2]]))[0]

        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]

        val, _error = dblquad(
            integrand,
            x_min, x_max,
            lambda x: y_min, lambda x: y_max,
            epsabs=1e-5, epsrel=1e-5
        )
        return val

    @property
    def true_integral(self) -> float:
        """Cached true integral computed via scipy.dblquad."""
        return self._true_integral

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the 2D diagonal resonance.

        Args:
            x: Array of shape (N, 2)

        Returns:
            Array of shape (N,)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]

        # f(x1, x2) = BW(x1 + x2 - 1.0, center=0, width=0.03) * BW(x1 - x2, center=0, width=0.1)
        term1 = self.bw_anti_diag(x1 + x2 - 1.0)
        term2 = self.bw_diag(x1 - x2)

        return term1 * term2

    def plot_heatmap(self) -> plt.Figure:
        """
        2D heatmap showing the function landscape.
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]

        x_range = np.linspace(x_min, x_max, 250)
        y_range = np.linspace(y_min, y_max, 250)
        X, Y = np.meshgrid(x_range, y_range)

        # Flatten grid points
        points = np.c_[X.ravel(), Y.ravel()]
        Z = self(points).reshape(X.shape)

        # Plot heatmap
        im = ax.imshow(
            Z,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            cmap='magma',
            aspect='auto'
        )

        fig.colorbar(im, ax=ax, label='$f(x_1, x_2)$')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('2D Diagonal Correlated Resonance')

        # Clean academic style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_with_vegas_grid(self, grid_edges_x: np.ndarray, grid_edges_y: np.ndarray) -> plt.Figure:
        """
        Overlay the VEGAS adaptive grid as white lines on the heatmap.
        This visually shows VEGAS building a square grid that cannot align with a diagonal ridge.
        """
        fig = self.plot_heatmap()
        ax = fig.gca()

        # Plot vertical and horizontal grid lines
        for gx in grid_edges_x:
            ax.axvline(gx, color='white', linewidth=0.5, alpha=0.6)
        for gy in grid_edges_y:
            ax.axhline(gy, color='white', linewidth=0.5, alpha=0.6)

        ax.set_title('Diagonal Resonance with Overlaid VEGAS Grid')
        return fig
