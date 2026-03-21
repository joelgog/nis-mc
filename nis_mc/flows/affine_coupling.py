"""
Affine Coupling module.

Implements a single, minimal affine coupling layer using NumPy for teaching.

Architecture: y = sigmoid(s · logit(u) + t) where u ~ U(0,1).

Key property: by applying logit FIRST (mapping U(0,1) → Logistic(0,1))
before the affine shift-and-scale, the output y covers the FULL interval
(0,1) for any finite (s, t). This is essential for unbiased importance
sampling over the entire integration domain.

Behaviour:
  s=1, t=0  →  y = u  (identity, uniform proposal)
  s→0, t=0  →  y concentrates around 0.5
  s→0, t=L  →  y concentrates around sigmoid(L)
"""

import numpy as np


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-7, 1.0 - 1e-7)
    return np.log(x / (1.0 - x))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class AffineCouplingLayer1D:
    """
    A minimal 1D Affine Coupling Layer for Normalizing Flows.

    Maps u ~ U(0,1) to y ~ q(y) via:
        z = logit(u)              # U(0,1) -> Logistic(0,1)
        w = s * z + t             # shift & scale
        y = sigmoid(w)            # -> (0,1), full support

    Attributes:
        s (float): Scale parameter. s=1 gives uniform; smaller s concentrates.
        t (float): Translation. sigmoid(t) is the mode of the distribution.
    """

    def __init__(self, s: float = 1.0, t: float = 0.0):
        self.s = s
        self.t = t

    def forward(self, u: np.ndarray) -> np.ndarray:
        """
        Forward map: u ~ U(0,1)  →  y in (0,1) with full support.

            y = sigmoid(s · logit(u) + t)

        Special cases:
            s=1, t=0  →  y = sigmoid(logit(u)) = u  (identity)
            s<1, t=0  →  distribution concentrates around 0.5
        """
        z = _logit(u)            # U(0,1) → Logistic(0,1)
        w = self.s * z + self.t  # affine transform
        y = _sigmoid(w)          # → (0,1)
        return y

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse map: y  →  u

            u = sigmoid((logit(y) - t) / s)
        """
        w = _logit(y)
        z = (w - self.t) / self.s
        u = _sigmoid(z)
        return u

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        """
        Log density log q(y) under the logit-sigmoid flow.

        Derivation (change-of-variables from u ~ U(0,1)):
            y = sigmoid(s · logit(u) + t)
            dy/du = s · y(1-y) / (u(1-u))
            log q(y) = -log|dy/du|
                     = log(u) + log(1-u) - log|s| - log(y) - log(1-y)
            where u = sigmoid((logit(y) - t) / s)  [the inverse]

        Properties:
            - Integrates to 1 over (0,1) for any (s,t)  ✓
            - s=1,t=0:  log q(y) = 0  (uniform)  ✓
            - Small s:  density peaks sharply at sigmoid(t)  ✓
        """
        y = np.clip(y, 1e-7, 1.0 - 1e-7)
        u = self.inverse(y)
        u = np.clip(u, 1e-7, 1.0 - 1e-7)

        log_q_y = (
            np.log(u) + np.log(1.0 - u)
            - np.log(np.abs(self.s))
            - np.log(y) - np.log(1.0 - y)
        )
        return log_q_y

    def diagram_ascii(self):
        """Prints an ASCII diagram of the transformation."""
        print(f"""
        AffineCouplingLayer1D (logit-sigmoid):
        ======================================

           u ~ U(0,1)     z=logit(u)    w=s·z+t    y=sigmoid(w)
          [Base Space]  → Logistic(0,1) →  affine  →  [Target Space]

        Current Parameters:
          s = {self.s:.4f}   (s<1 concentrates, s=1 is uniform)
          t = {self.t:.4f}   (mode at sigmoid(t) = {_sigmoid(self.t):.4f})

        Jacobian |dy/du| = s · y(1-y) / (u(1-u))
        """)
