import torch
import torch.nn as nn
import math

class AffineCouplingLayer2D(nn.Module):
    """
    2D Affine Coupling Layer.
    Transforms x -> z (forward) and z -> x (inverse).
    Mask determines which dimension is passed through unchanged and used to parameterize
    the transformation of the other dimension.
    """
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))

        # MLP: [1 -> 64 -> 64 -> 2] (outputs scale + shift)
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        """
        Data -> Latent.
        Returns: z, log_det_jacobian
        """
        x_id = x * self.mask

        # Extract the unmasked column to feed into the MLP.
        # Since mask is e.g. [1, 0], x_id[:, 0] is non-zero, x_id[:, 1] is zero.
        # We can sum along dim=1 to get the (N, 1) input, or just select based on mask.
        unmasked = x[:, self.mask == 1].view(-1, 1)

        out = self.mlp(unmasked)
        # Scale output further through tanh*2 for stability
        s = torch.tanh(out[:, 0]) * 2.0
        t = out[:, 1]

        # Shape expansion for broadcasting
        s = s.unsqueeze(1)
        t = t.unsqueeze(1)

        # The shifted/scaled part applies to the masked-out dimensions
        z = x_id + (1 - self.mask) * (x * torch.exp(s) + t)

        log_det = torch.sum((1 - self.mask) * s, dim=1)
        return z, log_det

    def inverse(self, z):
        """
        Latent -> Data.
        Returns: x, log_det_jacobian_inv
        """
        z_id = z * self.mask
        unmasked = z[:, self.mask == 1].view(-1, 1)

        out = self.mlp(unmasked)
        s = torch.tanh(out[:, 0]) * 2.0
        t = out[:, 1]

        s = s.unsqueeze(1)
        t = t.unsqueeze(1)

        x = z_id + (1 - self.mask) * ((z - t) * torch.exp(-s))

        log_det_inv = torch.sum((1 - self.mask) * (-s), dim=1)
        return x, log_det_inv

class RealNVP2D(nn.Module):
    """
    RealNVP normalizing flow for 2D integration over [0,1]^2.
    Stack of alternating AffineCouplingLayer2D blocks with logit preprocessing.
    """
    def __init__(self):
        super().__init__()

        masks = [[1, 0], [0, 1], [1, 0], [0, 1]]
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer2D(mask) for mask in masks
        ])

        # Standard Gaussian prior p(z) ~ N(0, I)
        self.register_buffer("prior_mean", torch.zeros(2))
        self.register_buffer("prior_var", torch.ones(2))

    def log_prob(self, x):
        """
        Computes log q(x) for x in [0, 1]^2.
        Includes the Jacobian of the logit transform.
        """
        # Clamp to avoid log(0) or log(1)
        x = torch.clamp(x, 1e-6, 1.0 - 1e-6)

        # Logit transform: [0, 1]^2 -> R^2
        x_logit = torch.log(x / (1 - x))

        # Jacobian of logit transform
        # dx_logit / dx = 1 / x + 1 / (1 - x) = 1 / (x * (1 - x))
        # log_det_logit = sum_i log(1 / (x_i * (1 - x_i))) = -sum_i (log x_i + log(1 - x_i))
        log_det_logit = -torch.sum(torch.log(x) + torch.log(1 - x), dim=1)

        # Pass through the flow blocks
        z = x_logit
        log_det_flow = 0.0
        for layer in self.coupling_layers:
            z, log_det_layer = layer.forward(z)
            log_det_flow += log_det_layer

        # Log probability under the prior
        log_prob_prior = -0.5 * (
            torch.sum((z - self.prior_mean) ** 2 / self.prior_var, dim=1)
            + 2 * math.log(2 * math.pi)
        )

        # Total log probability
        return log_prob_prior + log_det_flow + log_det_logit

    def sample(self, n):
        """
        Draws n samples from the flow, returning x in [0, 1]^2.
        Returns: x, log_q_x
        """
        z = torch.randn(n, 2, device=self.prior_mean.device) * torch.sqrt(self.prior_var) + self.prior_mean

        log_det_inv_flow = 0.0
        for layer in reversed(self.coupling_layers):
            z, log_det_inv_layer = layer.inverse(z)
            log_det_inv_flow += log_det_inv_layer

        # apply sigmoid to get back to [0, 1]^2
        x = torch.sigmoid(z)

        # Return samples
        return x
