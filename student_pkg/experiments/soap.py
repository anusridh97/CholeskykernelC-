"""Minimal clean-room SOAP optimizer, for use as a baseline.

SOAP (Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam", 2024)
runs Adam in the eigenbasis of the Shampoo preconditioner. For a 2-D weight
W ∈ R^{m×n} with gradient G:

  * maintain EMA Shampoo factors  L = E[G Gᵀ] (m×m),  R = E[Gᵀ G] (n×n);
  * every ``precondition_frequency`` steps, refresh their eigenvectors Q_L, Q_R
    (this eigendecomposition is SOAP's expensive, amortized step — the O(r³)
    cost our incremental method replaces with an O(r²) every-step refresh);
  * rotate the gradient into the eigenbasis  G' = Q_Lᵀ G Q_R, run Adam moments
    there, and rotate the update back.

1-D parameters (biases, norms) fall back to plain Adam. This is a compact
reference implementation for benchmarking, not the authors' tuned version.
"""

from __future__ import annotations

import torch


class SOAP:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, precondition_frequency=10):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.freq = precondition_frequency
        self.t = 0
        self.state = {id(p): {} for p in self.params}

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def _eig(self, A):
        # Symmetric eigendecomposition; return eigenvectors (columns).
        _, Q = torch.linalg.eigh(A)
        return Q

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            st = self.state[id(p)]

            if g.dim() != 2:
                # Adam fallback for non-matrix params.
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                    st["v"] = torch.zeros_like(g)
                st["m"].mul_(self.b1).add_(g, alpha=1 - self.b1)
                st["v"].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                mh = st["m"] / (1 - self.b1 ** self.t)
                vh = st["v"] / (1 - self.b2 ** self.t)
                upd = mh / (vh.sqrt() + self.eps)
                if self.wd:
                    p.mul_(1 - self.lr * self.wd)
                p.add_(upd, alpha=-self.lr)
                continue

            m, n = g.shape
            if "L" not in st:
                st["L"] = torch.zeros(m, m, device=g.device, dtype=g.dtype)
                st["R"] = torch.zeros(n, n, device=g.device, dtype=g.dtype)
                st["QL"] = torch.eye(m, device=g.device, dtype=g.dtype)
                st["QR"] = torch.eye(n, device=g.device, dtype=g.dtype)
                st["m"] = torch.zeros_like(g)
                st["v"] = torch.zeros_like(g)

            # Update Shampoo factors (EMA).
            st["L"].mul_(self.b2).add_(g @ g.mT, alpha=1 - self.b2)
            st["R"].mul_(self.b2).add_(g.mT @ g, alpha=1 - self.b2)
            # Periodically refresh the eigenbases (the amortized O(r^3) step).
            if self.t % self.freq == 1 or self.t == 1:
                st["QL"] = self._eig(st["L"])
                st["QR"] = self._eig(st["R"])

            QL, QR = st["QL"], st["QR"]
            g_rot = QL.mT @ g @ QR                       # rotate into eigenbasis
            st["m"].mul_(self.b1).add_(g_rot, alpha=1 - self.b1)
            st["v"].mul_(self.b2).addcmul_(g_rot, g_rot, value=1 - self.b2)
            mh = st["m"] / (1 - self.b1 ** self.t)
            vh = st["v"] / (1 - self.b2 ** self.t)
            upd_rot = mh / (vh.sqrt() + self.eps)
            upd = QL @ upd_rot @ QR.mT                   # rotate back
            if self.wd:
                p.mul_(1 - self.lr * self.wd)
            p.add_(upd, alpha=-self.lr)

    def num_blocks(self):
        return None
