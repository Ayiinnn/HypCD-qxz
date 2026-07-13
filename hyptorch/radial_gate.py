"""
Stage-1 v2 radial gate: annular reparameterization of ``ToPoincare`` that
PRESERVES the per-sample norm (depth) signal of the backbone.

Recap of the failure being fixed
--------------------------------
Post-LN ViT norms (~20-30) all exceed the clip radius ``cr`` (~1-2.3), so the
baseline ``clip -> expmap0``:
  * projects EVERY sample onto one shell  -> entailment feasible set is empty
    (same-depth exterior angle > 90 deg >> scaled aperture ~17 deg);
  * has Jacobian (r/||x||)(I - uu^T)      -> radial gradient exactly zero,
    both for any depth parameter AND for the backbone.

v1 of this module (per-level shell ``z = expmap0(s_level * x/||x||)``) fixed
the parameter side (s_level learnable) but the ``x/||x||`` normalization kept
the BACKBONE-side radial null-space: the trainable last block still receives
zero gradient through the feature norm, and DINO's (small but real) per-sample
norm variation is destroyed.

v2 (this file): rescale instead of normalize
--------------------------------------------
    t_i   = log( r_i / r0_level ),            r_i = ||x_i||,  r0 = init median
    tt_i  = T * tanh( t_i / T )               (smooth radial trust region)
    s_i   = s_level * exp( kappa_level * tt_i )
    s'_i  = min( s_i, R_guard )               (hard guard, rarely active)
    z_i   = expmap0( s'_i * x_i / r_i )

  * ``s_level``  : per-level bounded learnable anchor (as in v1);
  * ``kappa``    : per-level bounded learnable ADMISSION GAIN of the backbone
                   norm signal. kappa = 1  ==  the plain rescale
                   ``v = (s_level / r0) * x``  inside the trust band
                   (exp(log(r/r0)) = r/r0);  kappa -> 0  ==  v1 exactly.
                   The model can amplify, keep, or reject the DINO depth
                   signal -- "allow, don't force".
  * ``T=log(band)``: replaces "linear until an always-hard clip" in the tails
                   by a smooth saturation; typical samples (|t| << T) pass
                   through ~identity, outliers are bounded with small-but-
                   nonzero gradient. This IS the purpose of feature clipping
                   (bounded depth, gradients alive) achieved implicitly.
  * ``R_guard``  : the literal hard cap, sized to be inactive in normal
                   operation (occasional-cap semantics of Guo et al. CVPR'22,
                   restored from the degenerate always-active regime).

Nesting / safety ladder:   baseline clip  ==  v1 at init  ==  v2 with kappa=0;
v2 with kappa=1 == the plain median-rescale. All members share: direction
pathway parameter-free and untouched (the gate rescales along x_i only, so it
can never rotate an embedding), depths confined to a bounded annulus, level
anchors + radial order loss for inter-level hierarchy.

Orthogonality property: t is centered by the init MEDIAN, so batch-mean(tt)~0.
A uniform "push this level deeper/shallower" pressure therefore lands on
``s_level`` and (to first order) NOT on ``kappa``; only pressure CORRELATED
with the per-sample norm moves ``kappa``. The two scalars are near-orthogonal
coordinates: level anchor vs. signal admission.

New parameters: 2 anchors + 2 gains = 4 scalars. r0 is a frozen buffer
(median of the first batch per level = the initial post-LN median), stored in
the state_dict for eval/resume consistency.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import hyptorch.pmath as pmath
from hyptorch.nn import get_curvature


# --------------------------------------------------------------------------- #
# Entailment-cone validity floor (unchanged from v1).
# half_aperture(x) = asin( 2K / (sqrt(c) ||x_space||) ) needs
# ||x_space|| >= 2K/sqrt(c); mapping the Lorentz lift of a Poincare point back
# through expmap0 gives the smallest valid tangent norm.
# --------------------------------------------------------------------------- #
def min_valid_tangent_norm(c: float, min_radius: float = 0.1, margin: float = 1.15) -> float:
    c = float(c)
    a = min_radius / math.sqrt(c)
    y = (-1.0 + math.sqrt(1.0 + 4.0 * c * a * a)) / (2.0 * c * a)
    s = math.atanh(min(math.sqrt(c) * y, 1.0 - 1e-6)) / math.sqrt(c)
    return margin * s


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


class RadialToPoincare(nn.Module):
    """Drop-in replacement for ``ToPoincare(clip_r=cr)``; see module docstring.

    forward(x, level='image') -> Poincare-ball embedding. The default level
    keeps untouched call sites (e.g. ``test()``) working unchanged.
    """

    def __init__(
        self,
        c,
        cr: float,
        levels=("image", "object"),
        s_min_ratio: float = 0.35,
        s_max_ratio: float = 1.10,
        min_radius: float = 0.1,
        riemannian: bool = False,
        validity_margin: float = 1.15,
        # ---- v2: per-sample norm signal ----
        kappa_init: float = 1.0,
        kappa_max: float = 2.0,
        norm_band: float = 1.3,
        guard_ratio: float = 1.5,
        detach_norm: bool = False,
    ):
        super().__init__()
        if callable(c) and not torch.is_tensor(c):
            c_val = float(c())
        elif torch.is_tensor(c):
            c_val = float(c.detach().cpu())
        else:
            c_val = float(c)
        if cr is None or cr <= 0:
            raise ValueError("RadialToPoincare needs the baseline clip radius cr > 0.")
        if not norm_band > 1.0:
            raise ValueError("norm_band must be > 1 (it is a multiplicative trust band).")

        self.c = c
        self.cr = float(cr)
        self.levels = tuple(levels)
        self.detach_norm = bool(detach_norm)

        floor = min_valid_tangent_norm(c_val, min_radius=min_radius, margin=validity_margin)
        self.s_min = max(s_min_ratio * self.cr, floor)
        self.s_max = s_max_ratio * self.cr
        self.validity_floor = floor
        if not (self.s_min < self.cr < self.s_max):
            raise ValueError(
                "need s_min < cr < s_max, got s_min={:.4f} cr={:.4f} s_max={:.4f} "
                "(validity floor {:.4f}).".format(self.s_min, self.cr, self.s_max, floor))

        # hard guard: rarely active by sizing. With the trust band, the pre-
        # guard depth is already bounded by s_max * band**kappa_max; the guard
        # only bites the (kappa high) x (norm tail) corner -- the intended
        # occasional-cap semantics of feature clipping.
        self.r_guard = guard_ratio * self.cr
        if not self.r_guard > self.s_max:
            raise ValueError("guard_ratio * cr must exceed s_max_ratio * cr.")

        # per-level anchor s_level in (s_min, s_max), init == cr (v1 semantics).
        p = (self.cr - self.s_min) / (self.s_max - self.s_min)
        self.alpha = nn.ParameterDict({
            lv: nn.Parameter(torch.tensor([_logit(p)], dtype=torch.float32)) for lv in self.levels
        })
        # per-level admission gain kappa in (0, kappa_max), init == kappa_init.
        self.kappa_max = float(kappa_max)
        self.raw_kappa = nn.ParameterDict({
            lv: nn.Parameter(torch.tensor([_logit(kappa_init / self.kappa_max)], dtype=torch.float32))
            for lv in self.levels
        })
        # trust region half-width in log-norm units.
        self.T = math.log(norm_band)

        # frozen per-level norm reference r0 (median of the FIRST batch seen =
        # the initial post-LN median). Buffer => saved/loaded with state_dict.
        for lv in self.levels:
            self.register_buffer("r0_" + lv, torch.zeros(1))

        # mirror ToPoincare's optional Riemannian gradient rescaling.
        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c_val
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

        # read-only per-level batch stats for logging: {level: (mean, std, guard_rate)}
        self.last = {}

    # ---- learnable scalars ------------------------------------------------- #
    def depth(self, level: str) -> torch.Tensor:
        """Level anchor s_level (shape (1,), differentiable). Order loss uses this."""
        if level not in self.alpha:
            raise KeyError("unknown level '{}' (have {})".format(level, self.levels))
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.alpha[level])

    def kappa(self, level: str) -> torch.Tensor:
        """Admission gain kappa_level in (0, kappa_max) (shape (1,), differentiable)."""
        if level not in self.raw_kappa:
            raise KeyError("unknown level '{}' (have {})".format(level, self.levels))
        return self.kappa_max * torch.sigmoid(self.raw_kappa[level])

    def _r0(self, level: str, r: torch.Tensor) -> torch.Tensor:
        buf = getattr(self, "r0_" + level)
        if bool((buf <= 0).all()):
            with torch.no_grad():
                buf.copy_(r.detach().float().median().view(1))
        return buf

    # ---- forward ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, level: str = "image") -> torch.Tensor:
        if level not in self.levels:
            raise KeyError("unknown level '{}' (have {})".format(level, self.levels))
        c = get_curvature(self.c, x)
        self.riemannian.c = c.detach() if torch.is_tensor(c) else c

        r = torch.norm(x, dim=-1, keepdim=True) + 1e-5              # (N, 1)
        r0 = self._r0(level, r)

        rr = r.detach() if self.detach_norm else r
        t = torch.log(rr.float() / r0)                               # log norm-ratio
        t = self.T * torch.tanh(t / self.T)                          # smooth trust region
        s = self.depth(level) * torch.exp(self.kappa(level) * t)     # per-sample depth

        # hard guard. Since ||v|| = s by construction (v = x * s / r), the
        # norm-clip  v * min(1, R/||v||)  is exactly  clamp(s, max=R).
        s_bar = torch.clamp(s, max=self.r_guard)

        with torch.no_grad():
            self.last[level] = (
                float(s_bar.mean()), float(s_bar.std()),
                float((s > self.r_guard).float().mean()),
            )

        v = x * (s_bar / r.float()).to(x.dtype)
        return self.grad_fix(pmath.project(pmath.expmap0(v, c=c), c=c))

    # ---- logging helpers ----------------------------------------------------- #
    def depths_str(self) -> str:
        parts = []
        for lv in self.levels:
            r0 = float(getattr(self, "r0_" + lv))
            parts.append("[{}] s={:.4f} k={:.3f} r0={:.2f}".format(
                lv, self.depth(lv).item(), self.kappa(lv).item(), r0))
            if lv in self.last:
                m, sd, g = self.last[lv]
                parts[-1] += " depth={:.3f}+-{:.3f} guard={:.1%}".format(m, sd, g)
        return " | ".join(parts)

    def stats_str(self) -> str:
        def one(lv):
            if lv in self.last:
                m, sd, g = self.last[lv]
                return "{:.3f}+-{:.3f}".format(m, sd) + ("!" if g > 0 else "")
            return "-"
        return "s({})={} k=({})".format(
            "/".join(lv[:3] for lv in self.levels),
            "/".join(one(lv) for lv in self.levels),
            "/".join("{:.2f}".format(self.kappa(lv).item()) for lv in self.levels))

    def extra_repr(self):
        return ("c={}, cr={}, levels={}, s in [{:.4f},{:.4f}] (validity floor {:.4f}), "
                "band={:.3f}, kappa_max={}, r_guard={:.4f}, detach_norm={}").format(
                    self.c, self.cr, self.levels, self.s_min, self.s_max,
                    self.validity_floor, math.exp(self.T), self.kappa_max,
                    self.r_guard, self.detach_norm)

    def describe(self) -> str:
        return ("radial gate v2 | z = expmap0( min(s_lv * (r/r0)^[k_lv via tanh band], R_guard) * x/r ) | "
                "anchors init cr={:.4f}, annulus [{:.4f},{:.4f}], band x{:.2f}, "
                "R_guard={:.4f} (occasional-cap), kappa init {} in (0,{}) | k=0 == v1 shell, "
                "k=1 == plain (s/r0)*x rescale | {}").format(
                    self.cr, self.s_min, self.s_max, math.exp(self.T), self.r_guard,
                    "/".join("{:.2f}".format(self.kappa(lv).item()) for lv in self.levels),
                    self.kappa_max, self.depths_str())


# --------------------------------------------------------------------------- #
# Radial ordering loss on the LEVEL ANCHORS (unchanged from v1): deterministic
# in the two anchor scalars, guarantees inter-level separation emerges; the
# per-sample spread (kappa * norm signal) rides on top of the anchors.
# --------------------------------------------------------------------------- #
def radial_order_loss(gate: RadialToPoincare, parent_level: str, child_level: str,
                      margin: float = 0.2) -> torch.Tensor:
    return torch.relu(gate.depth(parent_level) - gate.depth(child_level) + margin).squeeze()