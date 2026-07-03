"""
Hyperbolic entailment-cone primitives for HypCD (Poincare ball).

HypCD embeds features on the Poincare ball (``hyptorch.nn.ToPoincare``), whereas
the entailment cone of HyCoCLIP / MERU is defined on the Lorentz hyperboloid.
Rather than re-deriving the cone in the Poincare ball, we *reuse* the
battle-tested Lorentz formulas from HyCoCLIP verbatim (``oxy_angle`` and
``half_aperture`` below are copied from ``hycoclip/lorentz.py``) and map the
Poincare features onto the hyperboloid with a small, exact bridge
(``poincare_to_lorentz``). Everything is carried out with HypCD's own curvature
``c`` so the cone lives in the *same* feature space as the rest of the model.

References
----------
* Ganea et al., "Hyperbolic Entailment Cones for Learning Hierarchical
  Embeddings", ICML 2018.
* Desai et al. (MERU) / Pal et al. (HyCoCLIP) -- ``oxy_angle`` / ``half_aperture``.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Curvature handling (mirror of hyptorch.nn.get_curvature, kept local to avoid
# an import cycle with hyptorch.nn).
# --------------------------------------------------------------------------- #
def resolve_c(c, ref: Tensor | None = None):
    """Resolve a curvature that may be a float, a tensor or a LearnableCurvature."""
    if callable(c) and not torch.is_tensor(c):
        c = c()
    if ref is None:
        return c
    if torch.is_tensor(c):
        return c.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(c, dtype=ref.dtype, device=ref.device)


# --------------------------------------------------------------------------- #
# Poincare ball  ->  Lorentz hyperboloid (same curvature magnitude ``c``).
#
# HypCD's ball (``pmath.project``) has radius ``1/sqrt(c)``. The exact map onto
# the hyperboloid with constraint <x,x>_L = -1/c (the convention used by the
# HyCoCLIP Lorentz functions, which recompute the time component from the space
# component) is:
#
#     x_space = 2 * y / (1 - c * ||y||^2)
#
# (one can verify x_time = sqrt(1/c + ||x_space||^2) = (1 + c||y||^2) /
#  (sqrt(c)(1 - c||y||^2)), i.e. the standard stereographic lift).
# --------------------------------------------------------------------------- #
def poincare_to_lorentz(y: Tensor, c, eps: float = 1e-6) -> Tensor:
    """Lift Poincare-ball points ``y`` to the Lorentz hyperboloid space component."""
    c = resolve_c(c, y)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    denom = (1.0 - c * y2).clamp_min(eps)
    return 2.0 * y / denom


# --------------------------------------------------------------------------- #
# Lorentz entailment-cone primitives (copied from hycoclip/lorentz.py).
# They take *space components* of hyperboloid points and recompute the time
# component internally, so they accept the output of ``poincare_to_lorentz``.
# --------------------------------------------------------------------------- #
def half_aperture(
    x: Tensor, curv, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """Half-aperture of the entailment cone with apex at ``x`` (values in (0, pi/2))."""
    curv = resolve_c(curv, x)
    rc = (curv ** 0.5) if not torch.is_tensor(curv) else curv.clamp_min(eps) ** 0.5
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * rc + eps)
    return torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))


def oxy_angle(x: Tensor, y: Tensor, curv, eps: float = 1e-8) -> Tensor:
    """Exterior angle at ``x`` in the hyperbolic triangle O-x-y (values in (0, pi))."""
    curv = resolve_c(curv, x)
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1))

    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl ** 2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


# --------------------------------------------------------------------------- #
# Entailment loss (HyCoCLIP relu-cone form), operating on Poincare features.
# --------------------------------------------------------------------------- #
def entailment_cone_loss(
    parent: Tensor,
    child: Tensor,
    c,
    aperture_scale: float = 1.2,
    min_radius: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    """``relu(oxy_angle(parent, child) - aperture_scale * half_aperture(parent))``.

    ``parent`` is the apex of the cone (the *more generic* concept, pulled toward
    the origin); ``child`` is pushed to lie inside that cone (farther from the
    origin). Both are Poincare-ball points in the *same* ball (curvature ``c``).
    """
    parent_s = poincare_to_lorentz(parent, c)
    child_s = poincare_to_lorentz(child, c)
    angle = oxy_angle(parent_s, child_s, curv=c)
    aper = half_aperture(parent_s, curv=c, min_radius=min_radius)
    loss = torch.clamp(angle - aperture_scale * aper, min=0)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# --------------------------------------------------------------------------- #
# Pairwise variants (added for class-level supervision of the object branch).
#
# ``oxy_angle`` above is elementwise (row i of x with row i of y). For
# supervision structures that pair every parent with several children (e.g.
# cross-view same-instance pairs, or all labelled same-class pairs, as in
# SupCon-style supervision) we need the full (N, M) angle matrix. The math is
# identical -- only the inner product / norms are batched with a matmul.
# --------------------------------------------------------------------------- #
def oxy_angle_pairwise(x: Tensor, y: Tensor, curv, eps: float = 1e-8) -> Tensor:
    """Pairwise exterior angle: entry (i, j) = oxy_angle(x[i], y[j]).

    ``x`` (N, D) and ``y`` (M, D) are *space components* of hyperboloid points
    (the output of :func:`poincare_to_lorentz`). Returns an (N, M) tensor.
    Row-wise it matches :func:`oxy_angle`: ``oxy_angle_pairwise(x, y).diag()``
    equals ``oxy_angle(x, y)`` (up to floating-point associativity).
    """
    curv = resolve_c(curv, x)
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1))          # (N,)
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1))          # (M,)

    xy = x @ y.transpose(-1, -2)                                       # (N, M)
    c_xyl = curv * (xy - x_time[:, None] * y_time[None, :])            # (N, M)

    acos_numer = y_time[None, :] + c_xyl * x_time[:, None]             # (N, M)
    acos_denom = torch.sqrt(torch.clamp(c_xyl ** 2 - 1, min=eps))      # (N, M)
    x_norm = torch.norm(x, dim=-1)                                     # (N,)
    acos_input = acos_numer / (x_norm[:, None] * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


def entailment_cone_violation_pairwise(
    parent: Tensor,
    child: Tensor,
    c,
    aperture_scale: float = 1.2,
    min_radius: float = 0.1,
) -> Tensor:
    """Pairwise cone violations: entry (i, j) is the relu-cone violation of
    child ``j`` w.r.t. the cone whose apex is parent ``i``.

    Inputs are Poincare-ball points (same ball, curvature ``c``); this is the
    (N, M) generalization of :func:`entailment_cone_loss` with
    ``reduction='none'``. The caller applies its own pair weighting.
    """
    parent_s = poincare_to_lorentz(parent, c)
    child_s = poincare_to_lorentz(child, c)
    angle = oxy_angle_pairwise(parent_s, child_s, curv=c)              # (N, M)
    aper = half_aperture(parent_s, curv=c, min_radius=min_radius)      # (N,)
    return torch.clamp(angle - aperture_scale * aper[:, None], min=0)