import math

import torch
import torch.nn as nn
import torch.nn.init as init

import hyptorch.pmath as pmath


# ============================================================================
# HNN++ (added) — everything between the "HNN++ (added)" banners is the only
# change in this file relative to ``nn_org.py``. No existing class below is
# modified, so the fixed-curvature baseline is byte-for-byte preserved.
#
# This adds the Poincaré fully-connected layer of Hyperbolic Neural Networks++
# (Shimizu et al., ICLR 2021), Eqs. (6)-(7), as a drop-in replacement for the
# HNN/Ganea-2018 Möbius affine layer (``HypLinear``) used as the SimGCD head.
# ============================================================================
def _resolve_c(c, ref):
    r"""Resolve a curvature spec to a scalar tensor on ``ref``'s device/dtype.

    Accepts:
      * a python ``float``                  (the fixed-c ``_org`` case),
      * a ``torch.Tensor`` (gradients kept) (a learnable curvature passed in
        per-forward as ``classifier(x, c=<tensor>)``), or
      * a zero-arg ``callable`` returning either of the above (e.g. a
        ``LearnableCurvature`` module / ``MultiCurvatureManager.value``).

    Keeping this resolver here lets :class:`PoincareFC` stay a drop-in for the
    fixed-curvature setup while already being ready for the future migration to
    the single-/compound-learnable-curvature training scripts (which call the
    classifier with a per-role curvature tensor via ``c=...``). Mirrors the
    role of ``get_curvature`` in ``nn.py`` / ``nn_mc.py`` without importing the
    ``LearnableCurvature`` class into this baseline file.
    """
    if callable(c) and not torch.is_tensor(c):
        c = c()
    return torch.as_tensor(c).type_as(ref)


class PoincareFC(nn.Module):
    r"""Poincaré FC layer (HNN++, Shimizu et al. 2021), Eqs. (6)-(7).

    Drop-in replacement for :class:`HypLinear` (the HNN Möbius affine layer
    ``expmap0(A logmap0(x)) ⊕_c b``). Identical constructor signature and I/O
    contract: maps a point on ``B^in_c`` to a point on ``B^out_c``. The SimGCD
    head therefore needs no other change — the output coordinates are still
    used directly as class logits (``logits.argmax(1)``).

    Parameters (same count as the Euclidean / HNN linear layer = ``n*m + m``):
        ``weight`` Z : (out_features, in_features), each row ``z_k ∈ T_0 B^n_c``
        ``bias``   r : (out_features,)             one scalar bias per output

    Unlike :class:`HypLinear` — whose per-output contour shape is fixed by the
    row direction alone, with norm/bias only scaling/shifting it — here each
    output coordinate's discriminative surface is a genuine *Poincaré*
    hyperplane (the geodesic hyperplane through ``expmap0(r_k [z_k])`` and
    orthogonal to ``z_k``), so the bias ``r_k`` reshapes the contour along the
    geodesics (paper Fig. 3).

    Notes
    -----
    * Single weight matrix ``Z`` (as in the paper's text and as in
      :class:`HypLinear`), *not* the weight-normalised (decoupled
      norm/direction) parameterisation used in the official ``mil-tokyo`` repo.
      This is intentional: it isolates the FC-formula change (HNN vs HNN++) for
      a fair comparison against the :class:`HypLinear` baseline. The numerical
      computation below is otherwise identical to the official
      ``poincare_linear`` / ``unidirectional_poincare_mlr``.
    * ``forward`` accepts a per-call ``c`` override (float or grad tensor); for
      the future compound-curvature migration, call
      ``classifier(x, c=c_dict["c_cls_*"])`` exactly as ``nn_mc`` does.
    """

    def __init__(self, in_features, out_features, c, bias=True, clamp=15.0):
        super(PoincareFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        # Ceiling on sqrt(c)*v_k before sinh(): beyond it the output already
        # saturates at the ball boundary, so this only guards against overflow
        # (same spirit / value as the tanh clamp in ``pmath``).
        self.clamp = clamp
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))  # Z
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))             # r
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # HNN++ App. E.1: each element of Z ~ N(0, std=(2 n m)^{-1/2}); r := 0.
        std = (2.0 * self.in_features * self.out_features) ** -0.5
        init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        c = _resolve_c(c, x)                       # -> scalar tensor (grad preserved)
        rc = c.clamp_min(1e-15).sqrt()             # sqrt(c)

        # --- Unidirectional Poincaré MLR score v_k(x)  (paper Eq. 6) ---------
        # With lambda^c_x = 2/(1 - c|x|^2) folded in, the argument of arcsinh is
        #   [ 2 <rc x, [z_k]> cosh(2 rc r_k) - (1 + c|x|^2) sinh(2 rc r_k) ]
        #   / (1 - c|x|^2)
        z_norm = self.weight.norm(dim=1).clamp_min(1e-15)        # (out,)   ||z_k||
        z_unit = self.weight / z_norm.unsqueeze(1)               # (out,in) [z_k]
        cx2 = c * x.pow(2).sum(dim=-1, keepdim=True)             # (...,1)  c|x|^2
        rcxz = rc * (x @ z_unit.t())                            # (...,out) <rc x,[z_k]>

        if self.bias is not None:
            two_rc_r = 2.0 * rc * self.bias                      # (out,) 2 sqrt(c) r_k
        else:
            two_rc_r = torch.zeros_like(z_norm)
        cosh = torch.cosh(two_rc_r)                              # (out,)
        sinh = torch.sinh(two_rc_r)                              # (out,)

        num = 2.0 * rcxz * cosh - (1.0 + cx2) * sinh             # (...,out)
        inner = num / (1.0 - cx2).clamp_min(1e-15)               # (...,out)
        vk = (2.0 * z_norm / rc) * pmath.arsinh(inner)           # (...,out) v_k(x)

        # --- Poincaré FC output  (paper Eq. 7) ------------------------------
        # w_k = sinh(sqrt(c) v_k)/sqrt(c);  y = w / (1 + sqrt(1 + c|w|^2)) in B^out_c
        w = torch.sinh((rc * vk).clamp(-self.clamp, self.clamp)) / rc   # (...,out)
        w2 = w.pow(2).sum(dim=-1, keepdim=True)                         # (...,1)
        y = w / (1.0 + torch.sqrt(1.0 + c * w2))                       # (...,out)
        return pmath.project(y, c=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )
# ============================ end HNN++ (added) =============================


class HyperbolicMLR(nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(self, ball_dim, n_classes, c):
        super(HyperbolicMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

    def forward(self, x, c=None):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
        p_vals_poincare = pmath.expmap0(self.p_vals, c=c)
        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
        logits = pmath._hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits

    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.ball_dim, self.n_classes, self.c
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))


class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = pmath.mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return pmath.project(mv, c=c)
        else:
            bias = pmath.expmap0(self.bias, c=c)
            return pmath.project(pmath.mobius_add(mv, bias), c=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )


class ConcatPoincareLayer(nn.Module):
    def __init__(self, d1, d2, d_out, c):
        super(ConcatPoincareLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        self.l1 = HypLinear(d1, d_out, bias=False, c=c)
        self.l2 = HypLinear(d2, d_out, bias=False, c=c)
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.mobius_add(self.l1(x1), self.l2(x2), c=c)

    def extra_repr(self):
        return "dims {} and {} ---> dim {}".format(self.d1, self.d2, self.d_out)


class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, c):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.dist(x1, x2, c=c, keepdim=True)

    def extra_repr(self):
        return "c={}".format(self.c)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True, clip_r=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c
        
        self.clip_r = clip_r
        
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac =  torch.minimum(
                torch.ones_like(x_norm), 
                self.clip_r / x_norm
            )
            x = x * fac
            
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):

        super(FromPoincare, self).__init__()

        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.logmap(xp, x, c=self.c)
        return pmath.logmap0(x, c=self.c)

    def extra_repr(self):
        return "train_c={}, train_x={}".format(self.train_c, self.train_x)