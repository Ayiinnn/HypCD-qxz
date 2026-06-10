import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import hyptorch.pmath as pmath

class LearnableCurvature(nn.Module): # mod
    def __init__(self, init_c=0.05, min_c=1e-5):
        super(LearnableCurvature, self).__init__()
        self.min_c = min_c
        init_c = torch.tensor(float(init_c), dtype=torch.float32)
        raw = torch.log(torch.expm1(torch.clamp(init_c - min_c, min=1e-12)))
        self.raw_c = nn.Parameter(raw.view(1))

    def forward(self):
        return F.softplus(self.raw_c) + self.min_c

    def extra_repr(self):
        return "c={}".format(self().detach().cpu().item())


def get_curvature(c, ref=None): # mod
    if isinstance(c, LearnableCurvature):
        c = c()
    elif callable(c) and not torch.is_tensor(c):
        c = c()

    if ref is None:
        return c
    if torch.is_tensor(c):
        return c.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(c, dtype=ref.dtype, device=ref.device)


class MultiCurvatureManager(nn.Module): # mod
    ROLE_NAMES = ("c_rep_sup", "c_rep_unsup", "c_cls_sup", "c_cls_unsup")

    def __init__(self, init_c=0.05, train_c=True, min_c=1e-5):
        super().__init__()
        self.train_c = train_c
        self.min_c = min_c

        if train_c:
            self.curvatures = nn.ModuleDict({
                name: LearnableCurvature(init_c=init_c, min_c=min_c)
                for name in self.ROLE_NAMES
            })
        else:
            self.curvatures = nn.ModuleDict()
            for name in self.ROLE_NAMES:
                self.register_buffer(f"{name}_const", torch.tensor([float(init_c)]))

    def value(self, name):
        if self.train_c:
            return self.curvatures[name]()
        return getattr(self, f"{name}_const")

    def values(self):
        return {name: self.value(name) for name in self.ROLE_NAMES}

    @staticmethod
    def _active(strength, epoch, unbind_epoch):
        strength = float(strength)
        return strength > 0 and (unbind_epoch < 0 or epoch < unbind_epoch)

    def _freeze_on(self, args, epoch):
        mode = getattr(args, "c_tie_mode", "none")
        if mode == "none" or not self.train_c:
            return False

        freeze_rep_cls = (
            float(getattr(args, "c_tie_rep_cls", 0.0)) > 0
            and epoch < getattr(args, "c_unfreeze_rep_cls_epoch", 0)
        )
        freeze_sup_unsup = (
            float(getattr(args, "c_tie_sup_unsup", 0.0)) > 0
            and epoch < getattr(args, "c_unfreeze_sup_unsup_epoch", 0)
        )
        return freeze_rep_cls or freeze_sup_unsup

    def effective(self, epoch, args):
        """
        return four curvatures used in given epoch
        do alias in hard mode:
          c_cls_sup -> c_rep_sup
          c_cls_unsup -> c_rep_unsup
          c_rep_unsup -> c_rep_sup
          c_cls_unsup -> c_cls_sup
        """
        c = self.values()
        mode = getattr(args, "c_tie_mode", "none")

        tie_rep_cls = self._active(
            getattr(args, "c_tie_rep_cls", 0.0),
            epoch,
            getattr(args, "c_unbind_rep_cls_epoch", -1),
        )
        tie_sup_unsup = self._active(
            getattr(args, "c_tie_sup_unsup", 0.0),
            epoch,
            getattr(args, "c_unbind_sup_unsup_epoch", -1),
        )

        if mode == "hard":
            # rep_sup=cls_sup, rep_unsup=cls_unsup
            if tie_rep_cls:
                c["c_cls_sup"] = c["c_rep_sup"]
                c["c_cls_unsup"] = c["c_rep_unsup"]

            # rep_sup=rep_unsup, cls_sup=cls_unsup
            if tie_sup_unsup:
                c["c_rep_unsup"] = c["c_rep_sup"]
                c["c_cls_unsup"] = c["c_cls_sup"]

        if self._freeze_on(args, epoch):
            c = {k: v.detach() for k, v in c.items()}

        return c

    def _raw_from_c(self, c_value, target_module):
        c_value = torch.as_tensor(
            c_value,
            dtype=target_module.raw_c.dtype,
            device=target_module.raw_c.device,
        )
        raw = torch.log(torch.expm1(torch.clamp(c_value - target_module.min_c, min=1e-12)))
        return raw.view_as(target_module.raw_c)

    @torch.no_grad()
    def _set_value(self, name, c_value):
        if self.train_c:
            m = self.curvatures[name]
            m.raw_c.copy_(self._raw_from_c(c_value, m))
        else:
            getattr(self, f"{name}_const").copy_(torch.as_tensor(c_value).view(1))

    @torch.no_grad()
    def hard_sync(self, epoch, args):
        """
        under hard tying, the non-anchor raw_c does not receive gradients.
        If it is not synchronized, then at the moment of untying it will jump out from the old init_c, causing an abrupt change in c.
        Therefore, synchronize it once after every optimizer.step().
        """
        if getattr(args, "c_tie_mode", "none") != "hard":
            return

        tie_rep_cls = self._active(
            getattr(args, "c_tie_rep_cls", 0.0),
            epoch,
            getattr(args, "c_unbind_rep_cls_epoch", -1),
        )
        tie_sup_unsup = self._active(
            getattr(args, "c_tie_sup_unsup", 0.0),
            epoch,
            getattr(args, "c_unbind_sup_unsup_epoch", -1),
        )

        if tie_rep_cls:
            self._set_value("c_cls_sup", self.value("c_rep_sup").detach())
            self._set_value("c_cls_unsup", self.value("c_rep_unsup").detach())

        if tie_sup_unsup:
            self._set_value("c_rep_unsup", self.value("c_rep_sup").detach())
            self._set_value("c_cls_unsup", self.value("c_cls_sup").detach())

    @staticmethod
    def _soft_weight(strength, epoch, unfreeze_epoch, unbind_epoch):
        strength = float(strength)
        if strength <= 0:
            return 0.0
        if epoch < unfreeze_epoch:
            return 0.0
        if unbind_epoch >= 0 and epoch >= unbind_epoch:
            return 0.0
        if unbind_epoch < 0:
            return strength

        denom = max(1, unbind_epoch - unfreeze_epoch)
        progress = max(0.0, float(epoch - unfreeze_epoch) / float(denom))
        return strength * max(0.0, 1.0 - progress)

    def tie_loss(self, epoch, args, device):
        """
        soft mode: L = lambda(t) * (log c_i - log c_j)^2
        hard/none mode: L = 0
        """
        if getattr(args, "c_tie_mode", "none") != "soft":
            return torch.zeros((), device=device)

        loss = torch.zeros((), device=device)

        w_rep_cls = self._soft_weight(
            getattr(args, "c_tie_rep_cls", 0.0),
            epoch,
            getattr(args, "c_unfreeze_rep_cls_epoch", 0),
            getattr(args, "c_unbind_rep_cls_epoch", -1),
        )
        if w_rep_cls > 0:
            loss = loss + w_rep_cls * (
                (torch.log(self.value("c_rep_sup")) - torch.log(self.value("c_cls_sup"))).pow(2)
                + (torch.log(self.value("c_rep_unsup")) - torch.log(self.value("c_cls_unsup"))).pow(2)
            ).sum()

        w_sup_unsup = self._soft_weight(
            getattr(args, "c_tie_sup_unsup", 0.0),
            epoch,
            getattr(args, "c_unfreeze_sup_unsup_epoch", 0),
            getattr(args, "c_unbind_sup_unsup_epoch", -1),
        )
        if w_sup_unsup > 0:
            loss = loss + w_sup_unsup * (
                (torch.log(self.value("c_rep_sup")) - torch.log(self.value("c_rep_unsup"))).pow(2)
                + (torch.log(self.value("c_cls_sup")) - torch.log(self.value("c_cls_unsup"))).pow(2)
            ).sum()

        return loss

    def log_string(self):
        vals = {k: float(v.detach().cpu()) for k, v in self.values().items()}
        return " ".join([f"{k}:{vals[k]:.6f}" for k in self.ROLE_NAMES])

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

    def forward(self, x, c=None): # mod
        if c is None:
            c = self.c
        c = get_curvature(c, x)
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

    def forward(self, x, c=None): # mod
        if c is None:
            c = self.c
        c = get_curvature(c, x)
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
        c = get_curvature(c, x1)
        return pmath.mobius_add(self.l1(x1, c=c), self.l2(x2, c=c), c=c)

    def extra_repr(self):
        return "dims {} and {} ---> dim {}".format(self.d1, self.d2, self.d_out)


class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, c):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None): # mod
        if c is None:
            c = self.c
        c = get_curvature(c, x1)
        return pmath.dist(x1, x2, c=c, keepdim=True)

    def extra_repr(self):
        return "c={}".format(self.c)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True, clip_r=None): # mod
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
            self.c = LearnableCurvature(c)
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c
        
        self.clip_r = clip_r
        
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def get_c(self, ref=None): # mod
        return get_curvature(self.c, ref)

    def forward(self, x, c=None):
        if c is None:
            c = self.get_c(x)
        else:
            c = get_curvature(c, x)

        self.riemannian.c = c.detach() if torch.is_tensor(c) else c

        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm)
            x = x * fac

        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=c), c=c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=c), c=c))

        return self.grad_fix(pmath.project(pmath.expmap0(x, c=c), c=c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None): # mod

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
            self.c = LearnableCurvature(c)
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def get_c(self, ref=None): # mod
        return get_curvature(self.c, ref)

    def forward(self, x): # mod
        c = self.get_c(x)
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=c), c=c)
            return pmath.logmap(xp, x, c=c)
        return pmath.logmap0(x, c=c)

    def extra_repr(self):
        return "train_c={}, train_x={}".format(self.train_c, self.train_x)
    
    


# ============================================================================
#  Mixed-curvature PRODUCT-SPACE extension  (appended; nothing above is changed)
# ----------------------------------------------------------------------------
#  Why this design (tailored to GCD + hyperbolic + the curvature experiments):
#
#   1) ONE shared trunk feature is split into several constant-curvature
#      factors (Euclidean / Poincare / Spherical). The product metric is the
#      additive  d^2 = sum_i  w_i * d_i^2 . The factors therefore *coexist in a
#      single representation* instead of living in disjoint branch spaces, so
#      old & new classes are always compared under one (full) metric. This is
#      exactly what hard sup/unsup branch-decoupling destroyed.
#
#   2) rep-vs-cls is expressed by *role weights* over factors (rep leans on the
#      curved factors, cls on the flat one) -- NOT by separate curvatures and
#      NOT by separate embeddings. sup-vs-unsup share the same embedding and the
#      same role weights, so the supervised-branch -> c->0 collapse cannot
#      happen by construction.
#
#   3) The single overloaded curvature pinned sqrt(c)*cr into a narrow band
#      (a homeostasis symptom). Here each hyperbolic factor keeps its own
#      effective radius sqrt(c_i)*clip_r_i, gently regularized toward a target,
#      so me_max (wants flat) and the distance loss (wants curved) act on
#      *different* factors and no longer fight over one c.
#
#   4) A full-metric ALIGNMENT loss (role="align", uniform weights, gate-free)
#      is the load-bearing term: it forces every factor to remain a mutually
#      consistent view of the shared representation -> anti-degeneracy + the
#      old/new anchoring that pure branch-decoupling could not keep.
#
#   5) Everything is N-factor from the start. Add a factor by extending the
#      spec list / spec string (e.g. a second Poincare factor `P:128` driven by
#      a future part-level loss + a new "part" role).
# ============================================================================


class FactorSpec:
    """Specification of a single product factor.

    kind  : 'euclidean' | 'poincare' | 'spherical'
    dim   : factor dimensionality
    init_c: initial curvature (Poincare) / initial angular scale (Spherical)
    clip_r: feature-clip radius before expmap0 (Poincare only; sets, with c,
            the effective radius sqrt(c)*clip_r)
    learn_c: whether this factor's curvature is trainable
    """

    def __init__(self, kind, dim, init_c=0.1, clip_r=2.0, learn_c=True, name=None):
        kind = kind.lower()
        assert kind in ("euclidean", "poincare", "spherical"), f"bad factor kind {kind}"
        self.kind = kind
        self.dim = int(dim)
        self.init_c = float(init_c)
        self.clip_r = float(clip_r)
        self.learn_c = bool(learn_c)
        self.name = name if name is not None else kind[:3]

    @property
    def is_curved(self):
        return self.kind in ("poincare", "spherical")

    def __repr__(self):
        return (f"FactorSpec({self.kind}, dim={self.dim}, c0={self.init_c}, "
                f"cr={self.clip_r}, learn_c={self.learn_c}, name={self.name})")


def parse_factor_specs(spec_str, default_c=0.1, default_clip_r=2.0, learn_c=True):
    """Parse a compact product-space spec string.

    Examples:
        "E:384,P:384"
        "E:256,P:384:c0.1:cr2.0,P:128:c0.5:cr1.5"
        "E:256,P:384,S:128"
    Tokens:  E=euclidean, P=poincare, S=spherical.
    Optional colon fields per factor:  cX -> init curvature,  crX -> clip radius.
    """
    kind_map = {"e": "euclidean", "p": "poincare", "s": "spherical"}
    specs = []
    for idx, tok in enumerate(spec_str.split(",")):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        kind = kind_map[parts[0].strip().lower()]
        dim = int(parts[1])
        c, cr = default_c, default_clip_r
        for extra in parts[2:]:
            extra = extra.strip().lower()
            if extra.startswith("cr"):
                cr = float(extra[2:])
            elif extra.startswith("c"):
                c = float(extra[1:])
        specs.append(FactorSpec(kind, dim, init_c=c, clip_r=cr,
                                learn_c=learn_c, name=f"f{idx}_{kind[:3]}"))
    return specs


class ProductManifold(nn.Module):
    """Splits a shared trunk feature into constant-curvature factors and exposes
    product-metric distances + role-based factor weighting.

    Role weights:
        "cls"   -> flat-leaning   (used by classification / me_max)
        "rep"   -> curved-leaning (used by contrastive / supcon / new-class)
        "align" -> uniform & gate-free, frozen (load-bearing anchor)
    Add a custom role (e.g. "part") with `add_role(name, init_logits)`.
    """

    def __init__(self, specs, in_dim, role_init_strength=0.7,
                 learn_gates=True, learn_role_weights=False, min_c=1e-5):
        super().__init__()
        self.specs = list(specs)
        self.in_dim = int(in_dim)
        self.n_factors = len(self.specs)
        self.dims = [s.dim for s in self.specs]
        self.total_dim = sum(self.dims)
        self.min_c = min_c

        offs, acc = [], 0
        for d in self.dims:
            offs.append((acc, acc + d)); acc += d
        self.offsets = offs

        # shared trunk -> per-factor tangent projection (keeps representation shared)
        self.projections = nn.ModuleList([nn.Linear(self.in_dim, d) for d in self.dims])

        # per-factor learnable curvature (poincare = curvature, spherical = angular scale)
        self.curv = nn.ModuleDict()
        self.clip_r = {}
        for i, s in enumerate(self.specs):
            if s.is_curved:
                m = LearnableCurvature(init_c=s.init_c, min_c=min_c)
                if not s.learn_c:
                    m.raw_c.requires_grad_(False)
                self.curv[str(i)] = m
            if s.kind == "poincare":
                self.clip_r[i] = s.clip_r

        # per-factor global importance gate (dataset-adaptive scale on d_i^2).
        # softplus(raw_gate) ~= 1 at init.
        raw_gate = math.log(math.expm1(1.0))
        self.gate_raw = nn.Parameter(torch.full((self.n_factors,), float(raw_gate)),
                                     requires_grad=learn_gates)
        self.learn_gates = learn_gates

        # role logits over factors. cls favors flat (euclidean) factors, rep favors curved.
        cls_logit = torch.zeros(self.n_factors)
        rep_logit = torch.zeros(self.n_factors)
        for i, s in enumerate(self.specs):
            if s.kind == "euclidean":
                cls_logit[i] = +role_init_strength
                rep_logit[i] = -role_init_strength
            elif s.kind == "poincare":
                cls_logit[i] = -role_init_strength
                rep_logit[i] = +role_init_strength
            else:  # spherical: neutral at init
                cls_logit[i] = 0.0
                rep_logit[i] = 0.0
        self.role_logits = nn.ParameterDict({
            "cls": nn.Parameter(cls_logit, requires_grad=learn_role_weights),
            "rep": nn.Parameter(rep_logit, requires_grad=learn_role_weights),
        })
        # align role: uniform over factors and never trained.
        self.register_buffer("align_logits", torch.zeros(self.n_factors))

    # ------------------------------------------------------------------ roles
    def add_role(self, name, init_logits=None, learnable=True):
        """Register an extra role (e.g. 'part'). init_logits: list/tensor [n_factors]."""
        if init_logits is None:
            init_logits = torch.zeros(self.n_factors)
        init_logits = torch.as_tensor(init_logits, dtype=torch.float32)
        self.role_logits[name] = nn.Parameter(init_logits, requires_grad=learnable)

    def role_weight(self, role, ref=None):
        """Effective per-factor weights for a role.
        'align' is uniform AND gate-free (so it always exercises every factor)."""
        if role == "align":
            w = torch.softmax(self.align_logits, dim=0)
        else:
            w = torch.softmax(self.role_logits[role], dim=0) * self.gates()
        if ref is not None:
            w = w.to(dtype=ref.dtype, device=ref.device)
        return w

    def gates(self):
        return F.softplus(self.gate_raw) + 1e-4

    # ------------------------------------------------------------- curvature
    def factor_c(self, i, ref=None):
        s = self.specs[i]
        if s.is_curved:
            c = self.curv[str(i)]()
        else:
            dev = ref.device if ref is not None else self.gate_raw.device
            c = torch.zeros((), dtype=torch.float32, device=dev)
        if ref is not None:
            c = c.to(dtype=ref.dtype, device=ref.device)
        return c

    def set_train_c(self, flag):
        for m in self.curv.values():
            m.raw_c.requires_grad_(bool(flag))

    @property
    def has_hyperbolic(self):
        return any(s.kind == "poincare" for s in self.specs)

    # ------------------------------------------------------------ projection
    def project(self, feat):
        """feat [B,in_dim] -> concatenated on-manifold coords [B,total_dim]."""
        outs = []
        for i, s in enumerate(self.specs):
            t = self.projections[i](feat)
            if s.kind == "euclidean":
                x = t
            elif s.kind == "poincare":
                c = self.factor_c(i, ref=t)
                cr = self.clip_r.get(i, None)
                if cr is not None:
                    n = torch.norm(t, dim=-1, keepdim=True) + 1e-5
                    t = t * torch.minimum(torch.ones_like(n), cr / n)
                x = pmath.project(pmath.expmap0(t, c=c), c=c)
            else:  # spherical -> unit sphere
                x = t / (torch.norm(t, dim=-1, keepdim=True) + 1e-9)
            outs.append(x)
        return torch.cat(outs, dim=-1)

    def split(self, X):
        return [X[..., a:b] for (a, b) in self.offsets]

    def _to_tangent(self, i, Xi):
        """Map on-manifold coords back to a Euclidean tangent (for cosine/angle)."""
        if self.specs[i].kind == "poincare":
            c = self.factor_c(i, ref=Xi)
            return pmath.logmap0(Xi, c=c)
        return Xi  # euclidean / (unit) spherical coords are already Euclidean vectors

    # ------------------------------------------------------- per-factor dists
    def _factor_sqdist_matrix(self, i, Ai, Bi):
        s = self.specs[i]
        if s.kind == "euclidean":
            d = torch.cdist(Ai, Bi, p=2)
            return d * d
        elif s.kind == "poincare":
            c = self.factor_c(i, ref=Ai)
            d = pmath.dist_matrix(Ai, Bi, c=c)
            return d * d
        else:  # spherical: scaled geodesic angle
            scale = self.factor_c(i, ref=Ai)
            Au = Ai / (Ai.norm(dim=-1, keepdim=True) + 1e-9)
            Bu = Bi / (Bi.norm(dim=-1, keepdim=True) + 1e-9)
            cosim = torch.clamp(Au @ Bu.t(), -1 + 1e-6, 1 - 1e-6)
            d = scale * torch.acos(cosim)
            return d * d

    def _factor_sqdist_pair(self, i, ai, bi):
        s = self.specs[i]
        if s.kind == "euclidean":
            d = torch.norm(ai - bi, dim=-1)
            return d * d
        elif s.kind == "poincare":
            c = self.factor_c(i, ref=ai)
            d = pmath.dist(ai, bi, c=c, keepdim=False)
            return d * d
        else:
            scale = self.factor_c(i, ref=ai)
            au = ai / (ai.norm(dim=-1, keepdim=True) + 1e-9)
            bu = bi / (bi.norm(dim=-1, keepdim=True) + 1e-9)
            cosim = torch.clamp((au * bu).sum(-1), -1 + 1e-6, 1 - 1e-6)
            d = scale * torch.acos(cosim)
            return d * d

    def sqdist_matrix(self, A, B, role="align"):
        Asp, Bsp = self.split(A), self.split(B)
        w = self.role_weight(role, ref=A)
        out = None
        for i in range(self.n_factors):
            term = w[i] * self._factor_sqdist_matrix(i, Asp[i], Bsp[i])
            out = term if out is None else out + term
        return out

    def dist_matrix(self, A, B, role="align"):
        """Product geodesic distance matrix = sqrt(sum_i w_i d_i^2)."""
        return torch.sqrt(self.sqdist_matrix(A, B, role) + 1e-12)

    def pair_dist(self, a, b, role="align"):
        asp, bsp = self.split(a), self.split(b)
        w = self.role_weight(role, ref=a)
        out = None
        for i in range(self.n_factors):
            term = w[i] * self._factor_sqdist_pair(i, asp[i], bsp[i])
            out = term if out is None else out + term
        return torch.sqrt(out + 1e-12)

    def angle_factor_index(self):
        for i, s in enumerate(self.specs):
            if s.kind == "euclidean":
                return i
        return min(range(self.n_factors), key=lambda i: self.specs[i].init_c)

    def angle_sim_matrix(self, A, B):
        """Cosine similarity in the designated 'angle' factor (the flat factor).
        This is the Euclidean end of the angle->distance curriculum."""
        i = self.angle_factor_index()
        Asp, Bsp = self.split(A), self.split(B)
        Ai = F.normalize(self._to_tangent(i, Asp[i]), dim=-1)
        Bi = F.normalize(self._to_tangent(i, Bsp[i]), dim=-1)
        return Ai @ Bi.t()

    # ------------------------------------------------------ regularizers/logs
    def effective_radii(self):
        out = {}
        for i, s in enumerate(self.specs):
            if s.kind == "poincare":
                c = float(self.curv[str(i)]().detach().cpu())
                out[s.name] = (c ** 0.5) * self.clip_r.get(i, 0.0)
        return out

    def radius_penalty(self, target):
        """sum_h (sqrt(c_h)*clip_r_h - target)^2 over poincare factors.
        Operationalizes the effective-radius homeostasis as an explicit (gentle)
        constraint, instead of letting one c emerge from a me_max/distance tug-of-war."""
        dev = self.gate_raw.device
        target = torch.as_tensor(target, dtype=torch.float32, device=dev)
        loss = torch.zeros((), device=dev)
        for i, s in enumerate(self.specs):
            if s.kind == "poincare":
                c = self.curv[str(i)]().to(dev)
                eff = torch.sqrt(c) * self.clip_r.get(i, 0.0)
                loss = loss + (eff.squeeze() - target) ** 2
        return loss

    def degeneracy_penalty(self, X, floor):
        """Penalize any factor whose batch coordinate std collapses below `floor`.
        Backstop against factor degeneration (a factor being silently ignored)."""
        Xsp = self.split(X)
        dev = X.device
        floor = torch.as_tensor(floor, dtype=torch.float32, device=dev)
        loss = torch.zeros((), device=dev)
        for i in range(self.n_factors):
            std = Xsp[i].float().std(dim=0).mean()
            loss = loss + F.relu(floor - std) ** 2
        return loss

    def log_string(self):
        parts = []
        gs = self.gates().detach().cpu()
        for i, s in enumerate(self.specs):
            if s.kind == "poincare":
                c = float(self.curv[str(i)]().detach().cpu())
                eff = (c ** 0.5) * self.clip_r.get(i, 0.0)
                parts.append(f"{s.name}[c={c:.4f},r={eff:.3f},g={float(gs[i]):.2f}]")
            elif s.kind == "spherical":
                c = float(self.curv[str(i)]().detach().cpu())
                parts.append(f"{s.name}[sph,s={c:.4f},g={float(gs[i]):.2f}]")
            else:
                parts.append(f"{s.name}[euc,g={float(gs[i]):.2f}]")
        wc = self.role_weight('cls').detach().cpu().tolist()
        wr = self.role_weight('rep').detach().cpu().tolist()
        parts.append("w_cls=" + ",".join(f"{v:.2f}" for v in wc))
        parts.append("w_rep=" + ",".join(f"{v:.2f}" for v in wr))
        return " ".join(parts)


class _CosineHead(nn.Module):
    """Cosine classifier head for a flat (Euclidean / spherical) factor."""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_classes, dim))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return x @ w.t()  # [B, n_classes], cosine logits in [-1, 1]


class ProductMLR(nn.Module):
    """Product-space classifier: one head per factor, combined by a role weighting.

    Poincare factor -> HyperbolicMLR ; flat factor -> cosine head.
    A per-head learnable logit_scale balances heterogeneous logit magnitudes.
    The manifold is referenced (not registered as a submodule) so its parameters
    are owned solely by the ProductManifold instance.
    """

    def __init__(self, specs, n_classes, manifold, role="cls"):
        super().__init__()
        self.specs = list(specs)
        self.role = role
        self.n_classes = int(n_classes)
        # hide manifold from nn.Module registration (avoid duplicate params/state)
        self._manifold_ref = (manifold,)

        self.heads = nn.ModuleList()
        for s in self.specs:
            if s.kind == "poincare":
                self.heads.append(HyperbolicMLR(ball_dim=s.dim, n_classes=n_classes, c=s.init_c))
            else:
                self.heads.append(_CosineHead(s.dim, n_classes))
        self.logit_scale = nn.Parameter(torch.ones(len(self.specs)))

    @property
    def manifold(self):
        return self._manifold_ref[0]

    def forward(self, X, role=None):
        role = role or self.role
        Xsp = self.manifold.split(X)
        w = self.manifold.role_weight(role, ref=X)
        logits = None
        for i, s in enumerate(self.specs):
            if s.kind == "poincare":
                c = self.manifold.factor_c(i, ref=Xsp[i])
                li = self.heads[i](Xsp[i], c=c)
            else:
                li = self.heads[i](Xsp[i])
            term = w[i] * (self.logit_scale[i] * li)
            logits = term if logits is None else logits + term
        return logits


def build_product(spec_str, in_dim, n_classes, default_c=0.1, default_clip_r=2.0,
                  train_c=True, learn_gates=True, learn_role_weights=False,
                  role_init_strength=0.7, min_c=1e-5):
    """Convenience builder: returns (ProductManifold, ProductMLR)."""
    specs = parse_factor_specs(spec_str, default_c=default_c,
                               default_clip_r=default_clip_r, learn_c=train_c)
    # spherical factors default to angular scale ~1.0 if left at the global default
    for s in specs:
        if s.kind == "spherical" and abs(s.init_c - default_c) < 1e-12:
            s.init_c = 1.0
    manifold = ProductManifold(specs, in_dim=in_dim,
                               role_init_strength=role_init_strength,
                               learn_gates=learn_gates,
                               learn_role_weights=learn_role_weights, min_c=min_c)
    manifold.set_train_c(train_c)
    classifier = ProductMLR(specs, n_classes, manifold, role="cls")
    return manifold, classifier