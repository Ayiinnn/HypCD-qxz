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
        fac = torch.minimum(
            torch.ones_like(x_norm),
            self.clip_r / x_norm
        )
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
    
    
