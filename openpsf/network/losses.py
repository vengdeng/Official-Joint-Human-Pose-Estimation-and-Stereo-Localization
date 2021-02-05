"""Losses."""

import torch

from ..data import (COCO_PERSON_SIGMAS, COCO_PERSON_SKELETON, KINEMATIC_TREE_SKELETON,
                    DENSER_COCO_PERSON_SKELETON)


def laplace_loss(x1, x2, logb, t1, t2, weight=None):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """
    norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)

    losses = 0.694 + logb + norm * torch.exp(-logb)
    if weight is not None:
        losses = losses * weight

    return torch.sum(losses)


def l1_loss(x1, x2, _, t1, t2, weight=None):
    """L1 loss.

    Loss for a single two-dimensional vector (x1, x2)
    true (t1, t2) vector.
    """
    losses = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


class SmoothL1Loss(object):
    def __init__(self, r_smooth, scale_required=True):
        self.r_smooth = r_smooth
        self.scale = None
        self.scale_required = scale_required

    def __call__(self, x1, x2, _, t1, t2, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """
        if self.scale_required and self.scale is None:
            raise Exception
        if self.scale is None:
            self.scale = 1.0

        r = self.r_smooth * self.scale
        d = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        smooth_regime = d < r

        smooth_loss = 0.5 / r[smooth_regime] * d[smooth_regime] ** 2
        linear_loss = d[smooth_regime == 0] - (0.5 * r[smooth_regime == 0])
        losses = torch.cat((smooth_loss, linear_loss))

        if weight is not None:
            losses = losses * weight

        self.scale = None
        return torch.sum(losses)


class SmootherL1Loss(object):
    def __init__(self, r_smooth, ln_threshold=3.0, scale_required=True):
        self.r_smooth = r_smooth
        self.ln_threshold = ln_threshold
        self.scale = None
        self.scale_required = scale_required

    def __call__(self, x1, x2, _, t1, t2, weight=None):
        if self.scale_required and self.scale is None:
            raise Exception
        if self.scale is None:
            self.scale = 1.0

        diff = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        diff = diff * self.r_smooth * self.scale  # TODO: update

        sq_mask = diff < 1.0
        ln_mask = diff >= self.ln_threshold
        lin_mask = (sq_mask | ln_mask) == 0

        diff[sq_mask] = 0.5 * torch.mul(diff[sq_mask], diff[sq_mask])
        diff[lin_mask] = diff[lin_mask] - 0.5

        ln_shift = self.ln_threshold - 1.0
        diff[ln_mask] = 0.5 + ln_shift + torch.log(diff[ln_mask] - ln_shift)

        diff = diff / self.r_smooth / self.scale
        self.scale = None
        return diff

class CompositeLoss(torch.nn.Module):
    def __init__(self, regression_loss, background_weight=1.0, skeleton=None,
                 multiplicity_correction=False, independence_scale=3.0,
                 n_vectors=2, n_scales=0, sigmas=None):
        super(CompositeLoss, self).__init__()

        self.background_weight = background_weight
        if skeleton is None:
            skeleton = COCO_PERSON_SKELETON

        self.multiplicity_correction = multiplicity_correction
        self.independence_scale = independence_scale

        self.n_vectors = n_vectors
        self.n_scales = n_scales
        if self.n_scales:
            assert len(sigmas) == n_scales

        if sigmas is not None:
            assert len(sigmas) == n_vectors
            scales_to_kp = torch.tensor(sigmas)
            scales_to_kp = torch.unsqueeze(scales_to_kp, -1)
            scales_to_kp = torch.unsqueeze(scales_to_kp, -1)
            self.register_buffer('scales_to_kp', scales_to_kp)
        else:
            self.scales_to_kp = None

        self.regression_loss = regression_loss or laplace_loss

    def forward(self, x, t):  # pylint: disable=arguments-differ
        x_intensity = x[0]
        x_regs = x[1:self.n_vectors + 1]
        x_spreads = x[self.n_vectors + 1:2 * self.n_vectors + 1]
        x_scales = []
        if self.n_scales:
            x_scales = x[2 * self.n_vectors + 1:2 * self.n_vectors + 1 + self.n_scales]

        target_intensity = t[0]
        target_regs = t[1:-1]
        target_scale = t[-1]

        bce_masks = torch.sum(target_intensity, dim=1, keepdim=True) > 0.5
        if torch.sum(bce_masks) < 1:
            return None, None, None

        batch_size = x_intensity.shape[0]

        bce_target = torch.masked_select(target_intensity[:, :-1], bce_masks)
        bce_weight = torch.ones_like(bce_target)
        bce_weight[bce_target == 0] = self.background_weight
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.masked_select(x_intensity, bce_masks),
            bce_target,
            weight=bce_weight,
        )

        reg_losses = [None for _ in target_regs]
        reg_masks = target_intensity[:, :-1] > 0.5
        if torch.sum(reg_masks) > 0:
            weight = None
            if self.multiplicity_correction:
                assert len(target_regs) == 2
                lengths = torch.norm(target_regs[0] - target_regs[1], dim=2)
                multiplicity = (lengths - 3.0) / self.independence_scale
                multiplicity = torch.clamp(multiplicity, min=1.0)
                multiplicity = torch.masked_select(multiplicity, reg_masks)
                weight = 1.0 / multiplicity

            reg_losses = []
            for i, (x_reg, x_spread, target_reg) in enumerate(zip(x_regs, x_spreads, target_regs)):
                if hasattr(self.regression_loss, 'scale'):
                    assert self.scales_to_kp is not None
                    self.regression_loss.scale = torch.masked_select(
                        torch.clamp(target_scale * self.scales_to_kp[i], 0.1, 1000.0),  # pylint: disable=unsubscriptable-object 夹紧张量
                        reg_masks,
                    )
                #print(x_reg[:, :, 1].shape)
                #print(target_reg[:, :, 1].shape)
                try:
                    # print(x_reg[:, :, 0].shape)
                    # print(reg_masks.shape)
                    # print(torch.masked_select(x_reg[:, :, 0], reg_masks))
                    reg_losses.append(self.regression_loss(
                        torch.masked_select(x_reg[:, :, 0], reg_masks),
                        torch.masked_select(x_reg[:, :, 1], reg_masks),
                        torch.masked_select(x_spread, reg_masks),
                        torch.masked_select(target_reg[:, :, 0], reg_masks),
                        torch.masked_select(target_reg[:, :, 1], reg_masks),
                        weight=weight,
                    ) / 1000.0 / batch_size)

                except:

                    reg_losses.append(self.regression_loss(
                        torch.masked_select(x_reg[:, :, 0], reg_masks),
                        0,
                        torch.masked_select(x_spread, reg_masks),
                        torch.masked_select(target_reg[:, :, 0], reg_masks),
                        0,
                        weight=weight,
                         ) / 1000.0 / batch_size)


        scale_losses = []
        if x_scales:
            scale_losses = [
                torch.nn.functional.l1_loss(
                    torch.masked_select(x_scale, reg_masks),
                    torch.masked_select(target_scale * scale_to_kp, reg_masks),
                    reduction='sum',
                ) / 1000.0 / batch_size
                for x_scale, scale_to_kp in zip(x_scales, self.scales_to_kp)
            ]

        return [ce_loss] + reg_losses + scale_losses


def cli(parser):
    group = parser.add_argument_group('losses')
    group.add_argument('--r-smooth', type=float, default=0.0,
                       help='r_{smooth} for SmoothL1 regressions')
    group.add_argument('--regression-loss', default='laplace',
                       choices=['smoothl1', 'smootherl1', 'l1', 'laplace'],
                       help='type of regression loss')
    group.add_argument('--background-weight', default=1.0, type=float,
                       help='BCE weight of background')
    group.add_argument('--paf-multiplicity-correction',
                       default=False, action='store_true',
                       help='use multiplicity correction for PAF loss')
    group.add_argument('--paf-independence-scale', default=3.0, type=float,
                       help='linear length scale of independence for PAF regression')


def factory(args):
    losses = []

    if args.regression_loss == 'smoothl1':
        reg_loss_ = torch.nn.SmoothL1Loss(reduction='none')
        reg_loss = SmoothL1Loss(args.r_smooth)
    elif args.regression_loss == 'smootherl1':
        reg_loss_ = None
        reg_loss = SmootherL1Loss(args.r_smooth)
    elif args.regression_loss == 'l1':
        reg_loss_ = torch.nn.L1Loss(reduction='sum')
        reg_loss = l1_loss
    elif args.regression_loss == 'laplace':
        reg_loss_ = torch.nn.L1Loss(reduction='sum')
        reg_loss = laplace_loss
    else:
        raise Exception('unknown regression loss type {}'.format(args.regression_loss))

    for head_name in args.headnets:
        if head_name in ('pif',):
            losses.append(CompositeLoss(reg_loss,
                                        background_weight=args.background_weight,
                                        n_vectors=1, n_scales=1,
                                        sigmas=[COCO_PERSON_SIGMAS]))
        elif head_name in ('paf',):
            losses.append(CompositeLoss(reg_loss,
                                        n_vectors=2, n_scales=0,
                                        background_weight=args.background_weight,
                                        multiplicity_correction=args.paf_multiplicity_correction,
                                        sigmas=[
                                            [COCO_PERSON_SIGMAS[j1i - 1]
                                             for j1i, _ in COCO_PERSON_SKELETON],
                                            [COCO_PERSON_SIGMAS[j2i - 1]
                                             for _, j2i in COCO_PERSON_SKELETON],
                                        ]))
        elif head_name in ('pafs', 'pafs19', 'pafsb'):
            losses.append(CompositeLoss(reg_loss,
                                        background_weight=args.background_weight,
                                        multiplicity_correction=args.paf_multiplicity_correction,
                                        independence_scale=args.paf_independence_scale))
        elif head_name in ('pisf',):
            losses.append(Associateloss(reg_loss,
                                        n_vectors=1, n_scales=1,
                                        background_weight=args.background_weight,
                                        multiplicity_correction=args.paf_multiplicity_correction,
                                        sigmas=[COCO_PERSON_SIGMAS]))
        elif head_name in ('psf',):
            losses.append(CompositeLoss(reg_loss,
                                        n_vectors=2, n_scales=0,
                                        background_weight=args.background_weight,
                                        multiplicity_correction=args.paf_multiplicity_correction,
                                        sigmas=[
                                            [COCO_PERSON_SIGMAS[i]
                                             for i in range(17)],
                                            [COCO_PERSON_SIGMAS[j]
                                             for j in range(17)],
                                        ]))

        elif head_name in ('skeleton',):
            pass
        else:
            raise Exception('unknown headname {} for loss'.format(head_name))

    return [l.to(device=args.device) for l in losses]
