"""Task-aligned assignment and detection losses.

``DetectionLoss`` covers DFL heads (``reg_max > 1``: CIoU + DFL + BCE) and
DFL-free heads (``reg_max == 1``: CIoU + image-normalised L1 + BCE).
``EndToEndLoss`` mixes the one-to-many and one-to-one branches of YOLO26, moving
weight to the one-to-one branch as training progresses.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .modules import dist2bbox, make_anchors


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    xy, wh = x[..., :2], x[..., 2:4] / 2
    return torch.cat((xy - wh, xy + wh), -1)


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(((x[..., :2] + x[..., 2:4]) / 2, x[..., 2:4] - x[..., :2]), -1)


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise CIoU for ``(..., 4)`` xyxy boxes; returns ``(..., 1)``."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
    w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU of ``(N, 4)`` and ``(M, 4)`` xyxy boxes -> ``(N, M)``."""
    (a1, a2), (b1, b2) = a.float().unsqueeze(1).chunk(2, 2), b.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox2dist(anchor_points, bbox, reg_max=None):
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    return dist.clamp_(0, reg_max - 0.01) if reg_max is not None else dist


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, stride=(8, 16, 32),
                 eps=1e-9, topk2=None):
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride_val = stride[1] if len(stride) > 1 else stride[0]
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs, self.n_max_boxes = pd_scores.shape[0], gt_bboxes.shape[1]
        if self.n_max_boxes == 0:
            return (
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
            )
        mask_in_gts = self._in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self._metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        mask_topk = self._topk(align_metric, mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt.bool()

        target_gt_idx, fg_mask, mask_pos = self._highest(mask_pos, overlaps, align_metric)
        target_labels, target_bboxes, target_scores = self._targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )
        align_metric = align_metric * mask_pos
        pos_align = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm = (align_metric * pos_overlaps / (pos_align + self.eps)).amax(-2).unsqueeze(-1)
        return target_bboxes, target_scores * norm, fg_mask.bool()

    def _in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        xywh = xyxy2xywh(gt_bboxes)
        small = (xywh[..., 2:] < self.stride_val) & mask_gt.bool()
        xywh[..., 2:] = torch.where(small, torch.full_like(xywh[..., 2:], self.stride_val),
                                    xywh[..., 2:])
        lt, rb = xywh2xyxy(xywh).unsqueeze(2).chunk(2, 3)
        mask = xy_centers[:, 0] - lt[..., 0] > eps
        mask &= xy_centers[:, 1] - lt[..., 1] > eps
        mask &= rb[..., 0] - xy_centers[:, 0] > eps
        mask &= rb[..., 1] - xy_centers[:, 1] > eps
        return mask

    def _metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask):
        na = pd_bboxes.shape[-2]
        mask = mask.bool()
        shape = (self.bs, self.n_max_boxes, na)
        b, g, a = mask.nonzero(as_tuple=True)
        scores = pd_scores[b, a, gt_labels[b, g, 0].long()]
        ious = bbox_ciou(gt_bboxes[b, g], pd_bboxes[b, a]).squeeze(-1).clamp_(0)
        overlaps = torch.zeros(shape, dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        align = torch.zeros(shape, dtype=pd_scores.dtype, device=pd_scores.device)
        overlaps[b, g, a] = ious.to(overlaps.dtype)
        align[b, g, a] = (scores.pow(self.alpha) * ious.pow(self.beta)).to(align.dtype)
        return align, overlaps

    def _topk(self, metrics, topk_mask):
        _, idx = torch.topk(metrics, self.topk, dim=-1, largest=True)
        idx = idx.masked_fill(~topk_mask, 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=idx.device)
        count.scatter_add_(-1, idx, torch.ones_like(idx, dtype=torch.int8))
        return count.masked_fill_(count > 1, 0)

    def _highest(self, mask_pos, overlaps, align_metric):
        fg = mask_pos.sum(-2)
        multi = (fg.unsqueeze(1) > 1).expand(-1, self.n_max_boxes, -1)
        best = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        best.scatter_(1, overlaps.argmax(1).unsqueeze(1), 1)
        mask_pos = torch.where(multi, best, mask_pos)
        if self.topk2 != self.topk:
            idx = torch.topk(align_metric * mask_pos, self.topk2, dim=-1).indices
            keep = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            keep.scatter_(-1, idx, 1)
            mask_pos = mask_pos * keep
        fg_mask, target_gt_idx = mask_pos.max(-2)
        return target_gt_idx, fg_mask, mask_pos

    def _targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch = torch.arange(self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        flat_idx = target_gt_idx + batch * self.n_max_boxes
        labels = gt_labels.long().flatten()[flat_idx].clamp_(0)
        bboxes = gt_bboxes.view(-1, 4)[flat_idx]
        scores = torch.zeros((*labels.shape, self.num_classes), dtype=torch.int8,
                             device=labels.device)
        scores.scatter_(2, labels.unsqueeze(-1), 1)
        scores = scores * (fg_mask[:, :, None] > 0)
        return labels, bboxes, scores


class DetectionLoss:
    """Loss for one head branch. Returns ``(loss * batch_size, items[box, cls, dfl])``."""

    def __init__(self, head, hyp: dict, topk: int = 10, topk2: int | None = None):
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.stride = head.stride
        self.hyp = hyp
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=self.nc, alpha=0.5, beta=6.0,
                                            stride=self.stride.tolist(), topk2=topk2)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.stride.device)

    def _targets(self, targets, batch_size, scale):
        device = scale.device
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=device)
        idx = targets[:, 0].long()
        counts = torch.bincount(idx, minlength=batch_size)
        out = torch.zeros(batch_size, int(counts.max()), 5, device=device)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
        within = torch.arange(targets.shape[0], device=device) - offsets[idx]
        out[idx, within] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5] * scale)
        return out

    def _decode(self, anchor_points, pred_dist):
        if self.reg_max > 1:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: dict, labels: torch.Tensor):
        device = preds["scores"].device
        loss = torch.zeros(3, device=device)
        pred_dist = preds["boxes"].float().permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].float().permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        bs = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=dtype) * self.stride[0]

        targets = self._targets(labels.to(device), bs, imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self._decode(anchor_points, pred_dist)
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_(min=1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        fg = fg_mask.nonzero(as_tuple=True)
        target_bboxes = target_bboxes / stride_tensor
        weight = target_scores[fg].sum(-1, keepdim=True)
        iou = bbox_ciou(pred_bboxes[fg], target_bboxes[fg])
        loss[0] = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.reg_max > 1:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            loss[2] = (
                _dfl(pred_dist[fg].view(-1, self.reg_max), target_ltrb[fg], self.reg_max) * weight
            ).sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes) * stride_tensor
            pred_ltrb = pred_dist * stride_tensor
            norm = torch.stack((imgsz[1], imgsz[0], imgsz[1], imgsz[0]))
            l1 = F.l1_loss(pred_ltrb[fg] / norm, target_ltrb[fg] / norm, reduction="none")
            loss[2] = (l1.mean(-1, keepdim=True) * weight).sum() / target_scores_sum

        loss[0] *= self.hyp["box"]
        loss[1] *= self.hyp["cls"]
        loss[2] *= self.hyp["dfl"]
        return loss.sum() * bs, loss.detach()


def _dfl(pred_dist, target, reg_max):
    target = target.clamp(0, reg_max - 1 - 0.01)
    tl = target.long()
    tr = tl + 1
    wl = tr - target
    wr = 1 - wl
    logp = F.log_softmax(pred_dist, dim=1)
    return -(
        logp.gather(1, tl.view(-1, 1)).view(tl.shape) * wl
        + logp.gather(1, tr.view(-1, 1)).view(tl.shape) * wr
    ).mean(-1, keepdim=True)


class EndToEndLoss:
    """YOLO26 dual-branch loss; call :meth:`update` once per epoch."""

    def __init__(self, head, hyp: dict, epochs: int):
        self.one2many = DetectionLoss(head, hyp, topk=10)
        self.one2one = DetectionLoss(head, hyp, topk=7, topk2=1)
        self.epochs = epochs
        self.updates = 0
        self.o2m_start, self.o2m_final = 0.8, 0.1
        self.o2m = self.o2m_start

    def __call__(self, preds: dict, labels: torch.Tensor):
        l_many, _ = self.one2many(preds["one2many"], labels)
        l_one, items = self.one2one(preds["one2one"], labels)
        return l_many * self.o2m + l_one * (1.0 - self.o2m), items

    def update(self) -> None:
        self.updates += 1
        frac = max(1 - self.updates / max(self.epochs - 1, 1), 0)
        self.o2m = frac * (self.o2m_start - self.o2m_final) + self.o2m_final


def build_loss(model, hyp: dict, epochs: int):
    head = model.head
    return EndToEndLoss(head, hyp, epochs) if head.end2end else DetectionLoss(head, hyp)
