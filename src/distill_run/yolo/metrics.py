"""Post-processing and COCO-style mAP (101-point, IoU 0.50:0.95)."""

from __future__ import annotations

import numpy as np
import torch

from .loss import box_iou, xywh2xyxy

IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


def non_max_suppression(pred, conf_thres=0.001, iou_thres=0.7, max_det=300, max_nms=30000,
                        multi_label=True):
    """``pred``: ``(B, 4 + nc, A)`` xywh+scores -> list of ``(n, 6)`` xyxy/conf/cls."""
    import torchvision

    nc = pred.shape[1] - 4
    pred = pred.transpose(1, 2)
    out = []
    for x in pred:
        boxes = xywh2xyxy(x[:, :4])
        scores = x[:, 4:]
        if multi_label and nc > 1:
            i, j = (scores > conf_thres).nonzero(as_tuple=True)
            det = torch.cat((boxes[i], scores[i, j, None], j[:, None].float()), 1)
        else:
            conf, j = scores.max(1, keepdim=True)
            keep = conf.view(-1) > conf_thres
            det = torch.cat((boxes, conf, j.float()), 1)[keep]
        if det.shape[0] > max_nms:
            det = det[det[:, 4].argsort(descending=True)[:max_nms]]
        if det.shape[0]:
            keep = torchvision.ops.batched_nms(det[:, :4], det[:, 4], det[:, 5].long(), iou_thres)
            det = det[keep[:max_det]]
        out.append(det)
    return out


def end2end_filter(pred, conf_thres=0.001):
    """``pred``: ``(B, k, 6)`` from an end-to-end head -> list of ``(n, 6)``."""
    return [p[p[:, 4] > conf_thres] for p in pred]


def match(det: torch.Tensor, gt_cls: torch.Tensor, gt_boxes: torch.Tensor) -> np.ndarray:
    """Return ``(n_det, 10)`` true-positive flags over the IoU thresholds."""
    correct = np.zeros((det.shape[0], len(IOU_THRESHOLDS)), dtype=bool)
    if det.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return correct
    iou = box_iou(gt_boxes, det[:, :4])
    iou = iou * (gt_cls[:, None] == det[:, 5][None, :])
    iou = iou.cpu().numpy()
    for k, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        gi, di = np.nonzero(iou >= threshold)
        if not len(gi):
            continue
        pairs = np.stack((gi, di, iou[gi, di]), 1)
        pairs = pairs[pairs[:, 2].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        correct[pairs[:, 1].astype(int), k] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x) if hasattr(np, "trapezoid")
                 else np.trapz(np.interp(x, mrec, mpre), x))


class DetectionMetrics:
    """Accumulates per-image matches and reports precision/recall/mAP."""

    def __init__(self, nc: int):
        self.nc = nc
        self.tp: list[np.ndarray] = []
        self.conf: list[np.ndarray] = []
        self.pred_cls: list[np.ndarray] = []
        self.target_cls: list[np.ndarray] = []

    def update(self, det: torch.Tensor, gt_cls: torch.Tensor, gt_boxes: torch.Tensor) -> None:
        self.tp.append(match(det, gt_cls, gt_boxes))
        self.conf.append(det[:, 4].cpu().numpy())
        self.pred_cls.append(det[:, 5].cpu().numpy())
        self.target_cls.append(gt_cls.cpu().numpy())

    def compute(self) -> dict[str, float]:
        tp = np.concatenate(self.tp) if self.tp else np.zeros((0, 10), bool)
        conf = np.concatenate(self.conf) if self.conf else np.zeros(0)
        pred_cls = np.concatenate(self.pred_cls) if self.pred_cls else np.zeros(0)
        target_cls = np.concatenate(self.target_cls) if self.target_cls else np.zeros(0)
        order = np.argsort(-conf)
        tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
        classes = np.unique(target_cls)
        ap = np.zeros((len(classes), tp.shape[1]))
        p_at, r_at = np.zeros(len(classes)), np.zeros(len(classes))
        for ci, c in enumerate(classes):
            sel = pred_cls == c
            n_gt = int((target_cls == c).sum())
            if not sel.any() or n_gt == 0:
                continue
            fpc = (1 - tp[sel]).cumsum(0)
            tpc = tp[sel].cumsum(0)
            recall = tpc / (n_gt + 1e-16)
            precision = tpc / (tpc + fpc)
            for k in range(tp.shape[1]):
                ap[ci, k] = compute_ap(recall[:, k], precision[:, k])
            # precision/recall at the confidence that maximises F1 on IoU 0.5
            f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + 1e-16)
            best = int(f1.argmax())
            p_at[ci], r_at[ci] = precision[best, 0], recall[best, 0]
        if not len(classes):
            return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}
        return {
            "precision": float(p_at.mean()),
            "recall": float(r_at.mean()),
            "mAP50": float(ap[:, 0].mean()),
            "mAP50-95": float(ap.mean()),
        }
