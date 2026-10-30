"""Network blocks.

Attribute names (``cv1``, ``m``, ``one2one_cv2`` ...) are part of the checkpoint
format and must not be renamed.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        h = dim + self.key_dim * num_heads * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q, k, v = (
            self.qkv(x)
            .view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n)
            .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        )
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class SPPF(nn.Module):
    """Spatial pyramid pooling; YOLO26 rows (``n``/``shortcut`` given) drop the cv1 activation."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False, cv1_act=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=cv1_act)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(feats, strides, offset=0.5):
    points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for feat, stride in zip(feats, strides):
        h, w = feat.shape[2:]
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)


class Detect(nn.Module):
    """Anchor-free head.

    ``legacy`` selects the plain 3x3 class branch of v5u/v8; newer families use
    depthwise convolutions. ``end2end`` adds a detached one-to-one copy of both
    branches (YOLO26) whose top-k output needs no NMS.

    Training returns the raw prediction dict(s). Eval returns ``(decoded, raw)``
    where ``decoded`` is ``(B, 4 + nc, A)`` xywh+scores for NMS heads, or
    ``(B, max_det, 6)`` xyxy/score/class for end-to-end heads.
    """

    max_det = 300

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=(), legacy=False):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.legacy = legacy
        self.register_buffer("stride", torch.zeros(self.nl), persistent=False)
        c2 = max((16, ch[0] // 4, reg_max * 4))
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for x in ch
        )
        if legacy:
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch
            )
        else:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, nc, 1),
                )
                for x in ch
            )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        self.end2end = end2end
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def _branch(self, x, box_head, cls_head):
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], -1
        )
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], -1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(self, x):
        preds = self._branch(x, self.cv2, self.cv3)
        if self.end2end:
            detached = [xi.detach() for xi in x] if self.training else x
            preds = {
                "one2many": preds,
                "one2one": self._branch(detached, self.one2one_cv2, self.one2one_cv3),
            }
        if self.training:
            return preds
        if self.end2end:
            return self._topk(self.decode(preds["one2one"])), preds
        return self.decode(preds), preds

    def decode(self, p):
        anchors, strides = (a.transpose(0, 1) for a in make_anchors(p["feats"], self.stride))
        dist = self.dfl(p["boxes"])
        boxes = dist2bbox(dist, anchors.unsqueeze(0), xywh=not self.end2end, dim=1) * strides
        return torch.cat((boxes, p["scores"].sigmoid()), 1)

    def _topk(self, y):
        y = y.permute(0, 2, 1)
        boxes, scores = y.split((4, self.nc), dim=-1)
        k = min(self.max_det, y.shape[1])
        index = scores.amax(dim=-1).topk(k, dim=1).indices
        scores = scores.gather(1, index[..., None].expand(-1, -1, self.nc))
        scores, flat = scores.flatten(1).topk(k, dim=1)
        index = index.gather(1, flat // self.nc)
        boxes = boxes.gather(1, index[..., None].expand(-1, -1, 4))
        return torch.cat((boxes, scores[..., None], (flat % self.nc)[..., None].float()), -1)

    def bias_init(self):
        heads = [(self.cv2, self.cv3)]
        if self.end2end:
            heads.append((self.one2one_cv2, self.one2one_cv3))
        for box_head, cls_head in heads:
            for a, b, s in zip(box_head, cls_head, self.stride):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)
