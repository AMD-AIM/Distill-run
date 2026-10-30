"""Architecture tables and the model builder.

Each family is a list of ``(from, repeats, block, args)`` rows plus per-scale
``(depth, width, max_channels)`` multipliers. The layouts match the published
checkpoints so state-dict keys line up (``model.<layer>.<...>``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
from torch import nn

from .modules import C2PSA, C3, SPPF, C2f, C3k2, Concat, Conv, Detect

_BLOCKS = {"Conv": Conv, "C2f": C2f, "C3": C3, "C3k2": C3k2, "SPPF": SPPF, "C2PSA": C2PSA}
_REPEAT_BLOCKS = (C2f, C3, C3k2, C2PSA)

_HEAD_NECK_11 = [
    (-1, 1, "Upsample", []),
    ([-1, 6], 1, "Concat", []),
    (-1, 2, "C3k2", [512, False]),
    (-1, 1, "Upsample", []),
    ([-1, 4], 1, "Concat", []),
    (-1, 2, "C3k2", [256, False]),
    (-1, 1, "Conv", [256, 3, 2]),
    ([-1, 13], 1, "Concat", []),
    (-1, 2, "C3k2", [512, False]),
    (-1, 1, "Conv", [512, 3, 2]),
    ([-1, 10], 1, "Concat", []),
    (-1, 2, "C3k2", [1024, True]),
    ([16, 19, 22], 1, "Detect", []),
]

_BACKBONE_11 = [
    (-1, 1, "Conv", [64, 3, 2]),
    (-1, 1, "Conv", [128, 3, 2]),
    (-1, 2, "C3k2", [256, False, 0.25]),
    (-1, 1, "Conv", [256, 3, 2]),
    (-1, 2, "C3k2", [512, False, 0.25]),
    (-1, 1, "Conv", [512, 3, 2]),
    (-1, 2, "C3k2", [512, True]),
    (-1, 1, "Conv", [1024, 3, 2]),
    (-1, 2, "C3k2", [1024, True]),
]


@dataclass(frozen=True)
class Architecture:
    scales: dict[str, tuple[float, float, int]]
    layers: list[tuple]
    reg_max: int = 16
    end2end: bool = False
    weights_name: str = "{family}{scale}"


ARCHITECTURES: dict[str, Architecture] = {
    "yolov5": Architecture(
        scales={
            "n": (0.33, 0.25, 1024),
            "s": (0.33, 0.50, 1024),
            "m": (0.67, 0.75, 1024),
            "l": (1.00, 1.00, 1024),
            "x": (1.33, 1.25, 1024),
        },
        layers=[
            (-1, 1, "Conv", [64, 6, 2, 2]),
            (-1, 1, "Conv", [128, 3, 2]),
            (-1, 3, "C3", [128]),
            (-1, 1, "Conv", [256, 3, 2]),
            (-1, 6, "C3", [256]),
            (-1, 1, "Conv", [512, 3, 2]),
            (-1, 9, "C3", [512]),
            (-1, 1, "Conv", [1024, 3, 2]),
            (-1, 3, "C3", [1024]),
            (-1, 1, "SPPF", [1024, 5]),
            (-1, 1, "Conv", [512, 1, 1]),
            (-1, 1, "Upsample", []),
            ([-1, 6], 1, "Concat", []),
            (-1, 3, "C3", [512, False]),
            (-1, 1, "Conv", [256, 1, 1]),
            (-1, 1, "Upsample", []),
            ([-1, 4], 1, "Concat", []),
            (-1, 3, "C3", [256, False]),
            (-1, 1, "Conv", [256, 3, 2]),
            ([-1, 14], 1, "Concat", []),
            (-1, 3, "C3", [512, False]),
            (-1, 1, "Conv", [512, 3, 2]),
            ([-1, 10], 1, "Concat", []),
            (-1, 3, "C3", [1024, False]),
            ([17, 20, 23], 1, "Detect", []),
        ],
        weights_name="yolov5{scale}u",
    ),
    "yolov8": Architecture(
        scales={
            "n": (0.33, 0.25, 1024),
            "s": (0.33, 0.50, 1024),
            "m": (0.67, 0.75, 768),
            "l": (1.00, 1.00, 512),
            "x": (1.00, 1.25, 512),
        },
        layers=[
            (-1, 1, "Conv", [64, 3, 2]),
            (-1, 1, "Conv", [128, 3, 2]),
            (-1, 3, "C2f", [128, True]),
            (-1, 1, "Conv", [256, 3, 2]),
            (-1, 6, "C2f", [256, True]),
            (-1, 1, "Conv", [512, 3, 2]),
            (-1, 6, "C2f", [512, True]),
            (-1, 1, "Conv", [1024, 3, 2]),
            (-1, 3, "C2f", [1024, True]),
            (-1, 1, "SPPF", [1024, 5]),
            (-1, 1, "Upsample", []),
            ([-1, 6], 1, "Concat", []),
            (-1, 3, "C2f", [512]),
            (-1, 1, "Upsample", []),
            ([-1, 4], 1, "Concat", []),
            (-1, 3, "C2f", [256]),
            (-1, 1, "Conv", [256, 3, 2]),
            ([-1, 12], 1, "Concat", []),
            (-1, 3, "C2f", [512]),
            (-1, 1, "Conv", [512, 3, 2]),
            ([-1, 9], 1, "Concat", []),
            (-1, 3, "C2f", [1024]),
            ([15, 18, 21], 1, "Detect", []),
        ],
    ),
    "yolo11": Architecture(
        scales={
            "n": (0.50, 0.25, 1024),
            "s": (0.50, 0.50, 1024),
            "m": (0.50, 1.00, 512),
            "l": (1.00, 1.00, 512),
            "x": (1.00, 1.50, 512),
        },
        layers=[
            *_BACKBONE_11,
            (-1, 1, "SPPF", [1024, 5]),
            (-1, 2, "C2PSA", [1024]),
            *_HEAD_NECK_11,
        ],
    ),
    "yolo26": Architecture(
        scales={
            "n": (0.50, 0.25, 1024),
            "s": (0.50, 0.50, 1024),
            "m": (0.50, 1.00, 512),
            "l": (1.00, 1.00, 512),
            "x": (1.00, 1.50, 512),
        },
        layers=[
            *_BACKBONE_11,
            (-1, 1, "SPPF", [1024, 5, 3, True]),
            (-1, 2, "C2PSA", [1024]),
            (-1, 1, "Upsample", []),
            ([-1, 6], 1, "Concat", []),
            (-1, 2, "C3k2", [512, True]),
            (-1, 1, "Upsample", []),
            ([-1, 4], 1, "Concat", []),
            (-1, 2, "C3k2", [256, True]),
            (-1, 1, "Conv", [256, 3, 2]),
            ([-1, 13], 1, "Concat", []),
            (-1, 2, "C3k2", [512, True]),
            (-1, 1, "Conv", [512, 3, 2]),
            ([-1, 10], 1, "Concat", []),
            (-1, 1, "C3k2", [1024, True, 0.5, True]),
            ([16, 19, 22], 1, "Detect", []),
        ],
        reg_max=1,
        end2end=True,
    ),
}

_ALIASES = {"yolov5": "yolov5", "yolo5": "yolov5", "yolov8": "yolov8", "yolo8": "yolov8",
            "yolo11": "yolo11", "yolov11": "yolo11", "yolo26": "yolo26", "yolov26": "yolo26"}
_NAME = re.compile(r"^(yolov?\d+)([nsmlx])u?$")

FAMILIES = tuple(ARCHITECTURES)
SCALES = ("n", "s", "m", "l", "x")


def parse_name(name: str) -> tuple[str, str] | None:
    """``yolo26n`` -> ("yolo26", "n"); ``yolov5su`` -> ("yolov5", "s"); else None."""
    match = _NAME.match(name.strip().lower())
    if not match or match.group(1) not in _ALIASES:
        return None
    return _ALIASES[match.group(1)], match.group(2)


def weights_filename(family: str, scale: str) -> str:
    return ARCHITECTURES[family].weights_name.format(family=family, scale=scale) + ".pt"


def make_divisible(x: float, divisor: int = 8) -> int:
    return int(math.ceil(x / divisor) * divisor)


def build_layers(family: str, scale: str, nc: int) -> tuple[nn.Sequential, list, list[int]]:
    arch = ARCHITECTURES[family]
    depth, width, max_channels = arch.scales[scale]
    legacy = True
    channels = [3]
    layers, froms, save = [], [], set()
    for i, (src, repeats, block_name, row_args) in enumerate(arch.layers):
        args = list(row_args)
        n = max(round(repeats * depth), 1) if repeats > 1 else repeats
        if block_name in _BLOCKS:
            block = _BLOCKS[block_name]
            c1 = channels[src]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if block in _REPEAT_BLOCKS:
                args.insert(2, n)
                n = 1
            if block is C3k2:
                legacy = False
                if scale in {"m", "l", "x"}:
                    args[3:4] = [True]
            kwargs = {}
            if block is SPPF and len(args) <= 3:
                kwargs["cv1_act"] = True
            module = (
                nn.Sequential(*(block(*args, **kwargs) for _ in range(n)))
                if n > 1
                else block(*args, **kwargs)
            )
        elif block_name == "Upsample":
            c2 = channels[src]
            module = nn.Upsample(None, 2, "nearest")
        elif block_name == "Concat":
            c2 = sum(channels[x] for x in src)
            module = Concat(1)
        elif block_name == "Detect":
            module = Detect(
                nc, arch.reg_max, arch.end2end, [channels[x] for x in src], legacy=legacy
            )
            c2 = None
        else:
            raise ValueError(f"unknown block {block_name}")
        for x in [src] if isinstance(src, int) else src:
            if x != -1:
                save.add(x % i)
        layers.append(module)
        froms.append(src)
        if i == 0:
            channels = []
        channels.append(c2)
    return nn.Sequential(*layers), froms, sorted(save)


class DetectionModel(nn.Module):
    """A YOLO detector whose ``model`` attribute holds the numbered layers."""

    def __init__(self, family: str, scale: str, nc: int = 80):
        super().__init__()
        if family not in ARCHITECTURES or scale not in SCALES:
            raise ValueError(f"unknown model {family}{scale}")
        self.family, self.scale, self.nc = family, scale, nc
        self.model, self._froms, self._save = build_layers(family, scale, nc)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
            elif isinstance(m, nn.SiLU):
                m.inplace = True
        head = self.head
        with torch.no_grad():
            self.eval()
            _, raw = self.forward(torch.zeros(1, 3, 256, 256))
            feats = (raw["one2many"] if head.end2end else raw)["feats"]
            head.stride.copy_(torch.tensor([256 / f.shape[-2] for f in feats]))
            self.train()
        head.bias_init()

    @property
    def head(self) -> Detect:
        return self.model[-1]

    @property
    def stride(self) -> torch.Tensor:
        return self.head.stride

    def forward(self, x):
        outputs: list = []
        for layer, src in zip(self.model, self._froms):
            if isinstance(src, list):
                x = [x if j == -1 else outputs[j] for j in src]
            elif src != -1:
                x = outputs[src]
            x = layer(x)
            outputs.append(x if len(outputs) in self._save else None)
        return x

    def describe(self) -> dict:
        params = sum(p.numel() for p in self.parameters())
        return {"family": self.family, "scale": self.scale, "nc": self.nc, "parameters": params}
