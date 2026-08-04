"""Small U-Net for eye-region segmentation.

Classes: background / sclera+skin / iris / pupil / corneal reflex.
Deliberately tiny (<600k params) so it runs per-frame on a laptop CPU inside
the analyzer loop, which is where the pipeline actually needs it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

CLASS_NAMES = ("background", "ocular", "iris", "pupil", "reflex")
N_CLASSES = len(CLASS_NAMES)
PUPIL_CLASS = CLASS_NAMES.index("pupil")
IRIS_CLASS = CLASS_NAMES.index("iris")
REFLEX_CLASS = CLASS_NAMES.index("reflex")


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class EyeSegNet(nn.Module):
    def __init__(self, base: int = 16) -> None:
        super().__init__()
        b = base
        self.enc1 = _block(1, b)
        self.enc2 = _block(b, b * 2)
        self.enc3 = _block(b * 2, b * 4)
        self.bottleneck = _block(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = _block(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = _block(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = _block(b * 2, b)
        self.head = nn.Conv2d(b, N_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # pad to a multiple of 8 so the three pooling steps are exact
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        bott = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.dec3(torch.cat([self.up3(bott), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.head(d1)
        return out[..., :h, :w]
