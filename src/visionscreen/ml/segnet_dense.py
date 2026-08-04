"""A compact dense encoder-decoder for eye segmentation.

The benchmark to beat (RITnet, 95.3% mIoU) reaches that with 248,900
parameters, so capacity is not the lever — feature reuse is. Dense blocks
concatenate every layer's output onto the block input, so each successive
convolution sees all previous features and needs far fewer channels of its own
to express the same thing.

Two departures from a textbook DenseNet, both aimed at this task:

* **Average pooling, not max**, for downsampling. Eye boundaries here are soft
  intensity ramps (limbus, sclera-to-eyelid), and max-pooling discards the
  gradient that localises them.
* **A shared 1x1 projection before each block's output.** It keeps the channel
  count flat across depth so the decoder stays cheap, which is where a naive
  dense net spends its parameter budget.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DenseBlock(nn.Module):
    """Four 3x3 convolutions, each seeing the concatenation of all before it."""

    def __init__(self, cin: int, growth: int, cout: int, layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        ch = cin
        for _ in range(layers):
            self.convs.append(nn.Sequential(
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(ch, growth, 3, padding=1, bias=False),
            ))
            ch += growth
        # flatten back to a fixed width so depth does not inflate the decoder
        self.project = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, cout, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for conv in self.convs:
            feats.append(conv(torch.cat(feats, 1)))
        return self.project(torch.cat(feats, 1))


class DenseEyeNet(nn.Module):
    def __init__(self, n_classes: int = 4, width: int = 32, growth: int = 12,
                 depth: int = 4, in_ch: int = 1):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1, bias=False)
        self.encoders = nn.ModuleList(
            [DenseBlock(width, growth, width) for _ in range(depth)]
        )
        self.bottleneck = DenseBlock(width, growth, width)
        # decoder takes [upsampled, skip] -> width
        self.decoders = nn.ModuleList(
            [DenseBlock(width * 2, growth, width) for _ in range(depth)]
        )
        self.head = nn.Conv2d(width, n_classes, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad = 1 << self.depth
        ph, pw = (-h) % pad, (-w) % pad
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")

        y = self.stem(x)
        skips = []
        for enc in self.encoders:
            y = enc(y)
            skips.append(y)
            y = F.avg_pool2d(y, 2)          # soft boundaries survive averaging
        y = self.bottleneck(y)
        for dec, skip in zip(self.decoders, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            y = dec(torch.cat([y, skip], 1))
        out = self.head(y)
        return out[..., :h, :w]


# ------------------------------------------------------------------ losses ---

def generalized_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """Dice weighted by inverse-square class frequency.

    Pupil is a few percent of the pixels; unweighted dice lets the model ignore
    it and still look good. The inverse-square weight is what makes the rare
    class carry comparable gradient to the background.
    """
    n_cls = logits.shape[1]
    probs = logits.softmax(1)
    onehot = F.one_hot(target, n_cls).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    counts = onehot.sum(dims)

    # Classes absent from this batch must be dropped, not down-weighted. With
    # 1/count^2 weighting an absent class takes weight 1/eps^2, which swamps
    # the denominator and pins the loss at 1.0 no matter what the model
    # predicts — the term then contributes pure noise. This bites constantly
    # here because pupil or sclera is missing from many closed-eye frames.
    present = counts > 0
    if not present.any():
        return logits.sum() * 0.0
    weights = torch.zeros_like(counts)
    weights[present] = 1.0 / counts[present].pow(2)
    # renormalise so the loss scale does not swing with how many classes
    # happen to appear in a batch
    weights = weights / weights.sum().clamp_min(eps)

    inter = (probs * onehot).sum(dims)
    denom = (probs + onehot).sum(dims)
    num = 2.0 * (weights * inter).sum()
    den = (weights * denom).sum()
    return 1.0 - num / den.clamp_min(eps)


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy re-weighted toward pixels near a class boundary.

    Almost all of the residual mIoU on this benchmark lives in a few pixels at
    each edge; a uniform loss spends its gradient on easy interiors instead.
    """
    n_cls = logits.shape[1]
    onehot = F.one_hot(target, n_cls).permute(0, 3, 1, 2).float()
    # a pixel is "on a boundary" where a 3x3 max and min of the label disagree
    pooled_max = F.max_pool2d(onehot, 3, 1, 1)
    pooled_min = -F.max_pool2d(-onehot, 3, 1, 1)
    edge = (pooled_max - pooled_min).sum(1, keepdim=True).clamp(0, 1)
    weight = 1.0 + 4.0 * edge.squeeze(1)
    ce = F.cross_entropy(logits, target, reduction="none")
    return (ce * weight).mean()


class CombinedLoss(nn.Module):
    """CE + boundary-weighted CE + generalized dice, as RITnet found necessary.

    Cross-entropy alone converges to a good pixel accuracy and a mediocre mIoU:
    it has no term that cares about the rare classes or the edges, which are
    exactly what mIoU measures.
    """

    def __init__(self, w_ce: float = 1.0, w_bd: float = 1.0, w_gdl: float = 1.0):
        super().__init__()
        self.w_ce, self.w_bd, self.w_gdl = w_ce, w_bd, w_gdl

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.w_ce * F.cross_entropy(logits, target)
                + self.w_bd * boundary_loss(logits, target)
                + self.w_gdl * generalized_dice_loss(logits, target))
