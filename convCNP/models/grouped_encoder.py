"""
Targeted (grouped) ConvCNP encoders.

The original `Encoder` (encoder.py) runs a single depthwise set-convolution over
*all* input channels at once. These encoders instead bucket the input channels
into labeled groups (by physical variable -- e.g. all temperature levels/hours
into a T group, all geopotential into a Z group, the lat/lon/season/elevation
scaffold into a scaffold group) and encode each group into a shared, labeled
embedding position before the CNN. This gives the model an explicit per-variable
representation and collapses the many atmospheric channels into a few group
embeddings.

Two mechanisms are provided so their efficiency can be compared head-to-head:

- ``GroupedSetConvEncoder``    : per-group set-conv, then a per-group linear
                                  projection to a shared embedding slot.
- ``ChannelAttentionEncoder``  : per-group set-conv, then a learnable attention
                                  pooling that collapses the group's channels to
                                  one labeled embedding.

Both reuse the set-convolution math (absolute-value depthwise conv + a
density->confidence `ProbabilityConverter`) from the original encoder and emit
the same ``(batch, lat, lon, out_channels)`` tensor the CNN decoder expects, so
the rest of the pipeline is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import ProbabilityConverter


class _AbsConv2d(nn.Conv2d):
    """Conv2d that uses the absolute value of its weights (set-conv density)."""

    def forward(self, input):
        return F.conv2d(
            input,
            self.weight.abs(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _GroupSetConv(nn.Module):
    """Set-convolution over a single channel group.

    Returns the density-normalized signal and a per-channel confidence map,
    mirroring the math in ``encoder.Encoder`` but for an arbitrary channel count.
    """

    def __init__(self, n_channels: int, kernel_size: int = 5):
        super().__init__()
        self.n_channels = n_channels
        self.conv = _AbsConv2d(
            n_channels,
            n_channels,
            groups=n_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.density_to_confidence = ProbabilityConverter(trainable_dim=n_channels)

    def forward(self, x, mask):
        batch_size, n_channels, h, w = x.shape
        num = self.conv(x * mask)
        denom = self.conv(mask)
        signal = num / torch.clamp(denom, min=1e-5)
        confidence = self.density_to_confidence(
            denom.view(-1, n_channels) * 0.1
        ).view(batch_size, n_channels, h, w)
        return signal, confidence


def _normalize_groups(groups):
    """Accept {label: [idx]} or [(label, [idx])] and return a list of tuples."""
    if isinstance(groups, dict):
        return list(groups.items())
    return list(groups)


class GroupedSetConvEncoder(nn.Module):
    """Per-group set-conv encoder.

    Each labeled group is set-convolved independently and projected to a shared
    ``group_dim`` embedding; the group embeddings are concatenated (each occupying
    its own labeled slice -- "the same position") and projected to ``out_channels``.

    Parameters
    ----------
    groups: mapping/label-list of (label, [channel indices]) defining the groups.
    out_channels: width fed to the CNN decoder (matches the flat encoder, 128).
    kernel_size: set-conv kernel size.
    """

    def __init__(self, groups, out_channels: int = 128, kernel_size: int = 5):
        super().__init__()
        groups = _normalize_groups(groups)
        self.group_labels = [label for label, _ in groups]
        self.out_channels = out_channels
        self.in_channels = sum(len(idx) for _, idx in groups)
        n_groups = len(groups)
        group_dim = max(8, out_channels // n_groups)
        self.group_dim = group_dim

        self.setconvs = nn.ModuleList(
            [_GroupSetConv(len(idx), kernel_size) for _, idx in groups]
        )
        self.group_proj = nn.ModuleList(
            [nn.Linear(2 * len(idx), group_dim) for _, idx in groups]
        )
        self.transform_to_cnn = nn.Linear(group_dim * n_groups, out_channels)

        for i, (_, idx) in enumerate(groups):
            self.register_buffer(
                f"group_idx_{i}", torch.as_tensor(list(idx), dtype=torch.long),
                persistent=False,
            )

    def forward(self, x, mask):
        group_embeds = []
        for i, setconv in enumerate(self.setconvs):
            idx = getattr(self, f"group_idx_{i}")
            xg = x.index_select(1, idx)
            mg = mask.index_select(1, idx)
            signal, confidence = setconv(xg, mg)
            hg = torch.cat([signal, confidence], dim=1)  # (b, 2*len, lat, lon)
            hg = self.group_proj[i](hg.permute(0, 2, 3, 1))  # (b, lat, lon, group_dim)
            group_embeds.append(hg)
        h = torch.cat(group_embeds, dim=-1)  # (b, lat, lon, group_dim * n_groups)
        return self.transform_to_cnn(h)  # (b, lat, lon, out_channels)


class ChannelAttentionEncoder(nn.Module):
    """Per-group set-conv encoder with learnable channel-attention pooling.

    Within each labeled group, a learnable attention over the group's channels
    collapses the set-conv signal/confidence to a single labeled embedding; the
    pooled group embeddings are concatenated and projected to ``out_channels``.

    Parameters mirror :class:`GroupedSetConvEncoder`.
    """

    def __init__(self, groups, out_channels: int = 128, kernel_size: int = 5):
        super().__init__()
        groups = _normalize_groups(groups)
        self.group_labels = [label for label, _ in groups]
        self.out_channels = out_channels
        self.in_channels = sum(len(idx) for _, idx in groups)
        n_groups = len(groups)
        group_dim = max(8, out_channels // n_groups)
        self.group_dim = group_dim

        self.setconvs = nn.ModuleList(
            [_GroupSetConv(len(idx), kernel_size) for _, idx in groups]
        )
        # Learnable attention logits over the channels of each group.
        self.attn_logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(len(idx))) for _, idx in groups]
        )
        # Pooled (signal, confidence) -> group embedding.
        self.group_proj = nn.ModuleList(
            [nn.Linear(2, group_dim) for _ in groups]
        )
        self.transform_to_cnn = nn.Linear(group_dim * n_groups, out_channels)

        for i, (_, idx) in enumerate(groups):
            self.register_buffer(
                f"group_idx_{i}", torch.as_tensor(list(idx), dtype=torch.long),
                persistent=False,
            )

    def forward(self, x, mask):
        group_embeds = []
        for i, setconv in enumerate(self.setconvs):
            idx = getattr(self, f"group_idx_{i}")
            xg = x.index_select(1, idx)
            mg = mask.index_select(1, idx)
            signal, confidence = setconv(xg, mg)  # (b, len, lat, lon)
            attn = torch.softmax(self.attn_logits[i], dim=0).view(1, -1, 1, 1)
            pooled_signal = (signal * attn).sum(dim=1, keepdim=True)  # (b, 1, lat, lon)
            pooled_conf = (confidence * attn).sum(dim=1, keepdim=True)
            hg = torch.cat([pooled_signal, pooled_conf], dim=1)  # (b, 2, lat, lon)
            hg = self.group_proj[i](hg.permute(0, 2, 3, 1))  # (b, lat, lon, group_dim)
            group_embeds.append(hg)
        h = torch.cat(group_embeds, dim=-1)
        return self.transform_to_cnn(h)  # (b, lat, lon, out_channels)


def build_grouped_encoder(kind: str, groups, out_channels: int = 128, kernel_size: int = 5):
    """Construct a targeted encoder by name.

    Args:
        kind: 'grouped_setconv' or 'channel_attention'.
        groups: mapping/label-list of (label, [channel indices]).
        out_channels: CNN-decoder width.
        kernel_size: set-conv kernel size.
    """
    if kind == "grouped_setconv":
        return GroupedSetConvEncoder(groups, out_channels=out_channels, kernel_size=kernel_size)
    if kind == "channel_attention":
        return ChannelAttentionEncoder(groups, out_channels=out_channels, kernel_size=kernel_size)
    raise ValueError(f"Unknown targeted encoder kind: {kind!r}")
