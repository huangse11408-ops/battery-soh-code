# -*- coding: utf-8 -*-
'''GRU backbone, anchor pooling and the SARG-GRU readout with four variants.'''

import copy

import torch
from torch import nn

VARIANTS = (
    "full",
    "no_ordered_differences_and_gating",
    "time_anchors",
    "no_voltage_branch",
)


class GRUBackbone(nn.Module):
    r'''Two-layer GRU over the 300-step trace plus a scalar regression head.'''

    def __init__(self, channels=3, hidden=64, dropout=0.1):
        super().__init__()
        self.encoder = nn.GRU(channels, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        _, state = self.encoder(x)
        return self.head(state[-1]).squeeze(-1)


def masked_mean(x, mask):
    return (x * mask.unsqueeze(-1)).sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp_min(1)


def progress(time):
    span = time[:, -1:] - time[:, :1]
    return (time - time[:, :1]) / span.clamp_min(1e-12), (span > 0).expand_as(time)


def pool(sequence, coordinate, anchors, sigma, cutoff, points=None):
    r'''Gaussian-weighted pooling of the sequence at fixed coordinate anchors.'''
    if points is None:
        points = torch.ones_like(coordinate, dtype=torch.bool)
    points = points & torch.isfinite(coordinate)
    lo = coordinate.masked_fill(~points, torch.inf).amin(dim=1, keepdim=True)
    hi = coordinate.masked_fill(~points, -torch.inf).amax(dim=1, keepdim=True)
    support = (anchors >= lo) & (anchors <= hi)
    safe = torch.where(points, coordinate, 0.0)
    delta = (safe[:, None, :] - anchors[None, :, None]) / sigma
    hits = (delta.abs() <= cutoff) & support[:, :, None] & points[:, None, :]
    weights = (-0.5 * delta.square()).exp() * hits
    observed = hits.any(2)
    weights = weights / weights.sum(2, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return torch.bmm(weights, sequence), observed


def pair_delta(z, valid):
    r'''Differences between adjacent anchor memories, masked by joint validity.'''
    pairs = valid[:, :-1] & valid[:, 1:]
    delta = (z[:, 1:] - z[:, :-1]) * pairs.unsqueeze(-1)
    return delta, pairs


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _resize_head(head, inputs):
    result = copy.deepcopy(head)
    result[0] = nn.Linear(inputs, head[0].out_features)
    return result


class SARGReadout(nn.Module):
    r'''Readout that fuses a time branch with a voltage-anchor memory branch.'''

    def __init__(self, backbone, variant, anchors, sigma, cutoff):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant
        self.encoder = backbone.encoder
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.sigma, self.cutoff = float(sigma), float(cutoff)

        width = self.encoder.hidden_size * (2 if self.encoder.bidirectional else 1)
        self.uses_voltage = variant != "no_voltage_branch"
        self.uses_differences = variant in {"full", "time_anchors"}
        self.uses_gate = variant in {"full", "time_anchors"}
        self.time_coordinate = variant == "time_anchors"

        self.time_projection = nn.Sequential(nn.Linear(2 * width, width), nn.GELU())
        self.head = _resize_head(backbone.head, width)
        if self.uses_voltage:
            self.voltage_projection = nn.Sequential(nn.Linear(2 * width, width), nn.GELU())
            if self.uses_differences:
                self.differences = nn.Sequential(
                    nn.Conv1d(width, width, kernel_size=3, padding=1), nn.GELU())
            if self.uses_gate:
                self.gate = nn.Linear(2 * width, width)

    def voltage_memory(self, z, valid):
        marginal = masked_mean(z, valid)
        pairs = valid[:, :-1] & valid[:, 1:]
        if self.uses_differences:
            delta, pairs = pair_delta(z, valid)
            encoded = self.differences(delta.transpose(1, 2)).transpose(1, 2)
            diff = masked_mean(encoded, pairs)
        else:
            diff = torch.zeros_like(marginal)
        memory = self.voltage_projection(torch.cat((marginal, diff), dim=1))
        return memory, valid.any(1), pairs

    def forward(self, x, coordinate, stages=None, diagnostics=False):
        sequence, _ = self.encoder(x)
        temporal = torch.cat((sequence[:, -1], sequence.mean(1)), dim=1)
        mt = self.time_projection(temporal)

        if not self.uses_voltage:
            y = self.head(mt).squeeze(-1)
            if diagnostics:
                empty = torch.zeros_like(mt)[:, None, :]
                return y, empty, empty
            return y

        if self.time_coordinate:
            coordinate, observed = progress(coordinate)
            z, valid = pool(sequence, coordinate, self.anchors, self.sigma, self.cutoff, observed)
        else:
            z, valid = pool(sequence, coordinate, self.anchors, self.sigma, self.cutoff)

        mv, has_memory, pairs = self.voltage_memory(z, valid)
        if self.uses_gate:
            gate = self.gate(torch.cat((mt, mv), 1)).sigmoid()
            fused = mt + gate * (mv - mt)
        else:
            fused = (mt + mv) * 0.5
        y = self.head(torch.where(has_memory[:, None], fused, mt)).squeeze(-1)
        if diagnostics:
            return y, valid[:, None, :], pairs[:, None, :]
        return y


def build_sarg(variant, anchor_config, channels=3, hidden=64, dropout=0.1):
    backbone = GRUBackbone(channels=channels, hidden=hidden, dropout=dropout)
    return SARGReadout(backbone, variant, anchor_config["anchors"],
                       anchor_config["sigma"], anchor_config["cutoff"])
