# -*- coding: utf-8 -*-
'''Scalar adapters around pinned upstream backbones.

Upstream sources are cloned by tools/fetch_upstream.py at the commits recorded in
vendor/SOURCES.md. Nothing is vendored here; each adapter loads exactly one
upstream module so that same-named helper packages cannot clash.
'''

import importlib
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace

from torch import nn

VENDOR = Path(__file__).resolve().parents[1] / "vendor"


def _namespace(name, folder):
    if name not in sys.modules:
        module = ModuleType(name)
        module.__path__ = [str(folder)]
        sys.modules[name] = module


def _source(name, folder, relative):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(name, Path(folder) / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DLinear(nn.Module):
    def __init__(self, cfg, length=300, channels=3):
        super().__init__()
        module = _source("official_dlinear", VENDOR / "dlinear", "models/DLinear.py")
        self.backbone = module.Model(SimpleNamespace(
            seq_len=length, pred_len=cfg["latent_length"], enc_in=channels,
            individual=cfg["individual"]))
        self.head = nn.Linear(channels * cfg["latent_length"], 1)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1)).squeeze(-1)


class MixLinear(nn.Module):
    def __init__(self, cfg, length=300, channels=3):
        super().__init__()
        module = _source("official_mixlinear", VENDOR / "mixlinear", "models/MixLinear.py")
        self.backbone = module.Model(SimpleNamespace(
            seq_len=length, pred_len=cfg["latent_length"], enc_in=channels,
            period_len=cfg["period_len"], lpf=cfg["lpf"], alpha=cfg["alpha"]))
        self.head = nn.Linear(channels * cfg["latent_length"], 1)

    def forward(self, x):
        with redirect_stdout(io.StringIO()):   # upstream prints shapes; math untouched
            latent = self.backbone(x)
        return self.head(latent.flatten(1)).squeeze(-1)


class PatchTST(nn.Module):
    def __init__(self, cfg, length=300, channels=3):
        super().__init__()
        folder = VENDOR / "patchtst/PatchTST_self_supervised/src"
        _namespace("official_patchtst", folder)
        cls = importlib.import_module("official_patchtst.models.patchTST").PatchTST
        self.patch_len, self.stride = cfg["patch_len"], cfg["stride"]
        self.backbone = cls(
            c_in=channels, target_dim=1, patch_len=self.patch_len, stride=self.stride,
            num_patch=(length - self.patch_len) // self.stride + 1,
            n_layers=cfg["e_layers"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"], dropout=cfg["dropout"], head_dropout=cfg["dropout"],
            head_type="regression", shared_embedding=True, y_range=None)

    def forward(self, x):
        patches = x.unfold(1, self.patch_len, self.stride)
        return self.backbone(patches).squeeze(-1)


class ITransformer(nn.Module):
    def __init__(self, cfg, length=300, channels=3):
        super().__init__()
        module = _source("official_itransformer", VENDOR / "itransformer", "model/iTransformer.py")
        self.backbone = module.Model(SimpleNamespace(
            seq_len=length, pred_len=64, enc_in=channels, output_attention=False,
            use_norm=cfg["use_norm"], d_model=cfg["d_model"], d_ff=cfg["d_ff"],
            n_heads=cfg["n_heads"], e_layers=cfg["e_layers"], embed="timeF", freq="h",
            dropout=cfg["dropout"], class_strategy="projection", factor=5, activation="gelu"))
        self.backbone.projector = nn.Identity()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(cfg["dropout"]),
                                  nn.Linear(channels * cfg["d_model"], 1))

    def forward(self, x):
        tokens = self.backbone.enc_embedding(x, None)
        hidden, _ = self.backbone.encoder(tokens, attn_mask=None)
        return self.head(hidden).squeeze(-1)


class ModernTCN(nn.Module):
    def __init__(self, cfg, length=300, channels=3):
        super().__init__()
        folder = VENDOR / "moderntcn/ModernTCN-classification"
        module = _source("official_moderntcn", folder, "models/ModernTCN.py")
        self.backbone = module.ModernTCN(
            task_name="regression", patch_size=cfg["patch_size"], patch_stride=cfg["patch_stride"],
            stem_ratio=1, downsample_ratio=2, ffn_ratio=cfg["ffn_ratio"],
            num_blocks=cfg["num_blocks"], large_size=cfg["large_size"],
            small_size=cfg["small_size"], dims=cfg["dims"], dw_dims=cfg["dims"],
            nvars=channels, backbone_dropout=cfg["dropout"], head_dropout=cfg["dropout"],
            use_multi_scale=False, revin=False, seq_len=length, c_in=channels,
            individual=False, target_window=64)
        self.backbone.head = nn.Identity()
        self.head = nn.Sequential(nn.Flatten(), nn.GELU(), nn.Dropout(cfg["dropout"]),
                                  nn.Linear(channels * self.backbone.head_nf, 1))

    def forward(self, x):
        return self.head(self.backbone.forward_feature(x.transpose(1, 2))).squeeze(-1)


class TimeMixer(nn.Module):
    def __init__(self, cfg, length=300, channels=3):
        super().__init__()
        module = _source("official_timemixer", VENDOR / "timemixer", "models/TimeMixer.py")
        self.backbone = module.Model(SimpleNamespace(
            task_name="classification", seq_len=length, label_len=0, pred_len=64,
            enc_in=channels, c_out=channels, num_class=1, d_model=cfg["d_model"],
            d_ff=cfg["d_ff"], e_layers=cfg["e_layers"], dropout=cfg["dropout"],
            channel_independence=0, down_sampling_layers=cfg["down_sampling_layers"],
            down_sampling_window=cfg["down_sampling_window"], down_sampling_method="avg",
            moving_avg=cfg["moving_avg"], decomp_method="moving_avg", top_k=3,
            use_future_temporal_feature=False, use_norm=0, embed="timeF", freq="h"))

    def forward(self, x):
        mask = x.new_ones(x.shape[:2])
        return self.backbone(x, mask, None, None).squeeze(-1)


class TimeMixerPP(nn.Module):
    def __init__(self, cfg, length=300, channels=3):
        super().__init__()
        base = VENDOR / "timemixerpp/pypots"
        for name, folder in (("pypots", base), ("pypots.nn", base / "nn"),
                             ("pypots.nn.modules", base / "nn/modules")):
            _namespace(name, folder)
        module = importlib.import_module("pypots.nn.modules.timemixerpp.backbone")
        self.backbone = module.BackboneTimeMixerPP(
            task_name="classification", n_steps=length, n_features=channels,
            n_pred_steps=64, n_pred_features=channels, n_layers=cfg["e_layers"],
            d_model=cfg["d_model"], d_ffn=cfg["d_ff"], n_heads=cfg["n_heads"],
            dropout=cfg["dropout"], top_k=cfg["top_k"], n_kernels=cfg["n_kernels"],
            channel_mixing=False, channel_independence=False,
            downsampling_layers=cfg["down_sampling_layers"],
            downsampling_window=cfg["down_sampling_window"],
            downsampling_method="avg", use_future_temporal_feature=False,
            use_norm=False, n_classes=1)

    def forward(self, x):
        mask = x.new_ones(x.shape[:2])
        return self.backbone.classification(x, mask).squeeze(-1)


REGISTRY = {
    "DLinear": DLinear,
    "MixLinear": MixLinear,
    "PatchTST": PatchTST,
    "iTransformer": ITransformer,
    "ModernTCN": ModernTCN,
    "TimeMixer": TimeMixer,
    "TimeMixerPP": TimeMixerPP,
}


def build_baseline(name, cfg, length=300, channels=3):
    if name not in REGISTRY:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, length=length, channels=channels)


def micro_batch(name, cfg):
    return int(cfg.get("micro_batch_size", 512))
