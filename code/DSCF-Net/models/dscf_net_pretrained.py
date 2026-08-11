"""
DSCF-Net variant with pretrained backbone (EfficientNet-B0 via torchvision).

Retains the ACS Fusion neck and multi-scale classification head
from DSCF-Net, while leveraging ImageNet-pretrained features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from .dscf_net import ACSFusion, PDC2k, DSCFHead


class PretrainedBackbone(nn.Module):
    """Pretrained EfficientNet-B0 backbone, outputs 3 multi-scale feature levels.

    Outputs:
        f0: 40ch @ 32×32 (H/8)  — from features[3]
        f1: 112ch @ 16×16 (H/16) — from features[5]
        f2: 320ch @ 8×8 (H/32)  — from features[7]
    """

    def __init__(self, freeze_stem=True):
        super().__init__()
        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Group features into meaningful stages
        # features[0:3]: 256→64, 3→24ch (stem + early MBConv)
        self.stage_early = nn.Sequential(*effnet.features[0:3])
        # features[3:5]: 64→16, 24→112ch
        self.stage_mid = nn.Sequential(effnet.features[3], effnet.features[4])
        # features[5:8]: 16→8, 112→320ch
        self.stage_late = nn.Sequential(
            effnet.features[5],
            effnet.features[6],
            effnet.features[7],
        )

        if freeze_stem:
            for p in self.stage_early.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.stage_early(x)     # 24ch @ 64×64
        f0 = self.stage_mid[0](x)   # 40ch @ 32×32
        x = self.stage_mid[1](f0)   # 80ch @ 16×16
        f1 = self.stage_late[0](x)  # 112ch @ 16×16
        x = self.stage_late[1](f1)  # 192ch @ 8×8
        f2 = self.stage_late[2](x)  # 320ch @ 8×8
        return [f0, f1, f2]


class DSCFNetPretrained(nn.Module):
    """DSCF-Net with pretrained EfficientNet backbone (torchvision).

    Backbone: EfficientNet-B0 (pretrained on ImageNet, ~4M params)
    Neck: ACS Fusion + PDC-2k (same as original DSCF-Net)
    Head: Multi-scale classification head

    Total: ~9M params (~4M backbone + ~5M neck/head)
    """

    def __init__(self, num_classes, neck_channels=128):
        super().__init__()

        backbone_channels = [40, 112, 320]  # EfficientNet-B0 feature dims

        self.backbone = PretrainedBackbone()

        # Lateral projections from backbone dims to neck_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False)
            for c in backbone_channels
        ])

        # ACS Fusion per level
        self.acs_fusions = nn.ModuleList([
            ACSFusion(neck_channels) for _ in range(3)
        ])

        # PDC-2k for post-fusion refinement (FPN path)
        self.fpn_refines = nn.ModuleList()
        self.fpn_reduces = nn.ModuleList()
        for i in range(3):
            if i < 2:
                self.fpn_refines.append(PDC2k(neck_channels * 2))
                self.fpn_reduces.append(nn.Sequential(
                    nn.Conv2d(neck_channels * 2, neck_channels, 1, bias=False),
                    nn.BatchNorm2d(neck_channels),
                ))
            else:
                self.fpn_refines.append(nn.Identity())
                self.fpn_reduces.append(nn.Identity())

        # PAN path
        self.pan_refines = nn.ModuleList()
        self.pan_reduces = nn.ModuleList()
        for i in range(3):
            if i < 2:
                self.pan_refines.append(PDC2k(neck_channels * 2))
                self.pan_reduces.append(nn.Sequential(
                    nn.Conv2d(neck_channels * 2, neck_channels, 1, bias=False),
                    nn.BatchNorm2d(neck_channels),
                ))
            else:
                self.pan_refines.append(nn.Identity())
                self.pan_reduces.append(nn.Identity())

        self.pan_downsamples = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(2)
        ])

        self.head = DSCFHead(in_channels=neck_channels, num_classes=num_classes, head_channels=neck_channels)

    def forward(self, x):
        feats = self.backbone(x)  # [C1@H/8, C2@H/16, C3@H/32]

        lateral = [conv(f) for conv, f in zip(self.lateral_convs, feats)]

        # FPN top-down
        fpn_feats = []
        prev = None
        for i in range(2, -1, -1):
            if prev is None:
                fpn_feats.insert(0, self.acs_fusions[i](lateral[i]))
            else:
                up = F.interpolate(prev, size=lateral[i].shape[2:], mode="bilinear", align_corners=False)
                fused = torch.cat([lateral[i], up], dim=1)
                refined = self.fpn_refines[i](fused)
                refined = self.fpn_reduces[i](refined)
                fpn_feats.insert(0, self.acs_fusions[i](refined))
            prev = fpn_feats[0] if i == 2 else fpn_feats[0]

        # PAN bottom-up
        pan_feats = []
        prev = None
        for i in range(3):
            if prev is None:
                pan_feats.append(fpn_feats[i])
            else:
                down = self.pan_downsamples[i - 1](prev)
                fused = torch.cat([fpn_feats[i], down], dim=1)
                refined = self.pan_refines[i - 1](fused)
                refined = self.pan_reduces[i - 1](refined)
                pan_feats.append(refined)
            prev = pan_feats[-1]

        return self.head(pan_feats)
