"""
Ablation study model variants for DSCF-Net (paper Section 4.4).

All variants use the CUSTOM DSCFNet backbone (no pretrained weights).
Each variant is a build_xxx(num_classes) → DSCFNet factory function.

Experiments:
    1. PDC-2k + ACS Fusion module interaction (Table 5)
    2. PDConv convolution type (Table 6)
    3. SCCFA component structure (Table 7)
    4. PDConv dilation rate combinations (Table 8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dscf_net import (
    DSCFNet, DSCFBackbone, DSCFNeck, DSCFHead,
    PDConv, PDBottleneck, PDCkk, PDC2k,
    ACSFusion, SCBlock, SCCFA, SpatialAdaptation, ChannelAdaptation,
)


# ===========================================================================
# Experiment 1: PDC-2k + ACS Fusion module interaction
# ===========================================================================

class BaselineConvBlock(nn.Module):
    """Standard residual conv block replacing PDC-2k."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return x + self.conv(x)


class PlainBackbone(nn.Module):
    """Backbone with standard conv blocks — no PDC-2k."""
    def __init__(self, embed_dims=(48, 96, 160, 224)):
        super().__init__()
        c1 = embed_dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.SiLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i in range(len(embed_dims)):
            in_ch = embed_dims[i-1] if i > 0 else c1
            out_ch = embed_dims[i]
            if i > 0:
                self.transitions.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
                ))
            self.stages.append(BaselineConvBlock(out_ch))

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.transitions[i-1](x)
            x = stage(x)
            feats.append(x)
        return feats


class PlainNeck(nn.Module):
    """Simple FPN+PAN neck — no ACS Fusion, no PDC-2k."""
    def __init__(self, in_channels, neck_channels=96):
        super().__init__()
        self.neck_channels = neck_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False) for c in in_channels
        ])

        def make_refine(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, neck_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(neck_channels), nn.SiLU(inplace=True),
            )

        self.fpn_refines = nn.ModuleList()
        for i in range(len(in_channels)):
            in_ch = neck_channels * 2 if i < len(in_channels)-1 else neck_channels
            self.fpn_refines.append(make_refine(in_ch))

        self.pan_refines = nn.ModuleList()
        for i in range(len(in_channels)):
            in_ch = neck_channels * 2 if i < len(in_channels)-1 else neck_channels
            self.pan_refines.append(make_refine(in_ch))

        self.pan_downs = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(len(in_channels)-1)
        ])

    def forward(self, features):
        feats = features[-3:]
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, feats)]

        fpn = []
        prev = None
        for i in range(len(lateral)-1, -1, -1):
            if prev is None:
                fpn.insert(0, self.fpn_refines[i](lateral[i]))
            else:
                up = F.interpolate(prev, size=lateral[i].shape[2:], mode='bilinear', align_corners=False)
                fpn.insert(0, self.fpn_refines[i](torch.cat([lateral[i], up], 1)))
            prev = fpn[0] if i == len(lateral)-1 else fpn[0]

        pan = []
        prev = None
        for i in range(len(fpn)):
            if prev is None:
                pan.append(fpn[i])
            else:
                down = self.pan_downs[i-1](prev)
                pan.append(self.pan_refines[i-1](torch.cat([fpn[i], down], 1)))
            prev = pan[-1]
        return pan


class _NeckWithPDC2k(nn.Module):
    """FPN+PAN neck with PDC2k refinement — no ACS Fusion."""
    def __init__(self, in_channels, neck_channels=96):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False) for c in in_channels
        ])
        self.fpn_refines = nn.ModuleList()
        self.fpn_reduces = nn.ModuleList()
        for i in range(len(in_channels)):
            if i < len(in_channels)-1:
                self.fpn_refines.append(PDC2k(neck_channels*2))
                self.fpn_reduces.append(nn.Sequential(
                    nn.Conv2d(neck_channels*2, neck_channels, 1, bias=False),
                    nn.BatchNorm2d(neck_channels),
                ))
            else:
                self.fpn_refines.append(nn.Identity())
                self.fpn_reduces.append(nn.Identity())
        self.pan_refines = nn.ModuleList()
        self.pan_reduces = nn.ModuleList()
        for i in range(len(in_channels)):
            if i < len(in_channels)-1:
                self.pan_refines.append(PDC2k(neck_channels*2))
                self.pan_reduces.append(nn.Sequential(
                    nn.Conv2d(neck_channels*2, neck_channels, 1, bias=False),
                    nn.BatchNorm2d(neck_channels),
                ))
            else:
                self.pan_refines.append(nn.Identity())
                self.pan_reduces.append(nn.Identity())
        self.pan_downs = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(len(in_channels)-1)
        ])
    def forward(self, features):
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, features[-3:])]
        fpn = []; prev = None
        for i in range(len(lateral)-1, -1, -1):
            if prev is None: fpn.insert(0, lateral[i])
            else:
                up = F.interpolate(prev, size=lateral[i].shape[2:], mode='bilinear', align_corners=False)
                fused = torch.cat([lateral[i], up], 1)
                r = self.fpn_refines[i](fused); r = self.fpn_reduces[i](r)
                fpn.insert(0, r)
            prev = fpn[0] if i==len(lateral)-1 else fpn[0]
        pan = []; prev = None
        for i in range(len(fpn)):
            if prev is None: pan.append(fpn[i])
            else:
                down = self.pan_downs[i-1](prev)
                fused = torch.cat([fpn[i], down], 1)
                r = self.pan_refines[i-1](fused); r = self.pan_reduces[i-1](r)
                pan.append(r)
            prev = pan[-1]
        return pan


class _NeckWithACS(nn.Module):
    """FPN+PAN neck with ACS Fusion — no PDC2k refinement."""
    def __init__(self, in_channels, neck_channels=96):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False) for c in in_channels
        ])
        self.acs_fusions = nn.ModuleList([ACSFusion(neck_channels) for _ in in_channels])
        self.fpn_refines = nn.ModuleList()
        for i in range(len(in_channels)):
            in_ch = neck_channels*2 if i<len(in_channels)-1 else neck_channels
            self.fpn_refines.append(nn.Sequential(
                nn.Conv2d(in_ch, neck_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(neck_channels), nn.SiLU(inplace=True),
            ))
        self.pan_refines = nn.ModuleList()
        for i in range(len(in_channels)):
            in_ch = neck_channels*2 if i<len(in_channels)-1 else neck_channels
            self.pan_refines.append(nn.Sequential(
                nn.Conv2d(in_ch, neck_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(neck_channels), nn.SiLU(inplace=True),
            ))
        self.pan_downs = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(len(in_channels)-1)
        ])
    def forward(self, features):
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, features[-3:])]
        fpn = []; prev = None
        for i in range(len(lateral)-1, -1, -1):
            if prev is None: fpn.insert(0, self.acs_fusions[i](lateral[i]))
            else:
                up = F.interpolate(prev, size=lateral[i].shape[2:], mode='bilinear', align_corners=False)
                fused = torch.cat([lateral[i], up], 1)
                refined = self.fpn_refines[i](fused)
                fpn.insert(0, self.acs_fusions[i](refined))
            prev = fpn[0] if i==len(lateral)-1 else fpn[0]
        pan = []; prev = None
        for i in range(len(fpn)):
            if prev is None: pan.append(fpn[i])
            else:
                down = self.pan_downs[i-1](prev)
                fused = torch.cat([fpn[i], down], 1)
                pan.append(self.pan_refines[i-1](fused))
            prev = pan[-1]
        return pan


def build_baseline(num_classes):
    """Exp1: Plain conv backbone + plain FPN neck (no ACS, no PDC2k)."""
    embed_dims = (48, 96, 160, 224)
    backbone = PlainBackbone(embed_dims)
    neck = PlainNeck(embed_dims[-3:], neck_channels=96)
    head = DSCFHead(96, num_classes, head_channels=96)
    return _assemble(backbone, neck, head)


def build_pdc2k_only(num_classes):
    """Exp1: Plain conv backbone + FPN neck with PDC2k refinement (no ACS)."""
    embed_dims = (48, 96, 160, 224)
    backbone = PlainBackbone(embed_dims)
    neck = _NeckWithPDC2k(embed_dims[-3:], neck_channels=96)
    head = DSCFHead(96, num_classes, head_channels=96)
    return _assemble(backbone, neck, head)


def build_acs_only(num_classes):
    """Exp1: Plain conv backbone + FPN neck with ACS Fusion (no PDC2k)."""
    embed_dims = (48, 96, 160, 224)
    backbone = PlainBackbone(embed_dims)
    neck = _NeckWithACS(embed_dims[-3:], neck_channels=96)
    head = DSCFHead(96, num_classes, head_channels=96)
    return _assemble(backbone, neck, head)


def build_full(num_classes):
    """Exp1: Plain conv backbone + FPN neck with ACS + PDC2k (full neck)."""
    embed_dims = (48, 96, 160, 224)
    backbone = PlainBackbone(embed_dims)
    neck = DSCFNeck(embed_dims[-3:], neck_channels=96)
    head = DSCFHead(96, num_classes, head_channels=96)
    return _assemble(backbone, neck, head)


# ===========================================================================
# Experiment 2: PDConv convolution type
# ===========================================================================

class StandardConvPDC(nn.Module):
    """PDConv replaced with a single standard 3×3 conv."""
    def __init__(self, channels, kernel_size=3, dilations=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return x + self.act(self.conv(self.bn(x)))


class DepthwiseConvPDC(nn.Module):
    """PDConv replaced with depthwise-separable conv."""
    def __init__(self, channels, kernel_size=3, dilations=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return x + self.pw(self.act(self.dw(self.bn(x))))


class _PDC2kWithConv(nn.Module):
    """Wrapper that rebuilds a DSCFNet with a custom PDConv class."""
    def __init__(self, num_classes, conv_cls, embed_dims=(48,96,160,224), neck_channels=96, num_pdckk=3):
        super().__init__()
        self.backbone = DSCFBackbone(embed_dims, num_pdckk=num_pdckk)
        self.neck = DSCFNeck(embed_dims[-3:], neck_channels=neck_channels)
        self.head = DSCFHead(neck_channels, num_classes, head_channels=neck_channels)
        _replace_pdconv(self, conv_cls)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)


def build_conv_standard(num_classes):
    return _PDC2kWithConv(num_classes, StandardConvPDC)


def build_conv_depthwise(num_classes):
    return _PDC2kWithConv(num_classes, DepthwiseConvPDC)


def build_conv_dilated(num_classes):
    return DSCFNet(num_classes, embed_dims=(48,96,160,224), neck_channels=96, num_pdckk=3)


# ===========================================================================
# Experiment 3: SCCFA component structure
# ===========================================================================

class SCCFANone(nn.Module):
    """Replace SCCFA with a plain 3×3 conv."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class SCCFASpatialOnly(nn.Module):
    """SpatialAdaptation only, no ChannelAdaptation."""
    def __init__(self, channels):
        super().__init__()
        self.spatial = SpatialAdaptation(channels)
    def forward(self, x):
        s1, s2 = self.spatial(x)
        return s1 + s2


class SCCFAChannelOnly(nn.Module):
    """ChannelAdaptation only, no SpatialAdaptation."""
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAdaptation(channels)
    def forward(self, x):
        return self.channel(x, x)


class _DSCFNetWithSCCFA(nn.Module):
    """Wrapper that rebuilds DSCFNet with a custom SCCFA class in SCBlock."""
    def __init__(self, num_classes, sccfa_cls, embed_dims=(48,96,160,224), neck_channels=96, num_pdckk=3):
        super().__init__()
        self.backbone = DSCFBackbone(embed_dims, num_pdckk=num_pdckk)
        self.neck = DSCFNeck(embed_dims[-3:], neck_channels=neck_channels)
        self.head = DSCFHead(neck_channels, num_classes, head_channels=neck_channels)
        _replace_sccfa(self, sccfa_cls)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)


def build_acs_nosccfa(num_classes):
    return _DSCFNetWithSCCFA(num_classes, SCCFANone)


def build_acs_spatial_only(num_classes):
    return _DSCFNetWithSCCFA(num_classes, SCCFASpatialOnly)


def build_acs_channel_only(num_classes):
    return _DSCFNetWithSCCFA(num_classes, SCCFAChannelOnly)


def build_acs_full(num_classes):
    return DSCFNet(num_classes, embed_dims=(48,96,160,224), neck_channels=96, num_pdckk=3)


# ===========================================================================
# Experiment 4: Dilation rate combinations
# ===========================================================================

class PDConvDilations(nn.Module):
    """PDConv with configurable dilation rates."""
    def __init__(self, channels, kernel_size=3, dilations=(2,4,6)):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size,
                      padding=d*(kernel_size//2), dilation=d,
                      groups=channels, bias=False)
            for d in dilations
        ])
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        x_bn = self.bn(x)
        out = x
        for conv in self.convs:
            out = out + self.act(conv(x_bn))
        return out


def build_dil_123(num_classes):
    return _PDC2kWithConv(num_classes, lambda ch, **kw: PDConvDilations(ch, dilations=(1,2,3), **kw))


def build_dil_246(num_classes):
    return _PDC2kWithConv(num_classes, lambda ch, **kw: PDConvDilations(ch, dilations=(2,4,6), **kw))


def build_dil_357(num_classes):
    return _PDC2kWithConv(num_classes, lambda ch, **kw: PDConvDilations(ch, dilations=(3,5,7), **kw))


# ===========================================================================
# Helpers
# ===========================================================================

def _assemble(backbone, neck, head):
    """Assemble backbone + neck + head into a single Module."""
    class Assembled(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.neck = neck
            self.head = head
        def forward(self, x):
            feats = self.backbone(x)
            feats = self.neck(feats)
            return self.head(feats)
    return Assembled()


def _replace_pdconv(module, conv_cls):
    """Recursively replace PDConv instances with conv_cls."""
    for name, child in list(module.named_children()):
        if isinstance(child, PDConv) and not isinstance(child, (StandardConvPDC, DepthwiseConvPDC, PDConvDilations)):
            new_conv = conv_cls(child.bn.num_features)
            # Copy kernel_size from first conv in original
            setattr(module, name, new_conv)
        else:
            _replace_pdconv(child, conv_cls)


def _replace_sccfa(module, sccfa_cls):
    """Recursively replace SCCFA instances in SCBlock with sccfa_cls."""
    for name, child in list(module.named_children()):
        if isinstance(child, SCBlock) and isinstance(child.sccfa, SCCFA):
            child.sccfa = sccfa_cls(child.sccfa.spatial_adapt.branch_hw.in_channels)
        else:
            _replace_sccfa(child, sccfa_cls)


# ===========================================================================
# Experiment 1 (pretrained): PDC2k + ACS Fusion with EfficientNet backbone
# ===========================================================================

class _PretrainedPlainNeck(nn.Module):
    """EfficientNet backbone + simple FPN+PAN neck (no ACS, no PDC2k)."""
    def __init__(self, in_channels=(40, 112, 320), neck_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False) for c in in_channels
        ])
        def _refine(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, neck_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(neck_channels), nn.SiLU(inplace=True),
            )
        self.fpn_refines = nn.ModuleList([
            _refine(neck_channels*2 if i<2 else neck_channels) for i in range(3)
        ])
        self.pan_refines = nn.ModuleList([
            _refine(neck_channels*2) for _ in range(2)
        ])
        self.pan_downs = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(2)
        ])
    def forward(self, features):
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, features[-3:])]
        fpn = []; prev = None
        for i in range(2, -1, -1):
            if prev is None: fpn.insert(0, self.fpn_refines[i](lateral[i]))
            else:
                up = F.interpolate(prev, size=lateral[i].shape[2:], mode='bilinear', align_corners=False)
                fpn.insert(0, self.fpn_refines[i](torch.cat([lateral[i], up], 1)))
            prev = fpn[0] if i==2 else fpn[0]
        pan = []; prev = None
        for i in range(3):
            if prev is None: pan.append(fpn[i])
            else:
                down = self.pan_downs[i-1](prev)
                pan.append(self.pan_refines[i-1](torch.cat([fpn[i], down], 1)))
            prev = pan[-1]
        return pan


class _PretrainedNeckPDC2kOnly(nn.Module):
    """EfficientNet backbone + FPN+PAN neck with PDC2k (no ACS Fusion)."""
    def __init__(self, in_channels=(40, 112, 320), neck_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False) for c in in_channels
        ])
        self.fpn_refines = nn.ModuleList()
        self.fpn_reduces = nn.ModuleList()
        for i in range(3):
            if i < 2:
                self.fpn_refines.append(PDC2k(neck_channels*2))
                self.fpn_reduces.append(nn.Sequential(
                    nn.Conv2d(neck_channels*2, neck_channels, 1, bias=False),
                    nn.BatchNorm2d(neck_channels),
                ))
            else:
                self.fpn_refines.append(nn.Identity())
                self.fpn_reduces.append(nn.Identity())
        self.pan_refines = nn.ModuleList()
        self.pan_reduces = nn.ModuleList()
        for i in range(3):
            if i < 2:
                self.pan_refines.append(PDC2k(neck_channels*2))
                self.pan_reduces.append(nn.Sequential(
                    nn.Conv2d(neck_channels*2, neck_channels, 1, bias=False),
                    nn.BatchNorm2d(neck_channels),
                ))
            else:
                self.pan_refines.append(nn.Identity())
                self.pan_reduces.append(nn.Identity())
        self.pan_downs = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(2)
        ])
    def forward(self, features):
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, features[-3:])]
        fpn = []; prev = None
        for i in range(2, -1, -1):
            if prev is None: fpn.insert(0, lateral[i])
            else:
                up = F.interpolate(prev, size=lateral[i].shape[2:], mode='bilinear', align_corners=False)
                fused = torch.cat([lateral[i], up], 1)
                refined = self.fpn_refines[i](fused)
                refined = self.fpn_reduces[i](refined)
                fpn.insert(0, refined)
            prev = fpn[0] if i==2 else fpn[0]
        pan = []; prev = None
        for i in range(3):
            if prev is None: pan.append(fpn[i])
            else:
                down = self.pan_downs[i-1](prev)
                fused = torch.cat([fpn[i], down], 1)
                refined = self.pan_refines[i-1](fused)
                refined = self.pan_reduces[i-1](refined)
                pan.append(refined)
            prev = pan[-1]
        return pan


class _PretrainedNeckACSOnly(nn.Module):
    def __init__(self, in_channels=(40, 112, 320), neck_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False) for c in in_channels
        ])
        def _refine(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, neck_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(neck_channels), nn.SiLU(inplace=True),
            )
        self.fpn_refines = nn.ModuleList([
            _refine(neck_channels*2) if i < 2 else nn.Identity() for i in range(3)
        ])
        self.acs_fusions = nn.ModuleList([
            ACSFusion(neck_channels) for _ in range(3)
        ])
        self.pan_refines = nn.ModuleList([
            _refine(neck_channels*2) for _ in range(2)
        ])
        self.pan_downs = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(2)
        ])
    def forward(self, features):
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, features[-3:])]
        fpn = []; prev = None
        for i in range(2, -1, -1):
            if prev is None:
                fpn.insert(0, self.acs_fusions[i](lateral[i]))
            else:
                up = F.interpolate(prev, size=lateral[i].shape[2:], mode='bilinear', align_corners=False)
                refined = self.fpn_refines[i](torch.cat([lateral[i], up], 1))
                fpn.insert(0, self.acs_fusions[i](refined))
            prev = fpn[0] if i == 2 else fpn[0]
        pan = []; prev = None
        for i in range(3):
            if prev is None:
                pan.append(fpn[i])
            else:
                down = self.pan_downs[i-1](prev)
                pan.append(self.pan_refines[i-1](torch.cat([fpn[i], down], 1)))
            prev = pan[-1]
        return pan


class _NoNeck(nn.Module):
    """No neck — directly GAP backbone features and classify."""
    def __init__(self, in_channels=(40, 112, 320), num_classes=30):
        super().__init__()
        self.gaps = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in in_channels])
        total_dim = sum(in_channels)
        self.fc = nn.Linear(total_dim, num_classes)
    def forward(self, features):
        pooled = [gap(f).flatten(1) for gap, f in zip(self.gaps, features)]
        return self.fc(torch.cat(pooled, dim=1))


class _NoNeckModel(nn.Module):
    """EfficientNet → GAP → FC (no neck, weakest baseline)."""
    def __init__(self, num_classes):
        super().__init__()
        from .dscf_net_pretrained import PretrainedBackbone
        self.backbone = PretrainedBackbone()
        self.classifier = _NoNeck(num_classes=num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


def _build_pretrained_base(num_classes, neck_cls, neck_channels=128):
    """Build a pretrained-backbone model with a given neck class."""
    from .dscf_net_pretrained import DSCFNetPretrained, PretrainedBackbone
    class CustomPretrained(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = PretrainedBackbone()
            self.neck = neck_cls(neck_channels=neck_channels)
            from .dscf_net import DSCFHead
            self.head = DSCFHead(neck_channels, num_classes, head_channels=neck_channels)
        def forward(self, x):
            feats = self.backbone(x)
            feats = self.neck(feats)
            return self.head(feats)
    return CustomPretrained()


def build_pretrained_no_neck(num_classes):
    """EfficientNet → GAP → FC (no neck — weakest baseline)."""
    return _NoNeckModel(num_classes)


def build_pretrained_baseline(num_classes):
    """EfficientNet + Plain Neck (no ACS, no PDC2k)."""
    return _build_pretrained_base(num_classes, _PretrainedPlainNeck, 128)


def build_pretrained_pdc2k_only(num_classes):
    """EfficientNet + Neck with PDC2k only (no ACS)."""
    return _build_pretrained_base(num_classes, _PretrainedNeckPDC2kOnly, 128)


def build_pretrained_acs_only(num_classes):
    return _build_pretrained_base(num_classes, _PretrainedNeckACSOnly, 128)


def build_pretrained_full(num_classes):
    """EfficientNet + Full Neck (ACS + PDC2k) = DSCFNetPretrained."""
    from .dscf_net_pretrained import DSCFNetPretrained
    return DSCFNetPretrained(num_classes, neck_channels=128)


# Registry for --ablation-variant argument
VARIANT_REGISTRY = {
    # Exp 1 (custom backbone)
    "baseline":          build_baseline,
    "pdc2k_only":        build_pdc2k_only,
    "acs_only":          build_acs_only,
    "full":              build_full,
    # Exp 1 (pretrained backbone)
    "pretrained_no_neck":       build_pretrained_no_neck,
    "pretrained_baseline":      build_pretrained_baseline,
    "pretrained_pdc2k_only":    build_pretrained_pdc2k_only,
    "pretrained_acs_only":      build_pretrained_acs_only,
    "pretrained_full":          build_pretrained_full,
    # Exp 2
    "conv_standard":     build_conv_standard,
    "conv_depthwise":    build_conv_depthwise,
    "conv_dilated":      build_conv_dilated,
    # Exp 3
    "acs_nosccfa":       build_acs_nosccfa,
    "acs_spatial_only":  build_acs_spatial_only,
    "acs_channel_only":  build_acs_channel_only,
    "acs_full":          build_acs_full,
    # Exp 4
    "dil_123":           build_dil_123,
    "dil_246":           build_dil_246,
    "dil_357":           build_dil_357,
}
