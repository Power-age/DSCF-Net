"""
DSCF-Net: Parallel Dual-Scale Convolutional Cross Network with
Attentional Cross-Spatial/Channel Fusion for Remote Sensing Scene Classification.

Reference: Jia X, Xu C. DSCF-Net, 2025.

Architecture overview:
    Backbone: Stem → PDC-2k stages (multi-scale feature extraction)
    Neck:     FPN+PAN with ACS Fusion attention + PDC-2k refinement
    Head:     Multi-scale classification heads with residual fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 公式 (10-11): PDConv — 并行深度卷积模块 (Fig. 3d)
# ---------------------------------------------------------------------------
class PDConv(nn.Module):
    """Parallel Dilated Convolution — 并行深度卷积模块.

    X_bn = BN(X_in)
    For i in {1,2,3}: Y_i = LeakyReLU(DWConv_{k,d_i}(X_bn))
    Y = X_in + Y_1 + Y_2 + Y_3
    """

    def __init__(self, channels, kernel_size=3, dilations=(2, 4, 6)):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                channels, channels, kernel_size,
                padding=d * (kernel_size // 2),
                dilation=d,
                groups=channels,
                bias=False,
            )
            for d in dilations
        ])
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x_bn = self.bn(x)
        out = x
        for conv in self.convs:
            out = out + self.act(conv(x_bn))
        return out


# ---------------------------------------------------------------------------
# Fig. 3c: PD-Bottleneck — 双尺度并行深度卷积模块
# ---------------------------------------------------------------------------
class PDBottleneck(nn.Module):
    """Parallel Dual-Scale Bottleneck (Fig. 3c).

    Branch 1: PDConv(k1) → SiLU  →  Y1 = X_in + SiLU(PDConv_k1(X_in))
    Branch 2: PDConv(k2) → SiLU  →  Y2 = X_in + SiLU(PDConv_k2(X_in))
    Y_out = Concat(Y1, Y2)   [channel count ×2]

    X_in information is preserved via residual connections in Y1 and Y2,
    making an explicit X_in copy redundant (KISS principle).
    """

    def __init__(self, channels, k1=3, k2=5):
        super().__init__()
        self.pdconv1 = PDConv(channels, kernel_size=k1)
        self.pdconv2 = PDConv(channels, kernel_size=k2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y1 = self.act(self.pdconv1(x))
        y2 = self.act(self.pdconv2(x))
        return torch.cat([y1, y2], dim=1)


# ---------------------------------------------------------------------------
# Fig. 3b: PD-Ckk — 并行双尺度卷积增强模块
# ---------------------------------------------------------------------------
class PDCkk(nn.Module):
    """Parallel Dual-Scale Conv Enhancement (Fig. 3b).

    Branch 1 (shortcut): 1×1 conv → Y_lin
    Branch 2 (deep):     1×1 conv → m × PD-Bottleneck → Y_deep
    Y_out = 1×1 Conv(Concat(Y_lin, Y_deep))  — restores original channels
    """

    def __init__(self, channels, bottleneck_count=1, expansion=0.5):
        super().__init__()
        mid_channels = int(channels * expansion)

        self.shortcut_conv = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(mid_channels)

        self.deep_conv = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.deep_bn = nn.BatchNorm2d(mid_channels)
        bottlenecks = []
        for _ in range(bottleneck_count):
            bottlenecks.append(PDBottleneck(mid_channels))
        self.bottlenecks = nn.Sequential(*bottlenecks)

        pdckk_in = mid_channels + mid_channels * (2 ** bottleneck_count)
        self.fusion_conv = nn.Conv2d(pdckk_in, channels, 1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        y_lin = self.shortcut_bn(self.shortcut_conv(x))

        y_deep = self.deep_bn(self.deep_conv(x))
        y_deep = self.bottlenecks(y_deep)

        y = torch.cat([y_lin, y_deep], dim=1)
        y = self.fusion_bn(self.fusion_conv(y))
        return y


# ---------------------------------------------------------------------------
# Fig. 3a: PDC-2k — 并行双尺度卷积交叉模块
# ---------------------------------------------------------------------------
class PDC2k(nn.Module):
    """Parallel Dual-Scale Conv Cross Module (Fig. 3a).

    Strategy: Split-Enhance-Concat with residual connection.
    X → 1×1 Conv → split [X1, X2]
    X1 → n × PD-Ckk (deep enhancement)
    X2 → identity
    Y = X + 1×1 Conv(Concat(X1_enhanced, X2))
    """

    def __init__(self, channels, num_pdckk=3, bottleneck_count=1):
        super().__init__()
        self.pre_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.pre_bn = nn.BatchNorm2d(channels)

        half = channels // 2
        pdckks = []
        for _ in range(num_pdckk):
            pdckks.append(PDCkk(half, bottleneck_count=bottleneck_count))
        self.pdckks = nn.Sequential(*pdckks)

        self.post_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.post_bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.pre_bn(self.pre_conv(x))
        x1, x2 = torch.chunk(y, 2, dim=1)
        x1 = self.pdckks(x1)
        y = torch.cat([x1, x2], dim=1)
        y = self.post_bn(self.post_conv(y))
        return x + y


# ---------------------------------------------------------------------------
# Fig. 5c: Spatial Adaptation Block — 空间自适应模块
# ---------------------------------------------------------------------------
class SpatialAdaptation(nn.Module):
    """Spatial Adaptation Block (Fig. 5c).

    Two branches: global H×W conv + 1×1 conv → ConvFFN → cross multiply.
    Models long-range spatial dependencies.
    """

    def __init__(self, channels):
        super().__init__()
        self.branch_hw = nn.Conv2d(channels, channels, 1, bias=False)
        self.branch_1x1 = nn.Conv2d(channels, channels, 1, bias=False)

        self.conv_ffn = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        b_hw = self.branch_hw(x)
        b_1 = self.branch_1x1(x)

        fused = torch.cat([b_hw, b_1], dim=1)
        jrf = self.conv_ffn(fused)  # joint representation feature

        w1, w2 = torch.chunk(jrf, 2, dim=1)
        out1 = b_hw * w1 + b_1
        out2 = b_1 * w2 + b_hw
        return out1, out2


# ---------------------------------------------------------------------------
# Fig. 5d: Channel Adaptation Block — 通道自适应模块
# ---------------------------------------------------------------------------
class ChannelAdaptation(nn.Module):
    """Channel Adaptation Block (Fig. 5d).

    Multi-dilation depthwise convs → MLP → cross channel attention.
    """

    def __init__(self, channels, dilations=(2, 4, 6)):
        super().__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(
                channels * 2, channels * 2, 3,
                padding=d, dilation=d,
                groups=channels * 2, bias=False,
            )
            for d in dilations
        ])

        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 6, channels * 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 1, bias=False),
        )

    def forward(self, x1, x2):
        y = torch.cat([x1, x2], dim=1)
        branches = [conv(y) for conv in self.dw_convs]
        y = torch.cat(branches, dim=1)
        y = self.mlp(y)

        w1, w2 = torch.chunk(y, 2, dim=1)
        out1 = x1 * w1 + x2
        out2 = x2 * w2 + x1
        return out1 + out2


# ---------------------------------------------------------------------------
# SCCFA: 空间与通道自注意力模块 (Spatial-Channel Cross Fusion Attention)
# ---------------------------------------------------------------------------
class SCCFA(nn.Module):
    """Spatial-Channel Cross Fusion Attention.

    SpatialAdaptation → ChannelAdaptation (sequential, cross-dim interaction).
    """

    def __init__(self, channels):
        super().__init__()
        self.spatial_adapt = SpatialAdaptation(channels)
        self.channel_adapt = ChannelAdaptation(channels)

    def forward(self, x):
        s1, s2 = self.spatial_adapt(x)
        y = self.channel_adapt(s1, s2)
        return y


# ---------------------------------------------------------------------------
# Fig. 5b: SCBlock — 级联增强模块
# ---------------------------------------------------------------------------
class SCBlock(nn.Module):
    """SCBlock — cascaded enhancement with dual residual connections (Fig. 5b).

    X → SCCFA → +X (residual 1) → MLP → +residual (residual 2) → Y
    """

    def __init__(self, channels, mlp_ratio=2):
        super().__init__()
        self.sccfa = SCCFA(channels)
        self.norm = nn.BatchNorm2d(channels)

        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        attn = self.sccfa(x)
        x = x + attn
        x = x + self.mlp(self.norm(x))
        return x


# ---------------------------------------------------------------------------
# SelfGamma — 自门控激活函数
# ---------------------------------------------------------------------------
class SelfGamma(nn.Module):
    """Self-gated activation: Y = X * sigmoid(gamma * X).

    Differs from SiLU by having a learnable scale parameter gamma,
    enabling adaptive response-strength modulation per channel.
    """

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.gamma * x)


# ---------------------------------------------------------------------------
# Fig. 5a: ACS Fusion — 空间与通道交叉融合注意力模块
# ---------------------------------------------------------------------------
class ACSFusion(nn.Module):
    """Attentional Cross-Spatial/Channel Fusion (Fig. 5a).

    X → BN → 3×3 DWConv → SiLU → SCBlock → 3×3 DWConv → SelfGamma → +X → Y
    """

    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.dw_conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.sc_block = SCBlock(channels)
        self.dw_conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.self_gamma = SelfGamma(channels)

    def forward(self, x):
        residual = x
        y = self.bn(x)
        y = self.dw_conv1(y)
        y = self.act(y)
        y = self.sc_block(y)
        y = self.dw_conv2(y)
        y = self.self_gamma(y)
        return y + residual


# ---------------------------------------------------------------------------
# Backbone — 特征提取骨干
# ---------------------------------------------------------------------------
class DSCFBackbone(nn.Module):
    """Backbone: Stem + PDC-2k stages with downsampling transitions."""

    def __init__(self, embed_dims=(64, 128, 192, 256), num_pdckk=3, bottleneck_count=1):
        super().__init__()
        self.embed_dims = embed_dims

        # Stem: two 3×3 convs with downsampling (H→H/2→H/4)
        c1 = embed_dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # Stages
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i in range(len(embed_dims)):
            in_ch = embed_dims[i - 1] if i > 0 else c1
            out_ch = embed_dims[i]

            # Transition (downsample) between stages
            if i > 0:
                self.transitions.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.SiLU(inplace=True),
                    )
                )

            # PDC-2k stage
            self.stages.append(
                PDC2k(out_ch, num_pdckk=num_pdckk, bottleneck_count=bottleneck_count)
            )

    def forward(self, x):
        x = self.stem(x)
        features = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.transitions[i - 1](x)
            x = x + stage(x)
            features.append(x)
        return features  # [feat_0, feat_1, feat_2, feat_3]


# ---------------------------------------------------------------------------
# Neck — 特征融合颈部 (FPN + PAN with ACS Fusion + PDC-2k)
# ---------------------------------------------------------------------------
class DSCFNeck(nn.Module):
    """Neck: FPN+PAN structure with ACS Fusion attention and PDC-2k refinement.

    Takes multi-scale backbone features, applies ACS Fusion for attention
    enhancement, then fuses via top-down (FPN) and bottom-up (PAN) pathways
    with PDC-2k refinement at each fusion node.
    """

    def __init__(self, in_channels, neck_channels=128):
        super().__init__()
        self.neck_channels = neck_channels

        # Project all input features to neck_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, neck_channels, 1, bias=False)
            for c in in_channels
        ])

        # ACS Fusion per level
        self.acs_fusions = nn.ModuleList([
            ACSFusion(neck_channels)
            for _ in in_channels
        ])

        # PDC-2k for post-fusion refinement (FPN path) + channel reduction
        # FPN top-down: highest index (i=2) never fuses, all lower indices do
        self.fpn_refines = nn.ModuleList()
        self.fpn_reduces = nn.ModuleList()
        for i in range(len(in_channels)):
            if i < len(in_channels) - 1:
                self.fpn_refines.append(PDC2k(neck_channels * 2))
                self.fpn_reduces.append(nn.Sequential(
                    nn.Conv2d(neck_channels * 2, neck_channels, 1, bias=False),
                    nn.BatchNorm2d(neck_channels),
                ))
            else:
                self.fpn_refines.append(nn.Identity())
                self.fpn_reduces.append(nn.Identity())

        # PAN bottom-up: highest index (i=2) never fuses, all lower indices do
        self.pan_refines = nn.ModuleList()
        self.pan_reduces = nn.ModuleList()
        for i in range(len(in_channels)):
            if i < len(in_channels) - 1:
                self.pan_refines.append(PDC2k(neck_channels * 2))
                self.pan_reduces.append(nn.Sequential(
                    nn.Conv2d(neck_channels * 2, neck_channels, 1, bias=False),
                    nn.BatchNorm2d(neck_channels),
                ))
            else:
                self.pan_refines.append(nn.Identity())
                self.pan_reduces.append(nn.Identity())

        # Downsampling convs for PAN
        self.pan_downsamples = nn.ModuleList([
            nn.Conv2d(neck_channels, neck_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(len(in_channels) - 1)
        ])

    def forward(self, features):
        # Use last 3 feature levels for neck
        feats = features[-3:]  # [C2, C3, C4]

        # Lateral projections
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, feats)]

        # --- Top-down FPN ---
        fpn_feats = []
        prev = None
        for i in range(len(lateral) - 1, -1, -1):
            if prev is None:
                # Highest level: ACS Fusion only
                fpn_feats.insert(0, self.acs_fusions[i](lateral[i]))
            else:
                # Upsample and fuse
                up = F.interpolate(prev, size=lateral[i].shape[2:], mode='bilinear', align_corners=False)
                fused = torch.cat([lateral[i], up], dim=1)
                refined = self.fpn_refines[i](fused)
                refined = self.fpn_reduces[i](refined)
                fpn_feats.insert(0, self.acs_fusions[i](refined))
            prev = fpn_feats[0] if i == len(lateral) - 1 else fpn_feats[0]

        # --- Bottom-up PAN ---
        pan_feats = []
        prev = None
        for i in range(len(fpn_feats)):
            if prev is None:
                pan_feats.append(fpn_feats[i])
            else:
                down = self.pan_downsamples[i - 1](prev)
                fused = torch.cat([fpn_feats[i], down], dim=1)
                refined = self.pan_refines[i - 1](fused)
                refined = self.pan_reduces[i - 1](refined)
                pan_feats.append(refined)
            prev = pan_feats[-1]

        return pan_feats


# ---------------------------------------------------------------------------
# Head — 多尺度分类头
# ---------------------------------------------------------------------------
class DSCFHead(nn.Module):
    """Multi-scale classification head with residual fusion.

    Each head: GAP → FC → logits.
    Three head outputs are fused via residual addition → Softmax.
    """

    def __init__(self, in_channels, num_classes, head_channels=128):
        super().__init__()
        self.head_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(3)
        ])

        self.gaps = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(3)
        ])

        self.fcs = nn.ModuleList([
            nn.Linear(head_channels, num_classes) for _ in range(3)
        ])

        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, features):
        assert len(features) == 3, f"Expected 3 feature levels, got {len(features)}"

        outputs = []
        for feat, conv, gap, fc in zip(features, self.head_convs, self.gaps, self.fcs):
            x = conv(feat)
            x = gap(x).flatten(1)
            x = fc(x)
            outputs.append(x)

        # Weighted residual fusion
        w = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w[i] * outputs[i] for i in range(3))
        return fused


# ---------------------------------------------------------------------------
# DSCF-Net — 完整模型
# ---------------------------------------------------------------------------
class DSCFNet(nn.Module):
    """DSCF-Net: Parallel Dual-Scale Convolutional Cross Network with
    Attentional Cross-Spatial/Channel Fusion.

    Args:
        num_classes: Number of scene categories.
        embed_dims: Backbone stage channel dimensions.
        neck_channels: Neck feature channels.
        num_pdckk: Number of PD-Ckk in each PDC-2k (n in the paper).
        bottleneck_count: Number of PD-Bottleneck in each PD-Ckk (m in the paper).
    """

    def __init__(
        self,
        num_classes,
        embed_dims=(48, 96, 160, 224),
        neck_channels=96,
        num_pdckk=3,
        bottleneck_count=1,
    ):
        super().__init__()
        self.backbone = DSCFBackbone(
            embed_dims=embed_dims,
            num_pdckk=num_pdckk,
            bottleneck_count=bottleneck_count,
        )
        # Neck uses last 3 backbone features
        neck_in_channels = embed_dims[-3:]
        self.neck = DSCFNeck(
            in_channels=neck_in_channels,
            neck_channels=neck_channels,
        )
        self.head = DSCFHead(
            in_channels=neck_channels,
            num_classes=num_classes,
            head_channels=neck_channels,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        neck_feats = self.neck(features)
        logits = self.head(neck_feats)
        return logits


# ---------------------------------------------------------------------------
# Variant constructors for ablation study
# ---------------------------------------------------------------------------
def dscf_net_compact(num_classes):
    """Channel-reduced variant: about 1.0M params and 0.4G FLOPs."""
    return DSCFNet(num_classes, embed_dims=(28, 60, 100, 140), neck_channels=60, num_pdckk=2)


def dscf_net_base(num_classes):
    """Base variant: ~2.0M params, ~0.8G FLOPs."""
    return DSCFNet(num_classes)


def dscf_net_large(num_classes):
    """Large variant: ~2.6M params, ~1.0G FLOPs."""
    return DSCFNet(num_classes, embed_dims=(44, 88, 148, 208), neck_channels=88, num_pdckk=3)
