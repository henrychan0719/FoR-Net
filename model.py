"""FoR-Net: Focus-on-Regions Network for Semantic Segmentation (arXiv:2605.02764)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=1, dilation=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class Selector(nn.Module):
    """Importance logit phi(F); S = sigmoid(phi(F)), Eq. (1)."""

    def __init__(self, c):
        super().__init__()
        self.body = ConvBNReLU(c, c // 2, 3)
        self.head = nn.Conv2d(c // 2, 1, 1)

    def forward(self, f):
        h = self.body(f)
        # keep the logit in fp32, otherwise Top-K sees many ties under AMP
        with torch.autocast(device_type=f.device.type, enabled=False):
            return self.head(h.float())


class GlobalContext(nn.Module):
    """F_ctx = F + psi(GAP(F)), Eq. (4)."""

    def __init__(self, c, reduction=4):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Conv2d(c, c // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // reduction, c, 1),
        )

    def forward(self, f):
        return f + self.psi(F.adaptive_avg_pool2d(f, 1))


class FoRNet(nn.Module):
    """ResNet-101 (output stride 8) with selector-driven Top-K multi-scale reasoning.

    forward(x) returns (logits, sel_logit, masks):
        logits     B x num_classes x H x W, upsampled to the input size
        sel_logit  B x 1 x h x w, selector logit on the stride-8 feature map
        masks      list of B x 1 x h x w binary Top-K masks, one per branch
    """

    def __init__(self, num_classes=19, channels=256, topk_ratios=(0.10, 0.20, 0.30, 0.40),
                 kernels=(1, 3, 5, 7), dilations=(1, 1, 1, 16), low_channels=48, pretrained=True):
        super().__init__()
        assert len(topk_ratios) == len(kernels) == len(dilations)
        self.topk_ratios = tuple(topk_ratios)

        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        res = models.resnet101(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = res.layer1, res.layer2, res.layer3, res.layer4

        self.trans = ConvBNReLU(2048, channels, 1)
        self.selector = Selector(channels)
        self.context = GlobalContext(channels)
        # C_i: 1x1 / 3x3 / 5x5 / dilated 7x7 (Table I)
        self.branches = nn.ModuleList(
            ConvBNReLU(channels, channels, k, d) for k, d in zip(kernels, dilations))

        # decoder D(F_agg, F_low), Eq. (7); F_low = layer1 (stride 4)
        self.low_proj = ConvBNReLU(256, low_channels, 1)
        self.decoder = nn.Sequential(
            ConvBNReLU(channels + low_channels, channels, 3),
            ConvBNReLU(channels, channels, 3),
        )
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    @torch.no_grad()
    def topk_masks(self, sel_logit):
        """M_i = TopK(S), Eq. (2). Exactly K locations per image; masks are nested."""
        B, _, H, W = sel_logit.shape
        n = H * W
        ks = [min(n, max(1, round(r * n))) for r in self.topk_ratios]
        flat = sel_logit.float().reshape(B, n)
        idx = flat.topk(max(ks), dim=1, sorted=True).indices
        masks = []
        for k in ks:
            m = torch.zeros_like(flat).scatter_(1, idx[:, :k], 1.0)
            masks.append(m.view(B, 1, H, W))
        return masks

    def forward(self, x):
        size = x.shape[-2:]
        x = self.stem(x)
        f_low = self.layer1(x)
        f = self.layer4(self.layer3(self.layer2(f_low)))
        f = self.trans(f)

        sel_logit = self.selector(f)
        s = torch.sigmoid(sel_logit)
        masks = self.topk_masks(sel_logit)
        f_ctx = self.context(f)

        f_agg = f
        for branch, m in zip(self.branches, masks):
            # hard mask in the forward pass, straight-through gradient to S
            m_ste = (m + (s - s.detach())).to(f_ctx.dtype)
            f_agg = f_agg + branch(f_ctx * m_ste)  # Eq. (5), (6)

        low = self.low_proj(f_low)
        y = F.interpolate(f_agg, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = self.classifier(self.decoder(torch.cat([y, low], dim=1)))
        y = F.interpolate(y.float(), size=size, mode="bilinear", align_corners=False)
        return y, sel_logit, masks
