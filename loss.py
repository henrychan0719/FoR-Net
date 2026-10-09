"""Training objective of FoR-Net, Eq. (8): L = L_CE + lambda1 * L_Dice + lambda2 * L_sel."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FoRNetLoss(nn.Module):
    def __init__(self, num_classes=19, ignore_index=255, lambda_dice=1.0, lambda_sel=0.5, band=2, decay=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lambda_dice = lambda_dice
        self.lambda_sel = lambda_sel
        self.band = band
        self.decay = decay

    def valid_mask(self, target):
        return (target >= 0) & (target < self.num_classes)

    def ce_loss(self, logits, target):
        valid = self.valid_mask(target)
        ce = F.cross_entropy(logits.float(), target, ignore_index=self.ignore_index, reduction="sum")
        return ce / valid.sum().clamp(min=1)

    def dice_loss(self, logits, target):
        """Multi-class soft Dice over valid pixels, averaged over classes present in the batch."""
        valid = self.valid_mask(target).unsqueeze(1)
        prob = logits.float().softmax(dim=1) * valid
        idx = target.masked_fill(~valid.squeeze(1), 0).unsqueeze(1)
        onehot = torch.zeros_like(prob).scatter_(1, idx, 1.0) * valid
        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        gsum = onehot.sum(dims)
        dice = (2.0 * inter + 1.0) / (prob.sum(dims) + gsum + 1.0)
        present = (gsum > 0).float()
        return 1.0 - (dice * present).sum() / present.sum().clamp(min=1)

    @torch.no_grad()
    def selector_target(self, target, size):
        """Soft target for the selector: class boundaries found at full resolution,
        pooled to the feature grid and dilated by `band` cells with geometric decay."""
        valid = self.valid_mask(target).unsqueeze(1)
        t = target.unsqueeze(1).float()
        big = float(self.num_classes + 1)
        mx = F.max_pool2d(torch.where(valid, t, torch.full_like(t, -1.0)), 3, 1, 1)
        mn = -F.max_pool2d(torch.where(valid, -t, torch.full_like(t, -big)), 3, 1, 1)
        edge = (valid & (mx != mn)).float()
        edge = F.adaptive_max_pool2d(edge, size)
        valid_cell = F.adaptive_max_pool2d(valid.float(), size)

        tgt, cur = edge, edge
        for k in range(1, self.band + 1):
            cur = F.max_pool2d(cur, 3, 1, 1)
            tgt = torch.maximum(tgt, cur * (self.decay ** k))
        return tgt * valid_cell, valid_cell

    def sel_loss(self, sel_logit, target):
        tgt, w = self.selector_target(target, sel_logit.shape[-2:])
        bce = F.binary_cross_entropy_with_logits(sel_logit.float(), tgt, reduction="none")
        return (bce * w).sum() / w.sum().clamp(min=1)

    def forward(self, logits, target, sel_logit):
        ce = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        sel = self.sel_loss(sel_logit, target)
        total = ce + self.lambda_dice * dice + self.lambda_sel * sel
        return total, {"ce": ce.detach(), "dice": dice.detach(), "sel": sel.detach()}
