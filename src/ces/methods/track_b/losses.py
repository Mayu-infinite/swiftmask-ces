
import torch
import torch.nn as nn

from ces.config import DICE_WEIGHT


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Dice coefficient:

        Dice = (2 * intersection + smooth) /
               (sum(prediction) + sum(target) + smooth)

    Loss = 1 - Dice
    """

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            -1
        )

        targets = targets.reshape(
            targets.shape[0],
            -1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class SegmentationLoss(nn.Module):
    """
    Combined BCE and Soft Dice loss.

    Total loss:

        L = (1 - alpha) * BCE + alpha * DiceLoss
    """

    def __init__(self, dice_weight=DICE_WEIGHT):
        super().__init__()

        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError(
                "dice_weight must be between 0 and 1."
            )

        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        total_loss = (
            (1.0 - self.dice_weight) * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss
