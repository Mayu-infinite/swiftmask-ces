import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ces.methods.track_b.model import SwiftMaskFCN


def test_model_produces_expected_shape():
    model = SwiftMaskFCN()
    model.eval()

    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 1, 128, 128)


def test_model_backward_pass_runs():
    model = SwiftMaskFCN()
    model.train()

    sample = torch.randn(2, 3, 128, 128)
    target = torch.rand(2, 1, 128, 128)
    prediction = model(sample)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target)
    loss.backward()

    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)
