
import io
import sys
import time

import torch

sys.path.insert(0, "src")
torch.set_num_threads(1)

from ces.methods.track_b.model import SwiftMaskFCN


def main():
    model = SwiftMaskFCN()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("\n===== SWIFTMASK V2 MODEL TEST =====")
    print(f"Parameters: {parameter_count:,}")

    # Test the supported image dimensions.
    model.eval()

    with torch.no_grad():
        for size in (128, 192, 256):
            sample = torch.randn(1, 3, size, size)
            output = model(sample)

            print(
                f"Input: {size}x{size} | "
                f"Output: {tuple(output.shape)}"
            )

            assert output.shape == (1, 1, size, size), (
                f"Unexpected output shape for input {size}"
            )

    print("Output shape tests: PASSED")

    # Test backward propagation.
    model.train()

    sample = torch.randn(2, 3, 128, 128)
    target = torch.rand(2, 1, 128, 128)

    prediction = model(sample)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction, target
    )

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_exist

    print(f"Backward pass: PASSED | Loss: {loss.item():.4f}")

    # Estimate the serialized state-dict size.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    print(f"Estimated state-dict size: {buffer.tell():,} bytes")

    # Preliminary single-thread CPU inference timing.
    model.eval()
    sample = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        for _ in range(3):
            model(sample)

        start_time = time.process_time()

        for _ in range(20):
            model(sample)

        elapsed = time.process_time() - start_time

    average_ms = elapsed * 1000 / 20

    print(f"CPU forward time: {average_ms:.3f} ms/image")
    print("\nSWIFTMASK V2 SMOKE TEST: PASSED")


if __name__ == "__main__":
    main()
