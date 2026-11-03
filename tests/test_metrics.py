import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ces.metrics.ces_score import calculate_ces_score


def test_ces_score_matches_formula():
    miou = 0.7
    boundary_f1 = 0.5
    cpu_time_ms = 20.0
    model_size_bytes = 0.5 * 1024 * 1024

    result = calculate_ces_score(miou, boundary_f1, cpu_time_ms, model_size_bytes)

    quality = 0.70 * miou + 0.30 * boundary_f1
    expected = quality * (
        1.0 - 0.20 * (cpu_time_ms / 50.0) - 0.10 * min(1.0, model_size_bytes / (4 * 1024 * 1024))
    )

    assert math.isclose(result["quality_score"], quality)
    assert math.isclose(result["ces_score"], expected)


def test_ces_zero_when_cpu_time_exceeds_budget():
    result = calculate_ces_score(0.8, 0.8, 60.0, 1000)
    assert result["ces_score"] == 0.0
    assert result["quality_score"] == 0.8 * 0.70 + 0.8 * 0.30
