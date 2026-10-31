
import time


def measure_cpu_time(function, runs=20, warmup=True):
    """
    Measure average CPU process time per inference.

    The warmup run is excluded from timing.

    Returns:
        {
            "total_cpu_seconds": ...,
            "average_cpu_ms": ...,
            "runs": ...
        }
    """

    if runs < 1:
        raise ValueError("runs must be at least 1.")

    if warmup:
        function()

    start = time.process_time()

    for _ in range(runs):
        function()

    end = time.process_time()

    total_cpu_seconds = end - start

    average_cpu_ms = (
        total_cpu_seconds / runs
    ) * 1000.0

    return {
        "total_cpu_seconds": float(total_cpu_seconds),
        "average_cpu_ms": float(average_cpu_ms),
        "runs": runs
    }


def measure_dataset_cpu_time(functions, warmup_function=None):
    """
    Measure total CPU process time over a collection of
    already prepared inference functions.

    Image reading and ground-truth loading must happen
    outside the measured functions.

    Returns:
        Total CPU seconds and average CPU milliseconds/image.
    """

    functions = list(functions)

    if not functions:
        raise ValueError("No inference functions supplied.")

    if warmup_function is not None:
        warmup_function()

    start = time.process_time()

    for function in functions:
        function()

    end = time.process_time()

    total_cpu_seconds = end - start

    average_cpu_ms = (
        total_cpu_seconds / len(functions)
    ) * 1000.0

    return {
        "total_cpu_seconds": float(total_cpu_seconds),
        "average_cpu_ms": float(average_cpu_ms),
        "num_images": len(functions)
    }
