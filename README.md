# SwiftMask-CES

## Cost-Efficient Segmentation under a Compute Budget

A lightweight foreground/background segmentation system optimized for the assignment constraints: low CPU inference time, small model size, and strong quality metrics.

## Final model

- Architecture: compact U2-inspired residual segmentation model
- Final checkpoint: [models/final/swiftmask_u2_v2.pt](models/final/swiftmask_u2_v2.pt)
- Parameters: 43,689
- Final threshold: 0.55

## Final validation results

Using the validation threshold sweep, the best operating threshold was 0.55.

- mIoU: 0.870597
- Boundary F1: 0.667994
- Quality Q: 0.809816
- CES: 0.739036

## Benchmark results

Validation benchmark under single-thread CPU evaluation:

- Average CPU time: 21.41 ms/image
- Model size: 290,947 bytes
- mIoU: 0.869044
- Boundary F1: 0.666302
- CES: 0.733409

## Captured-image evaluation

- mIoU: 0.729609
- Boundary F1: 0.396546
- Average CPU time: 24.31 ms/image
- CES: 0.564096

## Deliverables included

- Final trained checkpoint: [models/final/swiftmask_u2_v2.pt](models/final/swiftmask_u2_v2.pt)
- Final report: [reports/final_report.md](reports/final_report.md)
- Validation threshold comparison: [outputs/logs/threshold_optimization/threshold_comparison.csv](outputs/logs/threshold_optimization/threshold_comparison.csv)
- Best threshold JSON: [outputs/logs/threshold_optimization/best_threshold.json](outputs/logs/threshold_optimization/best_threshold.json)
- Validation benchmark JSON: [outputs/logs/val_benchmark.json](outputs/logs/val_benchmark.json)
- Captured evaluation JSON: [outputs/captured_evaluation/captured_results.json](outputs/captured_evaluation/captured_results.json)
- Validation visual comparisons: [outputs/visualizations/validation](outputs/visualizations/validation)
- Summary figures: [reports/figures/v2](reports/figures/v2)

## Project structure

- [src/ces](src/ces): model, dataset pipeline, evaluation utilities
- [scripts](scripts): training, validation, benchmark, and visualization scripts
- [models](models): checkpoints and final trained model
- [outputs](outputs): logs, predictions, and evaluation artifacts
- [reports](reports): final report and figures
- [tests](tests): validation and smoke tests

## Execution notes

The project is configured to use the best final threshold of 0.55 as the default inference threshold.

Typical commands:

```bash
export PYTHONPATH=src
./.venv/bin/python scripts/train_v2.py
./.venv/bin/python scripts/evaluate.py
./.venv/bin/python scripts/benchmark.py
./.venv/bin/python scripts/evaluate_captured.py
```

## Conclusion

This compact model satisfies the assignment constraints while maintaining strong segmentation quality. The earlier larger architecture was discarded because it violated the cost-efficiency requirements; the current compact model is the final, assignment-safe solution.