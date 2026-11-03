# SwiftMask CES — Final Report

## Final model

- Checkpoint: [models/final/swiftmask_u2_v2.pt](../models/final/swiftmask_u2_v2.pt)
- Architecture: compact U2-inspired residual segmentation model
- Trainable parameters: 43,689
- Final inference threshold: 0.55

## Validation results

The model was trained and evaluated with the compact architecture, then the threshold sweep selected 0.55 as the best operating point.

- Best validation threshold: 0.55
- mIoU: 0.870597
- Boundary F1: 0.667994
- Quality Q: 0.809816
- CES: 0.739036

## Benchmark results

Measured on the validation split under the assignment constraints using single-thread CPU evaluation.

- Average CPU time: 21.41 ms/image
- Model size: 290,947 bytes
- mIoU: 0.869044
- Boundary F1: 0.666302
- CES: 0.733409

## Captured-photo evaluation

- mIoU: 0.729609
- Boundary F1: 0.396546
- Average CPU time: 24.31 ms/image
- CES: 0.564096

## Visuals and summary artifacts

- Threshold sweep: [outputs/logs/threshold_optimization/threshold_comparison.csv](../outputs/logs/threshold_optimization/threshold_comparison.csv)
- Best threshold result: [outputs/logs/threshold_optimization/best_threshold.json](../outputs/logs/threshold_optimization/best_threshold.json)
- Validation benchmark: [outputs/logs/val_benchmark.json](../outputs/logs/val_benchmark.json)
- Captured evaluation: [outputs/captured_evaluation/captured_results.json](../outputs/captured_evaluation/captured_results.json)
- Validation overlays: [outputs/visualizations/validation](../outputs/visualizations/validation)

## Conclusion

This compact model meets the assignment constraints comfortably and yields strong validation performance while remaining computationally efficient. The larger earlier model was not the correct final choice because it violated the cost budget; the compact model is the proper final solution.
