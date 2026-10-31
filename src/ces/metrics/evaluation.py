
import numpy as np

from ces.metrics.iou import (
    update_confusion_matrix,
    compute_iou
)

from ces.metrics.boundary_f1 import (
    boundary_f1_score
)


class SegmentationEvaluator:
    """
    Evaluates binary segmentation predictions.

    mIoU:
        Aggregated confusion matrix across all images.

    Boundary F1:
        Calculated separately for each image and averaged.
    """

    def __init__(self, boundary_tolerance=2):

        self.boundary_tolerance = boundary_tolerance

        self.confusion = np.zeros(
            (2, 2),
            dtype=np.int64
        )

        self.boundary_scores = []

        self.image_count = 0

    def update(self, ground_truth, prediction):

        ground_truth = np.asarray(ground_truth).astype(bool)
        prediction = np.asarray(prediction).astype(bool)

        if ground_truth.shape != prediction.shape:
            raise ValueError(
                "Ground truth and prediction shapes differ."
            )

        update_confusion_matrix(
            self.confusion,
            ground_truth,
            prediction
        )

        score = boundary_f1_score(
            ground_truth,
            prediction,
            tolerance=self.boundary_tolerance
        )

        self.boundary_scores.append(score)

        self.image_count += 1

    def compute(self):

        if self.image_count == 0:
            raise RuntimeError("No images were evaluated.")

        iou_results = compute_iou(self.confusion)

        mean_boundary_f1 = float(
            np.mean(self.boundary_scores)
        )

        return {
            "num_images": self.image_count,
            "object_iou": iou_results["object_iou"],
            "background_iou": iou_results["background_iou"],
            "miou": iou_results["miou"],
            "boundary_f1": mean_boundary_f1,
            "confusion_matrix": self.confusion.tolist()
        }

    def reset(self):

        self.confusion.fill(0)
        self.boundary_scores.clear()
        self.image_count = 0
